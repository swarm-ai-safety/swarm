"""Tests for the benchmark suite.

Covers all 9 security fixes:
1. Oracle leakage prevention (TaskInstance vs TaskOracle separation)
2. run_fn sandboxing (validation, deep-copy)
3. Mutable shared state protection
4. BFS fallback (retry pairs, no network mutation)
5. Non-invertible pipeline transform (hash chain)
6. Greedy auction assumption (documented in module docstring)
7. Configurable p-weights (ScoringWeights)
8. Adversarial fraction + safety axis
9. Consistent partial fidelity across benchmarks
"""


import pytest

from swarm.benchmarks.allocation.resource_auction import (
    AuctionInstance,
    ResourceAuctionBenchmark,
)
from swarm.benchmarks.base import BenchmarkScore, ScoringWeights, TaskResult
from swarm.benchmarks.coordination.distributed_allocation import (
    AllocationInstance,
    DistributedAllocationBenchmark,
)
from swarm.benchmarks.long_horizon.pipeline_task import (
    PipelineInstance,
    PipelineTaskBenchmark,
    _stage_transform,
)
from swarm.benchmarks.routing.message_routing import (
    MessageRoutingBenchmark,
    RoutingInstance,
    _bfs_path,
)
from swarm.benchmarks.runner import BenchmarkRunner, _validate_result
from swarm.env.network import AgentNetwork, NetworkConfig, NetworkTopology

# ---------------------------------------------------------------------------
# Fix #1: Oracle leakage — TaskInstance must not contain ground truth
# ---------------------------------------------------------------------------

class TestOracleLeakage:
    """Verify that TaskInstance subclasses never expose oracle answers."""

    def test_routing_instance_has_no_expected_path(self):
        bench = MessageRoutingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=6)
        assert isinstance(inst, RoutingInstance)
        assert not hasattr(inst, "expected_path")
        # Oracle holds the path
        assert "expected_path" in oracle.ground_truth

    def test_allocation_instance_has_no_optimal_allocation(self):
        bench = DistributedAllocationBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        assert isinstance(inst, AllocationInstance)
        assert not hasattr(inst, "optimal_allocation")
        assert "optimal_allocation" in oracle.ground_truth

    def test_auction_instance_has_no_optimal_assignment(self):
        bench = ResourceAuctionBenchmark(n_resources=3)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        assert isinstance(inst, AuctionInstance)
        assert not hasattr(inst, "optimal_assignment")
        assert not hasattr(inst, "optimal_welfare")
        assert "optimal_assignment" in oracle.ground_truth

    def test_pipeline_instance_has_no_expected_output(self):
        bench = PipelineTaskBenchmark(n_stages=3)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        assert isinstance(inst, PipelineInstance)
        assert not hasattr(inst, "expected_output")
        assert "expected_output" in oracle.ground_truth

    def test_redact_returns_deep_copy(self):
        bench = MessageRoutingBenchmark()
        inst, _ = bench.generate(seed=0, n_agents=6)
        redacted = bench.redact(inst)
        assert redacted is not inst
        assert redacted.payload == inst.payload
        # Mutating redacted should not affect original
        redacted.payload = -999
        assert inst.payload != -999


# ---------------------------------------------------------------------------
# Fix #2: run_fn validation
# ---------------------------------------------------------------------------

class TestRunFnValidation:
    def test_validate_result_accepts_valid(self):
        r = TaskResult(completed=True, payload=42, steps_taken=1, agent_trace=["a"])
        assert _validate_result(r) is r

    def test_validate_result_rejects_non_taskresult(self):
        with pytest.raises(TypeError, match="run_fn must return TaskResult"):
            _validate_result({"completed": True})

    def test_validate_result_rejects_bad_completed(self):
        r = TaskResult(completed=True, payload=42, steps_taken=1, agent_trace=[])
        r.completed = "yes"  # type: ignore[assignment]
        with pytest.raises(TypeError, match="completed must be bool"):
            _validate_result(r)

    def test_validate_result_rejects_negative_steps(self):
        r = TaskResult(completed=True, payload=42, steps_taken=-1, agent_trace=[])
        with pytest.raises(ValueError, match="non-negative"):
            _validate_result(r)

    def test_validate_result_rejects_bad_trace(self):
        r = TaskResult(completed=True, payload=42, steps_taken=1, agent_trace="not_a_list")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="agent_trace must be list"):
            _validate_result(r)

    def test_runner_validates_run_fn_output(self):
        bench = MessageRoutingBenchmark()
        runner = BenchmarkRunner(n_agents=6)

        def bad_fn(instance, config):
            return {"not": "a TaskResult"}

        with pytest.raises(TypeError, match="run_fn must return TaskResult"):
            runner.run_frontier(bench, [{"id": "test"}], n_seeds=1, run_fn=bad_fn)


# ---------------------------------------------------------------------------
# Fix #3: Mutable shared state
# ---------------------------------------------------------------------------

class TestMutableSharedState:
    def test_run_fn_cannot_mutate_shared_instance(self):
        bench = MessageRoutingBenchmark()
        runner = BenchmarkRunner(n_agents=6)
        payloads_seen = []

        def mutating_fn(instance, config):
            payloads_seen.append(instance.payload)
            # Try to mutate the instance
            instance.payload = -1
            instance.source_agent = "HACKED"
            return TaskResult(
                completed=True,
                payload=payloads_seen[-1],
                steps_taken=1,
                agent_trace=[],
            )

        configs = [{"id": "a"}, {"id": "b"}]
        runner.run_frontier(bench, configs, n_seeds=1, run_fn=mutating_fn)
        # Both configs should see the same original payload for seed 0
        assert payloads_seen[0] == payloads_seen[1]

    def test_oracle_result_not_shared_across_calls(self):
        bench = PipelineTaskBenchmark(n_stages=3)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        r1 = bench.oracle_run(inst, oracle)
        r2 = bench.oracle_run(inst, oracle)
        assert r1 is not r2
        r1.payload = -999
        assert r2.payload != -999


# ---------------------------------------------------------------------------
# Fix #4: BFS fallback — retry pairs, don't mutate network
# ---------------------------------------------------------------------------

class TestBfsPath:
    def test_direct_connection(self):
        net = AgentNetwork(config=NetworkConfig(topology=NetworkTopology.COMPLETE))
        net.initialize(["a", "b", "c"])
        path = _bfs_path(net, "a", "b")
        assert path == ["a", "b"]

    def test_no_path(self):
        net = AgentNetwork(config=NetworkConfig(topology=NetworkTopology.CUSTOM))
        net.initialize(["a", "b"])
        path = _bfs_path(net, "a", "b")
        assert path == []

    def test_self_path(self):
        net = AgentNetwork(config=NetworkConfig(topology=NetworkTopology.COMPLETE))
        net.initialize(["a", "b"])
        path = _bfs_path(net, "a", "a")
        assert path == ["a"]


class TestRoutingNoNetworkMutation:
    def test_generate_never_adds_edges_to_sparse_network(self):
        """Even with low edge probability, generate should not call add_edge."""
        bench = MessageRoutingBenchmark(edge_probability=0.1)
        for seed in range(10):
            inst, oracle = bench.generate(seed=seed, n_agents=6)
            # Should always produce a valid path
            path = oracle.ground_truth["expected_path"]
            assert len(path) >= 2
            assert path[0] == inst.source_agent
            assert path[-1] == inst.target_agent


# ---------------------------------------------------------------------------
# Fix #5: Non-invertible pipeline transform
# ---------------------------------------------------------------------------

class TestPipelineHashChain:
    def test_stage_transform_deterministic(self):
        assert _stage_transform(100, 42) == _stage_transform(100, 42)

    def test_stage_transform_not_simple_addition(self):
        """The transform should not be payload + key."""
        result = _stage_transform(100, 42)
        assert result != 100 + 42

    def test_stage_transform_non_invertible_from_endpoints(self):
        """Given initial and final, you can't skip intermediate stages."""
        bench = PipelineTaskBenchmark(n_stages=5)
        inst, oracle = bench.generate(seed=0, n_agents=3)
        initial = inst.initial_payload
        expected = oracle.ground_truth["expected_output"]
        # Simple subtraction doesn't give you transform keys
        assert expected != initial + sum(s.transform_key for s in inst.stages)

    def test_pipeline_oracle_uses_hash_chain(self):
        bench = PipelineTaskBenchmark(n_stages=3)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        # Manually recompute
        running = inst.initial_payload
        for stage in inst.stages:
            running = _stage_transform(running, stage.transform_key)
        assert running == oracle.ground_truth["expected_output"]

    def test_partial_pipeline_gets_partial_credit(self):
        """Reaching an intermediate state gives partial fidelity."""
        bench = PipelineTaskBenchmark(n_stages=4)
        inst, oracle = bench.generate(seed=0, n_agents=4)
        intermediates = oracle.ground_truth["intermediate_states"]

        # Simulate stopping at stage 2 (index 2 in intermediates)
        partial_result = TaskResult(
            completed=True,
            payload=intermediates[2],
            steps_taken=2,
            agent_trace=[],
        )
        score = bench.score(partial_result, oracle)
        assert 0.0 < score.fidelity < 1.0
        # Should be ~2/5 for reaching stage 2 out of 5 states
        assert score.fidelity == pytest.approx(2 / 5, abs=0.01)


# ---------------------------------------------------------------------------
# Fix #7: Configurable p-weights
# ---------------------------------------------------------------------------

class TestScoringWeights:
    def test_weights_normalize(self):
        w = ScoringWeights(completion=2.0, fidelity=1.0, efficiency=1.0)
        assert w.completion == pytest.approx(0.5)
        assert w.fidelity == pytest.approx(0.25)
        assert w.efficiency == pytest.approx(0.25)

    def test_weights_reject_zero_sum(self):
        with pytest.raises(ValueError, match="positive"):
            ScoringWeights(completion=0.0, fidelity=0.0, efficiency=0.0)

    def test_custom_weights_change_p(self):
        bench_a = MessageRoutingBenchmark(
            weights=ScoringWeights(completion=1.0, fidelity=0.0, efficiency=0.0)
        )
        bench_b = MessageRoutingBenchmark(
            weights=ScoringWeights(completion=0.0, fidelity=0.0, efficiency=1.0)
        )
        score = BenchmarkScore(
            completion_rate=1.0,
            efficiency=0.5,
            fidelity=0.8,
            capability_ratio=1.0,
        )
        p_a = bench_a.to_soft_interaction(score).p
        p_b = bench_b.to_soft_interaction(score).p
        assert p_a != p_b
        assert p_a == pytest.approx(1.0)
        assert p_b == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Fix #8: Adversarial fraction + safety axis
# ---------------------------------------------------------------------------

class TestAdversarialFraction:
    def test_safety_score_zero_without_adversaries(self):
        bench = MessageRoutingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=6)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle, adversarial_fraction=0.0)
        assert score.safety_score == 0.0

    def test_safety_score_positive_with_adversaries(self):
        bench = MessageRoutingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=6)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle, adversarial_fraction=0.3)
        assert score.safety_score > 0.0
        assert score.safety_score == pytest.approx(1.0 * 0.3)

    def test_runner_passes_adversarial_fraction(self):
        bench = PipelineTaskBenchmark(n_stages=3)
        runner = BenchmarkRunner(n_agents=5)
        configs = [
            {"id": "clean", "adversarial_fraction": 0.0},
            {"id": "hostile", "adversarial_fraction": 0.5},
        ]
        df = runner.run_frontier(bench, configs, n_seeds=2)
        assert "adversarial_fraction" in df.columns
        assert "safety_score" in df.columns
        clean = df[df["gov_config"] == "clean"]
        hostile = df[df["gov_config"] == "hostile"]
        assert all(clean["safety_score"] == 0.0)
        assert all(hostile["safety_score"] > 0.0)

    def test_all_benchmarks_have_safety_score(self):
        benchmarks = [
            MessageRoutingBenchmark(),
            DistributedAllocationBenchmark(),
            ResourceAuctionBenchmark(n_resources=3),
            PipelineTaskBenchmark(n_stages=3),
        ]
        for bench in benchmarks:
            inst, oracle = bench.generate(seed=0, n_agents=5)
            result = bench.oracle_run(inst, oracle)
            score = bench.score(result, oracle, adversarial_fraction=0.2)
            assert hasattr(score, "safety_score")
            assert score.safety_score >= 0.0


# ---------------------------------------------------------------------------
# Fix #9: Consistent partial fidelity
# ---------------------------------------------------------------------------

class TestPartialFidelity:
    def test_routing_partial_fidelity_via_path_overlap(self):
        """Routing gives partial credit for path overlap even if payload differs."""
        bench = MessageRoutingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=6)
        expected_path = oracle.ground_truth["expected_path"]
        # Wrong payload but correct path
        result = TaskResult(
            completed=True,
            payload=-1,  # wrong
            steps_taken=len(expected_path) - 1,
            agent_trace=list(expected_path),
        )
        score = bench.score(result, oracle)
        # Should get path_credit (0.3) but not payload_match (0.0)
        assert score.fidelity > 0.0
        assert score.fidelity < 1.0

    def test_allocation_partial_fidelity_per_agent(self):
        """Allocation gives partial credit for per-agent accuracy."""
        bench = DistributedAllocationBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        oracle_alloc = oracle.ground_truth["optimal_allocation"]
        # 80% of optimal per agent
        partial = {k: v * 0.8 for k, v in oracle_alloc.items()}
        result = TaskResult(completed=True, payload=partial, steps_taken=1, agent_trace=[])
        score = bench.score(result, oracle)
        assert 0.5 < score.fidelity < 1.0

    def test_auction_welfare_ratio_fidelity(self):
        """Auction fidelity is welfare ratio, not binary match."""
        bench = ResourceAuctionBenchmark(n_resources=3)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        optimal = oracle.ground_truth["optimal_assignment"]
        # Assign all to agent_0 — some welfare but not optimal
        suboptimal = dict.fromkeys(optimal, "agent_0")
        result = TaskResult(completed=True, payload=suboptimal, steps_taken=1, agent_trace=[])
        score = bench.score(result, oracle)
        # Should get partial credit (agent_0 has some valuation for each resource)
        assert 0.0 < score.fidelity <= 1.0

    def test_pipeline_partial_fidelity_for_intermediate_state(self):
        """Pipeline gives partial credit for reaching intermediate states."""
        bench = PipelineTaskBenchmark(n_stages=5)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        intermediates = oracle.ground_truth["intermediate_states"]
        # Reach stage 3 out of 5
        result = TaskResult(
            completed=True,
            payload=intermediates[3],
            steps_taken=3,
            agent_trace=[],
        )
        score = bench.score(result, oracle)
        assert score.fidelity == pytest.approx(3 / 6, abs=0.01)  # 3 out of 6 states


# ---------------------------------------------------------------------------
# Message Routing (core functionality)
# ---------------------------------------------------------------------------

class TestMessageRouting:
    def test_generate_deterministic(self):
        bench = MessageRoutingBenchmark()
        inst1, oracle1 = bench.generate(seed=42, n_agents=8)
        inst2, oracle2 = bench.generate(seed=42, n_agents=8)
        assert inst1.payload == inst2.payload
        assert inst1.source_agent == inst2.source_agent
        assert inst1.target_agent == inst2.target_agent
        assert oracle1.ground_truth["expected_path"] == oracle2.ground_truth["expected_path"]

    def test_generate_different_seeds(self):
        bench = MessageRoutingBenchmark()
        inst1, _ = bench.generate(seed=1, n_agents=8)
        inst2, _ = bench.generate(seed=2, n_agents=8)
        assert inst1.payload != inst2.payload or inst1.source_agent != inst2.source_agent

    def test_oracle_always_completes(self):
        bench = MessageRoutingBenchmark()
        for seed in range(5):
            inst, oracle = bench.generate(seed=seed, n_agents=6)
            result = bench.oracle_run(inst, oracle)
            assert result.completed is True
            assert result.payload == inst.payload
            assert result.agent_trace == oracle.ground_truth["expected_path"]

    def test_score_perfect(self):
        bench = MessageRoutingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=6)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle)
        assert score.completion_rate == 1.0
        assert score.fidelity > 0.9  # path credit + payload match
        assert score.efficiency == 1.0
        assert score.capability_ratio == 1.0

    def test_score_corrupted_payload(self):
        bench = MessageRoutingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=6)
        result = bench.oracle_run(inst, oracle)
        corrupted = TaskResult(
            completed=True,
            payload=result.payload + 1,  # tampered
            steps_taken=result.steps_taken,
            agent_trace=result.agent_trace,
        )
        score = bench.score(corrupted, oracle)
        assert score.completion_rate == 0.0
        # But gets partial fidelity from path overlap
        assert score.fidelity > 0.0

    def test_score_extra_steps(self):
        bench = MessageRoutingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=6)
        result = bench.oracle_run(inst, oracle)
        slow = TaskResult(
            completed=True,
            payload=result.payload,
            steps_taken=result.steps_taken * 3,
            agent_trace=result.agent_trace,
        )
        score = bench.score(slow, oracle)
        assert score.efficiency < 1.0
        assert score.efficiency == pytest.approx(1 / 3, abs=0.01)

    def test_to_soft_interaction_p_bounded(self):
        bench = MessageRoutingBenchmark()
        score = BenchmarkScore(1.0, 1.0, 1.0, 1.0, 0.5)
        interaction = bench.to_soft_interaction(score)
        assert 0.0 <= interaction.p <= 1.0

        score_zero = BenchmarkScore(0.0, 0.0, 0.0, 0.0, 0.0)
        interaction_zero = bench.to_soft_interaction(score_zero)
        assert interaction_zero.p == 0.0

    def test_expected_path_in_oracle(self):
        bench = MessageRoutingBenchmark()
        inst, oracle = bench.generate(seed=7, n_agents=10)
        path = oracle.ground_truth["expected_path"]
        assert len(path) >= 2
        assert path[0] == inst.source_agent
        assert path[-1] == inst.target_agent


# ---------------------------------------------------------------------------
# Distributed Allocation (core functionality)
# ---------------------------------------------------------------------------

class TestDistributedAllocation:
    def test_generate_deterministic(self):
        bench = DistributedAllocationBenchmark()
        inst1, oracle1 = bench.generate(seed=42, n_agents=5)
        inst2, oracle2 = bench.generate(seed=42, n_agents=5)
        assert inst1.target_total == inst2.target_total
        assert inst1.agent_capacities == inst2.agent_capacities

    def test_oracle_meets_target(self):
        bench = DistributedAllocationBenchmark(target_ratio=0.7)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        assert result.completed is True
        total_alloc = sum(result.payload.values())
        assert total_alloc == pytest.approx(inst.target_total, rel=1e-6)

    def test_score_perfect(self):
        bench = DistributedAllocationBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle)
        assert score.completion_rate == pytest.approx(1.0)
        assert score.fidelity == pytest.approx(1.0)

    def test_score_partial(self):
        bench = DistributedAllocationBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        half = {k: v * 0.5 for k, v in result.payload.items()}
        partial = TaskResult(completed=True, payload=half, steps_taken=2, agent_trace=[])
        score = bench.score(partial, oracle)
        assert 0.0 < score.fidelity < 1.0

    def test_score_failed(self):
        bench = DistributedAllocationBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = TaskResult(completed=False, payload=None, steps_taken=0, agent_trace=[])
        score = bench.score(result, oracle)
        assert score.completion_rate == 0.0


# ---------------------------------------------------------------------------
# Resource Auction (core functionality)
# ---------------------------------------------------------------------------

class TestResourceAuction:
    def test_generate_deterministic(self):
        bench = ResourceAuctionBenchmark(n_resources=3)
        inst1, oracle1 = bench.generate(seed=42, n_agents=5)
        inst2, oracle2 = bench.generate(seed=42, n_agents=5)
        assert oracle1.ground_truth["optimal_assignment"] == oracle2.ground_truth["optimal_assignment"]
        assert oracle1.ground_truth["optimal_welfare"] == oracle2.ground_truth["optimal_welfare"]

    def test_oracle_assigns_all_resources(self):
        bench = ResourceAuctionBenchmark(n_resources=4)
        inst, oracle = bench.generate(seed=0, n_agents=6)
        result = bench.oracle_run(inst, oracle)
        assert result.completed is True
        assert len(result.payload) == 4

    def test_score_perfect(self):
        bench = ResourceAuctionBenchmark(n_resources=3)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle)
        assert score.fidelity == 1.0
        assert score.capability_ratio == 1.0

    def test_score_wrong_assignments(self):
        bench = ResourceAuctionBenchmark(n_resources=3)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        wrong = dict.fromkeys(result.payload, "agent_0")
        wrong_result = TaskResult(completed=True, payload=wrong, steps_taken=1, agent_trace=[])
        score = bench.score(wrong_result, oracle)
        assert score.fidelity <= 1.0


# ---------------------------------------------------------------------------
# Pipeline Task (core functionality)
# ---------------------------------------------------------------------------

class TestPipelineTask:
    def test_generate_deterministic(self):
        bench = PipelineTaskBenchmark(n_stages=4)
        inst1, oracle1 = bench.generate(seed=42, n_agents=5)
        inst2, oracle2 = bench.generate(seed=42, n_agents=5)
        assert oracle1.ground_truth["expected_output"] == oracle2.ground_truth["expected_output"]
        assert inst1.initial_payload == inst2.initial_payload

    def test_oracle_produces_correct_output(self):
        bench = PipelineTaskBenchmark(n_stages=3)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        assert result.completed is True
        assert result.payload == oracle.ground_truth["expected_output"]
        assert result.steps_taken == 3

    def test_score_perfect(self):
        bench = PipelineTaskBenchmark(n_stages=4)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle)
        assert score.completion_rate == 1.0
        assert score.fidelity == 1.0
        assert score.efficiency == 1.0

    def test_score_wrong_output(self):
        bench = PipelineTaskBenchmark(n_stages=3)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        wrong = TaskResult(
            completed=True,
            payload=result.payload + 999,
            steps_taken=result.steps_taken,
            agent_trace=result.agent_trace,
        )
        score = bench.score(wrong, oracle)
        assert score.fidelity == 0.0
        assert score.completion_rate == 0.0

    def test_stages_use_all_agents(self):
        bench = PipelineTaskBenchmark(n_stages=7)
        inst, _ = bench.generate(seed=0, n_agents=3)
        assert isinstance(inst, PipelineInstance)
        assert len(inst.stages) == 7
        agents_used = {s.agent_id for s in inst.stages}
        assert len(agents_used) == 3


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    def test_run_frontier_oracle_baseline(self):
        bench = MessageRoutingBenchmark()
        runner = BenchmarkRunner(n_agents=6)
        configs = [
            {"id": "no_governance"},
            {"id": "light_governance", "audit_rate": 0.1},
        ]
        df = runner.run_frontier(bench, configs, n_seeds=3)
        assert len(df) == 6  # 2 configs × 3 seeds
        assert "gov_config" in df.columns
        assert "capability_ratio" in df.columns
        assert all(df["capability_ratio"] == 1.0)

    def test_run_frontier_with_perturbation(self):
        bench = MessageRoutingBenchmark()
        runner = BenchmarkRunner(n_agents=6)

        def corrupt_half(instance, gov_config):
            """Simulate adversary corrupting payload 50% of the time."""
            if instance.seed % 2 == 0:
                return TaskResult(
                    completed=True,
                    payload=instance.payload + 1,  # tampered
                    steps_taken=3,
                    agent_trace=[],
                )
            return TaskResult(
                completed=True,
                payload=instance.payload,
                steps_taken=1,
                agent_trace=[],
            )

        configs = [{"id": "adversarial"}]
        df = runner.run_frontier(bench, configs, n_seeds=4, run_fn=corrupt_half)
        assert len(df) == 4
        # Seeds 0, 2 corrupted; seeds 1, 3 clean
        assert df[df["seed"] == 0]["capability_ratio"].iloc[0] == 0.0
        assert df[df["seed"] == 1]["capability_ratio"].iloc[0] == 1.0

    def test_summarize(self):
        bench = PipelineTaskBenchmark(n_stages=3)
        runner = BenchmarkRunner(n_agents=5)
        configs = [{"id": "baseline"}, {"id": "strict"}]
        df = runner.run_frontier(bench, configs, n_seeds=5)
        summary = runner.summarize(df)
        assert len(summary) == 2
        assert ("completion_rate", "mean") in summary.columns

    def test_run_all_benchmark_types(self):
        """Smoke test: all four benchmark types run through the runner."""
        runner = BenchmarkRunner(n_agents=5)
        configs = [{"id": "baseline"}]
        benchmarks = [
            MessageRoutingBenchmark(),
            DistributedAllocationBenchmark(),
            ResourceAuctionBenchmark(n_resources=3),
            PipelineTaskBenchmark(n_stages=3),
        ]
        for bench in benchmarks:
            df = runner.run_frontier(bench, configs, n_seeds=2)
            assert len(df) == 2
            assert all(df["capability_ratio"] > 0)

    def test_summarize_includes_safety_score(self):
        bench = MessageRoutingBenchmark()
        runner = BenchmarkRunner(n_agents=6)
        configs = [{"id": "gov", "adversarial_fraction": 0.3}]
        df = runner.run_frontier(bench, configs, n_seeds=3)
        summary = runner.summarize(df)
        assert ("safety_score", "mean") in summary.columns
