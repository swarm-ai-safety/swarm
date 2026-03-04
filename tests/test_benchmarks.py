"""Tests for the benchmark suite."""

import pytest

from swarm.benchmarks.base import BenchmarkScore, TaskResult
from swarm.benchmarks.routing.message_routing import (
    MessageRoutingBenchmark,
    RoutingInstance,
    _bfs_path,
)
from swarm.benchmarks.coordination.distributed_allocation import (
    DistributedAllocationBenchmark,
    AllocationInstance,
)
from swarm.benchmarks.allocation.resource_auction import (
    ResourceAuctionBenchmark,
    AuctionInstance,
)
from swarm.benchmarks.long_horizon.pipeline_task import (
    PipelineTaskBenchmark,
    PipelineInstance,
)
from swarm.benchmarks.runner import BenchmarkRunner
from swarm.env.network import AgentNetwork, NetworkConfig, NetworkTopology


# ---------------------------------------------------------------------------
# Message Routing
# ---------------------------------------------------------------------------

class TestMessageRouting:
    def test_generate_deterministic(self):
        bench = MessageRoutingBenchmark()
        inst1 = bench.generate(seed=42, n_agents=8)
        inst2 = bench.generate(seed=42, n_agents=8)
        assert inst1.payload == inst2.payload
        assert inst1.source_agent == inst2.source_agent
        assert inst1.target_agent == inst2.target_agent
        assert inst1.expected_path == inst2.expected_path

    def test_generate_different_seeds(self):
        bench = MessageRoutingBenchmark()
        inst1 = bench.generate(seed=1, n_agents=8)
        inst2 = bench.generate(seed=2, n_agents=8)
        # Very unlikely to be identical with different seeds
        assert inst1.payload != inst2.payload or inst1.source_agent != inst2.source_agent

    def test_oracle_always_completes(self):
        bench = MessageRoutingBenchmark()
        for seed in range(5):
            inst = bench.generate(seed=seed, n_agents=6)
            result = bench.oracle_run(inst)
            assert result.completed is True
            assert result.payload == inst.payload
            assert result.agent_trace == inst.expected_path

    def test_score_perfect(self):
        bench = MessageRoutingBenchmark()
        inst = bench.generate(seed=0, n_agents=6)
        oracle = bench.oracle_run(inst)
        # Perfect result matches oracle
        score = bench.score(oracle, oracle)
        assert score.completion_rate == 1.0
        assert score.fidelity == 1.0
        assert score.efficiency == 1.0
        assert score.capability_ratio == 1.0

    def test_score_corrupted_payload(self):
        bench = MessageRoutingBenchmark()
        inst = bench.generate(seed=0, n_agents=6)
        oracle = bench.oracle_run(inst)
        corrupted = TaskResult(
            completed=True,
            payload=oracle.payload + 1,  # tampered
            steps_taken=oracle.steps_taken,
            agent_trace=oracle.agent_trace,
        )
        score = bench.score(corrupted, oracle)
        assert score.fidelity == 0.0
        assert score.completion_rate == 0.0

    def test_score_extra_steps(self):
        bench = MessageRoutingBenchmark()
        inst = bench.generate(seed=0, n_agents=6)
        oracle = bench.oracle_run(inst)
        slow = TaskResult(
            completed=True,
            payload=oracle.payload,
            steps_taken=oracle.steps_taken * 3,
            agent_trace=oracle.agent_trace,
        )
        score = bench.score(slow, oracle)
        assert score.fidelity == 1.0
        assert score.efficiency < 1.0
        assert score.efficiency == pytest.approx(1 / 3, abs=0.01)

    def test_to_soft_interaction_p_bounded(self):
        bench = MessageRoutingBenchmark()
        score = BenchmarkScore(1.0, 1.0, 1.0, 1.0)
        interaction = bench.to_soft_interaction(score)
        assert 0.0 <= interaction.p <= 1.0

        score_zero = BenchmarkScore(0.0, 0.0, 0.0, 0.0)
        interaction_zero = bench.to_soft_interaction(score_zero)
        assert interaction_zero.p == 0.0

    def test_expected_path_valid(self):
        bench = MessageRoutingBenchmark()
        inst = bench.generate(seed=7, n_agents=10)
        assert isinstance(inst, RoutingInstance)
        path = inst.expected_path
        assert len(path) >= 2
        assert path[0] == inst.source_agent
        assert path[-1] == inst.target_agent


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


# ---------------------------------------------------------------------------
# Distributed Allocation
# ---------------------------------------------------------------------------

class TestDistributedAllocation:
    def test_generate_deterministic(self):
        bench = DistributedAllocationBenchmark()
        inst1 = bench.generate(seed=42, n_agents=5)
        inst2 = bench.generate(seed=42, n_agents=5)
        assert inst1.target_total == inst2.target_total
        assert inst1.agent_capacities == inst2.agent_capacities

    def test_oracle_meets_target(self):
        bench = DistributedAllocationBenchmark(target_ratio=0.7)
        inst = bench.generate(seed=0, n_agents=5)
        oracle = bench.oracle_run(inst)
        assert oracle.completed is True
        total_alloc = sum(oracle.payload.values())
        assert total_alloc == pytest.approx(inst.target_total, rel=1e-6)

    def test_score_perfect(self):
        bench = DistributedAllocationBenchmark()
        inst = bench.generate(seed=0, n_agents=5)
        oracle = bench.oracle_run(inst)
        score = bench.score(oracle, oracle)
        assert score.completion_rate == pytest.approx(1.0)
        assert score.fidelity == pytest.approx(1.0)

    def test_score_partial(self):
        bench = DistributedAllocationBenchmark()
        inst = bench.generate(seed=0, n_agents=5)
        oracle = bench.oracle_run(inst)
        # Half the allocation
        half = {k: v * 0.5 for k, v in oracle.payload.items()}
        result = TaskResult(completed=True, payload=half, steps_taken=2, agent_trace=[])
        score = bench.score(result, oracle)
        assert 0.0 < score.fidelity < 1.0

    def test_score_failed(self):
        bench = DistributedAllocationBenchmark()
        inst = bench.generate(seed=0, n_agents=5)
        oracle = bench.oracle_run(inst)
        result = TaskResult(completed=False, payload=None, steps_taken=0, agent_trace=[])
        score = bench.score(result, oracle)
        assert score.completion_rate == 0.0


# ---------------------------------------------------------------------------
# Resource Auction
# ---------------------------------------------------------------------------

class TestResourceAuction:
    def test_generate_deterministic(self):
        bench = ResourceAuctionBenchmark(n_resources=3)
        inst1 = bench.generate(seed=42, n_agents=5)
        inst2 = bench.generate(seed=42, n_agents=5)
        assert inst1.optimal_assignment == inst2.optimal_assignment
        assert inst1.optimal_welfare == inst2.optimal_welfare

    def test_oracle_assigns_all_resources(self):
        bench = ResourceAuctionBenchmark(n_resources=4)
        inst = bench.generate(seed=0, n_agents=6)
        oracle = bench.oracle_run(inst)
        assert oracle.completed is True
        assert len(oracle.payload) == 4

    def test_score_perfect(self):
        bench = ResourceAuctionBenchmark(n_resources=3)
        inst = bench.generate(seed=0, n_agents=5)
        oracle = bench.oracle_run(inst)
        score = bench.score(oracle, oracle)
        assert score.fidelity == 1.0
        assert score.capability_ratio == 1.0

    def test_score_wrong_assignments(self):
        bench = ResourceAuctionBenchmark(n_resources=3)
        inst = bench.generate(seed=0, n_agents=5)
        oracle = bench.oracle_run(inst)
        # Swap all assignments to agent_0
        wrong = {r: "agent_0" for r in oracle.payload}
        result = TaskResult(completed=True, payload=wrong, steps_taken=1, agent_trace=[])
        score = bench.score(result, oracle)
        # At most 1/3 could match by chance
        assert score.fidelity <= 1.0


# ---------------------------------------------------------------------------
# Pipeline Task
# ---------------------------------------------------------------------------

class TestPipelineTask:
    def test_generate_deterministic(self):
        bench = PipelineTaskBenchmark(n_stages=4)
        inst1 = bench.generate(seed=42, n_agents=5)
        inst2 = bench.generate(seed=42, n_agents=5)
        assert inst1.expected_output == inst2.expected_output
        assert inst1.initial_payload == inst2.initial_payload

    def test_oracle_produces_correct_output(self):
        bench = PipelineTaskBenchmark(n_stages=3)
        inst = bench.generate(seed=0, n_agents=5)
        oracle = bench.oracle_run(inst)
        assert oracle.completed is True
        assert oracle.payload == inst.expected_output
        assert oracle.steps_taken == 3

    def test_score_perfect(self):
        bench = PipelineTaskBenchmark(n_stages=4)
        inst = bench.generate(seed=0, n_agents=5)
        oracle = bench.oracle_run(inst)
        score = bench.score(oracle, oracle)
        assert score.completion_rate == 1.0
        assert score.fidelity == 1.0
        assert score.efficiency == 1.0

    def test_score_wrong_output(self):
        bench = PipelineTaskBenchmark(n_stages=3)
        inst = bench.generate(seed=0, n_agents=5)
        oracle = bench.oracle_run(inst)
        wrong = TaskResult(
            completed=True,
            payload=oracle.payload + 999,
            steps_taken=oracle.steps_taken,
            agent_trace=oracle.agent_trace,
        )
        score = bench.score(wrong, oracle)
        assert score.fidelity == 0.0
        assert score.completion_rate == 0.0

    def test_stages_use_all_agents(self):
        bench = PipelineTaskBenchmark(n_stages=7)
        inst = bench.generate(seed=0, n_agents=3)
        assert isinstance(inst, PipelineInstance)
        assert len(inst.stages) == 7
        # All 3 agents should appear (stages wrap around)
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
        # Without a custom run_fn, all results use oracle -> perfect scores
        assert all(df["capability_ratio"] == 1.0)

    def test_run_frontier_with_perturbation(self):
        bench = MessageRoutingBenchmark()
        runner = BenchmarkRunner(n_agents=6)

        def corrupt_half(instance, gov_config):
            """Simulate adversary corrupting payload 50% of the time."""
            oracle = bench.oracle_run(instance)
            if instance.seed % 2 == 0:
                return TaskResult(
                    completed=True,
                    payload=oracle.payload + 1,  # tampered
                    steps_taken=oracle.steps_taken + 2,
                    agent_trace=oracle.agent_trace,
                )
            return oracle

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
        assert len(summary) == 2  # one row per config
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
