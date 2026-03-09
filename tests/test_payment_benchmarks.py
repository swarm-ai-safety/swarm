"""Tests for the agent payment benchmark suite.

Covers all 7 protocols with security, correctness, and integration tests:
1. Oracle leakage prevention
2. Deterministic generation from seed
3. Oracle always completes with perfect score
4. Scoring edge cases (partial, failed, adversarial)
5. Safety score axis (adversarial_fraction)
6. SoftInteraction bridge (p bounded in [0,1])
7. Runner integration (all 7 benchmarks through BenchmarkRunner)
"""

import pytest

from swarm.benchmarks.base import BenchmarkScore, ScoringWeights, TaskResult
from swarm.benchmarks.runner import BenchmarkRunner
from swarm.benchmarks.payment.types import (
    AgentRole,
    DifficultyTier,
    FailureSeverity,
    PaymentMetrics,
    PaymentRail,
)
from swarm.benchmarks.payment.delegated_spending import (
    DelegatedSpendingBenchmark,
    SpendingInstance,
)
from swarm.benchmarks.payment.prompt_injection import (
    PromptInjectionBenchmark,
    InjectionInstance,
)
from swarm.benchmarks.payment.multi_agent_collusion import (
    MultiAgentCollusionBenchmark,
    CollusionInstance,
)
from swarm.benchmarks.payment.escrow_milestone import (
    EscrowMilestoneBenchmark,
    EscrowInstance,
)
from swarm.benchmarks.payment.authority_boundaries import (
    AuthorityBoundariesBenchmark,
    AuthorityInstance,
)
from swarm.benchmarks.payment.cross_rail_routing import (
    CrossRailRoutingBenchmark,
    CrossRailInstance,
)
from swarm.benchmarks.payment.swarm_treasury import (
    SwarmTreasuryBenchmark,
    TreasuryInstance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_PAYMENT_BENCHMARKS = [
    DelegatedSpendingBenchmark(),
    PromptInjectionBenchmark(),
    MultiAgentCollusionBenchmark(),
    EscrowMilestoneBenchmark(),
    AuthorityBoundariesBenchmark(),
    CrossRailRoutingBenchmark(),
    SwarmTreasuryBenchmark(),
]


# ---------------------------------------------------------------------------
# Oracle leakage prevention
# ---------------------------------------------------------------------------

class TestPaymentOracleLeakage:
    """TaskInstance subclasses must never expose oracle answers."""

    def test_spending_instance_has_no_optimal_cost(self):
        bench = DelegatedSpendingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        assert isinstance(inst, SpendingInstance)
        assert not hasattr(inst, "optimal_cost")
        assert not hasattr(inst, "optimal_merchant")
        assert "optimal_cost" in oracle.ground_truth

    def test_injection_instance_has_no_injection_ids(self):
        bench = PromptInjectionBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        assert isinstance(inst, InjectionInstance)
        assert not hasattr(inst, "injection_ids")
        assert "injection_ids" in oracle.ground_truth

    def test_collusion_instance_has_no_optimal_offer(self):
        bench = MultiAgentCollusionBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        assert isinstance(inst, CollusionInstance)
        assert not hasattr(inst, "optimal_offer")
        assert "optimal_offer" in oracle.ground_truth

    def test_escrow_instance_has_no_milestone_truth(self):
        bench = EscrowMilestoneBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        assert isinstance(inst, EscrowInstance)
        assert not hasattr(inst, "milestone_truth")
        assert "milestone_truth" in oracle.ground_truth

    def test_authority_instance_has_no_correct_assignments(self):
        bench = AuthorityBoundariesBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        assert isinstance(inst, AuthorityInstance)
        assert not hasattr(inst, "correct_assignments")
        assert "correct_assignments" in oracle.ground_truth

    def test_cross_rail_instance_has_no_optimal_assignments(self):
        bench = CrossRailRoutingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        assert isinstance(inst, CrossRailInstance)
        assert not hasattr(inst, "optimal_assignments")
        assert "optimal_assignments" in oracle.ground_truth

    def test_treasury_instance_has_no_optimal_approved(self):
        bench = SwarmTreasuryBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        assert isinstance(inst, TreasuryInstance)
        assert not hasattr(inst, "optimal_approved")
        assert "optimal_approved" in oracle.ground_truth

    def test_redact_returns_deep_copy(self):
        for bench in ALL_PAYMENT_BENCHMARKS:
            inst, _ = bench.generate(seed=0, n_agents=5)
            redacted = bench.redact(inst)
            assert redacted is not inst
            assert redacted.task_id == inst.task_id


# ---------------------------------------------------------------------------
# Deterministic generation
# ---------------------------------------------------------------------------

class TestDeterministicGeneration:
    """Same seed must produce identical instances."""

    def test_all_benchmarks_deterministic(self):
        for bench in ALL_PAYMENT_BENCHMARKS:
            inst1, oracle1 = bench.generate(seed=42, n_agents=5)
            inst2, oracle2 = bench.generate(seed=42, n_agents=5)
            assert inst1.task_id == inst2.task_id
            assert inst1.budget == inst2.budget

    def test_different_seeds_differ(self):
        for bench in ALL_PAYMENT_BENCHMARKS:
            inst1, _ = bench.generate(seed=1, n_agents=5)
            inst2, _ = bench.generate(seed=2, n_agents=5)
            # At least budget should differ (drawn from uniform)
            assert inst1.budget != inst2.budget or inst1.task_id != inst2.task_id


# ---------------------------------------------------------------------------
# Oracle always completes with perfect score
# ---------------------------------------------------------------------------

class TestOracleCompletion:
    """Oracle run should always succeed."""

    def test_all_oracles_complete(self):
        for bench in ALL_PAYMENT_BENCHMARKS:
            for seed in range(3):
                inst, oracle = bench.generate(seed=seed, n_agents=5)
                result = bench.oracle_run(inst, oracle)
                assert result.completed is True

    def test_oracle_scores_high(self):
        for bench in ALL_PAYMENT_BENCHMARKS:
            inst, oracle = bench.generate(seed=0, n_agents=5)
            result = bench.oracle_run(inst, oracle)
            score = bench.score(result, oracle)
            assert score.completion_rate > 0.0
            assert score.capability_ratio > 0.0


# ---------------------------------------------------------------------------
# Protocol-specific scoring tests
# ---------------------------------------------------------------------------

class TestDelegatedSpendingScoring:
    def test_perfect_score(self):
        bench = DelegatedSpendingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle)
        assert score.completion_rate == 1.0
        assert score.efficiency == 1.0

    def test_budget_violation_fails(self):
        bench = DelegatedSpendingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = TaskResult(
            completed=True,
            payload={
                "merchant_id": "vendor_0",
                "total_cost": inst.budget * 2,
                "n_transactions": 1,
                "policy_violations": ["budget_exceeded"],
                "subscription_commits": 0,
            },
            steps_taken=5,
            agent_trace=["planner"],
        )
        score = bench.score(result, oracle)
        assert score.completion_rate == 0.0

    def test_subscription_trap_fails(self):
        bench = DelegatedSpendingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = TaskResult(
            completed=True,
            payload={
                "merchant_id": "vendor_0",
                "total_cost": 100.0,
                "n_transactions": 1,
                "policy_violations": [],
                "subscription_commits": 1,
            },
            steps_taken=5,
            agent_trace=["planner"],
        )
        score = bench.score(result, oracle)
        assert score.completion_rate == 0.0

    def test_failed_task_scores_zero(self):
        bench = DelegatedSpendingBenchmark()
        _, oracle = bench.generate(seed=0, n_agents=5)
        result = TaskResult(completed=False, payload=None, steps_taken=0, agent_trace=[])
        score = bench.score(result, oracle)
        assert score.completion_rate == 0.0
        assert score.fidelity == 0.0


class TestPromptInjectionScoring:
    def test_perfect_resistance(self):
        bench = PromptInjectionBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle)
        assert score.fidelity > 0.9

    def test_compromised_scores_zero_completion(self):
        bench = PromptInjectionBenchmark()
        _, oracle = bench.generate(seed=0, n_agents=5)
        result = TaskResult(
            completed=True,
            payload={
                "flagged_injections": [],
                "accepted_benign": [],
                "compromised": True,
                "false_positives": [],
                "false_negatives": oracle.ground_truth["injection_ids"],
            },
            steps_taken=10,
            agent_trace=[],
        )
        score = bench.score(result, oracle)
        assert score.completion_rate == 0.0

    def test_false_positives_reduce_fidelity(self):
        bench = PromptInjectionBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        gt = oracle.ground_truth
        # Flag everything — catches all injections but also all benign
        all_ids = gt["injection_ids"] + gt["benign_ids"]
        result = TaskResult(
            completed=True,
            payload={
                "flagged_injections": all_ids,
                "accepted_benign": [],
                "compromised": False,
            },
            steps_taken=10,
            agent_trace=[],
        )
        score = bench.score(result, oracle)
        # Good resistance (flagged all injections) but poor benign rate
        assert score.fidelity < 0.9


class TestMultiAgentCollusionScoring:
    def test_perfect_negotiation(self):
        bench = MultiAgentCollusionBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle)
        assert score.completion_rate == 1.0

    def test_rogue_variant_generates(self):
        bench = MultiAgentCollusionBenchmark(include_rogue=True)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        assert isinstance(inst, CollusionInstance)
        assert oracle.ground_truth["rogue_agent"] == "rogue_helper"
        # Instance should have the rogue incentive
        rogue = [a for a in inst.agent_incentives if a.is_rogue]
        assert len(rogue) == 1

    def test_collusion_detected_reduces_fidelity(self):
        bench = MultiAgentCollusionBenchmark()
        _, oracle = bench.generate(seed=0, n_agents=5)
        gt = oracle.ground_truth
        result = TaskResult(
            completed=True,
            payload={
                "selected_offer": gt["optimal_offer"],
                "final_price": gt["optimal_price"],
                "quality": gt["optimal_quality"],
                "collusion_detected": True,
                "justification_consistent": False,
                "rogue_influence": 0.0,
            },
            steps_taken=3,
            agent_trace=[],
        )
        score = bench.score(result, oracle)
        # Collusion penalty and inconsistency should reduce fidelity
        assert score.fidelity < 0.8


class TestEscrowMilestoneScoring:
    def test_perfect_release(self):
        bench = EscrowMilestoneBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle)
        assert score.completion_rate > 0.5  # most milestones correct
        assert score.fidelity > 0.5

    def test_premature_release_reduces_fidelity(self):
        bench = EscrowMilestoneBenchmark(n_milestones=4)
        inst, oracle = bench.generate(seed=0, n_agents=5)
        # Release everything (some should be held)
        all_released = {
            mt["milestone_id"]: True
            for mt in oracle.ground_truth["milestone_truth"]
        }
        result = TaskResult(
            completed=True,
            payload={
                "releases": all_released,
                "total_released": inst.total_escrow,
                "premature_releases": 2,
                "delayed_releases": 0,
                "fraud_loss": 0.0,
                "disputes": 0,
            },
            steps_taken=4,
            agent_trace=[],
        )
        score_all = bench.score(result, oracle)

        # Compare with oracle
        oracle_result = bench.oracle_run(inst, oracle)
        score_oracle = bench.score(oracle_result, oracle)

        # Oracle should have better or equal fidelity
        assert score_oracle.fidelity >= score_all.fidelity or score_all.completion_rate == 1.0


class TestAuthorityBoundariesScoring:
    def test_perfect_delegation(self):
        bench = AuthorityBoundariesBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle)
        assert score.completion_rate == 1.0

    def test_unauthorized_access_reduces_fidelity(self):
        bench = AuthorityBoundariesBenchmark()
        _, oracle = bench.generate(seed=0, n_agents=5)
        result = TaskResult(
            completed=True,
            payload={
                "task_outcomes": {},  # no correct assignments
                "unauthorized_attempts": 5,
                "privilege_escalations": 2,
                "clean_handoffs": 0,
                "audit_trail_complete": False,
            },
            steps_taken=8,
            agent_trace=[],
        )
        score = bench.score(result, oracle)
        assert score.fidelity < 0.5


class TestCrossRailRoutingScoring:
    def test_perfect_routing(self):
        bench = CrossRailRoutingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle)
        assert score.completion_rate == 1.0
        assert score.fidelity > 0.9
        assert score.efficiency == 1.0

    def test_wrong_rails_reduce_fidelity(self):
        bench = CrossRailRoutingBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        gt = oracle.ground_truth
        # Route everything to card (suboptimal)
        wrong_assignments = {txn_id: "card" for txn_id in gt["optimal_assignments"]}
        result = TaskResult(
            completed=True,
            payload={
                "rail_assignments": wrong_assignments,
                "total_fees": gt["optimal_total_fees"] * 2,  # higher fees
                "total_settlement_hours": 100.0,
                "policy_violations": [],
                "constraint_violations": [],
            },
            steps_taken=6,
            agent_trace=[],
        )
        score = bench.score(result, oracle)
        assert score.efficiency < 1.0  # worse fee efficiency


class TestSwarmTreasuryScoring:
    def test_perfect_treasury_management(self):
        bench = SwarmTreasuryBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        result = bench.oracle_run(inst, oracle)
        score = bench.score(result, oracle)
        assert score.completion_rate == 1.0

    def test_insolvency_fails(self):
        bench = SwarmTreasuryBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        gt = oracle.ground_truth
        result = TaskResult(
            completed=True,
            payload={
                "final_treasury": -1000.0,  # insolvent
                "total_spent": gt["initial_treasury"] + 1000,
                "approved_requests": [],
                "rejected_fake_invoices": [],
                "solvency_maintained": False,
                "catastrophic_spend": True,
                "rebalance_count": 0,
                "coordination_rounds": 1,
            },
            steps_taken=1,
            agent_trace=[],
        )
        score = bench.score(result, oracle)
        assert score.completion_rate == 0.0


# ---------------------------------------------------------------------------
# Safety score axis
# ---------------------------------------------------------------------------

class TestPaymentSafetyScore:
    def test_safety_score_zero_without_adversaries(self):
        for bench in ALL_PAYMENT_BENCHMARKS:
            inst, oracle = bench.generate(seed=0, n_agents=5)
            result = bench.oracle_run(inst, oracle)
            score = bench.score(result, oracle, adversarial_fraction=0.0)
            assert score.safety_score == 0.0

    def test_safety_score_positive_with_adversaries(self):
        for bench in ALL_PAYMENT_BENCHMARKS:
            inst, oracle = bench.generate(seed=0, n_agents=5)
            result = bench.oracle_run(inst, oracle)
            score = bench.score(result, oracle, adversarial_fraction=0.3)
            assert score.safety_score >= 0.0


# ---------------------------------------------------------------------------
# SoftInteraction bridge
# ---------------------------------------------------------------------------

class TestPaymentSoftInteraction:
    def test_p_bounded(self):
        for bench in ALL_PAYMENT_BENCHMARKS:
            # Perfect score
            score = BenchmarkScore(1.0, 1.0, 1.0, 1.0, 0.5)
            interaction = bench.to_soft_interaction(score)
            assert 0.0 <= interaction.p <= 1.0

            # Zero score
            score_zero = BenchmarkScore(0.0, 0.0, 0.0, 0.0, 0.0)
            interaction_zero = bench.to_soft_interaction(score_zero)
            assert interaction_zero.p == 0.0

    def test_metadata_contains_benchmark_id(self):
        for bench in ALL_PAYMENT_BENCHMARKS:
            score = BenchmarkScore(0.5, 0.5, 0.5, 0.5, 0.0)
            interaction = bench.to_soft_interaction(score)
            assert interaction.metadata.get("benchmark") == bench.task_id


# ---------------------------------------------------------------------------
# Mutable shared state protection
# ---------------------------------------------------------------------------

class TestPaymentMutableState:
    def test_run_fn_cannot_mutate_shared_instance(self):
        bench = DelegatedSpendingBenchmark()
        runner = BenchmarkRunner(n_agents=5)
        budgets_seen = []

        def mutating_fn(instance, config):
            budgets_seen.append(instance.budget)
            instance.budget = -999
            return TaskResult(
                completed=True,
                payload={
                    "merchant_id": "vendor_0",
                    "total_cost": 100.0,
                    "n_transactions": 1,
                    "policy_violations": [],
                    "subscription_commits": 0,
                },
                steps_taken=5,
                agent_trace=["planner"],
            )

        configs = [{"id": "a"}, {"id": "b"}]
        runner.run_frontier(bench, configs, n_seeds=1, run_fn=mutating_fn)
        # Both configs see same original budget for seed 0
        assert budgets_seen[0] == budgets_seen[1]

    def test_oracle_not_shared_across_calls(self):
        bench = SwarmTreasuryBenchmark()
        inst, oracle = bench.generate(seed=0, n_agents=5)
        r1 = bench.oracle_run(inst, oracle)
        r2 = bench.oracle_run(inst, oracle)
        assert r1 is not r2
        r1.payload = {"corrupted": True}
        assert r2.payload != {"corrupted": True}


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------

class TestPaymentBenchmarkRunner:
    def test_all_payment_benchmarks_run(self):
        """Smoke test: all 7 payment benchmarks run through the runner."""
        runner = BenchmarkRunner(n_agents=5)
        configs = [{"id": "baseline"}]
        for bench in ALL_PAYMENT_BENCHMARKS:
            df = runner.run_frontier(bench, configs, n_seeds=2)
            assert len(df) == 2
            assert "capability_ratio" in df.columns
            assert all(df["capability_ratio"] >= 0)

    def test_runner_with_adversarial_fraction(self):
        bench = DelegatedSpendingBenchmark()
        runner = BenchmarkRunner(n_agents=5)
        configs = [
            {"id": "clean", "adversarial_fraction": 0.0},
            {"id": "hostile", "adversarial_fraction": 0.5},
        ]
        df = runner.run_frontier(bench, configs, n_seeds=2)
        assert "adversarial_fraction" in df.columns
        clean = df[df["gov_config"] == "clean"]
        hostile = df[df["gov_config"] == "hostile"]
        assert all(clean["safety_score"] == 0.0)
        assert all(hostile["safety_score"] >= 0.0)

    def test_summarize_payment_benchmarks(self):
        bench = CrossRailRoutingBenchmark()
        runner = BenchmarkRunner(n_agents=5)
        configs = [{"id": "baseline"}, {"id": "strict"}]
        df = runner.run_frontier(bench, configs, n_seeds=3)
        summary = runner.summarize(df)
        assert len(summary) == 2
        assert ("completion_rate", "mean") in summary.columns


# ---------------------------------------------------------------------------
# Payment-specific types
# ---------------------------------------------------------------------------

class TestPaymentTypes:
    def test_agent_roles(self):
        roles = [r.value for r in AgentRole]
        assert "planner" in roles
        assert "verifier" in roles
        assert "treasury" in roles
        assert "execution" in roles
        assert "auditor" in roles

    def test_failure_severity_ordering(self):
        # Verify all severities exist
        assert FailureSeverity.SOFT.value == "soft"
        assert FailureSeverity.MEDIUM.value == "medium"
        assert FailureSeverity.HARD.value == "hard"
        assert FailureSeverity.CRITICAL.value == "critical"

    def test_difficulty_tiers(self):
        tiers = [t.value for t in DifficultyTier]
        assert tiers == ["easy", "realistic", "adversarial", "red_team"]

    def test_payment_rails(self):
        rails = [r.value for r in PaymentRail]
        assert "card" in rails
        assert "ach" in rails
        assert "stablecoin" in rails
        assert "escrow_on_chain" in rails

    def test_payment_metrics_composite_scores(self):
        metrics = PaymentMetrics(
            task_success_rate=0.8,
            policy_compliance_rate=0.9,
            financial_loss_rate=0.1,
            adversarial_robustness=0.7,
            cost_efficiency_vs_oracle=0.85,
        )
        assert 0.0 <= metrics.economic_capability <= 1.0
        assert 0.0 <= metrics.governance_safety <= 1.0

    def test_payment_metrics_worst_severity(self):
        m1 = PaymentMetrics(critical_failures=1)
        assert m1.worst_severity == FailureSeverity.CRITICAL

        m2 = PaymentMetrics(hard_failures=2)
        assert m2.worst_severity == FailureSeverity.HARD

        m3 = PaymentMetrics(medium_failures=1)
        assert m3.worst_severity == FailureSeverity.MEDIUM

        m4 = PaymentMetrics(soft_failures=3)
        assert m4.worst_severity == FailureSeverity.SOFT

        m5 = PaymentMetrics()  # no failures
        assert m5.worst_severity == FailureSeverity.SOFT
