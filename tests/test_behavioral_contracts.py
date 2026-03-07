"""Tests for ABC behavioral contracts, drift detection, and compositionality."""

import pytest

from swarm.contracts.behavioral import (
    BehavioralContract,
    DriftDetector,
    InvariantCheck,
    Precondition,
    StageGuarantee,
    compute_pipeline_bound,
    compute_pipeline_bound_with_drift,
    default_recovery,
    max_drift_rate,
    min_resources,
    min_trust_score,
    p_in_bounds,
)
from swarm.contracts.contract import TruthfulAuctionContract
from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import SoftInteraction


def _make_agent(
    agent_id: str = "a1",
    resources: float = 100.0,
    reputation: float = 0.7,
    agent_type: AgentType = AgentType.HONEST,
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        agent_type=agent_type,
        resources=resources,
        reputation=reputation,
    )


def _make_interaction(
    p: float = 0.8,
    initiator: str = "a1",
    counterparty: str = "a2",
    metadata: dict | None = None,
) -> SoftInteraction:
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        p=p,
        v_hat=p * 2 - 1,  # Map p to v_hat range
        metadata=metadata or {},
    )


# ── Preconditions ──────────────────────────────────────────────────


class TestPreconditions:
    def test_min_resources_pass(self):
        pre = min_resources(50.0)
        agent = _make_agent(resources=100.0)
        assert pre.evaluate(agent) is True

    def test_min_resources_fail(self):
        pre = min_resources(150.0)
        agent = _make_agent(resources=100.0)
        assert pre.evaluate(agent) is False

    def test_min_trust_score_pass(self):
        pre = min_trust_score(0.5)
        agent = _make_agent(reputation=0.7)
        assert pre.evaluate(agent) is True

    def test_min_trust_score_fail(self):
        pre = min_trust_score(0.8)
        agent = _make_agent(reputation=0.5)
        assert pre.evaluate(agent) is False

    def test_custom_precondition(self):
        pre = Precondition(
            name="not_adversarial",
            check=lambda a: a.agent_type != AgentType.ADVERSARIAL,
        )
        honest = _make_agent(agent_type=AgentType.HONEST)
        adversarial = _make_agent(agent_type=AgentType.ADVERSARIAL)
        assert pre.evaluate(honest) is True
        assert pre.evaluate(adversarial) is False


# ── Invariants ─────────────────────────────────────────────────────


class TestInvariants:
    def test_p_in_bounds_pass(self):
        inv = p_in_bounds(0.3, 0.9)
        interaction = _make_interaction(p=0.5)
        assert inv.evaluate(interaction) is True

    def test_p_in_bounds_fail_low(self):
        inv = p_in_bounds(0.3, 0.9)
        interaction = _make_interaction(p=0.1)
        assert inv.evaluate(interaction) is False

    def test_p_in_bounds_fail_high(self):
        inv = p_in_bounds(0.3, 0.9)
        interaction = _make_interaction(p=0.95)
        assert inv.evaluate(interaction) is False

    def test_max_drift_rate_pass(self):
        inv = max_drift_rate(0.2)
        interaction = _make_interaction(metadata={"drift_rate": 0.1})
        assert inv.evaluate(interaction) is True

    def test_max_drift_rate_fail(self):
        inv = max_drift_rate(0.2)
        interaction = _make_interaction(metadata={"drift_rate": 0.3})
        assert inv.evaluate(interaction) is False

    def test_max_drift_rate_no_metadata(self):
        inv = max_drift_rate(0.2)
        interaction = _make_interaction(metadata={})
        # drift_rate defaults to 0.0, passes
        assert inv.evaluate(interaction) is True


# ── Recovery ───────────────────────────────────────────────────────


class TestRecovery:
    def test_default_recovery_low_severity(self):
        recovery = default_recovery()
        action = recovery.select_action(0.3)
        assert action is not None
        assert action.type == "penalty"

    def test_default_recovery_medium_severity(self):
        recovery = default_recovery()
        action = recovery.select_action(0.7)
        assert action is not None
        assert action.type == "tier_downgrade"

    def test_default_recovery_high_severity(self):
        recovery = default_recovery()
        action = recovery.select_action(0.95)
        assert action is not None
        assert action.type == "expel"

    def test_default_recovery_below_threshold(self):
        recovery = default_recovery()
        action = recovery.select_action(0.1)
        assert action is None


# ── BehavioralContract ─────────────────────────────────────────────


class TestBehavioralContract:
    def _make_contract(self) -> BehavioralContract:
        return BehavioralContract(
            governance=TruthfulAuctionContract(
                stake_fraction=0.05, audit_probability=0.2
            ),
            preconditions=[min_resources(10.0), min_trust_score(0.3)],
            invariants=[p_in_bounds(0.2, 1.0)],
        )

    def test_preconditions_pass(self):
        bc = self._make_contract()
        agent = _make_agent(resources=50.0, reputation=0.5)
        passed, failures = bc.check_preconditions(agent)
        assert passed is True
        assert failures == []

    def test_preconditions_fail(self):
        bc = self._make_contract()
        agent = _make_agent(resources=5.0, reputation=0.1)
        passed, failures = bc.check_preconditions(agent)
        assert passed is False
        assert len(failures) == 2  # Both preconditions fail

    def test_invariants_pass(self):
        bc = self._make_contract()
        interaction = _make_interaction(p=0.5)
        violations = bc.check_invariants(interaction)
        assert violations == []

    def test_invariants_fail(self):
        bc = self._make_contract()
        interaction = _make_interaction(p=0.1)
        violations = bc.check_invariants(interaction)
        assert len(violations) == 1
        assert violations[0].invariant_name == "p_bounds(0.2,1.0)"
        assert violations[0].severity == 0.8

    def test_execute_delegates_to_governance(self):
        bc = self._make_contract()
        interaction = _make_interaction(p=0.8)
        modified = bc.execute(interaction)
        # TruthfulAuctionContract reduces c_a, c_b by 50%
        assert modified.c_a <= interaction.c_a
        assert modified.metadata.get("contract") == "truthful_auction"

    def test_expel_on_severe_violation(self):
        bc = BehavioralContract(
            governance=TruthfulAuctionContract(),
            invariants=[
                InvariantCheck(
                    name="always_fail", check=lambda _: False, severity=0.95
                )
            ],
        )
        interaction = _make_interaction()
        bc.check_invariants(interaction)
        assert bc.is_expelled("a1") is True

    def test_violations_tracked(self):
        bc = self._make_contract()
        bc.check_invariants(_make_interaction(p=0.1))
        bc.check_invariants(_make_interaction(p=0.5))  # passes
        bc.check_invariants(_make_interaction(p=0.05))
        all_violations = bc.get_violations()
        assert len(all_violations) == 2
        agent_violations = bc.get_violations(agent_id="a1")
        assert len(agent_violations) == 2

    def test_reset_clears_state(self):
        bc = self._make_contract()
        bc.check_invariants(_make_interaction(p=0.1))
        assert len(bc.get_violations()) == 1
        bc.reset()
        assert len(bc.get_violations()) == 0

    def test_to_report_dict(self):
        bc = self._make_contract()
        report = bc.to_report_dict()
        assert report["governance"] == "truthful_auction"
        assert len(report["preconditions"]) == 2
        assert len(report["invariants"]) == 1


# ── DriftDetector ──────────────────────────────────────────────────


class TestDriftDetector:
    def test_no_drift_returns_none_until_enough_data(self):
        dd = DriftDetector(window_size=5, baseline_size=10)
        for _ in range(14):
            result = dd.record("a1", 0.8)
        assert result is None  # 14 < 10 + 5

    def test_no_drift_when_stable(self):
        dd = DriftDetector(window_size=5, baseline_size=10, drift_threshold=0.1)
        for _ in range(15):
            dd.record("a1", 0.8)
        drift = dd.get_drift("a1")
        assert drift is not None
        assert abs(drift) < 0.01  # Stable behavior, no drift

    def test_detects_degradation(self):
        dd = DriftDetector(window_size=5, baseline_size=10, drift_threshold=0.15)
        # Build trust with high p
        for _ in range(10):
            dd.record("a1", 0.9)
        # Then degrade
        for _ in range(5):
            dd.record("a1", 0.3)
        drift = dd.get_drift("a1")
        assert drift is not None
        assert drift > 0.15  # baseline 0.9 - recent 0.3 = 0.6
        assert dd.is_flagged("a1") is True

    def test_no_flag_for_improvement(self):
        dd = DriftDetector(window_size=5, baseline_size=10, drift_threshold=0.15)
        # Start low
        for _ in range(10):
            dd.record("a1", 0.3)
        # Improve
        for _ in range(5):
            dd.record("a1", 0.9)
        drift = dd.get_drift("a1")
        assert drift is not None
        assert drift < 0  # Improvement, not degradation
        assert dd.is_flagged("a1") is False

    def test_multi_agent_independent(self):
        dd = DriftDetector(window_size=3, baseline_size=5, drift_threshold=0.2)
        for _ in range(8):
            dd.record("a1", 0.9)
            dd.record("a2", 0.5)
        # a1 degrades, a2 stays stable
        for _ in range(3):
            dd.record("a1", 0.2)
            dd.record("a2", 0.5)
        # Need enough total data (5 + 3 = 8)
        assert dd.is_flagged("a1") is True
        assert dd.is_flagged("a2") is False

    def test_flagged_agents(self):
        dd = DriftDetector(window_size=3, baseline_size=5, drift_threshold=0.2)
        for _ in range(5):
            dd.record("a1", 0.9)
        for _ in range(3):
            dd.record("a1", 0.1)
        flagged = dd.get_flagged_agents()
        assert "a1" in flagged
        assert flagged["a1"] > 0.2

    def test_reset(self):
        dd = DriftDetector(window_size=3, baseline_size=5)
        for _ in range(8):
            dd.record("a1", 0.9)
        dd.reset()
        assert dd.get_drift("a1") is None
        assert dd.get_flagged_agents() == {}

    def test_reset_agent(self):
        dd = DriftDetector(window_size=3, baseline_size=5)
        for _ in range(8):
            dd.record("a1", 0.9)
            dd.record("a2", 0.9)
        dd.reset_agent("a1")
        assert dd.get_drift("a1") is None
        assert dd.get_drift("a2") is not None

    def test_validation(self):
        with pytest.raises(ValueError):
            DriftDetector(window_size=0)
        with pytest.raises(ValueError):
            DriftDetector(baseline_size=0)
        with pytest.raises(ValueError):
            DriftDetector(drift_threshold=1.5)


# ── Compositionality ───────────────────────────────────────────────


class TestCompositionality:
    def test_single_stage(self):
        stages = [StageGuarantee("stage1", p=0.95, delta=0.02)]
        bound = compute_pipeline_bound(stages)
        assert bound.p_pipeline == pytest.approx(0.95)
        assert bound.delta_pipeline == pytest.approx(0.02)
        assert bound.n_stages == 1

    def test_three_stage_pipeline(self):
        stages = [
            StageGuarantee("A", p=0.95, delta=0.01),
            StageGuarantee("B", p=0.90, delta=0.02),
            StageGuarantee("C", p=0.98, delta=0.005),
        ]
        bound = compute_pipeline_bound(stages)
        # p_pipeline = 0.95 * 0.90 * 0.98 = 0.8379
        assert bound.p_pipeline == pytest.approx(0.95 * 0.90 * 0.98, rel=1e-6)
        # delta_pipeline = 1 - (0.99 * 0.98 * 0.995)
        expected_delta = 1.0 - (0.99 * 0.98 * 0.995)
        assert bound.delta_pipeline == pytest.approx(expected_delta, rel=1e-6)
        assert bound.n_stages == 3
        assert len(bound.stage_details) == 3

    def test_perfect_stages(self):
        stages = [
            StageGuarantee("A", p=1.0, delta=0.0),
            StageGuarantee("B", p=1.0, delta=0.0),
        ]
        bound = compute_pipeline_bound(stages)
        assert bound.p_pipeline == pytest.approx(1.0)
        assert bound.delta_pipeline == pytest.approx(0.0)

    def test_degradation_with_more_stages(self):
        """More stages => lower pipeline compliance."""
        short = compute_pipeline_bound([
            StageGuarantee("A", p=0.95, delta=0.01),
        ])
        long = compute_pipeline_bound([
            StageGuarantee("A", p=0.95, delta=0.01),
            StageGuarantee("B", p=0.95, delta=0.01),
            StageGuarantee("C", p=0.95, delta=0.01),
        ])
        assert long.p_pipeline < short.p_pipeline
        assert long.delta_pipeline > short.delta_pipeline

    def test_empty_pipeline_raises(self):
        with pytest.raises(ValueError, match="at least one stage"):
            compute_pipeline_bound([])

    def test_invalid_stage_values(self):
        with pytest.raises(ValueError):
            StageGuarantee("bad", p=1.5, delta=0.0)
        with pytest.raises(ValueError):
            StageGuarantee("bad", p=0.5, delta=-0.1)

    def test_pipeline_with_drift(self):
        stages = [
            StageGuarantee("A", p=0.95, delta=0.01),
            StageGuarantee("B", p=0.90, delta=0.02),
        ]
        # No drift
        bound_t0 = compute_pipeline_bound_with_drift(stages, drift_rate=0.0, time_steps=10)
        assert bound_t0.p_pipeline == pytest.approx(0.95 * 0.90, rel=1e-6)

        # With drift D*=0.01 at t=10: p_A=0.85, p_B=0.80
        bound_t10 = compute_pipeline_bound_with_drift(stages, drift_rate=0.01, time_steps=10)
        assert bound_t10.p_pipeline == pytest.approx(0.85 * 0.80, rel=1e-6)
        assert bound_t10.p_pipeline < bound_t0.p_pipeline

    def test_drift_clamps_at_zero(self):
        stages = [StageGuarantee("A", p=0.5, delta=0.01)]
        bound = compute_pipeline_bound_with_drift(stages, drift_rate=0.1, time_steps=10)
        # p = 0.5 - 0.1*10 = -0.5 -> clamped to 0
        assert bound.p_pipeline == pytest.approx(0.0)

    def test_drift_empty_pipeline_raises(self):
        with pytest.raises(ValueError):
            compute_pipeline_bound_with_drift([], drift_rate=0.01, time_steps=5)
