"""Tests for the Diversity as Defense (DaD) governance lever."""

import math

import pytest
from pydantic import ValidationError

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.diversity import DiversityDefenseLever
from swarm.governance.engine import GovernanceEngine
from swarm.models.agent import AgentType
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestDiversityConfig:
    """Tests for DaD configuration validation."""

    def test_defaults_valid(self):
        """Default DaD config should pass validation."""
        config = GovernanceConfig()
        assert config.diversity_enabled is False
        assert config.diversity_rho_max == 0.5

    def test_invalid_rho_max(self):
        with pytest.raises(ValidationError, match="diversity_rho_max"):
            GovernanceConfig(diversity_rho_max=1.5)

    def test_invalid_rho_max_negative(self):
        with pytest.raises(ValidationError, match="diversity_rho_max"):
            GovernanceConfig(diversity_rho_max=-0.1)

    def test_invalid_entropy_min(self):
        with pytest.raises(ValidationError, match="diversity_entropy_min"):
            GovernanceConfig(diversity_entropy_min=-1.0)

    def test_invalid_adversarial_fraction_min(self):
        with pytest.raises(
            ValidationError, match="diversity_adversarial_fraction_min"
        ):
            GovernanceConfig(diversity_adversarial_fraction_min=1.5)

    def test_invalid_disagreement_tau(self):
        with pytest.raises(
            ValidationError, match="diversity_disagreement_tau"
        ):
            GovernanceConfig(diversity_disagreement_tau=-0.1)

    def test_invalid_error_threshold_p(self):
        with pytest.raises(
            ValidationError, match="diversity_error_threshold_p"
        ):
            GovernanceConfig(diversity_error_threshold_p=2.0)

    def test_invalid_correlation_penalty_rate(self):
        with pytest.raises(
            ValidationError, match="diversity_correlation_penalty_rate"
        ):
            GovernanceConfig(diversity_correlation_penalty_rate=-0.1)

    def test_invalid_entropy_penalty_rate(self):
        with pytest.raises(
            ValidationError, match="diversity_entropy_penalty_rate"
        ):
            GovernanceConfig(diversity_entropy_penalty_rate=-1.0)

    def test_invalid_audit_cost(self):
        with pytest.raises(ValidationError, match="diversity_audit_cost"):
            GovernanceConfig(diversity_audit_cost=-1.0)

    def test_invalid_correlation_window(self):
        with pytest.raises(
            ValidationError, match="diversity_correlation_window"
        ):
            GovernanceConfig(diversity_correlation_window=0)


# ---------------------------------------------------------------------------
# Static math helpers
# ---------------------------------------------------------------------------


class TestPopulationMix:
    """Tests for compute_population_mix."""

    def test_empty(self):
        mix = DiversityDefenseLever.compute_population_mix({})
        assert mix == {}

    def test_single_type(self):
        agents = {"a1": "honest", "a2": "honest", "a3": "honest"}
        mix = DiversityDefenseLever.compute_population_mix(agents)
        assert mix == {"honest": pytest.approx(1.0)}

    def test_uniform_two_types(self):
        agents = {"a1": "honest", "a2": "adversarial"}
        mix = DiversityDefenseLever.compute_population_mix(agents)
        assert mix["honest"] == pytest.approx(0.5)
        assert mix["adversarial"] == pytest.approx(0.5)

    def test_uneven_mix(self):
        agents = {
            "a1": "honest",
            "a2": "honest",
            "a3": "honest",
            "a4": "adversarial",
        }
        mix = DiversityDefenseLever.compute_population_mix(agents)
        assert mix["honest"] == pytest.approx(0.75)
        assert mix["adversarial"] == pytest.approx(0.25)


class TestShannonEntropy:
    """Tests for compute_shannon_entropy."""

    def test_single_type_zero_entropy(self):
        entropy = DiversityDefenseLever.compute_shannon_entropy({"a": 1.0})
        assert entropy == pytest.approx(0.0)

    def test_uniform_binary_max_entropy(self):
        entropy = DiversityDefenseLever.compute_shannon_entropy(
            {"a": 0.5, "b": 0.5}
        )
        assert entropy == pytest.approx(math.log(2))

    def test_uniform_four_types(self):
        mix = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        entropy = DiversityDefenseLever.compute_shannon_entropy(mix)
        assert entropy == pytest.approx(math.log(4))

    def test_skewed_lower_than_uniform(self):
        uniform_entropy = DiversityDefenseLever.compute_shannon_entropy(
            {"a": 0.5, "b": 0.5}
        )
        skewed_entropy = DiversityDefenseLever.compute_shannon_entropy(
            {"a": 0.9, "b": 0.1}
        )
        assert skewed_entropy < uniform_entropy

    def test_empty_mix(self):
        entropy = DiversityDefenseLever.compute_shannon_entropy({})
        assert entropy == pytest.approx(0.0)


class TestPairwiseCorrelation:
    """Tests for compute_pairwise_correlation."""

    def test_identical_sequences_correlation_one(self):
        errors = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
        rho = DiversityDefenseLever.compute_pairwise_correlation(errors, errors)
        assert rho == pytest.approx(1.0)

    def test_opposite_sequences_correlation_neg_one(self):
        ei = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
        ej = [0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
        rho = DiversityDefenseLever.compute_pairwise_correlation(ei, ej)
        assert rho == pytest.approx(-1.0)

    def test_uncorrelated_near_zero(self):
        # Large independent sequences should have correlation near 0
        import random

        rng = random.Random(42)
        ei = [rng.randint(0, 1) for _ in range(1000)]
        ej = [rng.randint(0, 1) for _ in range(1000)]
        rho = DiversityDefenseLever.compute_pairwise_correlation(ei, ej)
        assert abs(rho) < 0.15  # Should be near zero

    def test_constant_sequence_returns_zero(self):
        ei = [1, 1, 1, 1, 1]
        ej = [0, 1, 0, 1, 0]
        rho = DiversityDefenseLever.compute_pairwise_correlation(ei, ej)
        assert rho == pytest.approx(0.0)

    def test_too_short_returns_zero(self):
        rho = DiversityDefenseLever.compute_pairwise_correlation([1], [0])
        assert rho == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        rho = DiversityDefenseLever.compute_pairwise_correlation([], [])
        assert rho == pytest.approx(0.0)


class TestRiskSurrogate:
    """Tests for compute_risk_surrogate."""

    def test_zero_error_zero_risk(self):
        risk = DiversityDefenseLever.compute_risk_surrogate(0.0, 0.5, 10)
        assert risk == pytest.approx(0.0)

    def test_perfect_accuracy_zero_risk(self):
        risk = DiversityDefenseLever.compute_risk_surrogate(1.0, 0.5, 10)
        assert risk == pytest.approx(0.0)

    def test_risk_increases_with_correlation(self):
        risk_low = DiversityDefenseLever.compute_risk_surrogate(0.3, 0.1, 10)
        risk_high = DiversityDefenseLever.compute_risk_surrogate(0.3, 0.9, 10)
        assert risk_high > risk_low

    def test_risk_formula(self):
        # R = p(1-p)(1 + (N-1)*rho)
        p, rho, n = 0.3, 0.5, 10
        expected = p * (1 - p) * (1 + (n - 1) * rho)
        actual = DiversityDefenseLever.compute_risk_surrogate(p, rho, n)
        assert actual == pytest.approx(expected)

    def test_zero_agents(self):
        risk = DiversityDefenseLever.compute_risk_surrogate(0.3, 0.5, 0)
        assert risk == pytest.approx(0.0)

    def test_single_agent_no_correlation_effect(self):
        # With N=1, (N-1)*rho = 0 so R = p(1-p)
        risk = DiversityDefenseLever.compute_risk_surrogate(0.3, 0.9, 1)
        assert risk == pytest.approx(0.3 * 0.7)


class TestDisagreementRate:
    """Tests for compute_disagreement_rate."""

    def test_unanimous_zero_disagreement(self):
        d = DiversityDefenseLever.compute_disagreement_rate([1, 1, 1, 1])
        assert d == pytest.approx(0.0)

    def test_split_maximum_disagreement(self):
        d = DiversityDefenseLever.compute_disagreement_rate([1, 0, 1, 0])
        assert d == pytest.approx(0.5)

    def test_majority_partial_disagreement(self):
        d = DiversityDefenseLever.compute_disagreement_rate([1, 1, 1, 0])
        assert d == pytest.approx(0.25)

    def test_empty(self):
        d = DiversityDefenseLever.compute_disagreement_rate([])
        assert d == pytest.approx(0.0)

    def test_single_decision(self):
        d = DiversityDefenseLever.compute_disagreement_rate([1])
        assert d == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Lever lifecycle hooks
# ---------------------------------------------------------------------------


def _make_diverse_state() -> EnvState:
    """Create an EnvState with a mix of agent types."""
    state = EnvState()
    state.add_agent("h1", agent_type=AgentType.HONEST)
    state.add_agent("h2", agent_type=AgentType.HONEST)
    state.add_agent("h3", agent_type=AgentType.HONEST)
    state.add_agent("o1", agent_type=AgentType.OPPORTUNISTIC)
    state.add_agent("a1", agent_type=AgentType.ADVERSARIAL)
    return state


def _make_homogeneous_state() -> EnvState:
    """Create an EnvState with all-honest agents."""
    state = EnvState()
    for i in range(5):
        state.add_agent(f"h{i}", agent_type=AgentType.HONEST)
    return state


class TestDiversityLeverDisabled:
    """When disabled, the lever should be inert."""

    def test_on_interaction_no_effect(self):
        config = GovernanceConfig(diversity_enabled=False)
        lever = DiversityDefenseLever(config)
        state = EnvState()
        interaction = SoftInteraction(initiator="a1", p=0.3)
        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.0
        assert effect.cost_b == 0.0

    def test_on_epoch_start_no_effect(self):
        config = GovernanceConfig(diversity_enabled=False)
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()
        effect = lever.on_epoch_start(state, epoch=1)
        assert effect.cost_a == 0.0
        assert effect.cost_b == 0.0


class TestDiversityLeverErrorTracking:
    """Tests for error recording in on_interaction."""

    def test_records_error_without_ground_truth(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_error_threshold_p=0.5,
            diversity_disagreement_tau=1.0,  # No audit triggers
        )
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()

        # Low p → error
        interaction = SoftInteraction(initiator="h1", p=0.3)
        lever.on_interaction(interaction, state)

        history = lever.get_error_history()
        assert history["h1"] == [1]

    def test_records_non_error_without_ground_truth(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_error_threshold_p=0.5,
            diversity_disagreement_tau=1.0,
        )
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()

        # High p → no error
        interaction = SoftInteraction(initiator="h1", p=0.8)
        lever.on_interaction(interaction, state)

        history = lever.get_error_history()
        assert history["h1"] == [0]

    def test_records_error_with_ground_truth_mismatch(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_error_threshold_p=0.5,
            diversity_disagreement_tau=1.0,
        )
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()

        # p >= 0.5 predicts positive, but ground truth is -1 → error
        interaction = SoftInteraction(initiator="h1", p=0.7, ground_truth=-1)
        lever.on_interaction(interaction, state)

        history = lever.get_error_history()
        assert history["h1"] == [1]

    def test_records_non_error_with_ground_truth_match(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_error_threshold_p=0.5,
            diversity_disagreement_tau=1.0,
        )
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()

        # p >= 0.5 predicts positive, ground truth is +1 → correct
        interaction = SoftInteraction(initiator="h1", p=0.7, ground_truth=1)
        lever.on_interaction(interaction, state)

        history = lever.get_error_history()
        assert history["h1"] == [0]

    def test_window_trimming(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_correlation_window=5,
            diversity_disagreement_tau=1.0,
        )
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()

        for i in range(10):
            interaction = SoftInteraction(
                initiator="h1", p=0.3 if i % 2 == 0 else 0.8
            )
            lever.on_interaction(interaction, state)

        history = lever.get_error_history()
        assert len(history["h1"]) == 5


class TestDiversityLeverDisagreementAudit:
    """Tests for Rule 4: disagreement-triggered audit."""

    def test_audit_triggered_on_high_disagreement(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_disagreement_tau=0.3,
            diversity_audit_cost=0.2,
        )
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()

        # First, seed diverse error patterns so disagreement is high
        # Agent h1 gets error, h2 no error → disagreement = 0.5
        lever.on_interaction(
            SoftInteraction(initiator="h1", p=0.3), state
        )
        effect = lever.on_interaction(
            SoftInteraction(initiator="h2", p=0.8), state
        )

        # Disagreement rate should be 0.5 (1 error, 1 non-error)
        assert effect.details["disagreement_rate"] == pytest.approx(0.5)
        assert effect.details["audit_triggered"] is True
        assert effect.cost_a == pytest.approx(0.2)

    def test_no_audit_when_below_tau(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_disagreement_tau=0.8,
            diversity_audit_cost=0.2,
        )
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()

        # All agents make the same decision (all errors)
        lever.on_interaction(
            SoftInteraction(initiator="h1", p=0.3), state
        )
        effect = lever.on_interaction(
            SoftInteraction(initiator="h2", p=0.3), state
        )

        assert effect.details["audit_triggered"] is False
        assert effect.cost_a == 0.0


class TestDiversityLeverEpochStart:
    """Tests for epoch-start diversity constraint checks."""

    def test_correlation_cap_violation_adds_cost(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_rho_max=0.1,
            diversity_correlation_penalty_rate=1.0,
            diversity_entropy_min=0.0,  # No entropy constraint
        )
        lever = DiversityDefenseLever(config)
        state = _make_homogeneous_state()

        # Seed highly correlated errors for all agents
        for _ in range(20):
            for agent_id in state.agents:
                lever._error_history[agent_id].append(1)
        for _ in range(20):
            for agent_id in state.agents:
                lever._error_history[agent_id].append(0)

        effect = lever.on_epoch_start(state, epoch=1)

        # With identical error patterns, rho should be ~1.0
        # Excess = rho - 0.1 ≈ 0.9, penalty = 0.9 * 1.0 = 0.9
        assert effect.cost_a > 0
        assert effect.cost_b > 0
        assert "correlation_cap" in effect.details["violations"]

    def test_entropy_floor_violation_adds_cost(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_entropy_min=1.0,  # High entropy requirement
            diversity_entropy_penalty_rate=1.0,
            diversity_rho_max=1.0,  # No correlation constraint
        )
        lever = DiversityDefenseLever(config)
        state = _make_homogeneous_state()  # All honest → entropy = 0

        effect = lever.on_epoch_start(state, epoch=1)

        assert effect.cost_a > 0
        assert "entropy_floor" in effect.details["violations"]

    def test_no_violations_no_cost(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_rho_max=1.0,  # Very permissive
            diversity_entropy_min=0.0,  # No entropy requirement
            diversity_adversarial_fraction_min=0.0,
        )
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()

        effect = lever.on_epoch_start(state, epoch=1)

        assert effect.cost_a == 0.0
        assert effect.cost_b == 0.0
        assert effect.details["violations"] == []

    def test_adversarial_fraction_warning(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_adversarial_fraction_min=0.3,  # Need 30% adversarial
            diversity_rho_max=1.0,
            diversity_entropy_min=0.0,
        )
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()  # 1/5 = 20% adversarial

        effect = lever.on_epoch_start(state, epoch=1)

        # Warning only, no cost
        assert "adversarial_fraction_low" in effect.details["violations"]
        # Cost comes only from correlation and entropy violations
        assert effect.details["adversarial_fraction_satisfied"] is False

    def test_adversarial_fraction_satisfied(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_adversarial_fraction_min=0.1,  # Need 10%
            diversity_rho_max=1.0,
            diversity_entropy_min=0.0,
        )
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()  # 1/5 = 20% adversarial

        effect = lever.on_epoch_start(state, epoch=1)

        assert effect.details["adversarial_fraction_satisfied"] is True

    def test_metrics_snapshot_populated(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_rho_max=1.0,
            diversity_entropy_min=0.0,
        )
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()

        lever.on_epoch_start(state, epoch=1)

        metrics = lever.get_metrics()
        assert metrics is not None
        assert "honest" in metrics.population_mix
        assert metrics.population_mix["honest"] == pytest.approx(0.6)
        assert metrics.entropy > 0
        assert metrics.risk_surrogate >= 0


class TestDiversityLeverMetrics:
    """Tests for the DiversityMetrics dataclass and full computation."""

    def test_diverse_population_higher_entropy(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_rho_max=1.0,
            diversity_entropy_min=0.0,
        )
        lever_diverse = DiversityDefenseLever(config)
        lever_homo = DiversityDefenseLever(config)

        state_diverse = _make_diverse_state()
        state_homo = _make_homogeneous_state()

        lever_diverse.on_epoch_start(state_diverse, epoch=1)
        lever_homo.on_epoch_start(state_homo, epoch=1)

        m_diverse = lever_diverse.get_metrics()
        m_homo = lever_homo.get_metrics()

        assert m_diverse.entropy > m_homo.entropy
        assert m_homo.entropy == pytest.approx(0.0)

    def test_correlated_errors_increase_risk(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_rho_max=1.0,
            diversity_entropy_min=0.0,
        )
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()

        # Correlated errors: all agents err on same tasks
        for _ in range(20):
            for agent_id in state.agents:
                lever._error_history[agent_id].append(1)
        for _ in range(20):
            for agent_id in state.agents:
                lever._error_history[agent_id].append(0)

        lever.on_epoch_start(state, epoch=1)
        metrics_corr = lever.get_metrics()

        # Independent errors
        lever2 = DiversityDefenseLever(config)
        import random

        rng = random.Random(42)
        for _ in range(40):
            for agent_id in state.agents:
                lever2._error_history[agent_id].append(rng.randint(0, 1))

        lever2.on_epoch_start(state, epoch=1)
        metrics_indep = lever2.get_metrics()

        assert metrics_corr.mean_correlation > metrics_indep.mean_correlation
        assert metrics_corr.risk_surrogate > metrics_indep.risk_surrogate


class TestDiversityLeverClearHistory:
    """Tests for clear_history."""

    def test_clears_all_state(self):
        config = GovernanceConfig(diversity_enabled=True)
        lever = DiversityDefenseLever(config)
        state = _make_diverse_state()

        lever.on_interaction(
            SoftInteraction(initiator="h1", p=0.3), state
        )
        lever.on_epoch_start(state, epoch=1)

        assert lever.get_metrics() is not None
        assert len(lever.get_error_history()) > 0

        lever.clear_history()

        assert lever.get_metrics() is None
        assert len(lever.get_error_history()) == 0


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


class TestDiversityEngineIntegration:
    """Tests for DaD lever integration with GovernanceEngine."""

    def test_lever_registered_in_engine(self):
        config = GovernanceConfig(diversity_enabled=True)
        engine = GovernanceEngine(config)
        names = engine.get_registered_lever_names()
        assert "diversity_defense" in names

    def test_engine_diversity_lever_accessor(self):
        config = GovernanceConfig(diversity_enabled=True)
        engine = GovernanceEngine(config)
        lever = engine.get_diversity_lever()
        assert lever is not None
        assert isinstance(lever, DiversityDefenseLever)

    def test_engine_diversity_metrics_accessor(self):
        config = GovernanceConfig(diversity_enabled=True)
        engine = GovernanceEngine(config)
        state = _make_diverse_state()

        # Before any epoch, metrics should be None
        assert engine.get_diversity_metrics() is None

        # After epoch start, metrics should be populated
        engine.apply_epoch_start(state, epoch=1)
        metrics = engine.get_diversity_metrics()
        assert metrics is not None

    def test_engine_applies_diversity_on_interaction(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_disagreement_tau=0.3,
            diversity_audit_cost=0.5,
        )
        engine = GovernanceEngine(config)
        state = _make_diverse_state()

        # Create disagreement by feeding different error patterns
        engine.apply_interaction(
            SoftInteraction(initiator="h1", p=0.3, accepted=True), state
        )
        effect = engine.apply_interaction(
            SoftInteraction(initiator="h2", p=0.8, accepted=True), state
        )

        # The diversity lever should contribute audit cost
        assert effect.cost_a > 0

    def test_engine_applies_diversity_on_epoch_start(self):
        config = GovernanceConfig(
            diversity_enabled=True,
            diversity_entropy_min=2.0,  # Very high, will violate
            diversity_entropy_penalty_rate=1.0,
            diversity_rho_max=1.0,
        )
        engine = GovernanceEngine(config)
        state = _make_homogeneous_state()

        effect = engine.apply_epoch_start(state, epoch=1)

        # Should have entropy violation cost
        assert effect.cost_a > 0


# ---------------------------------------------------------------------------
# Governance corollary tests: diversity beats scaling
# ---------------------------------------------------------------------------


class TestDiversityBeatsScaling:
    """Tests verifying the core governance corollary.

    With constant high correlation, adding more identical agents does not
    reduce risk.  But reducing correlation (via diversity) does.
    """

    def test_risk_grows_with_n_when_rho_constant(self):
        """Var(S) ~ N^2 * rho * p(1-p) when rho is high and constant."""
        p = 0.3
        rho = 0.8
        risk_10 = DiversityDefenseLever.compute_risk_surrogate(p, rho, 10)
        risk_100 = DiversityDefenseLever.compute_risk_surrogate(p, rho, 100)
        # Risk should increase with N when rho is constant
        assert risk_100 > risk_10

    def test_risk_controlled_when_rho_decreases(self):
        """If rho ~ 0, Var(S) ~ N * p(1-p); risk grows linearly."""
        p = 0.3
        # Compare: 100 agents with rho=0.01 vs 10 agents with rho=0.8
        risk_100_diverse = DiversityDefenseLever.compute_risk_surrogate(
            p, 0.01, 100
        )
        risk_10_homo = DiversityDefenseLever.compute_risk_surrogate(
            p, 0.8, 10
        )
        # 100 diverse agents should have lower risk than 10 homogeneous ones
        assert risk_100_diverse < risk_10_homo
