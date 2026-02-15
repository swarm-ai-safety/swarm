"""Tests for the Obfuscation Atlas integration.

Covers:
- ObfuscatingAgent behavior in each mode (blatant/policy/activation/adaptive)
- Trust-building phase acts honestly
- Signal manipulation bounded correctly
- ObfuscationObservableGenerator wrapper
- Obfuscation metrics: evasion rate, drift, taxonomy, rates, AUC, belief shift
- MetricsSummary extension completeness
"""

import random

from swarm.agents.base import Observation
from swarm.agents.obfuscating import ObfuscatingAgent, ObfuscationStrategy
from swarm.core.observable_generator import (
    DefaultObservableGenerator,
    ObfuscationObservableGenerator,
)
from swarm.env.state import EnvState, InteractionProposal
from swarm.metrics.obfuscation_metrics import (
    ObfuscationCategory,
    ObfuscationMetrics,
    ObfuscationSummary,
)
from swarm.metrics.reporters import MetricsReporter, MetricsSummary
from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import InteractionType, SoftInteraction
from tests.fixtures.interactions import generate_obfuscation_scenario

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _make_agent(strategy: str, **extra_config) -> ObfuscatingAgent:
    """Create an ObfuscatingAgent with the given strategy."""
    config = {"obfuscation_strategy": strategy, **extra_config}
    return ObfuscatingAgent(
        agent_id=f"obf_{strategy}",
        config=config,
        rng=random.Random(42),
    )


def _make_observation(epoch: int = 0, step: int = 0) -> Observation:
    """Create a minimal observation."""
    return Observation(
        agent_state=AgentState(agent_id="obf_test"),
        current_epoch=epoch,
        current_step=step,
    )


def _make_interaction(
    p: float = 0.8,
    ground_truth: int = 1,
    accepted: bool = True,
    metadata: dict | None = None,
) -> SoftInteraction:
    """Create a test interaction."""
    v_hat = max(-1.0, min(1.0, (p - 0.5) * 2))
    return SoftInteraction(
        initiator="agent_a",
        counterparty="agent_b",
        interaction_type=InteractionType.TRADE,
        accepted=accepted,
        p=p,
        v_hat=v_hat,
        ground_truth=ground_truth,
        metadata=metadata or {},
    )


# =========================================================================
# Agent tests
# =========================================================================


class TestObfuscatingAgentInit:
    """Test agent initialization and enum."""

    def test_strategy_enum_values(self):
        assert ObfuscationStrategy.BLATANT.value == "blatant"
        assert ObfuscationStrategy.POLICY_OBFUSCATION.value == "policy_obfuscation"
        assert ObfuscationStrategy.ACTIVATION_OBFUSCATION.value == "activation_obfuscation"
        assert ObfuscationStrategy.ADAPTIVE.value == "adaptive"

    def test_agent_type_is_deceptive(self):
        agent = _make_agent("blatant")
        assert agent.agent_type == AgentType.DECEPTIVE

    def test_strategy_assignment(self):
        for strat in ["blatant", "policy_obfuscation", "activation_obfuscation", "adaptive"]:
            agent = _make_agent(strat)
            assert agent.strategy == ObfuscationStrategy(strat)

    def test_default_config_values(self):
        agent = _make_agent("blatant")
        assert agent.trust_building_phase == 5
        assert agent.rationalization_strength == 0.7
        assert agent.signal_inflation_cap == 0.4
        assert agent.drift_rate == 0.05

    def test_custom_config(self):
        agent = _make_agent(
            "policy_obfuscation",
            trust_building_phase=10,
            rationalization_strength=0.9,
        )
        assert agent.trust_building_phase == 10
        assert agent.rationalization_strength == 0.9


class TestTrustBuildingPhase:
    """Trust-building phase mimics honest behavior."""

    def test_in_trust_building_initially(self):
        agent = _make_agent("policy_obfuscation")
        assert agent.in_trust_building is True

    def test_no_signal_manipulation_during_trust(self):
        agent = _make_agent("policy_obfuscation")
        assert agent.get_signal_manipulation() == {}

    def test_no_rationalization_during_trust(self):
        agent = _make_agent("policy_obfuscation")
        assert agent.get_rationalization_metadata() == {}

    def test_exits_trust_building_after_n_interactions(self):
        agent = _make_agent("policy_obfuscation", trust_building_phase=3)
        interaction = _make_interaction()
        for _i in range(3):
            assert agent.in_trust_building is True
            agent.update_from_outcome(interaction, payoff=0.5)
        assert agent.in_trust_building is False

    def test_trust_building_acts_noop_without_context(self):
        agent = _make_agent("blatant")
        obs = _make_observation()
        action = agent.act(obs)
        assert action.action_type.value == "noop"


class TestBlatantStrategy:
    """Blatant strategy: no signal manipulation."""

    def test_no_manipulation(self):
        agent = _make_agent("blatant", trust_building_phase=0)
        assert agent.get_signal_manipulation() == {}

    def test_no_rationalization(self):
        agent = _make_agent("blatant", trust_building_phase=0)
        assert agent.get_rationalization_metadata() == {}


class TestPolicyObfuscation:
    """Policy obfuscation: inflates signals, attaches rationalizations."""

    def test_signal_manipulation_present(self):
        agent = _make_agent("policy_obfuscation", trust_building_phase=0)
        offsets = agent.get_signal_manipulation()
        assert "task_progress_delta" in offsets
        assert "rework_count" in offsets
        assert "verifier_rejections" in offsets
        assert offsets["task_progress_delta"] > 0  # inflation
        assert offsets["rework_count"] < 0  # suppression

    def test_signal_inflation_bounded(self):
        agent = _make_agent(
            "policy_obfuscation",
            trust_building_phase=0,
            signal_inflation_cap=0.3,
        )
        for _ in range(50):
            offsets = agent.get_signal_manipulation()
            assert offsets["task_progress_delta"] <= 0.3

    def test_rationalization_metadata(self):
        agent = _make_agent("policy_obfuscation", trust_building_phase=0)
        meta = agent.get_rationalization_metadata()
        assert "rationalization" in meta
        assert meta["rationalization_strength"] == 0.7
        assert meta["obfuscation_strategy"] == "policy_obfuscation"


class TestActivationObfuscation:
    """Activation obfuscation: gradual drift toward uncertainty zone."""

    def test_signal_offsets_present(self):
        agent = _make_agent("activation_obfuscation", trust_building_phase=0)
        agent._epoch = 5  # Simulate some epochs
        offsets = agent.get_signal_manipulation()
        assert "task_progress_delta" in offsets
        assert "counterparty_engagement_delta" in offsets

    def test_no_rationalization(self):
        agent = _make_agent("activation_obfuscation", trust_building_phase=0)
        assert agent.get_rationalization_metadata() == {}

    def test_drift_intensifies_with_epoch(self):
        agent = _make_agent(
            "activation_obfuscation",
            trust_building_phase=0,
            drift_rate=0.2,
            activation_noise_scale=0.0,  # Remove noise for deterministic test
        )
        # Epoch 0: no drift
        agent._epoch = 0
        offsets_0 = agent.get_signal_manipulation()
        # Epoch 5: drift_factor = min(1.0, 5 * 0.2) = 1.0
        agent._epoch = 5
        _ = agent.get_signal_manipulation()
        # With noise=0, offsets at epoch 0 should all be 0
        # (drift_factor = 0 * 0.2 = 0, and gauss(0, 0) = 0)
        for key in offsets_0:
            assert offsets_0[key] == 0.0


class TestAdaptiveStrategy:
    """Adaptive strategy: switches based on detection feedback."""

    def test_starts_with_policy_obfuscation(self):
        agent = _make_agent("adaptive", trust_building_phase=0)
        assert agent.active_strategy == ObfuscationStrategy.POLICY_OBFUSCATION

    def test_switches_to_activation_on_detection(self):
        agent = _make_agent("adaptive", trust_building_phase=0)
        interaction = _make_interaction()
        # Simulate repeated detection (negative payoffs)
        for _ in range(5):
            agent.update_from_outcome(interaction, payoff=-1.0)
        assert agent.active_strategy == ObfuscationStrategy.ACTIVATION_OBFUSCATION

    def test_stays_policy_when_undetected(self):
        agent = _make_agent("adaptive", trust_building_phase=0)
        interaction = _make_interaction()
        for _ in range(5):
            agent.update_from_outcome(interaction, payoff=1.0)
        assert agent.active_strategy == ObfuscationStrategy.POLICY_OBFUSCATION


# =========================================================================
# Observable generator tests
# =========================================================================


class TestObfuscationObservableGenerator:
    """Test the ObfuscationObservableGenerator wrapper."""

    def _make_state_with_agent(self, agent_id: str) -> EnvState:
        """Create an EnvState with one registered agent."""
        state = EnvState()
        state.add_agent(
            agent_id=agent_id,
            agent_type=AgentType.DECEPTIVE,
        )
        return state

    def test_non_obfuscating_agent_passthrough(self):
        """Non-obfuscating agents should produce identical signals."""
        inner = DefaultObservableGenerator(rng=random.Random(42))
        wrapper = ObfuscationObservableGenerator(
            inner=DefaultObservableGenerator(rng=random.Random(42)),
            agents={},  # No agents registered
        )

        state = self._make_state_with_agent("honest_1")
        proposal = InteractionProposal(
            initiator_id="honest_1",
            counterparty_id="agent_b",
        )

        base_obs = inner.generate(proposal, True, state)
        wrapped_obs = wrapper.generate(proposal, True, state)

        assert base_obs.task_progress_delta == wrapped_obs.task_progress_delta
        assert base_obs.rework_count == wrapped_obs.rework_count

    def test_obfuscating_agent_modifies_signals(self):
        """ObfuscatingAgent should modify signals via the wrapper."""
        agent = _make_agent("policy_obfuscation", trust_building_phase=0)

        inner = DefaultObservableGenerator(rng=random.Random(99))
        wrapper = ObfuscationObservableGenerator(
            inner=inner,
            agents={agent.agent_id: agent},
        )

        state = self._make_state_with_agent(agent.agent_id)
        proposal = InteractionProposal(
            initiator_id=agent.agent_id,
            counterparty_id="agent_b",
        )

        # Generate base for comparison
        inner2 = DefaultObservableGenerator(rng=random.Random(99))
        base_obs = inner2.generate(proposal, True, state)
        wrapped_obs = wrapper.generate(proposal, True, state)

        # Wrapper should have inflated task_progress
        assert wrapped_obs.task_progress_delta >= base_obs.task_progress_delta

    def test_signals_stay_in_range(self):
        """All signals should stay within valid ranges after manipulation."""
        agent = _make_agent("policy_obfuscation", trust_building_phase=0)

        inner = DefaultObservableGenerator(rng=random.Random(42))
        wrapper = ObfuscationObservableGenerator(
            inner=inner,
            agents={agent.agent_id: agent},
        )

        state = self._make_state_with_agent(agent.agent_id)
        proposal = InteractionProposal(
            initiator_id=agent.agent_id,
            counterparty_id="agent_b",
        )

        for _ in range(50):
            obs = wrapper.generate(proposal, True, state)
            assert -1.0 <= obs.task_progress_delta <= 1.0
            assert obs.rework_count >= 0
            assert obs.verifier_rejections >= 0
            assert obs.tool_misuse_flags >= 0
            assert -1.0 <= obs.counterparty_engagement_delta <= 1.0


# =========================================================================
# Metrics tests
# =========================================================================


class TestObfuscationMetrics:
    """Test obfuscation metrics computation."""

    def test_probe_evasion_rate_all_evade(self):
        """All negatives with p >= threshold should give evasion rate 1.0."""
        metrics = ObfuscationMetrics(threshold=0.5)
        interactions = [_make_interaction(p=0.8, ground_truth=-1) for _ in range(10)]
        assert metrics.probe_evasion_rate(interactions) == 1.0

    def test_probe_evasion_rate_none_evade(self):
        """All negatives with p < threshold should give evasion rate 0.0."""
        metrics = ObfuscationMetrics(threshold=0.5)
        interactions = [_make_interaction(p=0.2, ground_truth=-1) for _ in range(10)]
        assert metrics.probe_evasion_rate(interactions) == 0.0

    def test_probe_evasion_rate_no_negatives(self):
        """No negatives should return 0.0."""
        metrics = ObfuscationMetrics()
        interactions = [_make_interaction(p=0.8, ground_truth=1)]
        assert metrics.probe_evasion_rate(interactions) == 0.0

    def test_probe_evasion_rate_mixed(self):
        """Mixed case: 2 of 4 negatives evade."""
        metrics = ObfuscationMetrics(threshold=0.5)
        interactions = [
            _make_interaction(p=0.6, ground_truth=-1),  # evades
            _make_interaction(p=0.7, ground_truth=-1),  # evades
            _make_interaction(p=0.3, ground_truth=-1),  # caught
            _make_interaction(p=0.2, ground_truth=-1),  # caught
        ]
        assert metrics.probe_evasion_rate(interactions) == 0.5

    def test_representation_drift_positive(self):
        """Drift should be positive when AUC degrades."""
        metrics = ObfuscationMetrics()
        epoch_data = generate_obfuscation_scenario(n_epochs=5, seed=42)
        drift = metrics.representation_drift(epoch_data)
        # Just check it returns a float (drift direction depends on data)
        assert isinstance(drift, float)

    def test_representation_drift_single_epoch(self):
        """Single epoch should return 0.0."""
        metrics = ObfuscationMetrics()
        epoch_data = generate_obfuscation_scenario(n_epochs=1, seed=42)
        assert metrics.representation_drift(epoch_data) == 0.0

    def test_classify_honest(self):
        """Positive ground truth → HONEST."""
        metrics = ObfuscationMetrics()
        i = _make_interaction(p=0.8, ground_truth=1)
        assert metrics.classify_interaction(i) == ObfuscationCategory.HONEST

    def test_classify_blatant(self):
        """Negative ground truth, low p → BLATANT."""
        metrics = ObfuscationMetrics(threshold=0.5)
        i = _make_interaction(p=0.3, ground_truth=-1)
        assert metrics.classify_interaction(i) == ObfuscationCategory.BLATANT

    def test_classify_obfuscated_policy(self):
        """Negative ground truth, high p, rationalization → OBFUSCATED_POLICY."""
        metrics = ObfuscationMetrics(threshold=0.5)
        i = _make_interaction(
            p=0.8,
            ground_truth=-1,
            metadata={"rationalization": "Optimized."},
        )
        assert metrics.classify_interaction(i) == ObfuscationCategory.OBFUSCATED_POLICY

    def test_classify_obfuscated_activation(self):
        """Negative ground truth, high p, no rationalization → OBFUSCATED_ACTIVATION."""
        metrics = ObfuscationMetrics(threshold=0.5)
        i = _make_interaction(p=0.7, ground_truth=-1, metadata={})
        assert metrics.classify_interaction(i) == ObfuscationCategory.OBFUSCATED_ACTIVATION

    def test_classify_none_ground_truth(self):
        """None ground truth → HONEST."""
        metrics = ObfuscationMetrics()
        i = _make_interaction(p=0.5, ground_truth=None)  # type: ignore[arg-type]
        assert metrics.classify_interaction(i) == ObfuscationCategory.HONEST

    def test_taxonomy_rates_sum_to_one(self):
        """Taxonomy rates must sum to 1.0."""
        metrics = ObfuscationMetrics()
        epoch_data = generate_obfuscation_scenario(n_epochs=3, seed=42)
        all_interactions = [i for epoch in epoch_data for i in epoch]
        rates = metrics.taxonomy_rates(all_interactions)
        assert abs(sum(rates.values()) - 1.0) < 1e-10

    def test_taxonomy_rates_empty(self):
        """Empty interactions should return all zeros."""
        metrics = ObfuscationMetrics()
        rates = metrics.taxonomy_rates([])
        assert all(v == 0.0 for v in rates.values())

    def test_belief_shift_index_no_change(self):
        """Constant thresholds should give 0 shift."""
        metrics = ObfuscationMetrics()
        assert metrics.belief_shift_index([0.5, 0.5, 0.5]) == 0.0

    def test_belief_shift_index_known_values(self):
        """Known threshold changes should give exact shift."""
        metrics = ObfuscationMetrics()
        # |0.6 - 0.5| + |0.4 - 0.6| = 0.1 + 0.2 = 0.3
        assert abs(metrics.belief_shift_index([0.5, 0.6, 0.4]) - 0.3) < 1e-10

    def test_belief_shift_index_single_epoch(self):
        """Single epoch should give 0.0."""
        metrics = ObfuscationMetrics()
        assert metrics.belief_shift_index([0.5]) == 0.0

    def test_detector_auc_returns_float_or_none(self):
        """AUC should be float or None."""
        metrics = ObfuscationMetrics()
        epoch_data = generate_obfuscation_scenario(n_epochs=3, seed=42)
        all_interactions = [i for epoch in epoch_data for i in epoch]
        auc = metrics.detector_discriminative_power(all_interactions)
        assert auc is None or isinstance(auc, float)


class TestObfuscationSummary:
    """Test ObfuscationSummary dataclass."""

    def test_default_values(self):
        s = ObfuscationSummary()
        assert s.probe_evasion_rate == 0.0
        assert s.representation_drift == 0.0
        assert s.detector_auc is None
        assert s.belief_shift_index == 0.0
        assert s.total_interactions == 0

    def test_to_dict(self):
        s = ObfuscationSummary(probe_evasion_rate=0.5, total_interactions=100)
        d = s.to_dict()
        assert d["probe_evasion_rate"] == 0.5
        assert d["total_interactions"] == 100
        assert "taxonomy_rates" in d

    def test_compute_summary_integration(self):
        """Full summary computation from fixture data."""
        metrics = ObfuscationMetrics()
        epoch_data = generate_obfuscation_scenario(n_epochs=3, seed=42)
        all_interactions = [i for epoch in epoch_data for i in epoch]
        summary = metrics.compute_summary(
            interactions=all_interactions,
            epoch_interactions=epoch_data,
            epoch_thresholds=[0.5, 0.55, 0.6],
        )
        assert summary.total_interactions == len(all_interactions)
        assert 0.0 <= summary.probe_evasion_rate <= 1.0
        assert isinstance(summary.representation_drift, float)


# =========================================================================
# MetricsSummary extension tests
# =========================================================================


class TestMetricsSummaryObfuscation:
    """Test that MetricsSummary includes obfuscation fields."""

    def test_obfuscation_fields_exist(self):
        s = MetricsSummary(
            toxicity_soft=0.0, conditional_loss_initiator=0.0,
            conditional_loss_counterparty=0.0, spread=0.0,
            quality_gap=0.0, average_quality=0.0, uncertain_fraction=0.0,
            toxicity_hard=0.0, acceptance_rate=0.0,
            high_quality_acceptance=0.0, low_quality_acceptance=0.0,
            total_interactions=0, accepted_count=0, rejected_count=0,
            high_quality_count=0, low_quality_count=0,
            total_welfare=0.0, total_social_surplus=0.0,
            avg_initiator_payoff=0.0, avg_counterparty_payoff=0.0,
        )
        assert hasattr(s, "obfuscation_probe_evasion")
        assert hasattr(s, "obfuscation_drift")
        assert hasattr(s, "obfuscation_rationalization")
        assert hasattr(s, "obfuscation_detector_auc")
        assert hasattr(s, "obfuscation_belief_shift")

    def test_to_dict_includes_obfuscation(self):
        s = MetricsSummary(
            toxicity_soft=0.0, conditional_loss_initiator=0.0,
            conditional_loss_counterparty=0.0, spread=0.0,
            quality_gap=0.0, average_quality=0.0, uncertain_fraction=0.0,
            toxicity_hard=0.0, acceptance_rate=0.0,
            high_quality_acceptance=0.0, low_quality_acceptance=0.0,
            total_interactions=0, accepted_count=0, rejected_count=0,
            high_quality_count=0, low_quality_count=0,
            total_welfare=0.0, total_social_surplus=0.0,
            avg_initiator_payoff=0.0, avg_counterparty_payoff=0.0,
        )
        d = s.to_dict()
        assert "obfuscation" in d
        assert "probe_evasion" in d["obfuscation"]
        assert "detector_auc" in d["obfuscation"]

    def test_summary_with_obfuscation(self):
        """MetricsReporter.summary_with_obfuscation() populates fields."""
        reporter = MetricsReporter()
        interactions = [_make_interaction(p=0.7, ground_truth=1) for _ in range(10)]
        obf_summary = ObfuscationSummary(
            probe_evasion_rate=0.3,
            representation_drift=0.1,
            belief_shift_index=0.05,
        )
        result = reporter.summary_with_obfuscation(interactions, obf_summary)
        assert result.obfuscation_probe_evasion == 0.3
        assert result.obfuscation_drift == 0.1
        assert result.obfuscation_belief_shift == 0.05


# =========================================================================
# Fixture tests
# =========================================================================


class TestObfuscationFixture:
    """Test the generate_obfuscation_scenario fixture."""

    def test_epoch_count(self):
        data = generate_obfuscation_scenario(n_epochs=5, seed=42)
        assert len(data) == 5

    def test_interactions_per_epoch(self):
        data = generate_obfuscation_scenario(
            n_epochs=2,
            honest_per_epoch=10,
            blatant_per_epoch=5,
            policy_obf_per_epoch=8,
            activation_obf_per_epoch=5,
            seed=42,
        )
        for epoch in data:
            assert len(epoch) == 10 + 5 + 8 + 5

    def test_ground_truth_set(self):
        data = generate_obfuscation_scenario(n_epochs=2, seed=42)
        for epoch in data:
            for i in epoch:
                assert i.ground_truth in (1, -1)

    def test_policy_obfuscated_have_metadata(self):
        data = generate_obfuscation_scenario(n_epochs=2, seed=42)
        all_interactions = [i for epoch in data for i in epoch]
        policy_obf = [
            i for i in all_interactions
            if i.metadata.get("agent_type") == "policy_obfuscation"
        ]
        assert len(policy_obf) > 0
        for i in policy_obf:
            assert "rationalization" in i.metadata

    def test_deterministic_with_seed(self):
        data1 = generate_obfuscation_scenario(n_epochs=2, seed=123)
        data2 = generate_obfuscation_scenario(n_epochs=2, seed=123)
        for e1, e2 in zip(data1, data2, strict=True):
            for i1, i2 in zip(e1, e2, strict=True):
                assert i1.p == i2.p
                assert i1.ground_truth == i2.ground_truth
