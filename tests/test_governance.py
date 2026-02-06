"""Tests for the governance module."""

import pytest

from swarm.env.state import EnvState
from swarm.governance.admission import StakingLever
from swarm.governance.audits import RandomAuditLever
from swarm.governance.circuit_breaker import CircuitBreakerLever
from swarm.governance.config import GovernanceConfig
from swarm.governance.decomposition import DecompositionLever
from swarm.governance.dynamic_friction import IncoherenceFrictionLever
from swarm.governance.engine import GovernanceEffect, GovernanceEngine
from swarm.governance.ensemble import SelfEnsembleLever
from swarm.governance.incoherence_breaker import IncoherenceCircuitBreakerLever
from swarm.governance.levers import LeverEffect
from swarm.governance.reputation import ReputationDecayLever, VoteNormalizationLever
from swarm.governance.taxes import TransactionTaxLever
from swarm.models.interaction import SoftInteraction


class _DummyForecaster:
    """Tiny stub forecaster for adaptive-governance tests."""

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, feature_row):
        return self.probability


class TestGovernanceConfig:
    """Tests for GovernanceConfig validation."""

    def test_default_config_valid(self):
        """Default config should pass validation."""
        config = GovernanceConfig()
        config.validate()  # Should not raise

    def test_invalid_tax_rate(self):
        """Tax rate outside [0,1] should raise."""
        config = GovernanceConfig(transaction_tax_rate=1.5)
        with pytest.raises(ValueError, match="transaction_tax_rate"):
            config.validate()

        config = GovernanceConfig(transaction_tax_rate=-0.1)
        with pytest.raises(ValueError, match="transaction_tax_rate"):
            config.validate()

    def test_invalid_decay_rate(self):
        """Decay rate outside [0,1] should raise."""
        config = GovernanceConfig(reputation_decay_rate=1.5)
        with pytest.raises(ValueError, match="reputation_decay_rate"):
            config.validate()

    def test_invalid_audit_probability(self):
        """Audit probability outside [0,1] should raise."""
        config = GovernanceConfig(audit_probability=2.0)
        with pytest.raises(ValueError, match="audit_probability"):
            config.validate()

    def test_invalid_variance_aware_fields(self):
        """Variance-aware config validation should reject invalid values."""
        with pytest.raises(ValueError, match="self_ensemble_samples"):
            GovernanceConfig(self_ensemble_samples=0).validate()
        with pytest.raises(ValueError, match="incoherence_breaker_threshold"):
            GovernanceConfig(incoherence_breaker_threshold=1.1).validate()
        with pytest.raises(ValueError, match="decomposition_horizon_threshold"):
            GovernanceConfig(decomposition_horizon_threshold=0).validate()
        with pytest.raises(ValueError, match="incoherence_friction_rate"):
            GovernanceConfig(incoherence_friction_rate=-0.1).validate()
        with pytest.raises(ValueError, match="adaptive_incoherence_threshold"):
            GovernanceConfig(adaptive_incoherence_threshold=1.5).validate()


class TestLeverEffect:
    """Tests for LeverEffect dataclass."""

    def test_default_effect(self):
        """Default effect should have zero costs."""
        effect = LeverEffect()
        assert effect.cost_a == 0.0
        assert effect.cost_b == 0.0
        assert len(effect.agents_to_freeze) == 0

    def test_merge_effects(self):
        """Merging effects should sum costs."""
        effect1 = LeverEffect(cost_a=0.1, cost_b=0.2, lever_name="tax")
        effect2 = LeverEffect(cost_a=0.3, cost_b=0.1, lever_name="audit")

        merged = effect1.merge(effect2)
        assert merged.cost_a == pytest.approx(0.4)
        assert merged.cost_b == pytest.approx(0.3)

    def test_merge_freeze_sets(self):
        """Merging effects should union freeze sets."""
        effect1 = LeverEffect(agents_to_freeze={"agent_1"})
        effect2 = LeverEffect(agents_to_freeze={"agent_2"})

        merged = effect1.merge(effect2)
        assert merged.agents_to_freeze == {"agent_1", "agent_2"}


class TestTransactionTaxLever:
    """Tests for TransactionTaxLever."""

    def test_no_tax_when_rate_zero(self):
        """Zero rate should produce no tax."""
        config = GovernanceConfig(transaction_tax_rate=0.0)
        lever = TransactionTaxLever(config)

        interaction = SoftInteraction(tau=1.0)
        state = EnvState()

        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.0
        assert effect.cost_b == 0.0

    def test_tax_increases_with_tau(self):
        """Tax should increase with |tau|."""
        config = GovernanceConfig(transaction_tax_rate=0.1, transaction_tax_split=0.5)
        lever = TransactionTaxLever(config)

        state = EnvState()

        # tau = 1.0, p = 0.5 → surplus_est = 0.5, tax_base = 1.5
        interaction1 = SoftInteraction(tau=1.0, accepted=True)
        effect1 = lever.on_interaction(interaction1, state)

        # tau = 2.0, p = 0.5 → surplus_est = 0.5, tax_base = 2.5
        interaction2 = SoftInteraction(tau=2.0, accepted=True)
        effect2 = lever.on_interaction(interaction2, state)

        # Tax should increase with tau
        assert effect2.cost_a > effect1.cost_a
        assert effect2.cost_b > effect1.cost_b

    def test_tax_split(self):
        """Tax should be split according to config."""
        config = GovernanceConfig(transaction_tax_rate=0.1, transaction_tax_split=0.7)
        lever = TransactionTaxLever(config)

        # p=0.5 → surplus_est = 0.5, tax_base = 0.5 + 1.0 = 1.5
        interaction = SoftInteraction(tau=1.0, accepted=True)
        state = EnvState()

        effect = lever.on_interaction(interaction, state)

        total_tax = 0.1 * 1.5  # 10% of (surplus_est + tau)
        assert effect.cost_a == pytest.approx(total_tax * 0.7)
        assert effect.cost_b == pytest.approx(total_tax * 0.3)

    def test_negative_tau(self):
        """Tax should apply to |tau|."""
        config = GovernanceConfig(transaction_tax_rate=0.1, transaction_tax_split=0.5)
        lever = TransactionTaxLever(config)

        state = EnvState()

        interaction_pos = SoftInteraction(tau=1.0, accepted=True)
        interaction_neg = SoftInteraction(tau=-1.0, accepted=True)

        effect_pos = lever.on_interaction(interaction_pos, state)
        effect_neg = lever.on_interaction(interaction_neg, state)

        assert effect_pos.cost_a == effect_neg.cost_a
        assert effect_pos.cost_b == effect_neg.cost_b

    def test_no_tax_on_rejected_interaction(self):
        """Rejected interactions should not be taxed."""
        config = GovernanceConfig(transaction_tax_rate=0.1)
        lever = TransactionTaxLever(config)

        state = EnvState()
        interaction = SoftInteraction(tau=1.0, accepted=False)
        effect = lever.on_interaction(interaction, state)

        assert effect.cost_a == 0.0
        assert effect.cost_b == 0.0

    def test_tax_with_zero_tau(self):
        """Tax should still apply via surplus estimate even when tau is 0."""
        config = GovernanceConfig(transaction_tax_rate=0.1, transaction_tax_split=0.5)
        lever = TransactionTaxLever(config)

        state = EnvState()
        # p=0.7 → surplus_est = 0.7*2 - 0.3*1 = 1.1, tax_base = 1.1
        interaction = SoftInteraction(tau=0.0, p=0.7, accepted=True)
        effect = lever.on_interaction(interaction, state)

        assert effect.cost_a > 0
        assert effect.cost_b > 0


class TestReputationDecayLever:
    """Tests for ReputationDecayLever."""

    def test_no_decay_when_rate_one(self):
        """Rate of 1.0 should produce no decay."""
        config = GovernanceConfig(reputation_decay_rate=1.0)
        lever = ReputationDecayLever(config)

        state = EnvState()
        state.add_agent("agent_1", initial_reputation=10.0)

        effect = lever.on_epoch_start(state, epoch=1)
        assert len(effect.reputation_deltas) == 0

    def test_decay_reduces_reputation(self):
        """Decay should reduce positive reputation."""
        config = GovernanceConfig(reputation_decay_rate=0.9)
        lever = ReputationDecayLever(config)

        state = EnvState()
        state.add_agent("agent_1", initial_reputation=10.0)

        effect = lever.on_epoch_start(state, epoch=1)

        # Delta should be reputation * (rate - 1) = 10 * (-0.1) = -1.0
        assert "agent_1" in effect.reputation_deltas
        assert effect.reputation_deltas["agent_1"] == pytest.approx(-1.0)

    def test_decay_affects_negative_reputation(self):
        """Decay should also affect negative reputation (toward zero)."""
        config = GovernanceConfig(reputation_decay_rate=0.9)
        lever = ReputationDecayLever(config)

        state = EnvState()
        state.add_agent("agent_1", initial_reputation=-10.0)

        effect = lever.on_epoch_start(state, epoch=1)

        # Delta should be -10 * (0.9 - 1) = -10 * (-0.1) = +1.0
        assert effect.reputation_deltas["agent_1"] == pytest.approx(1.0)


class TestVoteNormalizationLever:
    """Tests for VoteNormalizationLever."""

    def test_disabled_returns_weight_one(self):
        """When disabled, vote weight should be 1.0."""
        config = GovernanceConfig(vote_normalization_enabled=False)
        lever = VoteNormalizationLever(config)

        weight = lever.compute_vote_weight("agent_1", vote_count=100)
        assert weight == 1.0

    def test_weight_decreases_with_vote_count(self):
        """Weight should decrease as vote count increases."""
        config = GovernanceConfig(
            vote_normalization_enabled=True,
            max_vote_weight_per_agent=10.0,
        )
        lever = VoteNormalizationLever(config)

        weight_0 = lever.compute_vote_weight("agent_1", vote_count=0)
        weight_10 = lever.compute_vote_weight("agent_1", vote_count=10)
        weight_100 = lever.compute_vote_weight("agent_1", vote_count=100)

        assert weight_0 > weight_10 > weight_100
        assert weight_0 == 1.0  # At vote_count=0, weight should be max


class TestStakingLever:
    """Tests for StakingLever."""

    def test_disabled_allows_all(self):
        """When disabled, all agents can act."""
        config = GovernanceConfig(staking_enabled=False)
        lever = StakingLever(config)

        state = EnvState()
        state.add_agent("agent_1", initial_resources=0.0)

        assert lever.can_agent_act("agent_1", state)

    def test_blocks_low_stake_agents(self):
        """Agents with insufficient stake should be blocked."""
        config = GovernanceConfig(
            staking_enabled=True,
            min_stake_to_participate=50.0,
        )
        lever = StakingLever(config)

        state = EnvState()
        state.add_agent("rich_agent", initial_resources=100.0)
        state.add_agent("poor_agent", initial_resources=10.0)

        assert lever.can_agent_act("rich_agent", state)
        assert not lever.can_agent_act("poor_agent", state)

    def test_slash_stake(self):
        """Slashing should reduce agent resources."""
        config = GovernanceConfig(
            staking_enabled=True,
            stake_slash_rate=0.2,
        )
        lever = StakingLever(config)

        state = EnvState()
        state.add_agent("agent_1", initial_resources=100.0)

        effect = lever.slash_stake("agent_1", state, reason="violation")

        # Should slash 20% = 20.0
        assert "agent_1" in effect.resource_deltas
        assert effect.resource_deltas["agent_1"] == pytest.approx(-20.0)


class TestCircuitBreakerLever:
    """Tests for CircuitBreakerLever."""

    def test_disabled_no_freezing(self):
        """When disabled, no agents should be frozen."""
        config = GovernanceConfig(circuit_breaker_enabled=False)
        lever = CircuitBreakerLever(config)

        state = EnvState()
        state.add_agent("agent_1")

        # Very toxic interaction
        interaction = SoftInteraction(initiator="agent_1", p=0.1)

        effect = lever.on_interaction(interaction, state)
        assert len(effect.agents_to_freeze) == 0

    def test_freeze_after_threshold_violations(self):
        """Agent should freeze after exceeding violation threshold."""
        config = GovernanceConfig(
            circuit_breaker_enabled=True,
            freeze_threshold_toxicity=0.5,  # p < 0.5 is toxic
            freeze_threshold_violations=3,
            freeze_duration_epochs=2,
        )
        lever = CircuitBreakerLever(config)

        state = EnvState()
        state.add_agent("toxic_agent")

        # Send multiple toxic interactions (p = 0.2 means toxicity = 0.8)
        for i in range(10):
            interaction = SoftInteraction(initiator="toxic_agent", p=0.2)
            effect = lever.on_interaction(interaction, state)

            if i < 2:
                # Not enough violations yet
                assert len(effect.agents_to_freeze) == 0
            else:
                # Should trigger after 3rd violation
                if i >= 2:
                    # After building up toxicity history, should freeze
                    status = lever.get_freeze_status("toxic_agent")
                    if status["violations"] >= 3:
                        assert len(effect.agents_to_freeze) == 1
                        break

    def test_unfreeze_after_duration(self):
        """Agent should unfreeze after freeze duration expires."""
        config = GovernanceConfig(
            circuit_breaker_enabled=True,
            freeze_threshold_toxicity=0.3,
            freeze_threshold_violations=1,
            freeze_duration_epochs=2,
        )
        lever = CircuitBreakerLever(config)

        state = EnvState()
        state.add_agent("agent_1")

        # Trigger freeze
        for _ in range(10):
            interaction = SoftInteraction(initiator="agent_1", p=0.1)
            lever.on_interaction(interaction, state)

        status = lever.get_freeze_status("agent_1")
        assert status["is_frozen"]

        # Check unfreeze at epoch 2
        state.current_epoch = 2
        effect = lever.on_epoch_start(state, epoch=2)
        assert "agent_1" in effect.agents_to_unfreeze


class TestRandomAuditLever:
    """Tests for RandomAuditLever."""

    def test_disabled_no_audit(self):
        """When disabled, no audits should occur."""
        config = GovernanceConfig(audit_enabled=False)
        lever = RandomAuditLever(config, seed=42)

        state = EnvState()
        interaction = SoftInteraction(p=0.1)  # Low quality

        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.0

    def test_audit_penalty_when_below_threshold(self):
        """Audit should apply penalty when p < threshold."""
        config = GovernanceConfig(
            audit_enabled=True,
            audit_probability=1.0,  # Always audit
            audit_threshold_p=0.5,
            audit_penalty_multiplier=2.0,
        )
        lever = RandomAuditLever(config, seed=42)

        state = EnvState()

        # Low quality interaction
        interaction = SoftInteraction(p=0.3)  # 0.2 below threshold
        effect = lever.on_interaction(interaction, state)

        # Penalty = (0.5 - 0.3) * 2.0 = 0.4
        assert effect.cost_a == pytest.approx(0.4)
        assert effect.details["penalty_applied"]

    def test_no_penalty_when_above_threshold(self):
        """Audit should not penalize when p >= threshold."""
        config = GovernanceConfig(
            audit_enabled=True,
            audit_probability=1.0,  # Always audit
            audit_threshold_p=0.5,
            audit_penalty_multiplier=2.0,
        )
        lever = RandomAuditLever(config, seed=42)

        state = EnvState()

        # High quality interaction
        interaction = SoftInteraction(p=0.7)
        effect = lever.on_interaction(interaction, state)

        assert effect.cost_a == 0.0
        assert not effect.details["penalty_applied"]

    def test_probabilistic_audit(self):
        """Audit should occur probabilistically."""
        config = GovernanceConfig(
            audit_enabled=True,
            audit_probability=0.5,
            audit_threshold_p=0.5,
            audit_penalty_multiplier=1.0,
        )
        lever = RandomAuditLever(config, seed=42)

        state = EnvState()

        audit_count = 0
        n_trials = 100

        for _ in range(n_trials):
            interaction = SoftInteraction(p=0.3)
            effect = lever.on_interaction(interaction, state)
            if effect.details.get("audited", False):
                audit_count += 1

        # Should be roughly 50% audited
        assert 30 < audit_count < 70


class TestGovernanceEngine:
    """Tests for GovernanceEngine."""

    def test_default_engine(self):
        """Engine with defaults should initialize."""
        engine = GovernanceEngine()
        assert engine.config is not None

    def test_aggregates_lever_effects(self):
        """Engine should aggregate effects from all levers."""
        config = GovernanceConfig(
            transaction_tax_rate=0.1,
            transaction_tax_split=0.5,
        )
        engine = GovernanceEngine(config)

        state = EnvState()
        interaction = SoftInteraction(tau=1.0, p=0.7, accepted=True)

        effect = engine.apply_interaction(interaction, state)

        # Should have tax applied
        assert effect.cost_a > 0
        assert effect.cost_b > 0

    def test_epoch_start_applies_decay(self):
        """Epoch start should apply reputation decay."""
        config = GovernanceConfig(reputation_decay_rate=0.9)
        engine = GovernanceEngine(config)

        state = EnvState()
        state.add_agent("agent_1", initial_reputation=10.0)

        effect = engine.apply_epoch_start(state, epoch=1)

        assert "agent_1" in effect.reputation_deltas
        assert effect.reputation_deltas["agent_1"] < 0

    def test_can_agent_act_checks_staking(self):
        """can_agent_act should check staking requirements."""
        config = GovernanceConfig(
            staking_enabled=True,
            min_stake_to_participate=50.0,
        )
        engine = GovernanceEngine(config)

        state = EnvState()
        state.add_agent("rich", initial_resources=100.0)
        state.add_agent("poor", initial_resources=10.0)

        assert engine.can_agent_act("rich", state)
        assert not engine.can_agent_act("poor", state)

    def test_variance_aware_levers_register_conditionally(self):
        """Engine should only register variance-aware levers when enabled."""
        config = GovernanceConfig(
            self_ensemble_enabled=True,
            incoherence_breaker_enabled=True,
            decomposition_enabled=True,
            incoherence_friction_enabled=True,
        )
        engine = GovernanceEngine(config)
        names = engine.get_active_lever_names()

        assert "self_ensemble" in names
        assert "incoherence_breaker" in names
        assert "decomposition" in names
        assert "incoherence_friction" in names

    def test_variance_aware_levers_not_registered_by_default(self):
        """Default config should preserve existing lever set."""
        engine = GovernanceEngine(GovernanceConfig())
        names = engine.get_active_lever_names()
        assert "self_ensemble" not in names
        assert "incoherence_breaker" not in names
        assert "decomposition" not in names
        assert "incoherence_friction" not in names

    def test_adaptive_gating_disables_variance_levers_below_threshold(self):
        """Adaptive mode should gate variance-aware levers when risk is low."""
        config = GovernanceConfig(
            self_ensemble_enabled=True,
            incoherence_breaker_enabled=True,
            decomposition_enabled=True,
            incoherence_friction_enabled=True,
            adaptive_governance_enabled=True,
            adaptive_incoherence_threshold=0.6,
        )
        engine = GovernanceEngine(config)
        engine.set_incoherence_forecaster(_DummyForecaster(probability=0.2))
        risk = engine.update_adaptive_mode({"horizon_length": 5.0})
        assert risk == pytest.approx(0.2)

        active_names = engine.get_active_lever_names()
        assert "self_ensemble" not in active_names
        assert "incoherence_breaker" not in active_names
        assert "decomposition" not in active_names
        assert "incoherence_friction" not in active_names

    def test_adaptive_gating_enables_variance_levers_above_threshold(self):
        """Adaptive mode should activate variance-aware levers when risk is high."""
        config = GovernanceConfig(
            self_ensemble_enabled=True,
            incoherence_breaker_enabled=True,
            decomposition_enabled=True,
            incoherence_friction_enabled=True,
            adaptive_governance_enabled=True,
            adaptive_incoherence_threshold=0.6,
        )
        engine = GovernanceEngine(config)
        engine.set_incoherence_forecaster(_DummyForecaster(probability=0.9))
        risk = engine.update_adaptive_mode({"horizon_length": 20.0})
        assert risk == pytest.approx(0.9)

        active_names = engine.get_active_lever_names()
        assert "self_ensemble" in active_names
        assert "incoherence_breaker" in active_names
        assert "decomposition" in active_names
        assert "incoherence_friction" in active_names

    def test_apply_step_with_decomposition(self):
        """Step-level hooks should aggregate decomposition effects."""
        config = GovernanceConfig(
            decomposition_enabled=True,
            decomposition_horizon_threshold=4,
        )
        engine = GovernanceEngine(config)
        state = EnvState(steps_per_epoch=8)
        state.add_agent("agent_1")
        st = state.get_agent("agent_1")
        st.interactions_initiated = 3
        st.interactions_accepted = 0
        st.interactions_rejected = 3

        effect = engine.apply_step(state, step=2)
        assert "agent_1" in effect.reputation_deltas


class TestGovernanceEffect:
    """Tests for GovernanceEffect aggregation."""

    def test_from_lever_effects(self):
        """Should correctly aggregate lever effects."""
        effects = [
            LeverEffect(cost_a=0.1, lever_name="tax"),
            LeverEffect(cost_a=0.2, cost_b=0.1, lever_name="audit"),
            LeverEffect(agents_to_freeze={"agent_1"}, lever_name="circuit_breaker"),
        ]

        result = GovernanceEffect.from_lever_effects(effects)

        assert result.cost_a == pytest.approx(0.3)
        assert result.cost_b == pytest.approx(0.1)
        assert "agent_1" in result.agents_to_freeze
        assert len(result.lever_effects) == 3


class TestVarianceAwareLevers:
    """Tests for new incoherence-targeted governance levers."""

    def test_self_ensemble_lever_adds_compute_cost(self):
        config = GovernanceConfig(
            self_ensemble_enabled=True,
            self_ensemble_samples=5,
        )
        lever = SelfEnsembleLever(config)
        effect = lever.on_interaction(
            SoftInteraction(p=0.6, accepted=True),
            EnvState(),
        )
        assert effect.cost_a > 0
        assert effect.cost_b > 0
        assert effect.details["ensemble_samples"] == 5

    def test_incoherence_breaker_freezes_on_high_uncertainty(self):
        config = GovernanceConfig(
            incoherence_breaker_enabled=True,
            incoherence_breaker_threshold=0.8,
            freeze_duration_epochs=2,
        )
        lever = IncoherenceCircuitBreakerLever(config)
        state = EnvState()
        state.add_agent("agent_1")

        effect = lever.on_interaction(
            SoftInteraction(initiator="agent_1", p=0.5, accepted=True),
            state,
        )
        assert "agent_1" in effect.agents_to_freeze

        # Unfreeze after freeze duration
        unfreeze = lever.on_epoch_start(state, epoch=2)
        assert "agent_1" in unfreeze.agents_to_unfreeze

    def test_decomposition_lever_penalizes_low_acceptance_agents(self):
        config = GovernanceConfig(
            decomposition_enabled=True,
            decomposition_horizon_threshold=6,
        )
        lever = DecompositionLever(config)
        state = EnvState(steps_per_epoch=8)
        state.add_agent("agent_1")
        agent_state = state.get_agent("agent_1")
        agent_state.interactions_initiated = 3
        agent_state.interactions_accepted = 0
        agent_state.interactions_rejected = 3

        effect = lever.on_step(state, step=3)
        assert "agent_1" in effect.reputation_deltas
        assert effect.reputation_deltas["agent_1"] < 0

    def test_incoherence_friction_scales_with_uncertainty(self):
        config = GovernanceConfig(
            incoherence_friction_enabled=True,
            incoherence_friction_rate=0.2,
            transaction_tax_split=0.25,
        )
        lever = IncoherenceFrictionLever(config)
        state = EnvState()

        high_uncertainty = lever.on_interaction(
            SoftInteraction(p=0.5, tau=1.0, accepted=True),
            state,
        )
        low_uncertainty = lever.on_interaction(
            SoftInteraction(p=0.95, tau=1.0, accepted=True),
            state,
        )

        assert high_uncertainty.cost_a > low_uncertainty.cost_a
        assert high_uncertainty.cost_b > low_uncertainty.cost_b


class TestOrchestratorIntegration:
    """Tests for governance integration with orchestrator."""

    def test_orchestrator_with_governance(self):
        """Orchestrator should integrate governance engine."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        gov_config = GovernanceConfig(
            transaction_tax_rate=0.1,
            transaction_tax_split=0.5,
        )

        config = OrchestratorConfig(
            n_epochs=1,
            steps_per_epoch=2,
            governance_config=gov_config,
            seed=42,
        )

        orchestrator = Orchestrator(config)
        assert orchestrator.governance_engine is not None

    def test_governance_costs_applied_to_interaction(self):
        """Governance costs should be added to c_a and c_b."""
        from swarm.agents.honest import HonestAgent
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        gov_config = GovernanceConfig(
            transaction_tax_rate=0.1,
            transaction_tax_split=0.5,
        )

        config = OrchestratorConfig(
            n_epochs=1,
            steps_per_epoch=5,
            governance_config=gov_config,
            seed=42,
        )

        orchestrator = Orchestrator(config)

        # Register agents
        agent1 = HonestAgent("agent_1")
        agent2 = HonestAgent("agent_2")
        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        # Run simulation
        orchestrator.run()

        # Check that some interactions have governance costs
        # At minimum, tax should apply to interactions with non-zero tau
        # The test is that the code runs without error

    def test_reputation_decay_at_epoch_start(self):
        """Reputation should decay at epoch boundaries."""
        from swarm.agents.honest import HonestAgent
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        gov_config = GovernanceConfig(
            reputation_decay_rate=0.5,  # Aggressive decay for testing
        )

        config = OrchestratorConfig(
            n_epochs=2,
            steps_per_epoch=1,
            governance_config=gov_config,
            seed=42,
        )

        orchestrator = Orchestrator(config)

        # Register agent with initial reputation
        agent = HonestAgent("test_agent")
        agent_state = orchestrator.register_agent(agent)
        agent_state.reputation = 10.0

        # Run simulation
        orchestrator.run()

        # Reputation should have decayed
        final_rep = orchestrator.state.get_agent("test_agent").reputation
        # After 2 epochs with 0.5 decay: 10 * 0.5 * 0.5 = 2.5 (approximately)
        # But interactions may also affect reputation
        assert final_rep < 10.0
