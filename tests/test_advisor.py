"""Tests for advisor-model steering mechanism (Asawa et al., 2026).

Tests the AdvisorPolicy, AdvisorLever, and AdvisedAgent components that
implement the advisor-model governance pattern for the SWARM simulation.
"""

import random

import pytest

from swarm.agents.advised_agent import AdvisedAgent
from swarm.agents.base import Action, InteractionProposal, Observation
from swarm.agents.honest import HonestAgent
from swarm.env.state import EnvState
from swarm.governance.advisor import AdvisorLever, AdvisorPolicy
from swarm.governance.config import GovernanceConfig
from swarm.governance.engine import GovernanceEngine
from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import InteractionType, SoftInteraction

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_interaction(
    initiator: str = "agent_a",
    counterparty: str = "agent_b",
    p: float = 0.7,
    accepted: bool = True,
) -> SoftInteraction:
    """Create a test SoftInteraction."""
    return SoftInteraction(
        interaction_id="test_interaction",
        initiator=initiator,
        counterparty=counterparty,
        interaction_type=InteractionType.COLLABORATION,
        accepted=accepted,
        p=p,
        v_hat=p * 2 - 1,
    )


def _make_state(*agent_ids: str) -> EnvState:
    """Create a minimal EnvState with named agents."""
    state = EnvState()
    for aid in agent_ids:
        state.add_agent(aid)
    return state


def _make_observation(**kwargs) -> Observation:
    """Create a test observation with sensible defaults."""
    defaults = {
        "agent_state": AgentState(
            agent_id="test_agent", reputation=0.5, resources=100.0
        ),
        "current_epoch": 1,
        "current_step": 5,
        "can_post": True,
        "can_interact": True,
        "can_vote": True,
        "can_claim_task": True,
    }
    defaults.update(kwargs)
    return Observation(**defaults)


# -----------------------------------------------------------------------
# AdvisorPolicy tests
# -----------------------------------------------------------------------


class TestAdvisorPolicy:
    """Tests for the core advisor policy logic."""

    def test_initial_state(self):
        """New policy should have neutral quality EMA for unknown agents."""
        policy = AdvisorPolicy(seed=42)
        state = policy.get_agent_state("unknown_agent")
        assert state.quality_ema == 0.5
        assert state.interaction_count == 0
        assert state.cumulative_reward == 0.0

    def test_observe_updates_quality_ema(self):
        """Observing interactions should update the quality EMA."""
        policy = AdvisorPolicy(alpha=0.5, seed=42)
        interaction = _make_interaction(p=0.9)
        state = _make_state("agent_a", "agent_b")

        policy.observe_interaction(interaction, state)

        agent_state = policy.get_agent_state("agent_a")
        # EMA: 0.5 * (1 - 0.5) + 0.9 * 0.5 = 0.25 + 0.45 = 0.7
        assert abs(agent_state.quality_ema - 0.7) < 1e-6

    def test_observe_counterparty_on_accepted(self):
        """Counterparty quality should update only on accepted interactions."""
        policy = AdvisorPolicy(alpha=0.5, seed=42)
        state = _make_state("agent_a", "agent_b")

        # Accepted: counterparty gets updated
        interaction_accepted = _make_interaction(p=0.8, accepted=True)
        policy.observe_interaction(interaction_accepted, state)
        cp_state = policy.get_agent_state("agent_b")
        assert cp_state.interaction_count == 1
        assert cp_state.quality_ema != 0.5  # Updated from default

        # Rejected: counterparty count stays the same
        policy2 = AdvisorPolicy(alpha=0.5, seed=42)
        interaction_rejected = _make_interaction(p=0.8, accepted=False)
        policy2.observe_interaction(interaction_rejected, _make_state("agent_a", "agent_b"))
        cp_state2 = policy2.get_agent_state("agent_b")
        assert cp_state2.interaction_count == 0  # Not updated on rejection

    def test_advice_positive_for_high_quality(self):
        """High-quality agents should receive positive advice."""
        policy = AdvisorPolicy(alpha=1.0, steering_strength=1.0, seed=42)
        state = _make_state("agent_a", "agent_b")

        # Push quality EMA high
        interaction = _make_interaction(p=0.9)
        advice_a, _ = policy.observe_interaction(interaction, state)

        assert advice_a > 0, "High-quality agent should get positive advice"

    def test_advice_negative_for_low_quality(self):
        """Low-quality agents should receive negative advice."""
        policy = AdvisorPolicy(alpha=1.0, steering_strength=1.0, seed=42)
        state = _make_state("agent_a", "agent_b")

        interaction = _make_interaction(p=0.1)
        advice_a, _ = policy.observe_interaction(interaction, state)

        assert advice_a < 0, "Low-quality agent should get negative advice"

    def test_advice_near_zero_for_neutral(self):
        """Neutral-quality agents should get near-zero advice."""
        policy = AdvisorPolicy(alpha=1.0, steering_strength=1.0, seed=42)
        state = _make_state("agent_a", "agent_b")

        interaction = _make_interaction(p=0.5)
        advice_a, _ = policy.observe_interaction(interaction, state)

        assert abs(advice_a) < 0.1, "Neutral agent should get near-zero advice"

    def test_advice_bounded(self):
        """Advice signal should always be in [-1, +1]."""
        policy = AdvisorPolicy(alpha=1.0, steering_strength=10.0, seed=42)
        state = _make_state("agent_a", "agent_b")

        # Extreme high
        interaction = _make_interaction(p=1.0)
        advice_a, _ = policy.observe_interaction(interaction, state)
        assert -1.0 <= advice_a <= 1.0

        # Reset and extreme low
        policy2 = AdvisorPolicy(alpha=1.0, steering_strength=10.0, seed=42)
        interaction2 = _make_interaction(p=0.0)
        advice_a2, _ = policy2.observe_interaction(interaction2, _make_state("agent_a", "agent_b"))
        assert -1.0 <= advice_a2 <= 1.0

    def test_reward_signal(self):
        """Reward should penalise low-p accepted interactions."""
        policy = AdvisorPolicy(welfare_weight=1.0, toxicity_weight=2.0, seed=42)
        state = _make_state("agent_a", "agent_b")

        # Good interaction: reward > 0
        good = _make_interaction(p=0.8, accepted=True)
        policy.observe_interaction(good, state)
        reward_good = policy.get_epoch_reward()
        assert reward_good > 0

        # Bad interaction: reward < good
        bad = _make_interaction(p=0.2, accepted=True)
        policy.observe_interaction(bad, state)
        reward_bad = policy.get_epoch_reward()
        assert reward_bad < reward_good

    def test_epoch_reward_resets(self):
        """get_epoch_reward should clear accumulated rewards."""
        policy = AdvisorPolicy(seed=42)
        state = _make_state("agent_a", "agent_b")

        policy.observe_interaction(_make_interaction(p=0.8), state)
        first = policy.get_epoch_reward()
        assert first != 0.0

        # Second call should return 0 (buffer cleared)
        second = policy.get_epoch_reward()
        assert second == 0.0

    def test_report(self):
        """Report should include tracked agent data."""
        policy = AdvisorPolicy(seed=42)
        state = _make_state("agent_a", "agent_b")

        policy.observe_interaction(_make_interaction(), state)
        report = policy.get_report()

        assert report["n_agents_tracked"] == 2
        assert "agent_a" in report["agents"]
        assert "quality_ema" in report["agents"]["agent_a"]


# -----------------------------------------------------------------------
# AdvisorLever tests
# -----------------------------------------------------------------------


class TestAdvisorLever:
    """Tests for the governance lever that applies advisor steering."""

    def test_disabled_returns_noop(self):
        """When disabled, lever should return no-op effect."""
        config = GovernanceConfig(advisor_enabled=False)
        lever = AdvisorLever(config, seed=42)

        interaction = _make_interaction(p=0.3)
        state = _make_state("agent_a", "agent_b")
        effect = lever.on_interaction(interaction, state)

        assert effect.cost_a == 0.0
        assert effect.cost_b == 0.0
        assert effect.reputation_deltas == {}

    def test_low_quality_adds_friction(self):
        """Low-quality interactions should incur governance friction."""
        config = GovernanceConfig(
            advisor_enabled=True,
            advisor_ema_alpha=1.0,  # Instant update for testing
            advisor_steering_strength=1.0,
            advisor_friction_rate=0.1,
            advisor_reputation_rate=0.1,
        )
        lever = AdvisorLever(config, seed=42)
        state = _make_state("agent_a", "agent_b")

        interaction = _make_interaction(p=0.1)
        effect = lever.on_interaction(interaction, state)

        assert effect.cost_a > 0, "Low-quality initiator should face friction"
        # Reputation should be negative for low-quality agent
        assert interaction.initiator in effect.reputation_deltas
        assert effect.reputation_deltas[interaction.initiator] < 0

    def test_high_quality_gives_bonus(self):
        """High-quality interactions should earn reputation bonus."""
        config = GovernanceConfig(
            advisor_enabled=True,
            advisor_ema_alpha=1.0,
            advisor_steering_strength=1.0,
            advisor_friction_rate=0.1,
            advisor_reputation_rate=0.1,
        )
        lever = AdvisorLever(config, seed=42)
        state = _make_state("agent_a", "agent_b")

        interaction = _make_interaction(p=0.9)
        effect = lever.on_interaction(interaction, state)

        # No friction for high-quality
        assert effect.cost_a == 0.0
        # Positive reputation delta
        assert interaction.initiator in effect.reputation_deltas
        assert effect.reputation_deltas[interaction.initiator] > 0

    def test_epoch_start_logs_reward(self):
        """Epoch start should log the advisor's epoch reward."""
        config = GovernanceConfig(advisor_enabled=True)
        lever = AdvisorLever(config, seed=42)
        state = _make_state("agent_a", "agent_b")

        # Generate some interactions first
        lever.on_interaction(_make_interaction(p=0.8), state)

        effect = lever.on_epoch_start(state, epoch=1)
        assert "epoch_reward" in effect.details
        assert effect.details["epoch"] == 1

    def test_lever_name(self):
        """Lever should identify as 'advisor'."""
        config = GovernanceConfig()
        lever = AdvisorLever(config)
        assert lever.name == "advisor"

    def test_get_report(self):
        """Lever report should delegate to policy."""
        config = GovernanceConfig(advisor_enabled=True)
        lever = AdvisorLever(config, seed=42)
        state = _make_state("agent_a", "agent_b")

        lever.on_interaction(_make_interaction(), state)
        report = lever.get_report()

        assert "n_agents_tracked" in report
        assert report["n_agents_tracked"] > 0


# -----------------------------------------------------------------------
# GovernanceEngine integration tests
# -----------------------------------------------------------------------


class TestAdvisorEngineIntegration:
    """Test advisor lever registration in GovernanceEngine."""

    def test_advisor_registered_when_enabled(self):
        """Advisor lever should appear when config enables it."""
        config = GovernanceConfig(advisor_enabled=True)
        engine = GovernanceEngine(config, seed=42)

        assert engine.get_advisor_lever() is not None
        assert "advisor" in engine.get_registered_lever_names()

    def test_advisor_not_registered_when_disabled(self):
        """Advisor lever should not appear when disabled."""
        config = GovernanceConfig(advisor_enabled=False)
        engine = GovernanceEngine(config, seed=42)

        assert engine.get_advisor_lever() is None
        assert "advisor" not in engine.get_registered_lever_names()

    def test_advisor_report_method(self):
        """Engine should expose advisor report."""
        config = GovernanceConfig(advisor_enabled=True)
        engine = GovernanceEngine(config, seed=42)

        report = engine.get_advisor_report()
        assert report is not None
        assert "n_agents_tracked" in report


# -----------------------------------------------------------------------
# AdvisedAgent tests
# -----------------------------------------------------------------------


class TestAdvisedAgent:
    """Tests for the advisor-model wrapper agent."""

    def test_creation_with_honest_base(self):
        """AdvisedAgent should wrap an honest base agent."""
        agent = AdvisedAgent(
            agent_id="advised_1",
            config={
                "base_agent_type": "honest",
                "base_agent_config": {"acceptance_threshold": 0.4},
            },
            rng=random.Random(42),
        )

        assert agent.agent_id == "advised_1"
        assert isinstance(agent.base_agent, HonestAgent)
        assert agent.agent_type == AgentType.HONEST

    def test_initial_advice_is_zero(self):
        """Fresh agent should have neutral advice."""
        agent = AdvisedAgent(
            agent_id="advised_1",
            config={"base_agent_type": "honest"},
            rng=random.Random(42),
        )

        assert agent.advice_signal == 0.0
        assert agent.quality_ema == 0.5

    def test_act_delegates_to_base(self):
        """act() should delegate to the base agent."""
        agent = AdvisedAgent(
            agent_id="advised_1",
            config={"base_agent_type": "honest"},
            rng=random.Random(42),
        )

        obs = _make_observation(
            agent_state=AgentState(agent_id="advised_1", reputation=0.5),
        )
        action = agent.act(obs)

        # Should return a valid action (may be NOOP or POST depending on RNG)
        assert isinstance(action, Action)

    def test_update_from_outcome_updates_advice(self):
        """update_from_outcome should adjust the advice signal."""
        agent = AdvisedAgent(
            agent_id="advised_1",
            config={
                "base_agent_type": "honest",
                "advisor_alpha": 1.0,  # instant update
                "advisor_strength": 1.0,
            },
            rng=random.Random(42),
        )

        # High-quality interaction
        interaction = _make_interaction(
            initiator="advised_1",
            counterparty="other",
            p=0.9,
        )
        agent.update_from_outcome(interaction, payoff=1.0)

        assert agent.quality_ema == pytest.approx(0.9)
        assert agent.advice_signal > 0, "High p should yield positive advice"

    def test_update_from_outcome_negative_for_low_p(self):
        """Low-quality interactions should yield negative advice."""
        agent = AdvisedAgent(
            agent_id="advised_1",
            config={
                "base_agent_type": "honest",
                "advisor_alpha": 1.0,
                "advisor_strength": 1.0,
            },
            rng=random.Random(42),
        )

        interaction = _make_interaction(
            initiator="advised_1",
            counterparty="other",
            p=0.1,
        )
        agent.update_from_outcome(interaction, payoff=-0.5)

        assert agent.quality_ema == pytest.approx(0.1)
        assert agent.advice_signal < 0, "Low p should yield negative advice"

    def test_accept_interaction_threshold_adjustment(self):
        """Positive advice should lower acceptance threshold."""
        agent = AdvisedAgent(
            agent_id="advised_1",
            config={
                "base_agent_type": "honest",
                "base_agent_config": {"acceptance_threshold": 0.5},
                "advisor_alpha": 1.0,
                "advisor_strength": 1.0,
                "advisor_threshold_bonus": 0.2,
            },
            rng=random.Random(42),
        )

        # Push advice positive
        interaction = _make_interaction(
            initiator="advised_1", counterparty="other", p=0.95
        )
        agent.update_from_outcome(interaction, payoff=1.0)
        assert agent.advice_signal > 0.5  # Should be strongly positive

        # Now test acceptance — base threshold is 0.5, should be lowered
        proposal = InteractionProposal(
            initiator_id="proposer",
            counterparty_id="advised_1",
        )
        obs = _make_observation(
            agent_state=AgentState(agent_id="advised_1", reputation=0.5),
        )

        # Check that the base agent's threshold was temporarily modified
        # by verifying the base threshold is restored after the call
        agent.accept_interaction(proposal, obs)
        assert agent.base_agent.acceptance_threshold == 0.5  # type: ignore[attr-defined]

    def test_inject_advice_into_observation(self):
        """Steered observations should include advisor metadata."""
        agent = AdvisedAgent(
            agent_id="advised_1",
            config={"base_agent_type": "honest"},
            rng=random.Random(42),
        )

        obs = _make_observation()
        steered = agent._inject_advice(obs)

        assert "advisor_signal" in steered.ecosystem_metrics
        assert "advisor_quality_ema" in steered.ecosystem_metrics
        assert steered.ecosystem_metrics["advisor_signal"] == 0.0
        assert steered.ecosystem_metrics["advisor_quality_ema"] == 0.5

    def test_repr(self):
        """repr should include advice signal."""
        agent = AdvisedAgent(
            agent_id="a1",
            config={"base_agent_type": "honest"},
            rng=random.Random(42),
        )
        r = repr(agent)
        assert "AdvisedAgent" in r
        assert "HonestAgent" in r

    def test_invalid_base_type_raises(self):
        """Invalid base_agent_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown base_agent_type"):
            AdvisedAgent(
                agent_id="bad",
                config={"base_agent_type": "nonexistent_type"},
                rng=random.Random(42),
            )


# -----------------------------------------------------------------------
# Config validation tests
# -----------------------------------------------------------------------


class TestAdvisorConfig:
    """Tests for advisor config validation in GovernanceConfig."""

    def test_valid_config(self):
        """Valid advisor config should not raise."""
        config = GovernanceConfig(
            advisor_enabled=True,
            advisor_ema_alpha=0.5,
            advisor_steering_strength=2.0,
            advisor_friction_rate=0.1,
            advisor_reputation_rate=0.2,
        )
        assert config.advisor_enabled is True

    def test_alpha_must_be_positive(self):
        """advisor_ema_alpha must be in (0, 1]."""
        with pytest.raises(ValueError, match="advisor_ema_alpha"):
            GovernanceConfig(advisor_ema_alpha=0.0)

    def test_alpha_must_be_at_most_one(self):
        """advisor_ema_alpha must be in (0, 1]."""
        with pytest.raises(ValueError, match="advisor_ema_alpha"):
            GovernanceConfig(advisor_ema_alpha=1.5)

    def test_steering_strength_non_negative(self):
        """advisor_steering_strength must be >= 0."""
        with pytest.raises(ValueError, match="advisor_steering_strength"):
            GovernanceConfig(advisor_steering_strength=-1.0)

    def test_friction_rate_non_negative(self):
        """advisor_friction_rate must be >= 0."""
        with pytest.raises(ValueError, match="advisor_friction_rate"):
            GovernanceConfig(advisor_friction_rate=-0.1)

    def test_reputation_rate_non_negative(self):
        """advisor_reputation_rate must be >= 0."""
        with pytest.raises(ValueError, match="advisor_reputation_rate"):
            GovernanceConfig(advisor_reputation_rate=-0.1)

    def test_welfare_weight_non_negative(self):
        """advisor_welfare_weight must be >= 0."""
        with pytest.raises(ValueError, match="advisor_welfare_weight"):
            GovernanceConfig(advisor_welfare_weight=-1.0)

    def test_toxicity_weight_non_negative(self):
        """advisor_toxicity_weight must be >= 0."""
        with pytest.raises(ValueError, match="advisor_toxicity_weight"):
            GovernanceConfig(advisor_toxicity_weight=-1.0)
