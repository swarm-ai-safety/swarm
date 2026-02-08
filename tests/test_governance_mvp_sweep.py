"""Tests for the governance MVP sweep and new levers (transparency, moderator)."""

import pytest
from pydantic import ValidationError

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.engine import GovernanceEngine
from swarm.governance.moderator_lever import ModeratorLever
from swarm.governance.transparency import TransparencyLever
from swarm.models.interaction import SoftInteraction

pytestmark = pytest.mark.slow

# ── TransparencyLever Tests ──────────────────────────────────────────────


class TestTransparencyLever:
    """Tests for TransparencyLever."""

    def test_disabled_no_effect(self):
        """When disabled, no reputation effect should apply."""
        config = GovernanceConfig(transparency_enabled=False)
        lever = TransparencyLever(config)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.2, accepted=True
        )

        effect = lever.on_interaction(interaction, state)
        assert len(effect.reputation_deltas) == 0

    def test_high_quality_gets_bonus(self):
        """High-quality interactions should receive a reputation bonus."""
        config = GovernanceConfig(
            transparency_enabled=True,
            transparency_bonus_rate=0.1,
            transparency_threshold_p=0.5,
        )
        lever = TransparencyLever(config)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.8, accepted=True
        )

        effect = lever.on_interaction(interaction, state)

        assert "agent_1" in effect.reputation_deltas
        # p=0.8, threshold=0.5, deviation=+0.3, delta = 0.3 * 0.1 = 0.03
        assert effect.reputation_deltas["agent_1"] == pytest.approx(0.03)

    def test_low_quality_gets_penalty(self):
        """Low-quality interactions should receive a reputation penalty."""
        config = GovernanceConfig(
            transparency_enabled=True,
            transparency_bonus_rate=0.1,
            transparency_threshold_p=0.5,
        )
        lever = TransparencyLever(config)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.2, accepted=True
        )

        effect = lever.on_interaction(interaction, state)

        assert "agent_1" in effect.reputation_deltas
        # p=0.2, threshold=0.5, deviation=-0.3, delta = -0.3 * 0.1 = -0.03
        assert effect.reputation_deltas["agent_1"] == pytest.approx(-0.03)

    def test_no_effect_on_rejected(self):
        """Rejected interactions should have no transparency effect."""
        config = GovernanceConfig(
            transparency_enabled=True,
            transparency_bonus_rate=0.1,
        )
        lever = TransparencyLever(config)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.2, accepted=False
        )

        effect = lever.on_interaction(interaction, state)
        assert len(effect.reputation_deltas) == 0

    def test_at_threshold_no_delta(self):
        """When p equals the threshold, delta should be zero."""
        config = GovernanceConfig(
            transparency_enabled=True,
            transparency_bonus_rate=0.1,
            transparency_threshold_p=0.5,
        )
        lever = TransparencyLever(config)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.5, accepted=True
        )

        effect = lever.on_interaction(interaction, state)
        assert effect.reputation_deltas.get("agent_1", 0.0) == pytest.approx(0.0)

    def test_higher_rate_stronger_effect(self):
        """Higher bonus rate should produce stronger reputation delta."""
        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.8, accepted=True
        )

        config_low = GovernanceConfig(
            transparency_enabled=True,
            transparency_bonus_rate=0.05,
        )
        lever_low = TransparencyLever(config_low)

        config_high = GovernanceConfig(
            transparency_enabled=True,
            transparency_bonus_rate=0.2,
        )
        lever_high = TransparencyLever(config_high)

        effect_low = lever_low.on_interaction(interaction, state)
        effect_high = lever_high.on_interaction(interaction, state)

        assert (
            abs(effect_high.reputation_deltas["agent_1"])
            > abs(effect_low.reputation_deltas["agent_1"])
        )


# ── ModeratorLever Tests ─────────────────────────────────────────────────


class TestModeratorLever:
    """Tests for ModeratorLever."""

    def test_disabled_no_effect(self):
        """When disabled, no moderator costs should apply."""
        config = GovernanceConfig(moderator_enabled=False)
        lever = ModeratorLever(config, seed=42)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.1, accepted=True
        )

        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.0
        assert effect.cost_b == 0.0

    def test_penalty_on_low_quality(self):
        """Moderator should penalize low-quality interactions when reviewed."""
        config = GovernanceConfig(
            moderator_enabled=True,
            moderator_review_probability=1.0,  # Always review
            moderator_penalty_multiplier=1.5,
            moderator_threshold_p=0.4,
        )
        lever = ModeratorLever(config, seed=42)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.2, accepted=True
        )

        effect = lever.on_interaction(interaction, state)

        # Penalty = (0.4 - 0.2) * 1.5 = 0.3, plus review overhead 0.01
        assert effect.cost_a == pytest.approx(0.31)
        assert effect.cost_b == pytest.approx(0.01)  # Only review overhead
        assert effect.details["penalty_applied"]

    def test_no_penalty_on_high_quality(self):
        """Moderator should not penalize high-quality interactions."""
        config = GovernanceConfig(
            moderator_enabled=True,
            moderator_review_probability=1.0,  # Always review
            moderator_penalty_multiplier=1.5,
            moderator_threshold_p=0.4,
        )
        lever = ModeratorLever(config, seed=42)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.7, accepted=True
        )

        effect = lever.on_interaction(interaction, state)

        # Only review overhead, no penalty
        assert effect.cost_a == pytest.approx(0.01)
        assert effect.cost_b == pytest.approx(0.01)
        assert not effect.details["penalty_applied"]

    def test_reputation_penalty_on_low_quality(self):
        """Moderator should apply reputation penalty for low quality."""
        config = GovernanceConfig(
            moderator_enabled=True,
            moderator_review_probability=1.0,
            moderator_penalty_multiplier=2.0,
            moderator_threshold_p=0.5,
        )
        lever = ModeratorLever(config, seed=42)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.3, accepted=True
        )

        effect = lever.on_interaction(interaction, state)

        # Reputation penalty = -(0.5 - 0.3) * 2.0 = -0.4
        assert "agent_1" in effect.reputation_deltas
        assert effect.reputation_deltas["agent_1"] == pytest.approx(-0.4)

    def test_probabilistic_review(self):
        """Moderator should review probabilistically."""
        config = GovernanceConfig(
            moderator_enabled=True,
            moderator_review_probability=0.5,
            moderator_threshold_p=0.5,
        )
        lever = ModeratorLever(config, seed=42)

        state = EnvState()

        review_count = 0
        n_trials = 100

        for _ in range(n_trials):
            interaction = SoftInteraction(
                initiator="agent_1", p=0.3, accepted=True
            )
            effect = lever.on_interaction(interaction, state)
            if effect.details.get("reviewed", False):
                review_count += 1

        # Should be roughly 50% reviewed
        assert 30 < review_count < 70

    def test_no_effect_on_rejected(self):
        """Rejected interactions should not be reviewed."""
        config = GovernanceConfig(
            moderator_enabled=True,
            moderator_review_probability=1.0,
        )
        lever = ModeratorLever(config, seed=42)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.1, accepted=False
        )

        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.0
        assert effect.cost_b == 0.0


# ── GovernanceConfig Validation Tests ────────────────────────────────────


class TestNewGovernanceConfigValidation:
    """Tests for new config field validation."""

    def test_valid_transparency_config(self):
        """Valid transparency config should pass validation."""
        config = GovernanceConfig(
            transparency_enabled=True,
            transparency_bonus_rate=0.1,
            transparency_threshold_p=0.5,
        )
        assert config is not None  # Pydantic auto-validates on creation

    def test_invalid_transparency_bonus_rate(self):
        """Negative bonus rate should raise."""
        with pytest.raises(ValidationError, match="transparency_bonus_rate"):
            GovernanceConfig(transparency_bonus_rate=-0.1)

    def test_invalid_transparency_threshold(self):
        """Threshold outside [0,1] should raise."""
        with pytest.raises(ValidationError, match="transparency_threshold_p"):
            GovernanceConfig(transparency_threshold_p=1.5)

    def test_valid_moderator_config(self):
        """Valid moderator config should pass validation."""
        config = GovernanceConfig(
            moderator_enabled=True,
            moderator_review_probability=0.3,
            moderator_penalty_multiplier=1.5,
            moderator_threshold_p=0.4,
        )
        assert config is not None  # Pydantic auto-validates on creation

    def test_invalid_moderator_review_probability(self):
        """Review probability outside [0,1] should raise."""
        with pytest.raises(ValidationError, match="moderator_review_probability"):
            GovernanceConfig(moderator_review_probability=2.0)

    def test_invalid_moderator_penalty_multiplier(self):
        """Negative penalty multiplier should raise."""
        with pytest.raises(ValidationError, match="moderator_penalty_multiplier"):
            GovernanceConfig(moderator_penalty_multiplier=-1.0)

    def test_invalid_moderator_threshold(self):
        """Threshold outside [0,1] should raise."""
        with pytest.raises(ValidationError, match="moderator_threshold_p"):
            GovernanceConfig(moderator_threshold_p=1.5)


# ── GovernanceEngine Integration ─────────────────────────────────────────


class TestNewLeversInEngine:
    """Tests that new levers are registered and work in the engine."""

    def test_transparency_registered(self):
        """Transparency lever should be registered in the engine."""
        config = GovernanceConfig(
            transparency_enabled=True,
            transparency_bonus_rate=0.1,
        )
        engine = GovernanceEngine(config)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.8, accepted=True
        )

        effect = engine.apply_interaction(interaction, state)

        # Should have a transparency effect in lever_effects
        lever_names = [e.lever_name for e in effect.lever_effects]
        assert "transparency" in lever_names

    def test_moderator_registered(self):
        """Moderator lever should be registered in the engine."""
        config = GovernanceConfig(
            moderator_enabled=True,
            moderator_review_probability=1.0,
        )
        engine = GovernanceEngine(config, seed=42)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.2, accepted=True
        )

        effect = engine.apply_interaction(interaction, state)

        lever_names = [e.lever_name for e in effect.lever_effects]
        assert "moderator" in lever_names

    def test_combined_levers(self):
        """Multiple levers should aggregate their effects."""
        config = GovernanceConfig(
            transaction_tax_rate=0.1,
            transparency_enabled=True,
            transparency_bonus_rate=0.1,
            moderator_enabled=True,
            moderator_review_probability=1.0,
        )
        engine = GovernanceEngine(config, seed=42)

        state = EnvState()
        interaction = SoftInteraction(
            initiator="agent_1", p=0.3, accepted=True
        )

        effect = engine.apply_interaction(interaction, state)

        # Should have effects from tax, transparency, and moderator
        lever_names = [e.lever_name for e in effect.lever_effects]
        assert "transaction_tax" in lever_names
        assert "transparency" in lever_names
        assert "moderator" in lever_names

        # Total costs should be sum of all lever costs
        assert effect.cost_a > 0


# ── MVP Sweep Tests ──────────────────────────────────────────────────────


class TestGovernanceMVPSweep:
    """Tests for the governance MVP sweep infrastructure."""

    def test_governance_configs_count(self):
        """Should define exactly 12 configurations."""
        from examples.governance_mvp_sweep import _governance_configs

        configs = _governance_configs()
        assert len(configs) == 12

    def test_governance_configs_unique_labels(self):
        """All configuration labels should be unique."""
        from examples.governance_mvp_sweep import _governance_configs

        configs = _governance_configs()
        labels = [c["label"] for c in configs]
        assert len(labels) == len(set(labels))

    def test_governance_configs_baseline_empty(self):
        """Baseline config should have no parameter overrides."""
        from examples.governance_mvp_sweep import _governance_configs

        configs = _governance_configs()
        baseline = configs[0]
        assert baseline["label"] == "baseline"
        assert baseline["params"] == {}

    def test_governance_configs_have_labels(self):
        """All configs should have non-empty labels."""
        from examples.governance_mvp_sweep import _governance_configs

        configs = _governance_configs()
        for config in configs:
            assert "label" in config
            assert len(config["label"]) > 0
            assert "params" in config

    def test_run_single_baseline(self):
        """Should successfully run a single baseline configuration."""
        from examples.governance_mvp_sweep import run_governance_sweep
        from swarm.scenarios import load_scenario

        scenario = load_scenario("scenarios/baseline.yaml")
        results = run_governance_sweep(
            scenario,
            runs_per_config=1,
            seed_base=42,
            n_epochs=2,
            progress=False,
        )

        # 12 configs × 1 run = 12 results
        assert len(results) == 12

        # Check baseline result
        baseline = results[0]
        assert baseline.label == "baseline"
        assert baseline.sweep_result.total_interactions >= 0

    def test_named_sweep_result_to_dict(self):
        """NamedSweepResult should include governance_label in dict."""
        from examples.governance_mvp_sweep import NamedSweepResult
        from swarm.analysis.sweep import SweepResult

        sweep_result = SweepResult(
            params={"test": 1},
            run_index=0,
            seed=42,
        )
        named = NamedSweepResult(label="test_label", sweep_result=sweep_result)
        d = named.to_dict()

        assert "governance_label" in d
        assert d["governance_label"] == "test_label"

    def test_sweep_with_multiple_runs(self):
        """Should support multiple runs per configuration."""
        from examples.governance_mvp_sweep import run_governance_sweep
        from swarm.scenarios import load_scenario

        scenario = load_scenario("scenarios/baseline.yaml")
        results = run_governance_sweep(
            scenario,
            runs_per_config=2,
            seed_base=42,
            n_epochs=2,
            progress=False,
        )

        # 12 configs × 2 runs = 24 results
        assert len(results) == 24

    def test_combined_defense_in_depth(self):
        """Combined config should have multiple governance params."""
        from examples.governance_mvp_sweep import _governance_configs

        configs = _governance_configs()
        combined = [c for c in configs if c["label"] == "combined_defense_in_depth"][0]

        params = combined["params"]
        # Should have multiple levers enabled
        assert "governance.reputation_decay_rate" in params
        assert "governance.transaction_tax_rate" in params
        assert "governance.audit_enabled" in params
        assert "governance.circuit_breaker_enabled" in params
        assert "governance.transparency_enabled" in params
        assert "governance.moderator_enabled" in params
