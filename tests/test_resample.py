"""Tests for the resample protocol governance lever (Bhatt et al., 2025)."""

import pytest
from pydantic import ValidationError

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.engine import GovernanceEngine
from swarm.governance.resample import ResampleLever
from swarm.models.interaction import SoftInteraction


def _make_config(**overrides):
    """Create a resample-enabled GovernanceConfig."""
    defaults = {
        "resample_enabled": True,
        "resample_suspicion_threshold": 0.6,
        "resample_n_samples": 3,
        "resample_trajectory_analysis": False,
        "resample_trajectory_window": 5,
        "resample_incrimination_threshold": 2.0,
        "resample_compute_cost": 0.01,
        "resample_evidence_decay": 0.9,
        "resample_evidence_rep_weight": 0.5,
    }
    defaults.update(overrides)
    return GovernanceConfig(**defaults)


def _make_interaction(p: float, initiator: str = "agent_a", **kwargs):
    """Create a SoftInteraction with given p."""
    defaults = {
        "initiator": initiator,
        "counterparty": "agent_b",
        "accepted": True,
        "p": p,
        "v_hat": 2 * p - 1,  # Map [0,1] → [-1,1]
    }
    defaults.update(kwargs)
    return SoftInteraction(**defaults)


class TestSuspicionScoring:
    """Tests for suspicion score computation."""

    def test_high_p_low_suspicion(self):
        """Interactions with high p should have low suspicion."""
        config = _make_config()
        lever = ResampleLever(config, seed=42)
        suspicion = lever.compute_suspicion(0.9, "agent_a")
        assert suspicion == pytest.approx(0.1, abs=1e-6)

    def test_low_p_high_suspicion(self):
        """Interactions with low p should have high suspicion."""
        config = _make_config()
        lever = ResampleLever(config, seed=42)
        suspicion = lever.compute_suspicion(0.2, "agent_a")
        assert suspicion == pytest.approx(0.8, abs=1e-6)

    def test_should_resample_below_threshold(self):
        """Actions below suspicion threshold should not trigger resample."""
        config = _make_config(resample_suspicion_threshold=0.6)
        lever = ResampleLever(config, seed=42)
        assert not lever.should_resample(0.9, "agent_a")  # suspicion=0.1

    def test_should_resample_above_threshold(self):
        """Actions above suspicion threshold should trigger resample."""
        config = _make_config(resample_suspicion_threshold=0.6)
        lever = ResampleLever(config, seed=42)
        assert lever.should_resample(0.2, "agent_a")  # suspicion=0.8

    def test_should_resample_disabled(self):
        """Resampling disabled should never trigger."""
        config = _make_config(resample_enabled=False)
        lever = ResampleLever(config, seed=42)
        assert not lever.should_resample(0.0, "agent_a")


class TestTrajectoryAnalysis:
    """Tests for trajectory-based suspicion boosting."""

    def test_declining_trajectory_boosts_suspicion(self):
        """A declining p trajectory should boost suspicion score."""
        config = _make_config(resample_trajectory_analysis=True, resample_trajectory_window=5)
        lever = ResampleLever(config, seed=42)

        # Seed a declining trajectory
        for p_val in [0.8, 0.7, 0.6, 0.5, 0.4]:
            lever._trajectory["agent_a"].append(p_val)

        # Suspicion for p=0.5 with declining trajectory should be > base
        suspicion_with_trajectory = lever.compute_suspicion(0.5, "agent_a")
        base_suspicion = 1.0 - 0.5  # = 0.5
        assert suspicion_with_trajectory > base_suspicion

    def test_stable_trajectory_no_boost(self):
        """A stable p trajectory should not boost suspicion."""
        config = _make_config(resample_trajectory_analysis=True, resample_trajectory_window=5)
        lever = ResampleLever(config, seed=42)

        # Seed a stable trajectory
        for _ in range(5):
            lever._trajectory["agent_a"].append(0.7)

        suspicion = lever.compute_suspicion(0.7, "agent_a")
        base_suspicion = 1.0 - 0.7  # = 0.3
        assert suspicion == pytest.approx(base_suspicion, abs=0.01)


class TestEvidenceAccumulation:
    """Tests for the incrimination mechanism."""

    def test_evidence_accumulates(self):
        """Evidence should grow with consistent suspicious resamples."""
        config = _make_config()
        lever = ResampleLever(config, seed=42)

        # Simulate resampled actions that are all suspicious (low p)
        delta = lever.accumulate_evidence("agent_a", 0.2, [0.1, 0.15, 0.2])
        assert delta > 0
        assert lever.get_evidence("agent_a") > 0

    def test_benign_resamples_low_evidence(self):
        """Resamples with high p should produce low evidence."""
        config = _make_config()
        lever = ResampleLever(config, seed=42)

        # All resamples look benign
        delta = lever.accumulate_evidence("agent_a", 0.8, [0.9, 0.85, 0.95])
        assert delta == pytest.approx(0.0, abs=1e-6)

    def test_incrimination_threshold(self):
        """Agent should be incriminated after sufficient evidence."""
        config = _make_config(resample_incrimination_threshold=1.0)
        lever = ResampleLever(config, seed=42)

        assert not lever.is_incriminated("agent_a")

        # Accumulate enough evidence (multiple rounds of suspicious resamples)
        for _ in range(10):
            lever.accumulate_evidence("agent_a", 0.1, [0.05, 0.1, 0.15])

        assert lever.is_incriminated("agent_a")

    def test_evidence_decay(self):
        """Evidence should decay each epoch."""
        config = _make_config(resample_evidence_decay=0.5)
        lever = ResampleLever(config, seed=42)
        state = EnvState()

        lever._evidence["agent_a"] = 1.0
        lever.on_epoch_start(state, epoch=1)

        assert lever.get_evidence("agent_a") == pytest.approx(0.5, abs=1e-6)


class TestOnInteraction:
    """Tests for the lever's interaction hook."""

    def test_non_resampled_interaction_no_effect(self):
        """Non-resampled interactions should have no resample cost."""
        config = _make_config()
        lever = ResampleLever(config, seed=42)
        state = EnvState()

        interaction = _make_interaction(p=0.8)
        effect = lever.on_interaction(interaction, state)

        assert effect.cost_a == 0.0
        assert len(effect.agents_to_freeze) == 0

    def test_resampled_interaction_has_cost(self):
        """Resampled interactions should incur compute cost."""
        config = _make_config(resample_compute_cost=0.05)
        lever = ResampleLever(config, seed=42)
        state = EnvState()

        interaction = _make_interaction(
            p=0.3,
            metadata={
                "resample_triggered": True,
                "resample_n_samples": 3,
                "resample_blocked": False,
                "resample_p_proxy": 0.2,
            },
        )
        effect = lever.on_interaction(interaction, state)

        assert effect.cost_a == pytest.approx(0.15, abs=1e-6)  # 0.05 * 3

    def test_blocked_action_reputation_penalty(self):
        """Blocked actions should incur reputation penalty."""
        config = _make_config()
        lever = ResampleLever(config, seed=42)
        state = EnvState()

        interaction = _make_interaction(
            p=0.3,
            metadata={
                "resample_triggered": True,
                "resample_n_samples": 3,
                "resample_blocked": True,
                "resample_p_proxy": 0.2,
            },
        )
        effect = lever.on_interaction(interaction, state)

        # Should have a reputation penalty
        assert interaction.initiator in effect.reputation_deltas
        assert effect.reputation_deltas[interaction.initiator] < 0

    def test_incriminated_agent_frozen(self):
        """Agent should be frozen when evidence exceeds threshold."""
        config = _make_config(resample_incrimination_threshold=0.5)
        lever = ResampleLever(config, seed=42)
        state = EnvState()

        # Pre-load evidence just below threshold
        lever._evidence["agent_a"] = 0.45

        interaction = _make_interaction(
            p=0.1,
            initiator="agent_a",
            metadata={
                "resample_triggered": True,
                "resample_n_samples": 3,
                "resample_blocked": False,
                "resample_p_proxy": 0.1,
                "resample_p_values": [0.05, 0.1, 0.08],
            },
        )
        effect = lever.on_interaction(interaction, state)

        assert "agent_a" in effect.agents_to_freeze

    def test_trajectory_updated(self):
        """Trajectory window should be updated on each interaction."""
        config = _make_config()
        lever = ResampleLever(config, seed=42)
        state = EnvState()

        for p in [0.8, 0.7, 0.6]:
            interaction = _make_interaction(p=p)
            lever.on_interaction(interaction, state)

        assert len(lever._trajectory["agent_a"]) == 3
        assert lever._trajectory["agent_a"] == [0.8, 0.7, 0.6]


class TestGovernanceEngineIntegration:
    """Tests for resample lever registration in GovernanceEngine."""

    def test_resample_lever_registered(self):
        """Resample lever should be registered when enabled."""
        config = _make_config()
        engine = GovernanceEngine(config=config, seed=42)

        assert engine.get_resample_lever() is not None
        assert "resample" in engine.get_registered_lever_names()

    def test_resample_lever_not_registered_when_disabled(self):
        """Resample lever should not be registered when disabled."""
        config = GovernanceConfig(resample_enabled=False)
        engine = GovernanceEngine(config=config, seed=42)

        assert engine.get_resample_lever() is None

    def test_resample_report(self):
        """Engine should return resample report."""
        config = _make_config()
        engine = GovernanceEngine(config=config, seed=42)

        report = engine.get_resample_report()
        assert report is not None
        assert "evidence" in report
        assert "resample_counts" in report
        assert "incriminated_agents" in report


class TestConfigValidation:
    """Tests for resample config parameter validation."""

    def test_invalid_suspicion_threshold(self):
        with pytest.raises(ValidationError, match="resample_suspicion_threshold"):
            GovernanceConfig(resample_suspicion_threshold=1.5)

    def test_invalid_n_samples(self):
        with pytest.raises(ValidationError, match="resample_n_samples"):
            GovernanceConfig(resample_n_samples=0)

    def test_invalid_trajectory_window(self):
        with pytest.raises(ValidationError, match="resample_trajectory_window"):
            GovernanceConfig(resample_trajectory_window=1)

    def test_invalid_incrimination_threshold(self):
        with pytest.raises(ValidationError, match="resample_incrimination_threshold"):
            GovernanceConfig(resample_incrimination_threshold=0)

    def test_invalid_evidence_decay(self):
        with pytest.raises(ValidationError, match="resample_evidence_decay"):
            GovernanceConfig(resample_evidence_decay=1.5)


class TestResampleReport:
    """Tests for the reporting and reset functionality."""

    def test_report_structure(self):
        """Report should have expected keys."""
        config = _make_config()
        lever = ResampleLever(config, seed=42)

        report = lever.get_report()
        assert "evidence" in report
        assert "resample_counts" in report
        assert "incriminated_agents" in report

    def test_reset_clears_state(self):
        """Reset should clear all accumulated state."""
        config = _make_config()
        lever = ResampleLever(config, seed=42)

        lever._evidence["agent_a"] = 5.0
        lever._resample_counts["agent_a"] = 10
        lever._trajectory["agent_a"] = [0.5, 0.4, 0.3]

        lever.reset()

        assert lever.get_evidence("agent_a") == 0.0
        assert len(lever._trajectory) == 0
        assert lever.get_report()["resample_counts"] == {}
