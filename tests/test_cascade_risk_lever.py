"""Tests for the CascadeRiskLever governance lever."""

import pytest

from swarm.env.state import EnvState
from swarm.governance.cascade import CascadeRiskLever
from swarm.governance.config import GovernanceConfig
from swarm.models.interaction import SoftInteraction


def _ix(
    iid: str,
    *,
    p: float = 0.5,
    initiator: str = "agent_a",
    counterparty: str = "agent_b",
    causal_parents: list[str] | None = None,
) -> SoftInteraction:
    return SoftInteraction(
        interaction_id=iid,
        p=p,
        v_hat=0.0,
        initiator=initiator,
        counterparty=counterparty,
        accepted=True,
        causal_parents=causal_parents or [],
    )


class TestCascadeRiskLeverDisabled:
    def test_noop_when_disabled(self):
        config = GovernanceConfig(cascade_risk_enabled=False)
        lever = CascadeRiskLever(config)
        state = EnvState()
        ix = _ix("A", p=0.1, causal_parents=["B"])
        effect = lever.on_interaction(ix, state)
        assert effect.cost_a == 0.0
        assert effect.reputation_deltas == {}


class TestCascadeRiskLeverEnabled:
    def _make_lever(self, **overrides):
        defaults = {
            "cascade_risk_enabled": True,
            "cascade_risk_threshold": 0.3,
            "cascade_risk_penalty_scale": 2.0,
            "cascade_risk_reputation_scale": 1.0,
            "cascade_risk_p_threshold": 0.3,
            "cascade_risk_window": 100,
        }
        defaults.update(overrides)
        config = GovernanceConfig(**defaults)
        return CascadeRiskLever(config)

    def test_no_penalty_for_root_interaction(self):
        """Interactions without causal_parents are never penalized."""
        lever = self._make_lever()
        state = EnvState()
        ix = _ix("A", p=0.1)  # low-p but no parents
        effect = lever.on_interaction(ix, state)
        assert effect.cost_a == 0.0

    def test_penalty_when_parent_has_bad_descendants(self):
        """Parent with many low-p descendants triggers cascade risk penalty."""
        lever = self._make_lever()
        state = EnvState()

        # Build a chain: root → bad1, bad2, bad3
        root = _ix("root", p=0.9, initiator="alice")
        lever.on_interaction(root, state)

        for i in range(3):
            bad = _ix(
                f"bad{i}", p=0.1, initiator="bob",
                causal_parents=["root"],
            )
            lever.on_interaction(bad, state)

        # Now an interaction that consumes root's artifact
        consumer = _ix(
            "consumer", p=0.5, initiator="carol",
            causal_parents=["root"],
        )
        effect = lever.on_interaction(consumer, state)

        # root has 4 descendants (bad0, bad1, bad2, consumer),
        # 3 of which have p < 0.3 → cascade_risk = 0.75
        # 0.75 > threshold (0.3) → penalty applied
        assert effect.cost_a > 0
        assert "alice" in effect.reputation_deltas
        assert effect.reputation_deltas["alice"] < 0
        assert effect.details["cascade_risk"] == pytest.approx(0.75)

    def test_no_penalty_when_descendants_are_good(self):
        """If descendants all have high p, no cascade risk penalty."""
        lever = self._make_lever()
        state = EnvState()

        root = _ix("root", p=0.9, initiator="alice")
        lever.on_interaction(root, state)

        good = _ix("good", p=0.8, initiator="bob", causal_parents=["root"])
        lever.on_interaction(good, state)

        consumer = _ix("consumer", p=0.7, initiator="carol", causal_parents=["root"])
        effect = lever.on_interaction(consumer, state)

        # All descendants have p > 0.3 → cascade_risk = 0.0
        assert effect.cost_a == 0.0

    def test_epoch_start_is_noop(self):
        """on_epoch_start is a no-op (GC moved to orchestrator)."""
        lever = self._make_lever()
        state = EnvState(steps_per_epoch=10)
        state.artifact_registry.publish(
            __import__("swarm.models.artifact", fromlist=["Artifact"]).Artifact(
                artifact_id="old", kind="r", step=0,
            )
        )
        assert len(state.artifact_registry) == 1
        lever.on_epoch_start(state, epoch=50)
        # GC is now orchestrator's responsibility, lever should not touch it
        assert len(state.artifact_registry) == 1


class TestCascadeRiskConfig:
    def test_valid_config(self):
        """All cascade risk config fields pass validation."""
        config = GovernanceConfig(
            cascade_risk_enabled=True,
            cascade_risk_threshold=0.5,
            cascade_risk_penalty_scale=1.0,
            cascade_risk_reputation_scale=0.5,
            cascade_risk_p_threshold=0.3,
            cascade_risk_window=200,
        )
        assert config.cascade_risk_enabled is True

    def test_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="cascade_risk_threshold"):
            GovernanceConfig(cascade_risk_threshold=1.5)

    def test_negative_penalty_scale(self):
        with pytest.raises(ValueError, match="cascade_risk_penalty_scale"):
            GovernanceConfig(cascade_risk_penalty_scale=-1.0)

    def test_p_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="cascade_risk_p_threshold"):
            GovernanceConfig(cascade_risk_p_threshold=-0.1)

    def test_window_too_small(self):
        with pytest.raises(ValueError, match="cascade_risk_window"):
            GovernanceConfig(cascade_risk_window=0)
