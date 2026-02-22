"""Tests for RBAC (Role-Based Access Control) governance lever."""

import pytest

from swarm.agents.base import ActionType, Role
from swarm.governance.config import GovernanceConfig
from swarm.governance.engine import GovernanceEngine
from swarm.governance.rbac import DEFAULT_ROLE_ACTION_MAP, RBACLever
from swarm.models.interaction import SoftInteraction


def _make_config(**overrides) -> GovernanceConfig:
    defaults = {
        "rbac_enabled": True,
        "rbac_violation_penalty": 0.5,
        "rbac_violation_reputation_penalty": -0.2,
        "rbac_high_stakes_actions": ["spawn_subagent", "post_bounty"],
        "rbac_security_clearance_required": 2,
        "rbac_high_stakes_penalty_multiplier": 2.0,
    }
    defaults.update(overrides)
    return GovernanceConfig(**defaults)


def _make_interaction(initiator: str, action_type: str, **kwargs) -> SoftInteraction:
    return SoftInteraction(
        initiator=initiator,
        counterparty=kwargs.get("counterparty", "other"),
        p=kwargs.get("p", 0.5),
        metadata={"action_type": action_type},
    )


class _FakeState:
    """Minimal stand-in for EnvState."""

    def __init__(self):
        self.agents = {}


# ---- Tests ----


class TestRBACDisabled:
    def test_disabled_lever_no_penalty(self):
        """When rbac_enabled=False, on_interaction returns zero cost."""
        config = _make_config(rbac_enabled=False)
        lever = RBACLever(config)
        lever.set_agent_roles({"a1": ["poster"]})

        # poster attempting a worker action — but lever disabled
        interaction = _make_interaction("a1", ActionType.CLAIM_TASK.value)
        effect = lever.on_interaction(interaction, _FakeState())
        assert effect.cost_a == 0.0
        assert effect.reputation_deltas == {}


class TestPermittedActions:
    def test_permitted_action_no_penalty(self):
        """An action within the agent's role incurs no cost."""
        config = _make_config()
        lever = RBACLever(config)
        lever.set_agent_roles({"w1": ["worker"]})

        interaction = _make_interaction("w1", ActionType.CLAIM_TASK.value)
        effect = lever.on_interaction(interaction, _FakeState())
        assert effect.cost_a == 0.0

    def test_multi_role_union(self):
        """Agent with multiple roles gets union of all permitted actions."""
        config = _make_config()
        lever = RBACLever(config)
        lever.set_agent_roles({"hybrid": ["worker", "verifier"]})

        permitted = lever.get_permitted_actions("hybrid")
        worker_actions = DEFAULT_ROLE_ACTION_MAP[Role.WORKER.value]
        verifier_actions = DEFAULT_ROLE_ACTION_MAP[Role.VERIFIER.value]
        for action in worker_actions:
            assert action in permitted
        for action in verifier_actions:
            assert action in permitted


class TestRoleViolation:
    def test_role_violation_penalty(self):
        """Attempting an action outside one's role incurs a penalty."""
        config = _make_config()
        lever = RBACLever(config)
        lever.set_agent_roles({"p1": ["poster"]})

        interaction = _make_interaction("p1", ActionType.CLAIM_TASK.value)
        effect = lever.on_interaction(interaction, _FakeState())
        assert effect.cost_a == pytest.approx(0.5)
        assert effect.reputation_deltas["p1"] == pytest.approx(-0.2)
        assert effect.details["violation"] is True

    def test_violation_counts_accumulate(self):
        """Epoch and lifetime violation counts increment."""
        config = _make_config()
        lever = RBACLever(config)
        lever.set_agent_roles({"p1": ["poster"]})
        state = _FakeState()

        for _ in range(3):
            interaction = _make_interaction("p1", ActionType.CLAIM_TASK.value)
            lever.on_interaction(interaction, state)

        assert lever.get_epoch_violations()["p1"] == 3
        assert lever.get_lifetime_violations()["p1"] == 3

    def test_epoch_violations_clear_lifetime_persists(self):
        """on_epoch_start clears epoch counts but lifetime persists."""
        config = _make_config()
        lever = RBACLever(config)
        lever.set_agent_roles({"p1": ["poster"]})
        state = _FakeState()

        interaction = _make_interaction("p1", ActionType.CLAIM_TASK.value)
        lever.on_interaction(interaction, state)
        assert lever.get_epoch_violations()["p1"] == 1

        lever.on_epoch_start(state, epoch=1)
        assert lever.get_epoch_violations().get("p1", 0) == 0
        assert lever.get_lifetime_violations()["p1"] == 1


class TestHighStakes:
    def test_high_stakes_without_clearance(self):
        """High-stakes action without clearance gets multiplied penalty."""
        config = _make_config()
        lever = RBACLever(config)
        # Worker has no spawn_subagent in default map, so it's both
        # a role violation AND high-stakes
        lever.set_agent_roles({"w1": ["worker"]})
        lever.set_security_clearances({"w1": 0})

        interaction = _make_interaction("w1", ActionType.SPAWN_SUBAGENT.value)
        effect = lever.on_interaction(interaction, _FakeState())
        # base 0.5 * multiplier 2.0 = 1.0
        assert effect.cost_a == pytest.approx(1.0)
        assert effect.details["high_stakes"] is True

    def test_high_stakes_with_clearance_permitted_role(self):
        """High-stakes action with sufficient clearance AND role = no penalty."""
        config = _make_config()
        lever = RBACLever(config)
        # Planner has spawn_subagent in default map
        lever.set_agent_roles({"plan1": ["planner"]})
        lever.set_security_clearances({"plan1": 2})

        interaction = _make_interaction("plan1", ActionType.SPAWN_SUBAGENT.value)
        effect = lever.on_interaction(interaction, _FakeState())
        assert effect.cost_a == 0.0

    def test_high_stakes_with_role_but_no_clearance(self):
        """Agent has the role for a high-stakes action but lacks clearance."""
        config = _make_config()
        lever = RBACLever(config)
        lever.set_agent_roles({"plan1": ["planner"]})
        lever.set_security_clearances({"plan1": 1})  # needs 2

        interaction = _make_interaction("plan1", ActionType.SPAWN_SUBAGENT.value)
        effect = lever.on_interaction(interaction, _FakeState())
        assert effect.cost_a == pytest.approx(1.0)
        assert effect.details["high_stakes"] is True


class TestCustomConfig:
    def test_custom_role_action_map(self):
        """Config override replaces default role-action map."""
        custom_map = {"worker": ["post", "reply"]}
        config = _make_config(rbac_role_action_map=custom_map)
        lever = RBACLever(config)
        lever.set_agent_roles({"w1": ["worker"]})

        permitted = lever.get_permitted_actions("w1")
        assert permitted == {"post", "reply"}

        # claim_task is no longer permitted for worker in custom map
        interaction = _make_interaction("w1", ActionType.CLAIM_TASK.value)
        effect = lever.on_interaction(interaction, _FakeState())
        assert effect.cost_a > 0


class TestMissingMetadata:
    def test_missing_action_type_no_op(self):
        """Interaction without action_type metadata is a graceful no-op."""
        config = _make_config()
        lever = RBACLever(config)
        lever.set_agent_roles({"w1": ["worker"]})

        interaction = SoftInteraction(
            initiator="w1", counterparty="other", p=0.5, metadata={}
        )
        effect = lever.on_interaction(interaction, _FakeState())
        assert effect.cost_a == 0.0


class TestGovernanceEngineIntegration:
    def test_engine_registers_rbac_lever(self):
        """GovernanceEngine creates RBACLever when rbac_enabled=True."""
        config = _make_config()
        engine = GovernanceEngine(config=config)

        lever = engine.get_rbac_lever()
        assert lever is not None
        assert lever.name == "rbac"

    def test_engine_set_roles_and_clearances(self):
        """Engine proxy methods reach the underlying lever."""
        config = _make_config()
        engine = GovernanceEngine(config=config)

        engine.set_rbac_agent_roles({"a1": ["worker"]})
        engine.set_rbac_security_clearances({"a1": 0})

        lever = engine.get_rbac_lever()
        assert lever is not None
        assert lever.get_permitted_actions("a1") == set(
            DEFAULT_ROLE_ACTION_MAP[Role.WORKER.value]
        )

    def test_engine_not_registered_when_disabled(self):
        """When rbac_enabled=False, no RBACLever is registered."""
        config = GovernanceConfig(rbac_enabled=False)
        engine = GovernanceEngine(config=config)
        assert engine.get_rbac_lever() is None
