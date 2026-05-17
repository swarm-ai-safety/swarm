"""Tests for hardware root-of-trust rejection handling."""

import pytest

from swarm.agents.base import Action, ActionType
from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.engine import GovernanceEngine
from swarm.governance.hardware_trust import (
    IRREVERSIBLE_ACTIONS,
    REVERSIBLE_ACTIONS,
    HaltReason,
    HardwareTrustLever,
    RecoveryMode,
)


@pytest.fixture
def config():
    return GovernanceConfig(
        hardware_trust_enabled=True,
        hardware_trust_propagation_enabled=True,
        hardware_trust_recovery_max_steps=5,
    )


@pytest.fixture
def lever(config):
    return HardwareTrustLever(config)


@pytest.fixture
def state():
    s = EnvState()
    s.add_agent("agent-1")
    s.add_agent("agent-2")
    s.add_agent("agent-3")
    return s


class TestHardwareTrustLever:
    """Test the HardwareTrustLever governance lever."""

    def test_lever_name(self, lever):
        assert lever.name == "hardware_trust"

    def test_receive_halt_constrains_source_agent(self, lever, state):
        effect = lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        # Constrained agents are NOT in agents_to_freeze (they can still
        # act with action-level filtering).  State-level freeze is reserved
        # for FROZEN mode escalation.
        assert len(effect.agents_to_freeze) == 0
        assert "agent-1" in lever.get_constrained_agents()
        assert lever.is_halted
        assert lever.get_active_halt() is not None
        assert lever.get_active_halt().halt_id == "halt-1"

    def test_receive_halt_global_when_no_source(self, lever, state):
        """When no source agent specified, halt constrains all agents."""
        effect = lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.TAMPER_DETECTED,
            state=state,
        )
        # All agents constrained, not state-level frozen
        assert len(effect.agents_to_freeze) == 0
        assert lever.get_constrained_agents() == {"agent-1", "agent-2", "agent-3"}

    def test_halt_records_causal_trace(self, lever, state):
        causal = ["event-a", "event-b", "event-c"]
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.ATTESTATION_FAILURE,
            state=state,
            source_agent_id="agent-2",
            causal_trace=causal,
        )
        record = lever.get_halt_record("halt-1")
        assert record is not None
        assert record.causal_trace == causal
        assert record.reason == HaltReason.ATTESTATION_FAILURE

    def test_halt_sets_constrained_recovery(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.POLICY_BREACH,
            state=state,
            source_agent_id="agent-1",
        )
        recovery = lever.get_recovery_state("agent-1")
        assert recovery is not None
        assert recovery.mode == RecoveryMode.CONSTRAINED
        assert recovery.halt_id == "halt-1"
        assert recovery.checkpoint_epoch == state.current_epoch
        assert recovery.checkpoint_step == state.current_step

    def test_constrained_mode_blocks_irreversible_actions(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        # Irreversible actions should be blocked
        for action in IRREVERSIBLE_ACTIONS:
            assert not lever.is_action_allowed("agent-1", action), (
                f"Action {action} should be blocked in constrained mode"
            )
        # Safe actions (noop, vote, reply) should be allowed
        assert lever.is_action_allowed("agent-1", "noop")
        assert lever.is_action_allowed("agent-1", "vote")
        assert lever.is_action_allowed("agent-1", "reply")

    def test_constrained_agent_can_still_act(self, lever, state):
        """Constrained agents can act (with filtered actions), not frozen."""
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        assert lever.can_agent_act("agent-1", state)

    def test_clear_halt_unfreezes_agents(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        effect = lever.clear_halt("halt-1", state)
        assert "agent-1" in effect.agents_to_unfreeze
        assert not lever.is_halted
        recovery = lever.get_recovery_state("agent-1")
        assert recovery.mode == RecoveryMode.NORMAL

    def test_clear_nonexistent_halt_is_noop(self, lever, state):
        effect = lever.clear_halt("nonexistent", state)
        assert len(effect.agents_to_unfreeze) == 0

    def test_clear_already_cleared_halt_is_noop(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        lever.clear_halt("halt-1", state)
        # Second clear should be noop
        effect = lever.clear_halt("halt-1", state)
        assert len(effect.agents_to_unfreeze) == 0

    def test_recovery_budget_escalates_to_frozen(self, lever, state):
        """After max recovery steps, agents escalate from constrained to frozen."""
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        # Simulate steps up to the limit
        for step in range(lever.config.hardware_trust_recovery_max_steps - 1):
            effect = lever.on_step(state, step)
            assert "agent-1" not in effect.agents_to_freeze

        # The final step should escalate
        effect = lever.on_step(state, 99)
        assert "agent-1" in effect.agents_to_freeze

        # Now agent should be in FROZEN mode
        recovery = lever.get_recovery_state("agent-1")
        assert recovery.mode == RecoveryMode.FROZEN
        assert not lever.can_agent_act("agent-1", state)

    def test_frozen_agent_cannot_act(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        # Manually set to frozen
        lever._recovery_states["agent-1"].mode = RecoveryMode.FROZEN
        assert not lever.can_agent_act("agent-1", state)

    def test_unaffected_agents_can_act(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        assert lever.can_agent_act("agent-2", state)
        assert lever.is_action_allowed("agent-2", "propose_interaction")

    def test_disabled_lever_is_transparent(self, state):
        config = GovernanceConfig(hardware_trust_enabled=False)
        lever = HardwareTrustLever(config)
        effect = lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        assert len(effect.agents_to_freeze) == 0
        assert lever.can_agent_act("agent-1", state)
        assert lever.is_action_allowed("agent-1", "propose_interaction")


class TestStopTokenPropagation:
    """Test that halt signals propagate across dependent agents."""

    def test_propagation_via_dependency_graph(self, lever, state):
        lever.register_dependency("agent-1", "agent-2")
        lever.register_dependency("agent-2", "agent-3")

        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        # All three should be constrained (agent-1 -> agent-2 -> agent-3)
        assert lever.get_constrained_agents() == {"agent-1", "agent-2", "agent-3"}

    def test_propagation_respects_state_agents(self, lever, state):
        """Only propagate to agents that exist in state."""
        lever.register_dependency("agent-1", "agent-2")
        lever.register_dependency("agent-1", "phantom-agent")

        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        assert "agent-1" in lever.get_constrained_agents()
        assert "agent-2" in lever.get_constrained_agents()
        assert "phantom-agent" not in lever.get_constrained_agents()

    def test_propagation_disabled(self, state):
        config = GovernanceConfig(
            hardware_trust_enabled=True,
            hardware_trust_propagation_enabled=False,
        )
        lever = HardwareTrustLever(config)
        lever.register_dependency("agent-1", "agent-2")

        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        # Only source agent, no propagation
        assert lever.get_constrained_agents() == {"agent-1"}

    def test_set_dependency_graph(self, lever, state):
        lever.set_dependency_graph({
            "agent-1": {"agent-2", "agent-3"},
        })
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        assert lever.get_constrained_agents() == {"agent-1", "agent-2", "agent-3"}


class TestTaskStatePreservation:
    """Test task state preservation and retrieval."""

    def test_preserve_and_retrieve_task_state(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        task_state = {"task_id": "t1", "progress": 0.5, "outputs": ["a", "b"]}
        lever.preserve_task_state("agent-1", task_state)

        retrieved = lever.get_preserved_task_state("agent-1")
        assert retrieved == task_state

    def test_retrieve_empty_for_unknown_agent(self, lever):
        assert lever.get_preserved_task_state("unknown") == {}


class TestQueryInterface:
    """Test the query/introspection methods."""

    def test_get_constrained_agents(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        assert "agent-1" in lever.get_constrained_agents()
        assert "agent-2" not in lever.get_constrained_agents()

    def test_get_frozen_agents(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        assert len(lever.get_frozen_agents()) == 0
        lever._recovery_states["agent-1"].mode = RecoveryMode.FROZEN
        assert "agent-1" in lever.get_frozen_agents()

    def test_get_all_halt_records(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        lever.receive_halt(
            halt_id="halt-2",
            reason=HaltReason.TAMPER_DETECTED,
            state=state,
            source_agent_id="agent-2",
        )
        records = lever.get_all_halt_records()
        assert len(records) == 2

    def test_multiple_halts_tracked_independently(self, lever, state):
        """Clearing one halt does not affect another active halt."""
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        lever.receive_halt(
            halt_id="halt-2",
            reason=HaltReason.TAMPER_DETECTED,
            state=state,
            source_agent_id="agent-2",
        )
        assert lever.is_halted
        assert len(lever.get_active_halts()) == 2

        # Clear halt-2; halt-1 remains active
        lever.clear_halt("halt-2", state)
        assert lever.is_halted
        active = lever.get_active_halts()
        assert len(active) == 1
        assert active[0].halt_id == "halt-1"

    def test_repeated_halt_does_not_overwrite_recovery(self, lever, state):
        """A second halt on the same agent preserves the first recovery entry."""
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        # Simulate some recovery steps
        lever.on_step(state, 0)
        lever.on_step(state, 1)
        recovery_before = lever.get_recovery_state("agent-1")
        assert recovery_before.recovery_steps_taken == 2
        assert recovery_before.halt_id == "halt-1"

        # Second halt targeting same agent
        lever.receive_halt(
            halt_id="halt-2",
            reason=HaltReason.TAMPER_DETECTED,
            state=state,
            source_agent_id="agent-1",
        )
        # Recovery state should NOT be overwritten
        recovery_after = lever.get_recovery_state("agent-1")
        assert recovery_after.halt_id == "halt-1"
        assert recovery_after.recovery_steps_taken == 2


class TestGovernanceEngineIntegration:
    """Test that the lever integrates properly with GovernanceEngine."""

    def test_engine_registers_lever(self):
        config = GovernanceConfig(hardware_trust_enabled=True)
        engine = GovernanceEngine(config)
        assert engine.get_hardware_trust_lever() is not None

    def test_engine_does_not_register_when_disabled(self):
        config = GovernanceConfig(hardware_trust_enabled=False)
        engine = GovernanceEngine(config)
        assert engine.get_hardware_trust_lever() is None

    def test_engine_can_agent_act_respects_hardware_halt(self):
        config = GovernanceConfig(hardware_trust_enabled=True)
        engine = GovernanceEngine(config)
        state = EnvState()
        state.add_agent("agent-1")

        lever = engine.get_hardware_trust_lever()
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        # Constrained agents can still act (they're filtered, not blocked)
        assert engine.can_agent_act("agent-1", state)

        # Escalate to frozen
        lever._recovery_states["agent-1"].mode = RecoveryMode.FROZEN
        assert not engine.can_agent_act("agent-1", state)

    def test_engine_is_action_allowed(self):
        config = GovernanceConfig(hardware_trust_enabled=True)
        engine = GovernanceEngine(config)
        state = EnvState()
        state.add_agent("agent-1")

        lever = engine.get_hardware_trust_lever()
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        # Irreversible actions blocked
        assert not engine.is_action_allowed("agent-1", "propose_interaction")
        # Safe actions allowed
        assert engine.is_action_allowed("agent-1", "noop")
        # Unaffected agents unrestricted
        assert engine.is_action_allowed("agent-2", "propose_interaction")

    def test_engine_step_tracks_recovery(self):
        config = GovernanceConfig(
            hardware_trust_enabled=True,
            hardware_trust_recovery_max_steps=3,
        )
        engine = GovernanceEngine(config)
        state = EnvState()
        state.add_agent("agent-1")

        lever = engine.get_hardware_trust_lever()
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )

        # Step through recovery
        for i in range(3):
            engine.apply_step(state, i)

        # Agent should now be frozen (budget exhausted)
        recovery = lever.get_recovery_state("agent-1")
        assert recovery.mode == RecoveryMode.FROZEN


class TestConfigValidation:
    """Test governance config validation for hardware trust fields."""

    def test_valid_config(self):
        config = GovernanceConfig(
            hardware_trust_enabled=True,
            hardware_trust_recovery_max_steps=10,
        )
        assert config.hardware_trust_recovery_max_steps == 10

    def test_invalid_recovery_max_steps(self):
        with pytest.raises(ValueError, match="hardware_trust_recovery_max_steps"):
            GovernanceConfig(hardware_trust_recovery_max_steps=0)


class TestAllowlistFailSafe:
    """Test that the allowlist approach blocks unknown action types."""

    def test_unknown_action_blocked_in_constrained_mode(self, lever, state):
        """New action types not on the allowlist are blocked by default."""
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        # A hypothetical new action type not on either list
        assert not lever.is_action_allowed("agent-1", "new_dangerous_action")

    def test_reversible_actions_allowed_in_constrained_mode(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        for action in REVERSIBLE_ACTIONS:
            assert lever.is_action_allowed("agent-1", action), (
                f"Reversible action {action} should be allowed"
            )


class TestEventEmission:
    """Test that hardware trust lifecycle emits events."""

    def test_receive_halt_emits_events(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        events = lever.drain_events()
        event_types = [e.event_type.value for e in events]
        assert "hardware_halt_received" in event_types
        assert "hardware_recovery_entered" in event_types

    def test_propagation_emits_event(self, lever, state):
        lever.register_dependency("agent-1", "agent-2")
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        events = lever.drain_events()
        event_types = [e.event_type.value for e in events]
        assert "hardware_halt_propagated" in event_types

    def test_clear_halt_emits_event(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        lever.drain_events()  # Clear receive events
        lever.clear_halt("halt-1", state)
        events = lever.drain_events()
        event_types = [e.event_type.value for e in events]
        assert "hardware_condition_cleared" in event_types

    def test_drain_events_clears_buffer(self, lever, state):
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=state,
            source_agent_id="agent-1",
        )
        first = lever.drain_events()
        assert len(first) > 0
        second = lever.drain_events()
        assert len(second) == 0


class TestOrchestratorIntegration:
    """Test that the hardware trust lever blocks actions at the orchestrator level."""

    def test_orchestrator_blocks_irreversible_action(self):
        """Exercise the actual orchestrator _execute_action code path."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        gov_config = GovernanceConfig(hardware_trust_enabled=True)
        orch_config = OrchestratorConfig(
            governance_config=gov_config,
            n_epochs=1,
            steps_per_epoch=1,
            log_events=False,
        )
        orch = Orchestrator(config=orch_config)
        orch.state.add_agent("agent-1")

        # Trigger a halt
        lever = orch.governance_engine.get_hardware_trust_lever()
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=orch.state,
            source_agent_id="agent-1",
        )

        # Try to execute an irreversible action through the orchestrator
        action = Action(
            action_type=ActionType.PROPOSE_INTERACTION,
            agent_id="agent-1",
            counterparty_id="agent-2",
        )
        result = orch._execute_action(action)
        assert result is False, "Irreversible action should be blocked by hardware trust"

    def test_orchestrator_allows_reversible_action_after_halt(self):
        """Reversible actions should still pass through the hardware trust check."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        gov_config = GovernanceConfig(hardware_trust_enabled=True)
        orch_config = OrchestratorConfig(
            governance_config=gov_config,
            n_epochs=1,
            steps_per_epoch=1,
            log_events=False,
        )
        orch = Orchestrator(config=orch_config)
        orch.state.add_agent("agent-1")

        lever = orch.governance_engine.get_hardware_trust_lever()
        lever.receive_halt(
            halt_id="halt-1",
            reason=HaltReason.INTEGRITY_VIOLATION,
            state=orch.state,
            source_agent_id="agent-1",
        )

        # VOTE is a reversible action — hardware trust should NOT block it
        assert orch.governance_engine.is_action_allowed("agent-1", "vote")
