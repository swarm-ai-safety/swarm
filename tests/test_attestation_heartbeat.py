"""Tests for the attestation heartbeat governance lever."""

import pytest
from pydantic import ValidationError

from swarm.env.state import EnvState
from swarm.governance.attestation_heartbeat import (
    AttestationHeartbeatLever,
    HeartbeatTracker,
)
from swarm.governance.config import GovernanceConfig
from swarm.governance.engine import GovernanceEngine
from swarm.models.agent import AgentType


def _make_state(*agent_ids: str) -> EnvState:
    """Create a minimal EnvState with the given agents."""
    state = EnvState()
    for aid in agent_ids:
        state.add_agent(aid, agent_type=AgentType.HONEST, initial_resources=10.0)
    return state


class TestHeartbeatTracker:
    """Tests for the HeartbeatTracker dataclass."""

    def test_defaults(self):
        t = HeartbeatTracker()
        assert t.last_attestation_step == -1
        assert t.missed_count == 0
        assert t.is_lapsed is False


class TestAttestationHeartbeatLever:
    """Tests for the AttestationHeartbeatLever."""

    def _make_lever(self, interval: int = 5, penalty: float = -0.1) -> AttestationHeartbeatLever:
        config = GovernanceConfig(
            attestation_heartbeat_enabled=True,
            attestation_heartbeat_interval=interval,
            attestation_heartbeat_reputation_penalty=penalty,
        )
        return AttestationHeartbeatLever(config)

    def test_name(self):
        lever = self._make_lever()
        assert lever.name == "attestation_heartbeat"

    def test_disabled_lever_allows_all(self):
        config = GovernanceConfig(attestation_heartbeat_enabled=False)
        lever = AttestationHeartbeatLever(config)
        lever.register_agent("a1", initial_step=-1)
        state = _make_state("a1")
        # Even without attestation, agent can act when disabled
        assert lever.can_agent_act("a1", state) is True
        effect = lever.on_step(state, step=100)
        assert len(effect.agents_to_freeze) == 0

    def test_untracked_agent_allowed(self):
        lever = self._make_lever()
        state = _make_state("a1")
        # Agent not registered — should be allowed
        assert lever.can_agent_act("a1", state) is True

    def test_attestation_resets_deadline(self):
        lever = self._make_lever(interval=5)
        lever.register_agent("a1", initial_step=0)
        state = _make_state("a1")

        # Step 4: within interval (deadline = 0 + 5 = 5), should be fine
        effect = lever.on_step(state, step=4)
        assert "a1" not in effect.agents_to_freeze

        # Attest at step 4
        lever.record_attestation("a1", step=4)

        # Step 9: within new interval (deadline = 4 + 5 = 9), should be fine
        effect = lever.on_step(state, step=9)
        assert "a1" not in effect.agents_to_freeze

    def test_lapsed_agent_frozen(self):
        lever = self._make_lever(interval=3)
        lever.register_agent("a1", initial_step=0)
        state = _make_state("a1")

        # Step 3: deadline is 0 + 3 = 3, step 3 is NOT > 3, so still OK
        effect = lever.on_step(state, step=3)
        assert "a1" not in effect.agents_to_freeze
        assert lever.can_agent_act("a1", state) is True

        # Step 4: now past deadline
        effect = lever.on_step(state, step=4)
        assert "a1" in effect.agents_to_freeze
        assert lever.can_agent_act("a1", state) is False

    def test_lapsed_agent_stays_frozen_until_reattestation(self):
        lever = self._make_lever(interval=3)
        lever.register_agent("a1", initial_step=0)
        state = _make_state("a1")

        # Lapse
        lever.on_step(state, step=4)
        assert lever.can_agent_act("a1", state) is False

        # Still frozen at next step
        lever.on_step(state, step=5)
        assert lever.can_agent_act("a1", state) is False

        # Re-attest
        lever.record_attestation("a1", step=5)
        assert lever.can_agent_act("a1", state) is True

    def test_missed_count_increments(self):
        lever = self._make_lever(interval=2)
        lever.register_agent("a1", initial_step=0)
        state = _make_state("a1")

        # First lapse
        lever.on_step(state, step=3)
        status = lever.get_status("a1")
        assert status["missed_count"] == 1

        # Re-attest and lapse again
        lever.record_attestation("a1", step=3)
        lever.on_step(state, step=6)
        status = lever.get_status("a1")
        assert status["missed_count"] == 2

    def test_reputation_penalty_applied(self):
        lever = self._make_lever(interval=2, penalty=-0.2)
        lever.register_agent("a1", initial_step=0)
        state = _make_state("a1")

        effect = lever.on_step(state, step=3)
        assert effect.reputation_deltas.get("a1") == -0.2

    def test_no_reputation_penalty_when_zero(self):
        lever = self._make_lever(interval=2, penalty=0.0)
        lever.register_agent("a1", initial_step=0)
        state = _make_state("a1")

        effect = lever.on_step(state, step=3)
        assert "a1" not in effect.reputation_deltas

    def test_penalty_only_on_newly_lapsed(self):
        """Penalty applies once on lapse, not every step afterward."""
        lever = self._make_lever(interval=2, penalty=-0.2)
        lever.register_agent("a1", initial_step=0)
        state = _make_state("a1")

        # First step past deadline — penalty applied
        effect = lever.on_step(state, step=3)
        assert effect.reputation_deltas.get("a1") == -0.2

        # Next step — still frozen but no repeated penalty
        effect = lever.on_step(state, step=4)
        assert "a1" in effect.agents_to_freeze
        assert "a1" not in effect.reputation_deltas

    def test_auto_registration_via_on_step(self):
        """Agents in state.agents are auto-tracked on first on_step."""
        lever = self._make_lever(interval=3)
        state = _make_state("a1", "a2")

        # No explicit register_agent — on_step auto-registers from state
        effect = lever.on_step(state, step=0)
        assert "a1" not in effect.agents_to_freeze
        assert "a2" not in effect.agents_to_freeze

        # After interval, both should lapse
        effect = lever.on_step(state, step=4)
        assert "a1" in effect.agents_to_freeze
        assert "a2" in effect.agents_to_freeze

    def test_auto_registration_via_can_agent_act(self):
        """Untracked agent is auto-registered on can_agent_act."""
        lever = self._make_lever(interval=3)
        state = _make_state("a1")

        # Auto-registers and allows (first interval starts now)
        assert lever.can_agent_act("a1", state) is True
        # Agent is now tracked
        status = lever.get_status("a1")
        assert status["is_lapsed"] is False

    def test_epoch_reset_clears_lapsed(self):
        config = GovernanceConfig(
            attestation_heartbeat_enabled=True,
            attestation_heartbeat_interval=3,
            attestation_heartbeat_reset_on_epoch=True,
        )
        lever = AttestationHeartbeatLever(config)
        lever.register_agent("a1", initial_step=0)
        state = _make_state("a1")

        # Lapse
        lever.on_step(state, step=4)
        assert lever.can_agent_act("a1", state) is False

        # Epoch reset
        effect = lever.on_epoch_start(state, epoch=1)
        assert "a1" in effect.agents_to_unfreeze
        assert lever.can_agent_act("a1", state) is True

    def test_epoch_reset_disabled(self):
        config = GovernanceConfig(
            attestation_heartbeat_enabled=True,
            attestation_heartbeat_interval=3,
            attestation_heartbeat_reset_on_epoch=False,
        )
        lever = AttestationHeartbeatLever(config)
        lever.register_agent("a1", initial_step=0)
        state = _make_state("a1")

        # Lapse
        lever.on_step(state, step=4)
        assert lever.can_agent_act("a1", state) is False

        # Epoch reset should NOT clear
        effect = lever.on_epoch_start(state, epoch=1)
        assert "a1" not in effect.agents_to_unfreeze
        assert lever.can_agent_act("a1", state) is False

    def test_get_lapsed_agents(self):
        lever = self._make_lever(interval=2)
        lever.register_agent("a1", initial_step=0)
        lever.register_agent("a2", initial_step=0)
        state = _make_state("a1", "a2")

        # Attest a2 but not a1
        lever.record_attestation("a2", step=2)
        lever.on_step(state, step=3)

        lapsed = lever.get_lapsed_agents()
        assert "a1" in lapsed
        assert "a2" not in lapsed

    def test_multiple_agents_independent(self):
        lever = self._make_lever(interval=3)
        lever.register_agent("a1", initial_step=0)
        lever.register_agent("a2", initial_step=1)
        state = _make_state("a1", "a2")

        # Step 4: a1 deadline=3 (lapsed), a2 deadline=4 (not lapsed)
        effect = lever.on_step(state, step=4)
        assert "a1" in effect.agents_to_freeze
        assert "a2" not in effect.agents_to_freeze

        # Step 5: a2 now lapsed too
        effect = lever.on_step(state, step=5)
        assert "a2" in effect.agents_to_freeze


class TestAttestationHeartbeatConfig:
    """Tests for heartbeat config validation."""

    def test_default_config_valid(self):
        config = GovernanceConfig()
        assert config.attestation_heartbeat_enabled is False
        assert config.attestation_heartbeat_interval == 5

    def test_invalid_interval(self):
        with pytest.raises(ValidationError, match="attestation_heartbeat_interval"):
            GovernanceConfig(attestation_heartbeat_interval=0)


class TestEngineIntegration:
    """Test that the heartbeat lever integrates with GovernanceEngine."""

    def test_engine_registers_lever_when_enabled(self):
        config = GovernanceConfig(attestation_heartbeat_enabled=True)
        engine = GovernanceEngine(config=config)
        assert "attestation_heartbeat" in engine.get_registered_lever_names()
        assert engine.get_attestation_heartbeat_lever() is not None

    def test_engine_omits_lever_when_disabled(self):
        config = GovernanceConfig(attestation_heartbeat_enabled=False)
        engine = GovernanceEngine(config=config)
        assert "attestation_heartbeat" not in engine.get_registered_lever_names()
        assert engine.get_attestation_heartbeat_lever() is None

    def test_engine_blocks_lapsed_agent(self):
        config = GovernanceConfig(
            attestation_heartbeat_enabled=True,
            attestation_heartbeat_interval=2,
        )
        engine = GovernanceEngine(config=config)
        state = _make_state("a1")

        # Register and let lapse through engine's step hook
        lever = engine.get_attestation_heartbeat_lever()
        assert lever is not None
        lever.register_agent("a1", initial_step=0)

        engine.apply_step(state, step=3)
        assert engine.can_agent_act("a1", state) is False

        # Re-attest
        lever.record_attestation("a1", step=3)
        assert engine.can_agent_act("a1", state) is True
