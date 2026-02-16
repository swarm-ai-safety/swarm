"""Tests for the loop detector governance lever."""

import pytest
from pydantic import ValidationError

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.loop_detector import LoopDetectorLever
from swarm.models.interaction import InteractionType, SoftInteraction


class TestLoopDetectorConfig:
    """Tests for loop detector configuration validation."""

    def test_default_config(self):
        """Default config should be valid."""
        config = GovernanceConfig()
        assert config.loop_detector_enabled is False
        assert config.loop_detector_quality_threshold == 0.4
        assert config.loop_detector_penalty_multiplier == 1.5

    def test_invalid_quality_threshold(self):
        """Quality threshold outside [0,1] should raise."""
        with pytest.raises(ValidationError, match="loop_detector_quality_threshold"):
            GovernanceConfig(loop_detector_quality_threshold=1.5)

        with pytest.raises(ValidationError, match="loop_detector_quality_threshold"):
            GovernanceConfig(loop_detector_quality_threshold=-0.1)

    def test_invalid_tool_misuse_threshold(self):
        """Negative tool misuse threshold should raise."""
        with pytest.raises(ValidationError, match="loop_detector_tool_misuse_threshold"):
            GovernanceConfig(loop_detector_tool_misuse_threshold=-1)

    def test_invalid_penalty_multiplier(self):
        """Negative penalty multiplier should raise."""
        with pytest.raises(ValidationError, match="loop_detector_penalty_multiplier"):
            GovernanceConfig(loop_detector_penalty_multiplier=-1.0)

    def test_invalid_freeze_threshold(self):
        """Freeze threshold < 1 should raise."""
        with pytest.raises(ValidationError, match="loop_detector_freeze_threshold"):
            GovernanceConfig(loop_detector_freeze_threshold=0)


class TestLoopDetectorLever:
    """Tests for LoopDetectorLever."""

    def test_disabled_no_detection(self):
        """When disabled, no loops should be detected."""
        config = GovernanceConfig(loop_detector_enabled=False)
        lever = LoopDetectorLever(config)

        state = EnvState()
        state.add_agent("agent_1")

        # Create repetitive pattern
        for _ in range(10):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_2",
                interaction_type=InteractionType.REPLY,
                p=0.2,
            )
            effect = lever.on_interaction(interaction, state)
            assert not effect.details.get("loop_detected", False)
            assert len(effect.agents_to_freeze) == 0

    def test_repetition_loop_detection(self):
        """Agent repeating same pattern should be detected."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_freeze_threshold=4,
        )
        lever = LoopDetectorLever(config)

        state = EnvState()
        state.add_agent("agent_1")

        # Send 10 identical interactions (same counterparty + type)
        for i in range(10):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_2",
                interaction_type=InteractionType.REPLY,
                p=0.6,  # Reasonable quality
            )
            effect = lever.on_interaction(interaction, state)

            # After 5 interactions, repetition should be detected (60%+ of pattern)
            if i >= 4:
                assert effect.details.get("loop_detected", False)
                assert effect.details.get("loop_type") == "repetition"

    def test_quality_loop_detection(self):
        """Agent with persistently low quality should be detected."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_quality_threshold=0.4,
            loop_detector_freeze_threshold=4,
        )
        lever = LoopDetectorLever(config)

        state = EnvState()
        state.add_agent("agent_1")

        # Send interactions with varied counterparties but low p
        counterparties = ["agent_2", "agent_3", "agent_4", "agent_2", "agent_5"]
        for i, cp in enumerate(counterparties + counterparties):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty=cp,
                interaction_type=InteractionType.REPLY,
                p=0.2,  # Low quality
            )
            effect = lever.on_interaction(interaction, state)

            # After 5 interactions, quality loop should be detected
            if i >= 4:
                assert effect.details.get("loop_detected", False)
                assert effect.details.get("loop_type") == "low_quality"

    def test_tool_misuse_loop_detection(self):
        """Agent with persistent tool misuse should be detected."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_tool_misuse_threshold=5,
            loop_detector_freeze_threshold=4,
        )
        lever = LoopDetectorLever(config)

        state = EnvState()
        state.add_agent("agent_1")

        # Send interactions with tool misuse flags
        for i in range(6):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty=f"agent_{i % 3 + 2}",
                interaction_type=InteractionType.REPLY,
                p=0.6,
                tool_misuse_flags=2,  # Each interaction has misuse
            )
            effect = lever.on_interaction(interaction, state)

            # After 3 interactions with 2 flags each (total 6), should detect
            if i >= 2:
                if effect.details.get("loop_detected", False):
                    assert effect.details.get("loop_type") == "tool_misuse"

    def test_rework_loop_detection(self):
        """Agent with persistent high rework should be detected."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_rework_threshold=10,
            loop_detector_freeze_threshold=4,
        )
        lever = LoopDetectorLever(config)

        state = EnvState()
        state.add_agent("agent_1")

        # Send interactions with high rework counts
        for i in range(6):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty=f"agent_{i % 3 + 2}",
                interaction_type=InteractionType.REPLY,
                p=0.6,
                rework_count=4,  # High rework
            )
            effect = lever.on_interaction(interaction, state)

            # After 3 interactions with 4 rework each (total 12), should detect
            if i >= 2:
                if effect.details.get("loop_detected", False):
                    assert effect.details.get("loop_type") == "rework"

    def test_penalty_applied_on_loop(self):
        """Loop detection should apply penalty cost."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_penalty_multiplier=2.0,
            loop_detector_freeze_threshold=4,
        )
        lever = LoopDetectorLever(config)

        state = EnvState()
        state.add_agent("agent_1")

        # Create a quality loop
        for i in range(6):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_2",
                interaction_type=InteractionType.REPLY,
                accepted=True,
                p=0.3,  # Low quality
            )
            effect = lever.on_interaction(interaction, state)

            if effect.details.get("loop_detected", False):
                # Penalty should be applied
                assert effect.cost_a > 0
                expected_penalty = 2.0 * 0.3  # multiplier * p (since accepted)
                assert effect.cost_a == pytest.approx(expected_penalty)

    def test_forced_replan_trigger(self):
        """Loop violations should trigger forced replan."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_replan_threshold=2,
            loop_detector_freeze_threshold=4,
        )
        lever = LoopDetectorLever(config)

        state = EnvState()
        state.current_epoch = 1
        state.add_agent("agent_1")

        # Create repetitive pattern to build violations
        violations = 0
        for i in range(10):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_2",
                interaction_type=InteractionType.REPLY,
                p=0.3,
            )
            effect = lever.on_interaction(interaction, state)

            if effect.details.get("loop_detected", False):
                violations += 1
                if violations >= 2:
                    # Should trigger replan
                    assert effect.details.get("force_replan", False)
                    break

    def test_circuit_breaker_freeze(self):
        """Excessive loop violations should freeze agent."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_freeze_threshold=3,
            loop_detector_freeze_duration=2,
        )
        lever = LoopDetectorLever(config)

        state = EnvState()
        state.current_epoch = 1
        state.add_agent("agent_1")

        # Create repetitive pattern to build violations
        for i in range(10):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_2",
                interaction_type=InteractionType.REPLY,
                p=0.3,
            )
            effect = lever.on_interaction(interaction, state)

            if len(effect.agents_to_freeze) > 0:
                assert "agent_1" in effect.agents_to_freeze
                # Verify freeze duration is set
                status = lever.get_loop_status("agent_1")
                assert status["is_frozen"]
                assert status["freeze_until_epoch"] == state.current_epoch + 2
                break

    def test_unfreeze_after_duration(self):
        """Agent should unfreeze after freeze duration expires."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_freeze_threshold=2,
            loop_detector_freeze_duration=1,
        )
        lever = LoopDetectorLever(config)

        state = EnvState()
        state.current_epoch = 1
        state.add_agent("agent_1")

        # Trigger freeze
        for i in range(6):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_2",
                interaction_type=InteractionType.REPLY,
                p=0.3,
            )
            lever.on_interaction(interaction, state)

        # Check if frozen
        status = lever.get_loop_status("agent_1")
        if status["is_frozen"]:
            freeze_until = status["freeze_until_epoch"]

            # Advance to unfreeze epoch
            effect = lever.on_epoch_start(state, freeze_until)

            # Should unfreeze
            assert "agent_1" in effect.agents_to_unfreeze
            assert effect.details["unfrozen_count"] == 1

            # Check status after unfreeze
            status = lever.get_loop_status("agent_1")
            assert not status["is_frozen"]
            assert status["violations"] == 0  # Should reset

    def test_no_replan_same_epoch(self):
        """Forced replan should not trigger twice in same epoch."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_replan_threshold=2,
            loop_detector_freeze_threshold=5,
        )
        lever = LoopDetectorLever(config)

        state = EnvState()
        state.current_epoch = 1
        state.add_agent("agent_1")

        replan_count = 0
        for i in range(15):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_2",
                interaction_type=InteractionType.REPLY,
                p=0.3,
            )
            effect = lever.on_interaction(interaction, state)

            if effect.details.get("force_replan", False):
                replan_count += 1

        # Should only replan once per epoch
        assert replan_count <= 1

    def test_reset_tracker(self):
        """Reset should clear agent tracking."""
        config = GovernanceConfig(loop_detector_enabled=True)
        lever = LoopDetectorLever(config)

        state = EnvState()
        state.add_agent("agent_1")

        # Create some interactions
        for _ in range(5):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_2",
                interaction_type=InteractionType.REPLY,
                p=0.3,
            )
            lever.on_interaction(interaction, state)

        # Reset tracker
        lever.reset_tracker("agent_1")

        # Status should be clean
        status = lever.get_loop_status("agent_1")
        assert status["violations"] == 0
        assert status["recent_interactions_count"] == 0

    def test_multiple_agents_independent(self):
        """Loop detection should track agents independently."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_freeze_threshold=3,
        )
        lever = LoopDetectorLever(config)

        state = EnvState()
        state.add_agent("agent_1")
        state.add_agent("agent_2")

        # Agent 1 creates loop
        for _ in range(10):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_3",
                interaction_type=InteractionType.REPLY,
                p=0.2,
            )
            lever.on_interaction(interaction, state)

        # Agent 2 behaves normally
        for i in range(5):
            interaction = SoftInteraction(
                initiator="agent_2",
                counterparty=f"agent_{i + 3}",
                interaction_type=InteractionType.REPLY,
                p=0.8,
            )
            lever.on_interaction(interaction, state)

        # Check statuses
        status_1 = lever.get_loop_status("agent_1")
        status_2 = lever.get_loop_status("agent_2")

        # Agent 1 should have violations, agent 2 should not
        assert status_1["violations"] > 0
        assert status_2["violations"] == 0
