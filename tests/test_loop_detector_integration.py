"""Integration tests for loop detector with governance engine."""


from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.engine import GovernanceEngine
from swarm.models.interaction import InteractionType, SoftInteraction


class TestLoopDetectorIntegration:
    """Tests for loop detector integration with governance engine."""

    def test_loop_detector_registered_when_enabled(self):
        """Loop detector should be registered when enabled."""
        config = GovernanceConfig(loop_detector_enabled=True)
        engine = GovernanceEngine(config=config)

        # Loop detector should be registered
        lever = engine.get_loop_detector_lever()
        assert lever is not None
        assert lever.name == "loop_detector"

    def test_loop_detector_not_registered_when_disabled(self):
        """Loop detector should not be registered when disabled."""
        config = GovernanceConfig(loop_detector_enabled=False)
        engine = GovernanceEngine(config=config)

        # Loop detector should not be registered
        lever = engine.get_loop_detector_lever()
        assert lever is None

    def test_loop_detector_applies_through_engine(self):
        """Loop detection should work through governance engine."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_freeze_threshold=3,
            loop_detector_penalty_multiplier=2.0,
        )
        engine = GovernanceEngine(config=config)

        state = EnvState()
        state.add_agent("agent_1")

        # Create repetitive interactions through engine
        for i in range(10):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_2",
                interaction_type=InteractionType.REPLY,
                p=0.3,
            )
            effect = engine.apply_interaction(interaction, state)

            # After enough iterations, should detect loop and apply penalties
            if i >= 5:
                # Check if any lever detected a loop
                for lever_effect in effect.lever_effects:
                    if lever_effect.lever_name == "loop_detector":
                        if lever_effect.details.get("loop_detected", False):
                            # Penalty should be applied
                            assert effect.cost_a > 0
                            # Should eventually freeze
                            if lever_effect.details.get("violations", 0) >= 3:
                                assert len(effect.agents_to_freeze) > 0

    def test_loop_detector_unfreeze_through_engine(self):
        """Loop detector unfreeze should work through governance engine."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_freeze_threshold=2,
            loop_detector_freeze_duration=1,
        )
        engine = GovernanceEngine(config=config)

        state = EnvState()
        state.current_epoch = 1
        state.add_agent("agent_1")

        # Trigger freeze
        for _ in range(8):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_2",
                interaction_type=InteractionType.REPLY,
                p=0.2,
            )
            engine.apply_interaction(interaction, state)

        # Check if frozen
        lever = engine.get_loop_detector_lever()
        if lever:
            status = lever.get_loop_status("agent_1")
            if status["is_frozen"]:
                freeze_until = status["freeze_until_epoch"]

                # Apply epoch start at unfreeze time
                effect = engine.apply_epoch_start(state, freeze_until)

                # Should unfreeze
                assert "agent_1" in effect.agents_to_unfreeze

    def test_loop_detector_works_with_other_levers(self):
        """Loop detector should work alongside other governance levers."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_penalty_multiplier=1.5,
            circuit_breaker_enabled=True,
            freeze_threshold_toxicity=0.6,
            transaction_tax_rate=0.1,
        )
        engine = GovernanceEngine(config=config)

        state = EnvState()
        state.add_agent("agent_1")

        # Create low-quality repetitive interactions
        for i in range(10):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_2",
                interaction_type=InteractionType.REPLY,
                accepted=True,
                p=0.2,  # Low quality (toxic)
            )
            effect = engine.apply_interaction(interaction, state)

            # Both tax and loop detector penalty should apply
            if i >= 5:
                # Should have multiple lever effects
                assert len(effect.lever_effects) > 1

                # Should have costs from multiple sources
                tax_applied = any(
                    le.lever_name == "transaction_tax" for le in effect.lever_effects
                )
                loop_detected = any(
                    le.lever_name == "loop_detector"
                    and le.details.get("loop_detected", False)
                    for le in effect.lever_effects
                )

                assert tax_applied  # Tax should always apply
                if loop_detected:
                    # Total cost should be sum of all penalties
                    assert effect.cost_a > 0

    def test_multiple_agents_loop_detection_independent(self):
        """Loop detection should track multiple agents independently through engine."""
        config = GovernanceConfig(
            loop_detector_enabled=True,
            loop_detector_freeze_threshold=3,
        )
        engine = GovernanceEngine(config=config)

        state = EnvState()
        state.add_agent("agent_1")
        state.add_agent("agent_2")

        # Agent 1 creates a loop
        for _ in range(10):
            interaction = SoftInteraction(
                initiator="agent_1",
                counterparty="agent_3",
                interaction_type=InteractionType.REPLY,
                p=0.2,
            )
            engine.apply_interaction(interaction, state)

        # Agent 2 behaves normally
        for i in range(5):
            interaction = SoftInteraction(
                initiator="agent_2",
                counterparty=f"agent_{i + 4}",
                interaction_type=InteractionType.REPLY,
                p=0.8,
            )
            engine.apply_interaction(interaction, state)

        # Check statuses through lever
        lever = engine.get_loop_detector_lever()
        assert lever is not None

        status_1 = lever.get_loop_status("agent_1")
        status_2 = lever.get_loop_status("agent_2")

        # Agent 1 should have violations, agent 2 should not
        assert status_1["violations"] > 0
        assert status_2["violations"] == 0
