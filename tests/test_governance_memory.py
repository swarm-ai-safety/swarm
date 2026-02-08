"""Tests for swarm.governance.memory levers."""

import pytest

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.memory import (
    CrossVerificationLever,
    PromotionGateLever,
    ProvenanceLever,
    WriteRateLimitLever,
)
from swarm.models.interaction import SoftInteraction


def _make_state() -> EnvState:
    state = EnvState()
    state.add_agent("agent_a")
    state.add_agent("agent_b")
    return state


class TestPromotionGateLever:
    def test_disabled(self):
        config = GovernanceConfig(memory_promotion_gate_enabled=False)
        lever = PromotionGateLever(config)
        interaction = SoftInteraction(
            metadata={"memory_promotion": True, "quality_score": 0.1}
        )
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 0.0

    def test_non_promotion_interaction(self):
        config = GovernanceConfig(memory_promotion_gate_enabled=True)
        lever = PromotionGateLever(config)
        interaction = SoftInteraction(metadata={})
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 0.0

    def test_blocks_low_quality(self):
        config = GovernanceConfig(
            memory_promotion_gate_enabled=True,
            memory_promotion_min_quality=0.7,
            memory_promotion_min_verifications=0,
        )
        lever = PromotionGateLever(config)
        interaction = SoftInteraction(
            metadata={
                "memory_promotion": True,
                "quality_score": 0.3,
                "verified_by": [],
            }
        )
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 1.0
        assert effect.details["blocked"] is True

    def test_blocks_insufficient_verifications(self):
        config = GovernanceConfig(
            memory_promotion_gate_enabled=True,
            memory_promotion_min_quality=0.0,
            memory_promotion_min_verifications=3,
        )
        lever = PromotionGateLever(config)
        interaction = SoftInteraction(
            metadata={
                "memory_promotion": True,
                "quality_score": 0.9,
                "verified_by": ["v1"],
            }
        )
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 1.0
        assert "verifications" in str(effect.details["reasons"])

    def test_allows_valid_promotion(self):
        config = GovernanceConfig(
            memory_promotion_gate_enabled=True,
            memory_promotion_min_quality=0.5,
            memory_promotion_min_verifications=1,
        )
        lever = PromotionGateLever(config)
        interaction = SoftInteraction(
            metadata={
                "memory_promotion": True,
                "quality_score": 0.8,
                "verified_by": ["v1", "v2"],
            }
        )
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 0.0

    def test_name(self):
        config = GovernanceConfig()
        lever = PromotionGateLever(config)
        assert lever.name == "memory_promotion_gate"


class TestWriteRateLimitLever:
    def test_disabled(self):
        config = GovernanceConfig(memory_write_rate_limit_enabled=False)
        lever = WriteRateLimitLever(config)
        interaction = SoftInteraction(
            initiator="agent_a",
            metadata={"memory_write": True},
        )
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 0.0

    def test_non_write_interaction(self):
        config = GovernanceConfig(memory_write_rate_limit_enabled=True)
        lever = WriteRateLimitLever(config)
        interaction = SoftInteraction(initiator="agent_a", metadata={})
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 0.0

    def test_within_limit(self):
        config = GovernanceConfig(
            memory_write_rate_limit_enabled=True,
            memory_write_rate_limit_per_epoch=5,
        )
        lever = WriteRateLimitLever(config)
        state = _make_state()
        interaction = SoftInteraction(
            initiator="agent_a",
            metadata={"memory_write": True},
        )
        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.0

    def test_exceeds_limit(self):
        config = GovernanceConfig(
            memory_write_rate_limit_enabled=True,
            memory_write_rate_limit_per_epoch=2,
        )
        lever = WriteRateLimitLever(config)
        state = _make_state()
        interaction = SoftInteraction(
            initiator="agent_a",
            metadata={"memory_write": True},
        )
        # Write 1 and 2 are OK
        lever.on_interaction(interaction, state)
        lever.on_interaction(interaction, state)
        # Write 3 exceeds limit
        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 1.0
        assert effect.details["writes"] == 3
        assert effect.details["cap"] == 2

    def test_epoch_reset(self):
        config = GovernanceConfig(
            memory_write_rate_limit_enabled=True,
            memory_write_rate_limit_per_epoch=1,
        )
        lever = WriteRateLimitLever(config)
        state = _make_state()
        interaction = SoftInteraction(
            initiator="agent_a",
            metadata={"memory_write": True},
        )
        lever.on_interaction(interaction, state)
        # Exceeds limit
        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 1.0
        # Reset on new epoch
        lever.on_epoch_start(state, epoch=1)
        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.0

    def test_get_remaining_writes(self):
        config = GovernanceConfig(
            memory_write_rate_limit_enabled=True,
            memory_write_rate_limit_per_epoch=5,
        )
        lever = WriteRateLimitLever(config)
        assert lever.get_remaining_writes("agent_a") == 5
        state = _make_state()
        interaction = SoftInteraction(
            initiator="agent_a",
            metadata={"memory_write": True},
        )
        lever.on_interaction(interaction, state)
        assert lever.get_remaining_writes("agent_a") == 4

    def test_name(self):
        config = GovernanceConfig()
        lever = WriteRateLimitLever(config)
        assert lever.name == "memory_write_rate_limit"


class TestCrossVerificationLever:
    def test_disabled(self):
        config = GovernanceConfig(memory_cross_verification_enabled=False)
        lever = CrossVerificationLever(config)
        interaction = SoftInteraction(
            initiator="verifier",
            metadata={"memory_verification": True, "entry_author": "author"},
        )
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 0.0

    def test_non_verification_interaction(self):
        config = GovernanceConfig(memory_cross_verification_enabled=True)
        lever = CrossVerificationLever(config)
        interaction = SoftInteraction(initiator="verifier", metadata={})
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 0.0

    def test_no_author(self):
        config = GovernanceConfig(memory_cross_verification_enabled=True)
        lever = CrossVerificationLever(config)
        interaction = SoftInteraction(
            initiator="verifier",
            metadata={"memory_verification": True},
        )
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 0.0

    def test_normal_verification(self):
        config = GovernanceConfig(memory_cross_verification_enabled=True)
        lever = CrossVerificationLever(config)
        state = _make_state()
        interaction = SoftInteraction(
            initiator="verifier",
            metadata={"memory_verification": True, "entry_author": "author"},
        )
        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.0

    def test_collusion_detected(self):
        config = GovernanceConfig(memory_cross_verification_enabled=True)
        lever = CrossVerificationLever(config)
        state = _make_state()
        interaction = SoftInteraction(
            initiator="verifier",
            metadata={"memory_verification": True, "entry_author": "author"},
        )
        # First 3 are OK
        for _ in range(3):
            lever.on_interaction(interaction, state)
        # 4th triggers collusion
        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.5
        assert "verifier" in effect.reputation_deltas
        assert effect.reputation_deltas["verifier"] == pytest.approx(-0.05)

    def test_epoch_reset_clears_pairs(self):
        config = GovernanceConfig(memory_cross_verification_enabled=True)
        lever = CrossVerificationLever(config)
        state = _make_state()
        interaction = SoftInteraction(
            initiator="verifier",
            metadata={"memory_verification": True, "entry_author": "author"},
        )
        for _ in range(3):
            lever.on_interaction(interaction, state)
        lever.on_epoch_start(state, epoch=1)
        # After reset, should be OK again
        effect = lever.on_interaction(interaction, state)
        assert effect.cost_a == 0.0

    def test_name(self):
        config = GovernanceConfig()
        lever = CrossVerificationLever(config)
        assert lever.name == "memory_cross_verification"


class TestProvenanceLever:
    def test_disabled(self):
        config = GovernanceConfig(memory_provenance_enabled=False)
        lever = ProvenanceLever(config)
        interaction = SoftInteraction(
            metadata={"memory_revert": True, "entry_author": "author"},
        )
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 0.0

    def test_non_revert_interaction(self):
        config = GovernanceConfig(memory_provenance_enabled=True)
        lever = ProvenanceLever(config)
        interaction = SoftInteraction(metadata={})
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 0.0

    def test_no_author(self):
        config = GovernanceConfig(memory_provenance_enabled=True)
        lever = ProvenanceLever(config)
        interaction = SoftInteraction(
            metadata={"memory_revert": True},
        )
        effect = lever.on_interaction(interaction, _make_state())
        assert effect.cost_a == 0.0

    def test_revert_penalty(self):
        config = GovernanceConfig(
            memory_provenance_enabled=True,
            memory_provenance_revert_penalty=0.2,
        )
        lever = ProvenanceLever(config)
        interaction = SoftInteraction(
            metadata={
                "memory_revert": True,
                "entry_author": "bad_author",
                "entry_id": "entry123",
            },
        )
        effect = lever.on_interaction(interaction, _make_state())
        assert "bad_author" in effect.reputation_deltas
        assert effect.reputation_deltas["bad_author"] == pytest.approx(-0.2)
        assert effect.details["reverted_author"] == "bad_author"
        assert effect.details["entry_id"] == "entry123"

    def test_name(self):
        config = GovernanceConfig()
        lever = ProvenanceLever(config)
        assert lever.name == "memory_provenance"
