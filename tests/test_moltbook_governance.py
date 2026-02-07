"""Tests for Moltbook governance levers."""

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.moltbook import ChallengeVerificationLever, MoltbookRateLimitLever
from swarm.models.interaction import SoftInteraction


def test_rate_limit_blocks_post_cooldown():
    config = GovernanceConfig(
        moltbook_rate_limit_enabled=True,
        moltbook_post_cooldown_steps=3,
    )
    lever = MoltbookRateLimitLever(config)
    assert lever.can_perform("agent", "post", 0)
    lever.record_action("agent", "post", 0)
    assert not lever.can_perform("agent", "post", 1)
    assert lever.can_perform("agent", "post", 3)


def test_daily_comment_cap_resets_on_epoch():
    config = GovernanceConfig(
        moltbook_rate_limit_enabled=True,
        moltbook_daily_comment_cap=2,
    )
    lever = MoltbookRateLimitLever(config)
    lever.record_action("agent", "comment", 0)
    lever.record_action("agent", "comment", 1)
    assert not lever.can_perform("agent", "comment", 2)

    lever.on_epoch_start(EnvState(), 1)
    assert lever.can_perform("agent", "comment", 0)


def test_challenge_lever_expires_pending():
    lever = ChallengeVerificationLever(GovernanceConfig())
    lever.register("post_1", "agent", expires_at_step=1)
    assert "post_1" in lever.get_pending_for_agent("agent")
    lever.on_step(EnvState(), 2)
    assert "post_1" not in lever.get_pending_for_agent("agent")


def test_challenge_lever_blocks_unverified_interaction():
    lever = ChallengeVerificationLever(GovernanceConfig())
    interaction = SoftInteraction(
        initiator="agent_a",
        counterparty="agent_b",
        accepted=True,
        metadata={"moltbook_status": "pending_verification", "post_id": "post_x"},
    )
    effect = lever.on_interaction(interaction, EnvState())
    assert effect.details.get("blocked") is True
