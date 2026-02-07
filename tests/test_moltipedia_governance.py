"""Tests for Moltipedia governance levers."""

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.moltipedia import (
    DailyPointCapLever,
    NoSelfFixLever,
    PageCooldownLever,
    PairCapLever,
)
from swarm.models.interaction import SoftInteraction


def _interaction(points: float, **metadata):
    return SoftInteraction(
        initiator="agent_a",
        counterparty="agent_b",
        accepted=True,
        metadata={"points": points, **metadata},
    )


def test_pair_cap_lever_blocks_after_max():
    config = GovernanceConfig(
        moltipedia_pair_cap_enabled=True,
        moltipedia_pair_cap_max=2,
    )
    lever = PairCapLever(config)
    state = EnvState()
    lever.on_epoch_start(state, 0)

    assert lever.on_interaction(_interaction(10.0), state).cost_a == 0.0
    assert lever.on_interaction(_interaction(10.0), state).cost_a == 0.0
    assert lever.on_interaction(_interaction(10.0), state).cost_a == 10.0


def test_page_cooldown_blocks_scoring():
    config = GovernanceConfig(
        moltipedia_page_cooldown_enabled=True,
        moltipedia_page_cooldown_steps=3,
    )
    lever = PageCooldownLever(config)
    state = EnvState()
    state.current_step = 1
    interaction = SoftInteraction(
        initiator="agent_a",
        counterparty="agent_b",
        accepted=True,
        metadata={"points": 8.0, "page_id": "page_1"},
    )
    assert lever.on_interaction(interaction, state).cost_a == 0.0
    state.current_step = 2
    assert lever.on_interaction(interaction, state).cost_a == 8.0


def test_daily_cap_limits_policy_fix_points():
    config = GovernanceConfig(
        moltipedia_daily_cap_enabled=True,
        moltipedia_daily_policy_fix_cap=10.0,
    )
    lever = DailyPointCapLever(config)
    state = EnvState()
    lever.on_epoch_start(state, 0)

    interaction = SoftInteraction(
        initiator="agent_a",
        counterparty="agent_b",
        accepted=True,
        metadata={"points": 8.0, "edit_type": "policy_fix"},
    )
    assert lever.on_interaction(interaction, state).cost_a == 0.0

    interaction2 = SoftInteraction(
        initiator="agent_a",
        counterparty="agent_b",
        accepted=True,
        metadata={"points": 8.0, "edit_type": "policy_fix"},
    )
    assert lever.on_interaction(interaction2, state).cost_a == 6.0


def test_no_self_fix_blocks_scoring():
    config = GovernanceConfig(moltipedia_no_self_fix=True)
    lever = NoSelfFixLever(config)
    state = EnvState()
    interaction = SoftInteraction(
        initiator="agent_a",
        counterparty="agent_b",
        accepted=True,
        metadata={"points": 15.0, "created_by": "agent_a"},
    )
    assert lever.on_interaction(interaction, state).cost_a == 15.0
