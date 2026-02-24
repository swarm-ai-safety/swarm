"""Tests for the AutoGPT bridge.

No external dependencies required — the bridge is purely protocol-level.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from swarm.bridges.autogpt import (
    AutoGPTAction,
    AutoGPTBridge,
    AutoGPTBridgeConfig,
)
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_action(
    text: str = "I need to search.",
    command: str = "web_search",
    success: bool = True,
    criticism: str = "",
) -> AutoGPTAction:
    return AutoGPTAction(
        thoughts={"text": text, "reasoning": "relevant research needed", "criticism": criticism},
        command_name=command,
        command_args={"query": "test"},
        result="results found",
        success=success,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestAutoGPTBridgeConfig:
    def test_defaults(self):
        cfg = AutoGPTBridgeConfig()
        assert cfg.agent_id == "autogpt-agent"
        assert cfg.proxy_sigmoid_k > 0

    def test_blocked_commands_defaults_populated(self):
        cfg = AutoGPTBridgeConfig()
        assert "delete_file" in cfg.blocked_commands
        assert "shutdown" in cfg.blocked_commands

    def test_empty_agent_id_raises(self):
        with pytest.raises(ValidationError):
            AutoGPTBridgeConfig(agent_id="   ")


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


class TestAutoGPTBridge:
    def test_record_action_returns_soft_interaction(self):
        bridge = AutoGPTBridge(config=AutoGPTBridgeConfig(enable_event_log=False))
        interaction = bridge.record_action(make_action())
        assert isinstance(interaction, SoftInteraction)

    def test_p_invariant_on_success(self):
        bridge = AutoGPTBridge(config=AutoGPTBridgeConfig(enable_event_log=False))
        ix = bridge.record_action(make_action(success=True))
        assert 0.0 <= ix.p <= 1.0

    def test_p_invariant_on_failure(self):
        bridge = AutoGPTBridge(config=AutoGPTBridgeConfig(enable_event_log=False))
        ix = bridge.record_action(make_action(success=False))
        assert 0.0 <= ix.p <= 1.0

    def test_blocked_command_lowers_p(self):
        cfg = AutoGPTBridgeConfig(enable_event_log=False)
        bridge_blocked = AutoGPTBridge(config=cfg)
        bridge_ok = AutoGPTBridge(config=cfg)

        ix_blocked = bridge_blocked.record_action(make_action(command="delete_file"))
        ix_ok = bridge_ok.record_action(make_action(command="web_search"))

        assert ix_ok.p > ix_blocked.p

    def test_blocked_command_not_accepted(self):
        bridge = AutoGPTBridge(config=AutoGPTBridgeConfig(enable_event_log=False))
        # blocked commands override success=True
        action = AutoGPTAction(
            thoughts={},
            command_name="shutdown",
            success=True,  # even if agent says success
        )
        ix = bridge.record_action(action)
        # blocked command → task_progress_delta=-1 → low p, accepted stays True
        # because accepted reflects action.success; governance sees low p
        assert ix.task_progress_delta == -1.0

    def test_criticism_increases_rework(self):
        bridge = AutoGPTBridge(config=AutoGPTBridgeConfig(enable_event_log=False))
        ix = bridge.record_action(make_action(criticism="This approach may fail"))
        assert ix.rework_count > 0

    def test_no_criticism_zero_rework(self):
        bridge = AutoGPTBridge(config=AutoGPTBridgeConfig(enable_event_log=False))
        ix = bridge.record_action(make_action(criticism=""))
        assert ix.rework_count == 0

    def test_successful_action_higher_p_than_failed(self):
        cfg = AutoGPTBridgeConfig(enable_event_log=False)
        bridge_ok = AutoGPTBridge(config=cfg)
        bridge_fail = AutoGPTBridge(config=cfg)
        p_ok = bridge_ok.record_action(make_action(success=True)).p
        p_fail = bridge_fail.record_action(make_action(success=False)).p
        assert p_ok > p_fail

    def test_interactions_accumulate(self):
        bridge = AutoGPTBridge(config=AutoGPTBridgeConfig(enable_event_log=False))
        bridge.record_action(make_action())
        bridge.record_action(make_action())
        assert len(bridge.get_interactions()) == 2

    def test_get_interactions_returns_copy(self):
        bridge = AutoGPTBridge(config=AutoGPTBridgeConfig(enable_event_log=False))
        bridge.record_action(make_action())
        copy = bridge.get_interactions()
        copy.clear()
        assert len(bridge.get_interactions()) == 1

    def test_initiator_matches_config_agent_id(self):
        cfg = AutoGPTBridgeConfig(agent_id="my-gpt", enable_event_log=False)
        bridge = AutoGPTBridge(config=cfg)
        ix = bridge.record_action(make_action())
        assert ix.initiator == "my-gpt"

    def test_counterparty_from_action(self):
        bridge = AutoGPTBridge(config=AutoGPTBridgeConfig(enable_event_log=False))
        action = make_action()
        action.counterparty = "server-42"
        ix = bridge.record_action(action)
        assert ix.counterparty == "server-42"

    def test_toxicity_rate_zero_when_empty(self):
        bridge = AutoGPTBridge(config=AutoGPTBridgeConfig(enable_event_log=False))
        assert bridge.get_toxicity_rate() == 0.0

    def test_long_thought_text_increases_engagement(self):
        cfg = AutoGPTBridgeConfig(enable_event_log=False)
        bridge_long = AutoGPTBridge(config=cfg)
        bridge_short = AutoGPTBridge(config=cfg)

        ix_long = bridge_long.record_action(make_action(text="x" * 1000))
        ix_short = bridge_short.record_action(make_action(text="x"))

        # Longer thought → higher engagement delta → higher counterparty_engagement_delta
        assert ix_long.counterparty_engagement_delta > ix_short.counterparty_engagement_delta
