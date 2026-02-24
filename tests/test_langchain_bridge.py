"""Tests for the LangChain bridge.

All tests use mock chains so that langchain itself is not required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from swarm.bridges.langchain import (
    LangChainBridge,
    LangChainBridgeConfig,
)
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chain(output: str = "hello", intermediate_steps: int = 0):
    """Build a mock chain with .invoke() that returns a dict."""
    chain = MagicMock()
    chain.invoke.return_value = {
        "output": output,
        "intermediate_steps": [None] * intermediate_steps,
    }
    return chain


def make_chain_run(output: str = "hello"):
    """Build a mock chain with .run() (legacy API)."""
    chain = MagicMock(spec=["run"])
    chain.run.return_value = output
    return chain


def make_failing_chain():
    """Build a mock chain whose .invoke() raises."""
    chain = MagicMock()
    chain.invoke.side_effect = RuntimeError("LLM refused")
    return chain


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestLangChainBridgeConfig:
    def test_default_config(self):
        cfg = LangChainBridgeConfig()
        assert cfg.agent_id == "langchain-agent"
        assert cfg.max_steps >= 1
        assert cfg.proxy_sigmoid_k > 0

    def test_invalid_max_steps(self):
        with pytest.raises(ValidationError):
            LangChainBridgeConfig(max_steps=0)

    def test_invalid_timeout(self):
        with pytest.raises(ValidationError):
            LangChainBridgeConfig(timeout_seconds=0)


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


class TestLangChainBridge:
    def test_successful_run_returns_soft_interaction(self):
        bridge = LangChainBridge(
            chain=make_chain("result"),
            config=LangChainBridgeConfig(enable_event_log=False),
        )
        interaction = bridge.run("do something")
        assert isinstance(interaction, SoftInteraction)
        assert interaction.accepted is True

    def test_p_invariant_holds_on_success(self):
        bridge = LangChainBridge(
            chain=make_chain("result"),
            config=LangChainBridgeConfig(enable_event_log=False),
        )
        interaction = bridge.run("task")
        assert 0.0 <= interaction.p <= 1.0

    def test_p_invariant_holds_on_failure(self):
        bridge = LangChainBridge(
            chain=make_failing_chain(),
            config=LangChainBridgeConfig(enable_event_log=False),
        )
        interaction = bridge.run("task")
        assert 0.0 <= interaction.p <= 1.0
        assert interaction.accepted is False

    def test_failed_chain_produces_interaction_not_exception(self):
        bridge = LangChainBridge(
            chain=make_failing_chain(),
            config=LangChainBridgeConfig(enable_event_log=False),
        )
        interaction = bridge.run("task")
        assert interaction is not None
        assert interaction.accepted is False

    def test_success_gives_higher_p_than_failure(self):
        cfg = LangChainBridgeConfig(enable_event_log=False)
        bridge_ok = LangChainBridge(chain=make_chain("long output " * 50), config=cfg)
        bridge_fail = LangChainBridge(chain=make_failing_chain(), config=cfg)
        p_ok = bridge_ok.run("task").p
        p_fail = bridge_fail.run("task").p
        assert p_ok > p_fail

    def test_legacy_run_method_is_used(self):
        bridge = LangChainBridge(
            chain=make_chain_run("output"),
            config=LangChainBridgeConfig(enable_event_log=False),
        )
        interaction = bridge.run("task")
        assert isinstance(interaction, SoftInteraction)
        assert interaction.accepted is True

    def test_chain_with_no_invoke_or_run_returns_failed_interaction(self):
        chain = MagicMock(spec=[])  # no invoke, no run
        bridge = LangChainBridge(
            chain=chain,
            config=LangChainBridgeConfig(enable_event_log=False),
        )
        # Bridge handles the error gracefully; returns interaction with accepted=False
        interaction = bridge.run("task")
        assert isinstance(interaction, SoftInteraction)
        assert interaction.accepted is False

    def test_interactions_accumulate(self):
        bridge = LangChainBridge(
            chain=make_chain("x"),
            config=LangChainBridgeConfig(enable_event_log=False),
        )
        bridge.run("a")
        bridge.run("b")
        assert len(bridge.get_interactions()) == 2

    def test_get_interactions_returns_copy(self):
        bridge = LangChainBridge(
            chain=make_chain("x"),
            config=LangChainBridgeConfig(enable_event_log=False),
        )
        bridge.run("a")
        copy = bridge.get_interactions()
        copy.clear()
        assert len(bridge.get_interactions()) == 1

    def test_counterparty_recorded(self):
        bridge = LangChainBridge(
            chain=make_chain("x"),
            config=LangChainBridgeConfig(enable_event_log=False),
        )
        interaction = bridge.run("task", counterparty="user-42")
        assert interaction.counterparty == "user-42"

    def test_initiator_is_agent_id(self):
        cfg = LangChainBridgeConfig(agent_id="my-chain", enable_event_log=False)
        bridge = LangChainBridge(chain=make_chain("x"), config=cfg)
        interaction = bridge.run("task")
        assert interaction.initiator == "my-chain"

    def test_quality_score_below_half_penalises_interaction(self):
        bridge = LangChainBridge(
            chain=make_chain("good output"),
            config=LangChainBridgeConfig(enable_event_log=False),
        )
        ix_low = bridge.run("task", quality_score=0.1)
        bridge2 = LangChainBridge(
            chain=make_chain("good output"),
            config=LangChainBridgeConfig(enable_event_log=False),
        )
        ix_high = bridge2.run("task", quality_score=0.9)
        assert ix_high.p >= ix_low.p

    def test_toxicity_rate_zero_when_no_interactions(self):
        bridge = LangChainBridge(
            chain=make_chain("x"),
            config=LangChainBridgeConfig(enable_event_log=False),
        )
        assert bridge.get_toxicity_rate() == 0.0

    def test_max_steps_exceeded_treats_as_failure(self):
        cfg = LangChainBridgeConfig(max_steps=2, enable_event_log=False)
        chain = make_chain("result", intermediate_steps=10)
        bridge = LangChainBridge(chain=chain, config=cfg)
        interaction = bridge.run("task")
        assert interaction.accepted is False
