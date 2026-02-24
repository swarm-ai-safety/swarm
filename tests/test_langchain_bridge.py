"""Tests for the LangChain bridge — verifies reputation_weight wiring."""

from __future__ import annotations

from swarm.bridges.langchain.bridge import LangChainBridge
from swarm.bridges.langchain.config import LangChainBridgeConfig
from swarm.core.payoff import PayoffConfig


class TestLangChainBridgeConfig:
    """Test LangChainBridgeConfig defaults and field values."""

    def test_defaults(self) -> None:
        config = LangChainBridgeConfig()
        assert config.proxy_sigmoid_k == 2.0
        assert config.reputation_weight == 1.0

    def test_custom_reputation_weight(self) -> None:
        config = LangChainBridgeConfig(reputation_weight=3.5)
        assert config.reputation_weight == 3.5

    def test_zero_reputation_weight(self) -> None:
        config = LangChainBridgeConfig(reputation_weight=0.0)
        assert config.reputation_weight == 0.0


class TestLangChainBridgeWiring:
    """Test that reputation_weight is correctly wired into PayoffConfig.w_rep."""

    def test_default_wiring(self) -> None:
        bridge = LangChainBridge()
        payoff_config = bridge.get_payoff_config()
        assert isinstance(payoff_config, PayoffConfig)
        assert payoff_config.w_rep == 1.0

    def test_reputation_weight_wired_to_w_rep(self) -> None:
        config = LangChainBridgeConfig(reputation_weight=2.5)
        bridge = LangChainBridge(config)
        payoff_config = bridge.get_payoff_config()
        assert payoff_config.w_rep == 2.5

    def test_zero_reputation_weight_wired(self) -> None:
        config = LangChainBridgeConfig(reputation_weight=0.0)
        bridge = LangChainBridge(config)
        payoff_config = bridge.get_payoff_config()
        assert payoff_config.w_rep == 0.0

    def test_get_payoff_config_returns_payoff_config_instance(self) -> None:
        bridge = LangChainBridge(LangChainBridgeConfig(reputation_weight=1.5))
        result = bridge.get_payoff_config()
        assert isinstance(result, PayoffConfig)

    def test_no_config_uses_defaults(self) -> None:
        bridge = LangChainBridge(config=None)
        payoff_config = bridge.get_payoff_config()
        assert payoff_config.w_rep == LangChainBridgeConfig().reputation_weight
