"""SWARM-LangChain Bridge — Governance-Aware LangChain Agent Integration.

Bridges LangChain-based agent chains/runnables with the SWARM safety
framework, wiring LangChainBridgeConfig parameters (including
reputation_weight) into the payoff engine.

Usage::

    from swarm.bridges.langchain import LangChainBridgeConfig, LangChainBridge

    config = LangChainBridgeConfig(reputation_weight=2.0)
    bridge = LangChainBridge(config)
    payoff_config = bridge.get_payoff_config()
    assert payoff_config.w_rep == 2.0

Requires the ``langchain`` optional dependency group::

    pip install swarm-safety[langchain]
"""

from swarm.bridges.langchain.bridge import LangChainBridge
from swarm.bridges.langchain.config import LangChainBridgeConfig

__all__ = [
    "LangChainBridgeConfig",
    "LangChainBridge",
]
