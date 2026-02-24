"""Main bridge connecting LangChain agents to SWARM.

LangChainBridge is the central adapter that:
1. Accepts a LangChainBridgeConfig (including reputation_weight)
2. Wires reputation_weight into PayoffConfig.w_rep
3. Exposes a PayoffConfig ready for use with SoftPayoffEngine
"""

from __future__ import annotations

import logging
from typing import Optional

from swarm.bridges.langchain.config import LangChainBridgeConfig
from swarm.core.payoff import PayoffConfig
from swarm.core.proxy import ProxyComputer

logger = logging.getLogger(__name__)


class LangChainBridge:
    """Bridge between LangChain agents and the SWARM payoff framework.

    Wires ``LangChainBridgeConfig.reputation_weight`` into
    ``PayoffConfig.w_rep`` so that reputation effects are consistently
    applied across all interactions scored by this bridge.

    Example::

        config = LangChainBridgeConfig(reputation_weight=2.0)
        bridge = LangChainBridge(config)
        payoff_config = bridge.get_payoff_config()
        # payoff_config.w_rep == 2.0
    """

    def __init__(self, config: Optional[LangChainBridgeConfig] = None) -> None:
        self._config = config or LangChainBridgeConfig()
        self._proxy = ProxyComputer(sigmoid_k=self._config.proxy_sigmoid_k)

    def get_payoff_config(self) -> PayoffConfig:
        """Return a PayoffConfig with w_rep wired from reputation_weight."""
        return PayoffConfig(w_rep=self._config.reputation_weight)
