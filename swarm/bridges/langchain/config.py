"""Configuration for the LangChain bridge."""

from dataclasses import dataclass


@dataclass
class LangChainBridgeConfig:
    """Configuration for the LangChain agent bridge.

    Attributes:
        reputation_weight: Weight applied to reputation in the payoff
            equation (maps to PayoffConfig.w_rep).  Increase to make
            reputation more influential; set to 0.0 to disable reputation
            effects entirely.
    """

    reputation_weight: float = 1.0
