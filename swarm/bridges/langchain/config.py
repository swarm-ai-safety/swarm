"""Configuration for the LangChain bridge."""

from dataclasses import dataclass, field

from swarm.governance.config import GovernanceConfig


@dataclass
class LangChainBridgeConfig:
    """Configuration for the LangChain agent bridge.

    Attributes:
        governance_config: SWARM governance configuration.
        proxy_sigmoid_k: Calibration sharpness for the proxy sigmoid.
        reputation_weight: Weight applied to reputation in the payoff
            equation (maps to PayoffConfig.w_rep).  Increase to make
            reputation more influential; set to 0.0 to disable reputation
            effects entirely.
    """

    governance_config: GovernanceConfig = field(default_factory=GovernanceConfig)
    proxy_sigmoid_k: float = 2.0
    reputation_weight: float = 1.0
