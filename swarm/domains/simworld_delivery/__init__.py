"""SimWorld Delivery domain for distributional safety research.

A multi-agent delivery economy inspired by SimWorld's Case 2 benchmark
(NeurIPS 2025). Agents bid on delivery orders, invest in tools, and
compete/cooperate in an urban environment with governance interventions.
"""

from swarm.domains.simworld_delivery.agents import (
    AggressivePolicy,
    CautiousPolicy,
    ConscientiousPolicy,
    DeliveryPolicy,
    OpportunisticPolicy,
)
from swarm.domains.simworld_delivery.config import DeliveryConfig
from swarm.domains.simworld_delivery.entities import (
    AgentState,
    Bid,
    DeliveryAction,
    DeliveryActionType,
    DeliveryEvent,
    DeliveryOrder,
    OrderStatus,
    PersonaType,
)
from swarm.domains.simworld_delivery.env import DeliveryEnvironment
from swarm.domains.simworld_delivery.metrics import (
    DeliveryMetrics,
    compute_delivery_metrics,
)
from swarm.domains.simworld_delivery.runner import DeliveryScenarioRunner

__all__ = [
    "AgentState",
    "AggressivePolicy",
    "Bid",
    "CautiousPolicy",
    "ConscientiousPolicy",
    "DeliveryAction",
    "DeliveryActionType",
    "DeliveryConfig",
    "DeliveryEnvironment",
    "DeliveryEvent",
    "DeliveryMetrics",
    "DeliveryOrder",
    "DeliveryPolicy",
    "DeliveryScenarioRunner",
    "OpportunisticPolicy",
    "OrderStatus",
    "PersonaType",
    "compute_delivery_metrics",
]
