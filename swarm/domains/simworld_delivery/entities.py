"""Entity definitions for the SimWorld Delivery domain.

Models a multi-agent delivery economy inspired by SimWorld's Case 2
benchmark (NeurIPS 2025). Agents bid on delivery orders, invest in
tools (scooters), and compete/cooperate in an urban environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class DeliveryActionType(Enum):
    """Actions a delivery agent can take."""

    BID = "bid"
    DELIVER = "deliver"
    BUY_SCOOTER = "buy_scooter"
    SHARE_ORDER = "share_order"
    WAIT = "wait"


class PersonaType(Enum):
    """Agent personality types from SimWorld's Big Five mapping."""

    CONSCIENTIOUS = "conscientious"
    OPEN = "open"
    AGGRESSIVE = "aggressive"
    CAUTIOUS = "cautious"
    OPPORTUNISTIC = "opportunistic"


class OrderStatus(Enum):
    """Status of a delivery order."""

    AVAILABLE = "available"
    ASSIGNED = "assigned"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class DeliveryOrder:
    """A delivery order in the economy."""

    order_id: str
    origin: Tuple[float, float] = (0.0, 0.0)
    destination: Tuple[float, float] = (0.0, 0.0)
    value: float = 10.0
    deadline_steps: int = 20
    weight: float = 1.0
    status: OrderStatus = OrderStatus.AVAILABLE
    assigned_agent: Optional[str] = None
    created_step: int = 0
    delivered_step: Optional[int] = None


@dataclass
class AgentState:
    """State of a delivery agent."""

    agent_id: str
    persona: PersonaType = PersonaType.CONSCIENTIOUS
    position: Tuple[float, float] = (0.0, 0.0)
    budget: float = 100.0
    speed: float = 1.0
    has_scooter: bool = False
    scooter_cost: float = 50.0

    # Current delivery
    current_order: Optional[str] = None
    carrying: bool = False

    # Performance tracking
    deliveries_completed: int = 0
    deliveries_failed: int = 0
    total_earnings: float = 0.0
    total_bids: int = 0
    overbids: int = 0
    orders_shared: int = 0
    idle_steps: int = 0
    total_distance: float = 0.0

    # Reputation
    reputation: float = 1.0
    customer_complaints: int = 0
    on_time_deliveries: int = 0
    late_deliveries: int = 0


@dataclass
class Bid:
    """A bid on a delivery order."""

    agent_id: str
    order_id: str
    amount: float
    step: int = 0


@dataclass
class DeliveryAction:
    """An agent's action for one step."""

    agent_id: str = ""
    action_type: DeliveryActionType = DeliveryActionType.WAIT
    order_id: str = ""
    bid_amount: float = 0.0
    share_with: str = ""


@dataclass
class DeliveryEvent:
    """An event in the delivery simulation for logging."""

    event_type: str
    step: int = 0
    epoch: int = 0
    agent_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
