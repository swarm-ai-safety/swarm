"""Entity definitions for the Gather-Trade-Build domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ResourceType(Enum):
    """Types of resources in the GTB world."""

    WOOD = "wood"
    STONE = "stone"
    COIN = "coin"


class GTBActionType(Enum):
    """Actions a worker can take in the GTB world."""

    MOVE = "move"
    GATHER = "gather"
    TRADE_BUY = "trade_buy"
    TRADE_SELL = "trade_sell"
    BUILD = "build"
    NOOP = "noop"
    # Strategic / adversarial actions
    SHIFT_INCOME = "shift_income"
    MISREPORT = "misreport"


class Direction(Enum):
    """Movement directions on the grid."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Resource:
    """A resource tile on the grid."""

    resource_type: ResourceType
    amount: float
    position: Tuple[int, int] = (0, 0)
    regen_rate: float = 0.0  # Per-step regeneration


@dataclass
class House:
    """A built house that generates income."""

    owner_id: str
    position: Tuple[int, int] = (0, 0)
    wood_cost: float = 0.0
    stone_cost: float = 0.0
    income_per_step: float = 1.0
    build_step: int = 0


@dataclass
class MarketOrder:
    """An order in the centralized market."""

    agent_id: str
    resource_type: ResourceType
    quantity: float
    price_per_unit: float  # In coins
    is_buy: bool = True  # True=buy, False=sell
    step: int = 0


@dataclass
class TradeResult:
    """Result of a completed trade."""

    buyer_id: str
    seller_id: str
    resource_type: ResourceType
    quantity: float
    price_per_unit: float
    total_price: float
    step: int = 0


@dataclass
class WorkerState:
    """State of a single worker agent in the GTB world."""

    agent_id: str
    position: Tuple[int, int] = (0, 0)

    # Inventory
    inventory: Dict[str, float] = field(default_factory=lambda: {
        ResourceType.WOOD.value: 0.0,
        ResourceType.STONE.value: 0.0,
        ResourceType.COIN.value: 0.0,
    })

    # Income tracking
    gross_income_this_epoch: float = 0.0
    reported_income_this_epoch: float = 0.0
    cumulative_income: float = 0.0
    tax_paid_this_epoch: float = 0.0

    # Income shifting (deferred income account)
    deferred_income: float = 0.0

    # Houses built
    houses_built: int = 0

    # Energy budget
    energy: float = 100.0
    max_energy: float = 100.0

    # Skill multiplier (heterogeneous agents)
    skill_gather: float = 1.0
    skill_build: float = 1.0

    # Audit history
    times_audited: int = 0
    times_caught: int = 0
    total_fines: float = 0.0

    # Collusion
    coalition_id: Optional[str] = None

    def get_resource(self, rtype: ResourceType) -> float:
        """Get amount of a resource."""
        return self.inventory.get(rtype.value, 0.0)

    def add_resource(self, rtype: ResourceType, amount: float) -> None:
        """Add resource to inventory."""
        key = rtype.value
        self.inventory[key] = self.inventory.get(key, 0.0) + amount

    def remove_resource(self, rtype: ResourceType, amount: float) -> bool:
        """Remove resource from inventory. Returns False if insufficient."""
        key = rtype.value
        current = self.inventory.get(key, 0.0)
        if current < amount - 1e-9:
            return False
        self.inventory[key] = max(0.0, current - amount)
        return True

    def reset_epoch(self) -> None:
        """Reset per-epoch accumulators."""
        self.gross_income_this_epoch = 0.0
        self.reported_income_this_epoch = 0.0
        self.tax_paid_this_epoch = 0.0


@dataclass
class GTBGridCell:
    """A single cell in the GTB grid."""

    position: Tuple[int, int]
    resource: Optional[Resource] = None
    house: Optional[House] = None
    occupants: List[str] = field(default_factory=list)


@dataclass
class GTBEvent:
    """An event in the GTB simulation for logging."""

    event_type: str  # gather, trade, build, tax, audit, shift, misreport, collusion
    step: int = 0
    epoch: int = 0
    agent_id: str = ""
    details: Dict = field(default_factory=dict)
