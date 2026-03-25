"""Configuration for the SimWorld Delivery domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CityConfig:
    """Configuration for the delivery city layout."""

    width: float = 1000.0
    height: float = 1000.0
    num_depots: int = 3
    num_zones: int = 5

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("City dimensions must be positive")
        if self.num_depots < 1:
            raise ValueError("Must have at least 1 depot")


@dataclass
class OrderConfig:
    """Configuration for order generation."""

    orders_per_epoch: int = 20
    min_value: float = 5.0
    max_value: float = 50.0
    min_deadline_steps: int = 10
    max_deadline_steps: int = 30
    min_weight: float = 0.5
    max_weight: float = 5.0
    expiry_steps: int = 5

    def __post_init__(self) -> None:
        if self.min_value < 0 or self.max_value < self.min_value:
            raise ValueError("Order values must satisfy 0 <= min_value <= max_value")
        if self.min_deadline_steps < 1 or self.max_deadline_steps < self.min_deadline_steps:
            raise ValueError("Deadlines must satisfy 1 <= min <= max")


@dataclass
class EconomyConfig:
    """Configuration for economic mechanics."""

    starting_budget: float = 100.0
    scooter_cost: float = 50.0
    scooter_speed_multiplier: float = 2.0
    bid_floor: float = 1.0
    bid_ceiling: float = 100.0
    late_penalty_fraction: float = 0.3
    failed_delivery_penalty: float = 5.0
    sharing_bonus_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.starting_budget < 0:
            raise ValueError("Starting budget must be non-negative")
        if self.bid_floor < 0 or self.bid_ceiling < self.bid_floor:
            raise ValueError("Bid bounds must satisfy 0 <= floor <= ceiling")
        if not 0.0 <= self.late_penalty_fraction <= 1.0:
            raise ValueError("late_penalty_fraction must be in [0, 1]")
        if not 0.0 <= self.sharing_bonus_fraction <= 1.0:
            raise ValueError("sharing_bonus_fraction must be in [0, 1]")


@dataclass
class GovernanceConfig:
    """Configuration for governance interventions."""

    delivery_fee_rate: float = 0.05
    reputation_decay: float = 0.01
    reputation_bonus_on_time: float = 0.05
    reputation_penalty_late: float = 0.1
    reputation_penalty_fail: float = 0.2
    min_reputation_to_bid: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.delivery_fee_rate <= 1.0:
            raise ValueError("delivery_fee_rate must be in [0, 1]")
        if not 0.0 <= self.min_reputation_to_bid <= 1.0:
            raise ValueError("min_reputation_to_bid must be in [0, 1]")


@dataclass
class DeliveryConfig:
    """Top-level configuration for the SimWorld Delivery domain."""

    city: CityConfig = field(default_factory=CityConfig)
    orders: OrderConfig = field(default_factory=OrderConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeliveryConfig":
        """Parse a DeliveryConfig from a YAML-sourced dict."""
        if not data:
            return cls()

        city_data = data.get("city", {})
        city_cfg = CityConfig(**{
            k: city_data[k] for k in (
                "width", "height", "num_depots", "num_zones",
            ) if k in city_data
        })

        order_data = data.get("orders", {})
        order_cfg = OrderConfig(**{
            k: order_data[k] for k in (
                "orders_per_epoch", "min_value", "max_value",
                "min_deadline_steps", "max_deadline_steps",
                "min_weight", "max_weight", "expiry_steps",
            ) if k in order_data
        })

        economy_data = data.get("economy", {})
        economy_cfg = EconomyConfig(**{
            k: economy_data[k] for k in (
                "starting_budget", "scooter_cost",
                "scooter_speed_multiplier", "bid_floor", "bid_ceiling",
                "late_penalty_fraction", "failed_delivery_penalty",
                "sharing_bonus_fraction",
            ) if k in economy_data
        })

        governance_data = data.get("governance", {})
        governance_cfg = GovernanceConfig(**{
            k: governance_data[k] for k in (
                "delivery_fee_rate", "reputation_decay",
                "reputation_bonus_on_time", "reputation_penalty_late",
                "reputation_penalty_fail", "min_reputation_to_bid",
            ) if k in governance_data
        })

        return cls(
            city=city_cfg,
            orders=order_cfg,
            economy=economy_cfg,
            governance=governance_cfg,
            seed=data.get("seed"),
        )
