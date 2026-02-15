"""Configuration for the Gather-Trade-Build domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaxBracket:
    """A single bracket in a piecewise tax schedule."""

    threshold: float = 0.0  # income above which this rate applies
    rate: float = 0.0  # marginal rate in this bracket


@dataclass
class TaxScheduleConfig:
    """Configuration for the tax schedule."""

    schedule_family: str = "piecewise"  # piecewise | flat | saez
    brackets: List[TaxBracket] = field(default_factory=lambda: [
        TaxBracket(threshold=0.0, rate=0.1),
        TaxBracket(threshold=10.0, rate=0.2),
        TaxBracket(threshold=25.0, rate=0.35),
    ])
    smoothing: float = 0.0  # sigmoid smoothing at bracket edges
    damping: float = 0.0  # rate-of-change damping for planner updates
    update_interval_epochs: int = 1  # how often planner updates
    allow_non_monotone: bool = False  # permit U-shaped schedules


@dataclass
class GamingConfig:
    """Configuration for strategic gaming mechanics."""

    income_shifting_enabled: bool = True
    max_shift_fraction_per_epoch: float = 0.3
    gaming_cost_fraction: float = 0.05  # friction on shifted amount
    bunching_detection_enabled: bool = True
    bunching_bin_width: float = 1.0  # histogram bin width near brackets


@dataclass
class MisreportingConfig:
    """Configuration for misreporting / evasion mechanics."""

    enabled: bool = True
    max_underreport_fraction: float = 0.5
    audit_probability: float = 0.2
    risk_based_audit_multiplier: float = 1.5
    fine_multiplier: float = 2.0  # fine = multiplier * evaded_tax
    reputation_penalty_per_catch: float = 0.1
    freeze_on_repeat: bool = True
    freeze_after_n_catches: int = 3
    freeze_duration_epochs: int = 2


@dataclass
class CollusionConfig:
    """Configuration for collusion mechanics."""

    enabled: bool = True
    max_coalition_size: int = 4
    min_coalition_size: int = 2
    detection_window_steps: int = 20
    similarity_threshold: float = 0.7
    suspicion_score_threshold: float = 0.6
    response_audit_multiplier: float = 2.0
    response_stake_increase: float = 1.5
    response_trade_restriction_epochs: int = 1


@dataclass
class MapConfig:
    """Configuration for the gridworld map."""

    height: int = 15
    width: int = 15
    wood_density: float = 0.2
    stone_density: float = 0.15
    resource_regen_rate: float = 0.1
    resource_max_amount: float = 5.0


@dataclass
class MarketConfig:
    """Configuration for the centralized market."""

    enabled: bool = True
    transaction_fee_rate: float = 0.02
    price_floor: float = 0.1
    price_ceiling: float = 100.0


@dataclass
class BuildConfig:
    """Configuration for house building."""

    wood_cost: float = 3.0
    stone_cost: float = 3.0
    income_per_house_per_step: float = 1.0
    max_houses_per_agent: int = 10


@dataclass
class PlannerConfig:
    """Configuration for the planner agent."""

    planner_type: str = "heuristic"  # heuristic | bandit | rl
    objective: str = "welfare"
    prod_weight: float = 1.0
    ineq_weight: float = 0.5
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    update_interval_epochs: int = 1


@dataclass
class GTBConfig:
    """Top-level configuration for the Gather-Trade-Build domain."""

    map: MapConfig = field(default_factory=MapConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    taxation: TaxScheduleConfig = field(default_factory=TaxScheduleConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    gaming: GamingConfig = field(default_factory=GamingConfig)
    misreporting: MisreportingConfig = field(default_factory=MisreportingConfig)
    collusion: CollusionConfig = field(default_factory=CollusionConfig)
    energy_per_step: float = 10.0
    energy_cost_move: float = 1.0
    energy_cost_gather: float = 1.0
    energy_cost_trade: float = 0.5
    energy_cost_build: float = 2.0
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GTBConfig":
        """Parse a GTBConfig from a YAML-sourced dict."""
        if not data:
            return cls()

        map_data = data.get("map", {})
        map_cfg = MapConfig(**{
            k: map_data[k] for k in (
                "height", "width", "wood_density", "stone_density",
                "resource_regen_rate", "resource_max_amount",
            ) if k in map_data
        })

        market_data = data.get("market", {})
        market_cfg = MarketConfig(**{
            k: market_data[k] for k in (
                "enabled", "transaction_fee_rate", "price_floor", "price_ceiling",
            ) if k in market_data
        })

        build_data = data.get("build", {})
        build_cfg = BuildConfig(**{
            k: build_data[k] for k in (
                "wood_cost", "stone_cost", "income_per_house_per_step",
                "max_houses_per_agent",
            ) if k in build_data
        })

        tax_data = data.get("taxation", {})
        brackets = []
        for b in tax_data.get("brackets", []):
            brackets.append(TaxBracket(
                threshold=b.get("threshold", 0.0),
                rate=b.get("rate", 0.0),
            ))
        tax_kwargs: Dict[str, Any] = {}
        if brackets:
            tax_kwargs["brackets"] = brackets
        for k in ("schedule_family", "smoothing", "damping",
                   "update_interval_epochs", "allow_non_monotone"):
            if k in tax_data:
                tax_kwargs[k] = tax_data[k]
        tax_cfg = TaxScheduleConfig(**tax_kwargs)

        planner_data = data.get("planner", {})
        planner_cfg = PlannerConfig(**{
            k: planner_data[k] for k in (
                "planner_type", "objective", "prod_weight", "ineq_weight",
                "learning_rate", "exploration_rate", "update_interval_epochs",
            ) if k in planner_data
        })

        gaming_data = data.get("gaming", {})
        gaming_cfg = GamingConfig(**{
            k: gaming_data[k] for k in (
                "income_shifting_enabled", "max_shift_fraction_per_epoch",
                "gaming_cost_fraction", "bunching_detection_enabled",
                "bunching_bin_width",
            ) if k in gaming_data
        })

        misreport_data = data.get("misreporting", {})
        misreport_cfg = MisreportingConfig(**{
            k: misreport_data[k] for k in (
                "enabled", "max_underreport_fraction", "audit_probability",
                "risk_based_audit_multiplier", "fine_multiplier",
                "reputation_penalty_per_catch", "freeze_on_repeat",
                "freeze_after_n_catches", "freeze_duration_epochs",
            ) if k in misreport_data
        })

        collusion_data = data.get("collusion", {})
        collusion_cfg = CollusionConfig(**{
            k: collusion_data[k] for k in (
                "enabled", "max_coalition_size", "min_coalition_size",
                "detection_window_steps", "similarity_threshold",
                "suspicion_score_threshold", "response_audit_multiplier",
                "response_stake_increase", "response_trade_restriction_epochs",
            ) if k in collusion_data
        })

        return cls(
            map=map_cfg,
            market=market_cfg,
            build=build_cfg,
            taxation=tax_cfg,
            planner=planner_cfg,
            gaming=gaming_cfg,
            misreporting=misreport_cfg,
            collusion=collusion_cfg,
            energy_per_step=data.get("energy_per_step", 10.0),
            energy_cost_move=data.get("energy_cost_move", 1.0),
            energy_cost_gather=data.get("energy_cost_gather", 1.0),
            energy_cost_trade=data.get("energy_cost_trade", 0.5),
            energy_cost_build=data.get("energy_cost_build", 2.0),
            seed=data.get("seed"),
        )
