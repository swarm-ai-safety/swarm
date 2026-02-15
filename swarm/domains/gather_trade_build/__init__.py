"""Gather-Trade-Build (GTB) domain for AI Economist scenarios.

An AI Economist-style gridworld economy where workers gather resources,
trade in markets, and build houses, while a Planner agent sets tax policy.
"""

from swarm.domains.gather_trade_build.agents import (
    CollusiveWorkerPolicy,
    EvasiveWorkerPolicy,
    GamingWorkerPolicy,
    GTBWorkerPolicy,
    HonestWorkerPolicy,
)
from swarm.domains.gather_trade_build.config import GTBConfig
from swarm.domains.gather_trade_build.entities import (
    GTBEvent,
    GTBGridCell,
    House,
    MarketOrder,
    Resource,
    ResourceType,
    TradeResult,
    WorkerState,
)
from swarm.domains.gather_trade_build.env import GTBAction, GTBEnvironment
from swarm.domains.gather_trade_build.metrics import GTBMetrics, compute_gtb_metrics
from swarm.domains.gather_trade_build.planner import PlannerAgent
from swarm.domains.gather_trade_build.reward import compute_worker_utility
from swarm.domains.gather_trade_build.runner import GTBScenarioRunner
from swarm.domains.gather_trade_build.tax_schedule import TaxSchedule

__all__ = [
    "CollusiveWorkerPolicy",
    "EvasiveWorkerPolicy",
    "GamingWorkerPolicy",
    "GTBAction",
    "GTBConfig",
    "GTBEnvironment",
    "GTBEvent",
    "GTBGridCell",
    "GTBMetrics",
    "GTBScenarioRunner",
    "GTBWorkerPolicy",
    "HonestWorkerPolicy",
    "House",
    "MarketOrder",
    "PlannerAgent",
    "Resource",
    "ResourceType",
    "TaxSchedule",
    "TradeResult",
    "WorkerState",
    "compute_gtb_metrics",
    "compute_worker_utility",
]
