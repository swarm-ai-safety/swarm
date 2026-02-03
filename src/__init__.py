"""Distributional AGI Safety Sandbox - Soft Label Payoff & Metrics System."""

from src.core.payoff import PayoffConfig, SoftPayoffEngine
from src.core.proxy import ProxyComputer, ProxyObservables, ProxyWeights
from src.metrics.reporters import MetricsReporter
from src.metrics.soft_metrics import SoftMetrics
from src.models.agent import AgentState, AgentType
from src.models.interaction import InteractionType, SoftInteraction

__all__ = [
    "SoftInteraction",
    "InteractionType",
    "AgentType",
    "AgentState",
    "SoftPayoffEngine",
    "PayoffConfig",
    "ProxyComputer",
    "ProxyWeights",
    "ProxyObservables",
    "SoftMetrics",
    "MetricsReporter",
]

__version__ = "0.1.0"
