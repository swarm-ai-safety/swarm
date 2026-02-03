"""Distributional AGI Safety Sandbox - Soft Label Payoff & Metrics System."""

from src.core.orchestrator import EpochMetrics, Orchestrator, OrchestratorConfig
from src.core.payoff import PayoffConfig, SoftPayoffEngine
from src.core.proxy import ProxyComputer, ProxyObservables, ProxyWeights
from src.metrics.reporters import MetricsReporter
from src.metrics.soft_metrics import SoftMetrics
from src.models.agent import AgentState, AgentType
from src.models.interaction import InteractionType, SoftInteraction

__all__ = [
    # Models
    "SoftInteraction",
    "InteractionType",
    "AgentType",
    "AgentState",
    # Core
    "SoftPayoffEngine",
    "PayoffConfig",
    "ProxyComputer",
    "ProxyWeights",
    "ProxyObservables",
    # Orchestrator
    "Orchestrator",
    "OrchestratorConfig",
    "EpochMetrics",
    # Metrics
    "SoftMetrics",
    "MetricsReporter",
]

__version__ = "0.1.0"
