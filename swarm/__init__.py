"""Distributional AGI Safety Sandbox - Soft Label Payoff & Metrics System."""

from swarm.core.orchestrator import EpochMetrics, Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables, ProxyWeights
from swarm.metrics.reporters import MetricsReporter
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.agent import AgentState, AgentStatus, AgentType
from swarm.models.interaction import InteractionType, SoftInteraction


# Lazy imports: ``swarm.research`` and ``swarm.evaluation`` depend on core
# modules that import from this package.  Deferring them avoids circular
# import errors while still allowing ``import swarm; swarm.research``.
def __getattr__(name: str):
    if name == "research":
        from swarm import research

        return research
    if name == "evaluation":
        from swarm import evaluation

        return evaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Models
    "SoftInteraction",
    "InteractionType",
    "AgentType",
    "AgentStatus",
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
    # Research
    "research",
    # Evaluation
    "evaluation",
]

__version__ = "1.0.0"
