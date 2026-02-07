"""Core computation modules for payoff, proxy, and orchestration."""

from swarm.core.orchestrator import EpochMetrics, Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables, ProxyWeights
from swarm.core.pseudo_verifiers import (
    CompositeVerifier,
    ConsistencyVerifier,
    FormatVerifier,
    HeuristicVerifier,
    ModelBasedVerifier,
    PseudoVerifier,
    VerificationResult,
    create_code_verifier,
    create_research_verifier,
    create_task_verifier,
)
from swarm.core.sigmoid import calibrated_sigmoid, inverse_sigmoid

__all__ = [
    "calibrated_sigmoid",
    "inverse_sigmoid",
    "ProxyComputer",
    "ProxyWeights",
    "ProxyObservables",
    "SoftPayoffEngine",
    "PayoffConfig",
    "Orchestrator",
    "OrchestratorConfig",
    "EpochMetrics",
    # Pseudo-verifiers (Bradley framework)
    "CompositeVerifier",
    "ConsistencyVerifier",
    "FormatVerifier",
    "HeuristicVerifier",
    "ModelBasedVerifier",
    "PseudoVerifier",
    "VerificationResult",
    "create_code_verifier",
    "create_research_verifier",
    "create_task_verifier",
]
