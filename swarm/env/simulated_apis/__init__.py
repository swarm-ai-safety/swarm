"""Service-like simulated APIs for swarm amplification safety experiments.

This package provides:
- Domain services (IAM, Payments, Incident Response) with stateful API calls
- Deterministic task generators that produce JSON bundles
- Checkers that score success/safety from final state + event logs
- Optional m-of-n gating for irreversible endpoints
"""

from swarm.env.simulated_apis.adversary import (
    AdversarySuggestion,
    AdversaryType,
    suggest_steering,
)
from swarm.env.simulated_apis.gating import ApprovalConfig, IrreversibleGate
from swarm.env.simulated_apis.logging import SimApiEpisodeLog
from swarm.env.simulated_apis.service import (
    ApiCallError,
    ApiCallResult,
    ApiEndpointSpec,
    SimulatedApiService,
)
from swarm.env.simulated_apis.spec import Domain, Split
from swarm.env.simulated_apis.suite import (
    TaskBundle,
    generate_task_bundle,
    score_task_bundle,
)

__all__ = [
    "AdversarySuggestion",
    "AdversaryType",
    "ApiCallError",
    "ApiCallResult",
    "ApiEndpointSpec",
    "ApprovalConfig",
    "Domain",
    "Split",
    "IrreversibleGate",
    "SimApiEpisodeLog",
    "SimulatedApiService",
    "suggest_steering",
    "TaskBundle",
    "generate_task_bundle",
    "score_task_bundle",
]
