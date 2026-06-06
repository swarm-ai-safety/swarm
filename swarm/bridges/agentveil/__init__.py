"""SWARM–AgentVeil Bridge.

Connects SWARM's governance and metrics framework to the Agent Veil
Protocol (AVP), enabling cryptographic identity, peer-attested reputation,
and sybil-resistant admission control on top of SWARM's probabilistic
labels.

See ``docs/bridges/agentveil.md`` for the full plan and failure-mode
catalog. This PR ships only the v1 skeleton: ``config.py`` and
``events.py``. ``client.py``, ``mapper.py``, ``policy.py``, and the
``AgentVeilBridge`` orchestrator land in follow-up PRs (issues
``wzo9`` / ``69rq`` / ``sdgo`` / ``whog``); live registry HTTP,
write-back attestations, and the dispute/arbitration flow land in v2+.

Planned architecture (target shape — most symbols below are not yet
exported from this package):

    AVP Registry (agentveil.dev)        [live mode only; v2]
        |
    AVPClient (client.py)               [v1: mock_mode only — TBD]
        |
    AgentVeilBridge._process_event()    [TBD]
        |   AVPPolicy (admission gate, rate limiter, write-back gate)
        |
    AVPMapper -> ProxyObservables -> ProxyComputer -> (v_hat, p)
        |
    SoftInteraction -> EventLog (incl. ATTESTATION_SUBMITTED) -> metrics
"""

from swarm.bridges.agentveil.config import AgentVeilConfig
from swarm.bridges.agentveil.events import (
    AgentVeilEvent,
    AgentVeilEventType,
    AttestationEvent,
    ReputationSnapshotEvent,
    TrustDecisionEvent,
)

__all__ = [
    "AgentVeilConfig",
    "AgentVeilEvent",
    "AgentVeilEventType",
    "AttestationEvent",
    "ReputationSnapshotEvent",
    "TrustDecisionEvent",
]
