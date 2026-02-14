"""SWARM-AgentLaboratory Bridge.

Connects SWARM's governance and metrics framework to AgentLaboratory,
enabling monitoring, scoring, and governance of autonomous research
workflows that orchestrate specialized LLM agents through literature
review, experimentation, and paper writing.

Architecture:
    AgentLab checkpoints / lab directories
        |
    AgentLabClient (pickle parser + directory scanner)
        |
    AgentLabBridge._process_event()
        |   AgentLabPolicy (phase gates, circuit breakers, cost caps)
        |
    AgentLabMapper -> ProxyObservables -> ProxyComputer -> (v_hat, p)
        |
    SoftInteraction -> EventLog + SWARM metrics pipeline
"""

from swarm.bridges.agent_lab.bridge import AgentLabBridge
from swarm.bridges.agent_lab.client import AgentLabClient
from swarm.bridges.agent_lab.config import (
    AgentLabClientConfig,
    AgentLabConfig,
)
from swarm.bridges.agent_lab.events import (
    AgentLabEvent,
    AgentLabEventType,
    DialogueEvent,
    ReviewEvent,
    SolverIterationEvent,
)
from swarm.bridges.agent_lab.mapper import AgentLabMapper
from swarm.bridges.agent_lab.policy import (
    AgentLabPolicy,
    PolicyDecision,
    PolicyResult,
)

__all__ = [
    "AgentLabBridge",
    "AgentLabClient",
    "AgentLabClientConfig",
    "AgentLabConfig",
    "AgentLabEvent",
    "AgentLabEventType",
    "AgentLabMapper",
    "AgentLabPolicy",
    "DialogueEvent",
    "PolicyDecision",
    "PolicyResult",
    "ReviewEvent",
    "SolverIterationEvent",
]
