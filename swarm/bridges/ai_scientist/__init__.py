"""SWARM-AI-Scientist Bridge.

Connects SWARM's governance and metrics framework to SakanaAI's
AI-Scientist, enabling monitoring, scoring, and governance of
fully automated research pipelines (Idea Generation -> Experiment ->
Writeup -> Review -> Improvement).

Architecture:
    AI-Scientist results directories (ideas.json, final_info.json, review.txt)
        |
    AIScientistClient (JSON/text parser)
        |
    AIScientistBridge._process_event()
        |   AIScientistPolicy (novelty gate, circuit breaker, cost cap, review threshold)
        |
    AIScientistMapper -> ProxyObservables -> ProxyComputer -> (v_hat, p)
        |
    SoftInteraction -> EventLog + SWARM metrics pipeline
"""

from swarm.bridges.ai_scientist.bridge import AIScientistBridge
from swarm.bridges.ai_scientist.client import AIScientistClient
from swarm.bridges.ai_scientist.config import (
    AIScientistClientConfig,
    AIScientistConfig,
)
from swarm.bridges.ai_scientist.events import (
    AIScientistEvent,
    AIScientistEventType,
    ExperimentRunEvent,
    IdeaEvent,
    ReviewEvent,
    WriteupEvent,
)
from swarm.bridges.ai_scientist.mapper import AIScientistMapper
from swarm.bridges.ai_scientist.policy import (
    AIScientistPolicy,
    PolicyDecision,
    PolicyResult,
)

__all__ = [
    "AIScientistBridge",
    "AIScientistClient",
    "AIScientistClientConfig",
    "AIScientistConfig",
    "AIScientistEvent",
    "AIScientistEventType",
    "AIScientistMapper",
    "AIScientistPolicy",
    "ExperimentRunEvent",
    "IdeaEvent",
    "PolicyDecision",
    "PolicyResult",
    "ReviewEvent",
    "WriteupEvent",
]
