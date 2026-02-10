"""SWARM-GasTown Bridge.

Connects SWARM's governance and metrics framework to a GasTown
multi-agent workspace, enabling safety scoring of bead lifecycle
and PR workflow events.

Architecture:
    SWARM Orchestrator (Python)
        └── GasTownBridge
                ├── BeadsClient   (SQLite read-only)
                ├── GitObserver   (git subprocess)
                ├── GasTownMapper (events → SoftInteraction)
                ├── GasTownPolicy (governance → gt CLI)
                └── GasTownAgent  (BaseAgent adapter)
"""

from swarm.bridges.gastown.agent import GasTownAgent
from swarm.bridges.gastown.bridge import GasTownBridge
from swarm.bridges.gastown.config import GasTownConfig
from swarm.bridges.gastown.events import GasTownEvent, GasTownEventType
from swarm.bridges.gastown.mapper import GasTownMapper

__all__ = [
    "GasTownBridge",
    "GasTownConfig",
    "GasTownEvent",
    "GasTownEventType",
    "GasTownMapper",
    "GasTownAgent",
]
