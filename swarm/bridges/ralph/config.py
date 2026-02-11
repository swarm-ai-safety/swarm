"""Configuration for the SWARM-Ralph bridge."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RalphConfig:
    """Configuration for consuming Ralph event exports.

    Attributes:
        events_path: Path to a Ralph JSONL event export.
        orchestrator_id: SWARM-side initiator id for generated interactions.
        poll_interval_seconds: Suggested poll interval for callers.
        proxy_sigmoid_k: Sigmoid sharpness for ProxyComputer.
        agent_role_map: Optional mapping from Ralph actor ids to SWARM ids.
        max_interactions: Cap on in-memory interactions retained by bridge.
        max_events: Cap on in-memory events retained by bridge.
    """

    events_path: str
    orchestrator_id: str = "ralph_orchestrator"
    poll_interval_seconds: float = 5.0
    proxy_sigmoid_k: float = 2.0
    agent_role_map: Dict[str, str] = field(default_factory=dict)
    max_interactions: int = 50000
    max_events: int = 50000
