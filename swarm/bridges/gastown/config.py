"""Configuration for the SWARM-GasTown bridge."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class GasTownConfig:
    """Configuration for the GasTown bridge.

    Attributes:
        workspace_path: Path to the GasTown workspace root (contains .beads/, worktrees).
        beads_db_path: Override for the .beads/beads.db location.
        gt_cli_path: Path to the ``gt`` CLI binary.
        poll_interval_seconds: How often to poll beads/git for new events.
        proxy_sigmoid_k: Sigmoid sharpness for ProxyComputer.
        agent_role_map: Maps GasTown agent names to SWARM agent IDs.
        max_interactions: Cap on stored interactions.
        max_events: Cap on stored events.
    """

    workspace_path: str = "."
    beads_db_path: Optional[str] = None
    gt_cli_path: str = "gt"
    poll_interval_seconds: float = 5.0
    proxy_sigmoid_k: float = 2.0
    base_branch: str = "origin/main"
    agent_role_map: Dict[str, str] = field(default_factory=dict)
    max_interactions: int = 50000
    max_events: int = 50000
