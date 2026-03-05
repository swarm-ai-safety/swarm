"""GovernanceModule: the stable governance plugin interface.

This is one of the 3 frozen interfaces. Governance modules receive the
current world state and proposed actions, and can modify actions, add
costs, trigger sanctions, and emit events.

Frozen interface contract:
    apply(world_state, proposed_actions) -> (modified_actions, interventions, events)
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Tuple

from swarm_gym.utils.types import Action, Event, Intervention


class GovernanceModule(abc.ABC):
    """Abstract base for governance plugins.

    Each module gets:
    - current world state snapshot
    - proposed actions
    - can modify actions, add costs, trigger sanctions, emit events

    Governance is a first-class part of the environment, not an afterthought.
    """

    name: str = "BaseGovernance"
    version: str = "1.0"

    @abc.abstractmethod
    def apply(
        self,
        world_state: Dict[str, Any],
        proposed_actions: List[Action],
    ) -> Tuple[List[Action], List[Intervention], List[Event]]:
        """Apply governance to proposed actions.

        Args:
            world_state: Current environment state snapshot.
            proposed_actions: Actions proposed by agents this step.

        Returns:
            Tuple of:
            - modified_actions: Actions after governance (may block, modify, add costs)
            - interventions: List of governance interventions taken
            - events: List of governance events emitted
        """

    def reset(self, seed: int = 0) -> None:
        """Reset module state for a new episode."""

    def get_state(self) -> Dict[str, Any]:
        """Return current governance parameters for snapshotting."""
        return {}

    def get_params(self) -> Dict[str, Any]:
        """Return configuration parameters for reporting."""
        return {}

    def to_report_dict(self) -> Dict[str, Any]:
        """Serialize for episode/summary JSON."""
        return {
            "name": self.name,
            "version": self.version,
            "params": self.get_params(),
        }
