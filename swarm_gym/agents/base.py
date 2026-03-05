"""AgentPolicy: the interface for agent decision-making.

Agent policies receive observations and return actions. They can be
scripted baselines, RL policies, or LLM-backed agents.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

from swarm_gym.utils.types import Action, AgentId, AgentRecord, Observation


class AgentPolicy(abc.ABC):
    """Abstract base for agent policies.

    A policy controls one or more agents in the environment.
    Single-agent policies (one agent) and population policies
    (multiple agents with different behaviors) are both supported.
    """

    name: str = "BasePolicy"

    @abc.abstractmethod
    def act(
        self,
        observations: Dict[AgentId, Observation],
    ) -> List[Action]:
        """Choose actions given current observations.

        Args:
            observations: Per-agent observations from the environment.

        Returns:
            List of actions (one per agent, or subset).
        """

    def reset(self, agent_ids: List[AgentId], seed: int = 0) -> None:
        """Reset policy state for a new episode.

        Args:
            agent_ids: The IDs of agents this policy controls.
            seed: Random seed for reproducibility.
        """

    def get_agent_records(self) -> List[AgentRecord]:
        """Return agent type information for reporting."""
        return []

    @property
    def policy_name(self) -> str:
        return self.name
