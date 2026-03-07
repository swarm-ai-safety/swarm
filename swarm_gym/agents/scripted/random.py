"""Random agent: uniform random actions from the action space."""

from __future__ import annotations

import random as _random
from typing import Dict, List

from swarm_gym.agents.base import AgentPolicy
from swarm_gym.utils.types import Action, AgentId, AgentRecord, Observation


class RandomPolicy(AgentPolicy):
    """Uniformly random action selection.

    For each agent, picks a random action type and random target.
    """

    name = "Random"

    def __init__(self, action_types: List[str] | None = None):
        self._action_types = action_types or [
            "cooperate", "defect", "message", "trade", "noop",
        ]
        self._agent_ids: List[AgentId] = []
        self._rng = _random.Random(0)

    def act(self, observations: Dict[AgentId, Observation]) -> List[Action]:
        actions: List[Action] = []
        for agent_id, obs in observations.items():
            action_type = self._rng.choice(self._action_types)
            target = None
            if obs.visible_agents and action_type != "noop":
                others = [a for a in obs.visible_agents if a != agent_id]
                if others:
                    target = self._rng.choice(others)
            actions.append(Action(
                agent_id=agent_id,
                type=action_type,
                target=target,
            ))
        return actions

    def reset(self, agent_ids: List[AgentId], seed: int = 0) -> None:
        self._agent_ids = list(agent_ids)
        self._rng = _random.Random(seed)

    def get_agent_records(self) -> List[AgentRecord]:
        return [AgentRecord(agent_id=aid, type="Random") for aid in self._agent_ids]
