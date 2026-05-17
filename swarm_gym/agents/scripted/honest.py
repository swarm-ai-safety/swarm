"""Honest agent: always cooperates, never deceives."""

from __future__ import annotations

import random as _random
from typing import Dict, List

from swarm_gym.agents.base import AgentPolicy
from swarm_gym.utils.types import Action, AgentId, AgentRecord, Observation


class HonestPolicy(AgentPolicy):
    """Always cooperates and reports truthfully.

    Prefers cooperation > trade > message > noop.
    Never escalates, lies, or hides.
    """

    name = "Honest"

    def __init__(self):
        self._agent_ids: List[AgentId] = []
        self._rng = _random.Random(0)

    def act(self, observations: Dict[AgentId, Observation]) -> List[Action]:
        actions: List[Action] = []
        for agent_id, obs in observations.items():
            others = [a for a in obs.visible_agents if a != agent_id]
            target = self._rng.choice(others) if others else None

            # Honest agents mostly cooperate, sometimes trade
            roll = self._rng.random()
            if roll < 0.7:
                actions.append(Action(agent_id=agent_id, type="cooperate", target=target))
            elif roll < 0.9:
                actions.append(Action(
                    agent_id=agent_id, type="trade", target=target,
                    asset="X", qty=1.0, price=self._rng.uniform(3.0, 8.0),
                ))
            else:
                actions.append(Action(
                    agent_id=agent_id, type="message", target=target,
                    content="Let's work together.",
                ))
        return actions

    def reset(self, agent_ids: List[AgentId], seed: int = 0) -> None:
        self._agent_ids = list(agent_ids)
        self._rng = _random.Random(seed)

    def get_agent_records(self) -> List[AgentRecord]:
        return [AgentRecord(agent_id=aid, type="Honest") for aid in self._agent_ids]
