"""Greedy agent: maximizes short-term personal gain."""

from __future__ import annotations

import random as _random
from typing import Dict, List

from swarm_gym.agents.base import AgentPolicy
from swarm_gym.utils.types import Action, AgentId, AgentRecord, Observation


class GreedyPolicy(AgentPolicy):
    """Myopic self-interest maximizer.

    Trades aggressively at favorable prices, defects when advantageous,
    cooperates only when resources are low.
    """

    name = "Greedy"

    def __init__(self):
        self._agent_ids: List[AgentId] = []
        self._rng = _random.Random(0)

    def act(self, observations: Dict[AgentId, Observation]) -> List[Action]:
        actions: List[Action] = []
        for agent_id, obs in observations.items():
            others = [a for a in obs.visible_agents if a != agent_id]
            target = self._rng.choice(others) if others else None

            # Greedy logic: trade at extractive prices, defect frequently
            roll = self._rng.random()
            if obs.own_resources < 20 and roll < 0.3:
                # Desperate: cooperate for survival
                actions.append(Action(agent_id=agent_id, type="cooperate", target=target))
            elif roll < 0.5:
                # Trade at high prices (extractive)
                actions.append(Action(
                    agent_id=agent_id, type="trade", target=target,
                    asset="X", qty=1.0, price=self._rng.uniform(8.0, 15.0),
                ))
            elif roll < 0.8:
                actions.append(Action(agent_id=agent_id, type="defect", target=target))
            else:
                actions.append(Action(agent_id=agent_id, type="noop"))
        return actions

    def reset(self, agent_ids: List[AgentId], seed: int = 0) -> None:
        self._agent_ids = list(agent_ids)
        self._rng = _random.Random(seed)

    def get_agent_records(self) -> List[AgentRecord]:
        return [AgentRecord(agent_id=aid, type="Greedy") for aid in self._agent_ids]
