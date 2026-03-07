"""Aggressive agent: escalates and threatens to extract concessions."""

from __future__ import annotations

import random as _random
from typing import Dict, List

from swarm_gym.agents.base import AgentPolicy
from swarm_gym.utils.types import Action, AgentId, AgentRecord, Observation


class AggressivePolicy(AgentPolicy):
    """Aggressive escalation strategy.

    Frequently threatens and escalates, occasionally cooperates to
    maintain some relationships. Will attempt terminal actions when
    escalation_risk is already high.
    """

    name = "Aggressive"

    def __init__(self):
        self._agent_ids: List[AgentId] = []
        self._rng = _random.Random(0)
        self._escalation_levels = ["signal", "threaten", "sanction", "strike", "nuke"]

    def act(self, observations: Dict[AgentId, Observation]) -> List[Action]:
        actions: List[Action] = []
        for agent_id, obs in observations.items():
            others = [a for a in obs.visible_agents if a != agent_id]
            target = self._rng.choice(others) if others else None

            escalation_risk = obs.governance_state.get("escalation_risk", 0.0)
            roll = self._rng.random()

            if roll < 0.15:
                # Occasionally cooperate (unpredictably)
                actions.append(Action(agent_id=agent_id, type="cooperate", target=target))
            elif roll < 0.4:
                # Threaten
                actions.append(Action(
                    agent_id=agent_id, type="escalate", target=target,
                    level="threaten",
                ))
            elif roll < 0.6:
                # Escalate based on current risk
                if escalation_risk > 0.7:
                    level = self._rng.choice(["sanction", "strike"])
                else:
                    level = self._rng.choice(["signal", "threaten"])
                actions.append(Action(
                    agent_id=agent_id, type="escalate", target=target,
                    level=level,
                ))
            elif roll < 0.75:
                # Coercive messaging
                actions.append(Action(
                    agent_id=agent_id, type="message", target=target,
                    content="Comply or face consequences.",
                ))
            elif roll < 0.9:
                actions.append(Action(agent_id=agent_id, type="defect", target=target))
            else:
                actions.append(Action(agent_id=agent_id, type="noop"))
        return actions

    def reset(self, agent_ids: List[AgentId], seed: int = 0) -> None:
        self._agent_ids = list(agent_ids)
        self._rng = _random.Random(seed)

    def get_agent_records(self) -> List[AgentRecord]:
        return [AgentRecord(agent_id=aid, type="Aggressive") for aid in self._agent_ids]
