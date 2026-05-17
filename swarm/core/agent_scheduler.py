"""Agent scheduling and eligibility filtering.

Extracted from ``Orchestrator`` to isolate the turn-order and
eligibility logic into a focused, testable component.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set


class AgentScheduler:
    """Determines which agents act each step and in what order.

    Parameters:
        schedule_mode: ``"round_robin"`` | ``"random"`` | ``"priority"``
        max_actions_per_step: Cap on how many agents act per step.
        rng: Seeded random instance for shuffle reproducibility.
    """

    def __init__(
        self,
        schedule_mode: str = "round_robin",
        max_actions_per_step: int = 20,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._mode = schedule_mode
        self._max = max_actions_per_step
        self._rng = rng or random.Random()

    def get_eligible(
        self,
        agents: Dict[str, Any],
        state: Any,  # EnvState
        *,
        governance_engine: Optional[Any] = None,
        dropped_agents: Optional[Set[str]] = None,
    ) -> List[str]:
        """Return an ordered list of agent IDs eligible to act this step."""
        schedule = self._get_order(agents, state)
        dropped = dropped_agents or set()

        eligible: List[str] = []
        for agent_id in schedule:
            if len(eligible) >= self._max:
                break
            if agent_id in dropped:
                continue
            if not state.can_agent_act(agent_id):
                continue
            if governance_engine is not None and not governance_engine.can_agent_act(
                agent_id, state
            ):
                continue
            eligible.append(agent_id)
        return eligible

    def _get_order(self, agents: Dict[str, Any], state: Any) -> List[str]:
        """Return agent IDs in scheduling order."""
        agent_ids = list(agents.keys())

        if self._mode == "random":
            self._rng.shuffle(agent_ids)
        elif self._mode == "priority":
            agent_ids.sort(
                key=lambda aid: (
                    agent_st.reputation
                    if (agent_st := state.get_agent(aid))
                    else 0
                ),
                reverse=True,
            )
        # else: round_robin (insertion order)

        return agent_ids
