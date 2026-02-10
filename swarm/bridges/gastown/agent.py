"""GasTownAgent — BaseAgent adapter for the SWARM orchestrator loop.

Wraps a GasTownBridge so a GasTown agent can participate in the
standard orchestrator act/accept_interaction/propose_interaction cycle.
"""

import logging
from typing import Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.bridges.gastown.bridge import GasTownBridge
from swarm.models.agent import AgentType

logger = logging.getLogger(__name__)


class GasTownAgent(BaseAgent):
    """A SWARM agent backed by a GasTown workspace agent.

    Each call to :meth:`act` polls the bridge for new interactions
    involving this agent and returns an appropriate SWARM Action.
    """

    def __init__(
        self,
        agent_id: str,
        bridge: GasTownBridge,
        gastown_name: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            roles=roles,
            config=config or {},
        )
        self._bridge = bridge
        self._gastown_name = gastown_name

    def act(self, observation: Observation) -> Action:
        """Poll the bridge and return an action based on latest interactions."""
        new_interactions = self._bridge.poll()

        # Filter to interactions for this agent
        mine = [
            i
            for i in new_interactions
            if i.counterparty == self.agent_id
        ]

        if not mine:
            return self.create_noop_action()

        latest = mine[-1]

        # If there are available tasks, claim one
        if observation.available_tasks:
            task = observation.available_tasks[0]
            return self.create_claim_task_action(task.get("task_id", ""))

        # Otherwise create a post summarising the completed work
        summary = latest.metadata.get("bead_title", "completed work")
        return self.create_post_action(f"Completed: {summary}"[:500])

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Accept interactions from agents with decent reputation."""
        return observation.agent_state.reputation >= -0.5

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """GasTown agents do not proactively propose interactions."""
        return None
