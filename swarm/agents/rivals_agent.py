"""Agent implementations for the Team-of-Rivals scenario.

RivalsProducerAgent: acts on RIVALS_PRODUCE when assigned as coder/chart_maker/writer.
RivalsCriticAgent: acts on RIVALS_REVIEW when assigned as critic_code/chart/output.
"""

import random
from typing import TYPE_CHECKING, Dict, List, Optional

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType

if TYPE_CHECKING:
    from swarm.agents.memory_config import MemoryConfig


class RivalsProducerAgent(BaseAgent):
    """Producer agent for the rivals pipeline.

    Acts on RIVALS_PRODUCE assignments. Configured with quality_level
    and trap_awareness parameters that affect artifact quality.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            roles=roles or [Role.WORKER],
            config=config or {},
            name=name,
            memory_config=memory_config,
            rng=rng,
        )
        self.quality_level = self.config.get("quality_level", 0.7)
        self.trap_awareness = self.config.get("trap_awareness", 0.5)
        self.role = self.config.get("role", "coder")

    def act(self, observation: Observation) -> Action:
        """Produce artifact when assigned, otherwise noop."""
        assignments = getattr(observation, "rivals_assignments", [])
        for assignment in assignments:
            if assignment.get("type") == "produce":
                return Action(
                    action_type=ActionType.RIVALS_PRODUCE,
                    agent_id=self.agent_id,
                    target_id=assignment.get("episode_id", ""),
                    metadata={
                        "quality_level": self.quality_level,
                        "trap_awareness": self.trap_awareness,
                    },
                )
        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        return True

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        return None


class RivalsCriticAgent(BaseAgent):
    """Critic agent for the rivals pipeline.

    Acts on RIVALS_REVIEW assignments. Configured with detection_rate
    and false_positive_rate parameters that affect review decisions.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            roles=roles or [Role.VERIFIER],
            config=config or {},
            name=name,
            memory_config=memory_config,
            rng=rng,
        )
        self.detection_rate = self.config.get("detection_rate", 0.7)
        self.false_positive_rate = self.config.get("false_positive_rate", 0.1)
        self.role = self.config.get("role", "critic_code")

    def act(self, observation: Observation) -> Action:
        """Review artifact when assigned, otherwise noop."""
        assignments = getattr(observation, "rivals_assignments", [])
        for assignment in assignments:
            if assignment.get("type") == "review":
                return Action(
                    action_type=ActionType.RIVALS_REVIEW,
                    agent_id=self.agent_id,
                    target_id=assignment.get("episode_id", ""),
                    metadata={
                        "detection_rate": self.detection_rate,
                        "false_positive_rate": self.false_positive_rate,
                    },
                )
        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        return True

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        return None
