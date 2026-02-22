"""Coding agent policy — models coding agent behavior in the swarm."""

from typing import TYPE_CHECKING, Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType

if TYPE_CHECKING:
    from swarm.agents.memory_config import MemoryConfig


class CodingAgent(BaseAgent):
    """Simulates a coding agent that claims tasks, submits code, and reviews.

    Config params:
        skill_level (float): Code quality baseline, 0-1. Default 0.7.
        review_honesty (float): How honestly the agent reviews, 0-1. Default 0.9.
        adversarial (bool): Whether this agent is adversarial. Default False.
        subtlety (float): How hard adversarial cheating is to detect, 0-1. Default 0.5.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng=None,
    ):
        config = config or {}
        is_adversarial = config.get("adversarial", False)
        agent_type = AgentType.ADVERSARIAL if is_adversarial else AgentType.HONEST

        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            roles=roles or [Role.WORKER, Role.VERIFIER],
            config=config,
            name=name,
            memory_config=memory_config,
            rng=rng,
        )

        self.skill_level: float = config.get("skill_level", 0.7)
        self.review_honesty: float = config.get("review_honesty", 0.9)
        self.adversarial: bool = is_adversarial
        self.subtlety: float = config.get("subtlety", 0.5)

    def act(self, observation: Observation) -> Action:
        """Decide action: pending proposals > active tasks > claim task > interact > noop."""
        # Handle pending proposals first
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            return self.create_accept_action(proposal["proposal_id"])

        # Submit output for active tasks
        if observation.active_tasks:
            task = observation.active_tasks[0]
            if task.get("status") == "in_progress":
                return self._submit_code(task)

        # Claim a new task
        if observation.can_claim_task and observation.available_tasks:
            new_task = self._pick_task(observation.available_tasks)
            if new_task:
                return self.create_claim_task_action(new_task["task_id"])

        # Review another agent's submission (via VERIFY_OUTPUT)
        if observation.active_tasks:
            for task in observation.active_tasks:
                if task.get("status") == "submitted" and task.get("claimed_by") != self.agent_id:
                    return self._review_code(task)

        # Propose code-review collaborations with visible agents
        if observation.can_interact and observation.visible_agents:
            return self._propose_code_review(observation)

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Accept code review requests and collaborations."""
        if proposal.interaction_type == InteractionType.COLLABORATION:
            return True
        trust = self.compute_counterparty_trust(proposal.initiator_id)
        return bool(trust >= 0.3)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose code review interactions."""
        trust = self.compute_counterparty_trust(counterparty_id)
        if trust < 0.2:
            return None
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Requesting code review.",
            offered_transfer=0.0,
        )

    # ---- internal helpers ----

    def _propose_code_review(self, observation: Observation) -> Action:
        """Propose a code review collaboration with a visible agent."""
        candidates = [
            a for a in observation.visible_agents
            if a.get("agent_id") != self.agent_id
        ]
        if not candidates:
            return self.create_noop_action()

        target = self._rng.choice(candidates)
        counterparty_id: str = target.get("agent_id", "")

        # Quality of the proposed interaction depends on agent behavior
        quality = self._compute_submission_quality()
        content = f"code_review_proposal:quality={quality:.2f}"

        return self.create_propose_action(
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content=content,
        )

    def _pick_task(self, tasks: List[Dict]) -> Optional[Dict]:
        """Pick the best task for this agent."""
        if not tasks:
            return None
        # Simple: pick first available
        return tasks[0]

    def _submit_code(self, task: Dict) -> Action:
        """Submit code for a task. Quality depends on skill and adversarial flag."""
        task_id = task.get("task_id", "")
        quality = self._compute_submission_quality()
        has_hidden_issues = self.adversarial and self._rng.random() < 0.8

        content = f"code_submission:{task_id[:8]}"
        metadata = {
            "quality": quality,
            "has_hidden_issues": has_hidden_issues,
            "subtlety": self.subtlety if has_hidden_issues else 0.0,
        }

        action = self.create_submit_output_action(task_id, content)
        action.metadata.update(metadata)
        return action

    def _review_code(self, task: Dict) -> Action:
        """Review another agent's code submission."""
        task_id = task.get("task_id", "")
        action = Action(
            action_type=self._get_verify_action_type(),
            agent_id=self.agent_id,
            target_id=task_id,
            content="code_review",
            metadata={
                "review_honesty": self.review_honesty,
                "reviewer_skill": self.skill_level,
            },
        )
        return action

    def _compute_submission_quality(self) -> float:
        """Compute code quality for a submission."""
        base = self.skill_level
        noise = self._rng.gauss(0, 0.1)
        quality = max(0.0, min(1.0, base + noise))

        if self.adversarial:
            # Adversarial agents inflate apparent quality
            apparent: float = min(1.0, quality + 0.2)
            return apparent
        return float(quality)

    @staticmethod
    def _get_verify_action_type():
        """Return the VERIFY_OUTPUT ActionType."""
        from swarm.agents.base import ActionType
        return ActionType.VERIFY_OUTPUT
