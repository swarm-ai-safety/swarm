"""Base agent interface and core abstractions."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.models.agent import AgentState, AgentType
from src.models.interaction import InteractionType, SoftInteraction


class ActionType(Enum):
    """Types of actions an agent can take."""

    # Feed actions
    POST = "post"
    REPLY = "reply"
    VOTE = "vote"

    # Interaction actions
    PROPOSE_INTERACTION = "propose_interaction"
    ACCEPT_INTERACTION = "accept_interaction"
    REJECT_INTERACTION = "reject_interaction"

    # Task actions
    CLAIM_TASK = "claim_task"
    SUBMIT_OUTPUT = "submit_output"
    VERIFY_OUTPUT = "verify_output"

    # Marketplace actions
    POST_BOUNTY = "post_bounty"
    PLACE_BID = "place_bid"
    ACCEPT_BID = "accept_bid"
    REJECT_BID = "reject_bid"
    WITHDRAW_BID = "withdraw_bid"
    FILE_DISPUTE = "file_dispute"

    # Special actions
    NOOP = "noop"  # Do nothing this turn


class Role(Enum):
    """Agent roles in the simulation."""

    PLANNER = "planner"
    WORKER = "worker"
    VERIFIER = "verifier"
    POSTER = "poster"
    MODERATOR = "moderator"


@dataclass
class Action:
    """An action taken by an agent."""

    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ActionType = ActionType.NOOP
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Action-specific payloads
    content: str = ""  # For posts, replies, outputs
    target_id: str = ""  # post_id, proposal_id, task_id, etc.
    counterparty_id: str = ""  # For interactions
    interaction_type: InteractionType = InteractionType.REPLY
    vote_direction: int = 0  # +1 for upvote, -1 for downvote

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize action."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "target_id": self.target_id,
            "counterparty_id": self.counterparty_id,
            "interaction_type": self.interaction_type.value,
            "vote_direction": self.vote_direction,
            "metadata": self.metadata,
        }


@dataclass
class Observation:
    """
    Observation provided to an agent for decision making.

    Contains the agent's view of the current state.
    """

    # Agent's own state
    agent_state: AgentState = field(default_factory=AgentState)

    # Simulation context
    current_epoch: int = 0
    current_step: int = 0

    # Rate limit info
    can_post: bool = True
    can_interact: bool = True
    can_vote: bool = True
    can_claim_task: bool = True

    # Feed view (ranked posts visible to agent)
    visible_posts: List[Dict] = field(default_factory=list)

    # Pending proposals (interactions proposed to this agent)
    pending_proposals: List[Dict] = field(default_factory=list)

    # Available tasks
    available_tasks: List[Dict] = field(default_factory=list)

    # Agent's active tasks
    active_tasks: List[Dict] = field(default_factory=list)

    # Recent interactions involving this agent
    recent_interactions: List[Dict] = field(default_factory=list)

    # Other agents (visible subset based on reputation/visibility)
    visible_agents: List[Dict] = field(default_factory=list)

    # Marketplace
    available_bounties: List[Dict] = field(default_factory=list)
    active_bids: List[Dict] = field(default_factory=list)
    active_escrows: List[Dict] = field(default_factory=list)
    pending_bid_decisions: List[Dict] = field(default_factory=list)

    # Signals about ecosystem health
    ecosystem_metrics: Dict = field(default_factory=dict)


@dataclass
class InteractionProposal:
    """A proposal for an interaction."""

    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    initiator_id: str = ""
    counterparty_id: str = ""
    interaction_type: InteractionType = InteractionType.COLLABORATION
    content: str = ""
    task_id: Optional[str] = None
    post_id: Optional[str] = None
    offered_transfer: float = 0.0  # tau offered by initiator
    metadata: Dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for agent policies.

    Agents observe the environment and take actions based on their
    behavioral policy.
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize agent.

        Args:
            agent_id: Unique identifier
            agent_type: Behavioral archetype
            roles: List of roles this agent can fulfill
            config: Agent-specific configuration
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.roles = roles or [Role.WORKER]
        self.config = config or {}

        # Internal state
        self._memory: List[Dict] = []
        self._interaction_history: List[SoftInteraction] = []

    @property
    def primary_role(self) -> Role:
        """Get the agent's primary role."""
        return self.roles[0] if self.roles else Role.WORKER

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """
        Decide on an action given the current observation.

        Args:
            observation: Current view of the environment

        Returns:
            Action to take
        """
        pass

    @abstractmethod
    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """
        Decide whether to accept a proposed interaction.

        Args:
            proposal: The interaction proposal
            observation: Current observation

        Returns:
            True to accept, False to reject
        """
        pass

    @abstractmethod
    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """
        Create an interaction proposal for a counterparty.

        Args:
            observation: Current observation
            counterparty_id: Target agent ID

        Returns:
            InteractionProposal or None if not proposing
        """
        pass

    def update_from_outcome(
        self,
        interaction: SoftInteraction,
        payoff: float,
    ) -> None:
        """
        Update internal state after an interaction completes.

        Args:
            interaction: The completed interaction
            payoff: Payoff received
        """
        self._interaction_history.append(interaction)
        self._memory.append({
            "type": "interaction_outcome",
            "interaction_id": interaction.interaction_id,
            "counterparty": (
                interaction.counterparty
                if interaction.initiator == self.agent_id
                else interaction.initiator
            ),
            "p": interaction.p,
            "payoff": payoff,
            "accepted": interaction.accepted,
            "timestamp": datetime.now().isoformat(),
        })

    def remember(self, memory_item: Dict) -> None:
        """Add an item to memory."""
        memory_item["timestamp"] = datetime.now().isoformat()
        self._memory.append(memory_item)

    def get_memory(self, limit: int = 100) -> List[Dict]:
        """Get recent memory items."""
        return self._memory[-limit:]

    def get_interaction_history(self, limit: int = 50) -> List[SoftInteraction]:
        """Get recent interaction history."""
        return self._interaction_history[-limit:]

    def compute_counterparty_trust(self, counterparty_id: str) -> float:
        """
        Compute trust score for a counterparty based on history.

        Args:
            counterparty_id: ID of the counterparty

        Returns:
            Trust score in [0, 1]
        """
        relevant = [
            i for i in self._interaction_history
            if (i.initiator == counterparty_id or i.counterparty == counterparty_id)
            and i.accepted
        ]

        if not relevant:
            return 0.5  # Neutral for unknown agents

        # Average p from past interactions
        avg_p = sum(i.p for i in relevant) / len(relevant)
        return avg_p

    def should_post(self, observation: Observation) -> bool:
        """Determine if agent should create a post."""
        return observation.can_post

    def should_vote(self, observation: Observation) -> bool:
        """Determine if agent should vote."""
        return observation.can_vote and len(observation.visible_posts) > 0

    def should_interact(self, observation: Observation) -> bool:
        """Determine if agent should initiate an interaction."""
        return observation.can_interact

    def should_claim_task(self, observation: Observation) -> bool:
        """Determine if agent should claim a task."""
        return observation.can_claim_task and len(observation.available_tasks) > 0

    def create_noop_action(self) -> Action:
        """Create a no-op action."""
        return Action(
            action_type=ActionType.NOOP,
            agent_id=self.agent_id,
        )

    def create_post_action(self, content: str) -> Action:
        """Create a post action."""
        return Action(
            action_type=ActionType.POST,
            agent_id=self.agent_id,
            content=content,
        )

    def create_reply_action(self, post_id: str, content: str) -> Action:
        """Create a reply action."""
        return Action(
            action_type=ActionType.REPLY,
            agent_id=self.agent_id,
            target_id=post_id,
            content=content,
        )

    def create_vote_action(self, post_id: str, direction: int) -> Action:
        """Create a vote action (+1 upvote, -1 downvote)."""
        return Action(
            action_type=ActionType.VOTE,
            agent_id=self.agent_id,
            target_id=post_id,
            vote_direction=direction,
        )

    def create_propose_action(
        self,
        counterparty_id: str,
        interaction_type: InteractionType,
        content: str = "",
        task_id: Optional[str] = None,
    ) -> Action:
        """Create an interaction proposal action."""
        return Action(
            action_type=ActionType.PROPOSE_INTERACTION,
            agent_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=interaction_type,
            content=content,
            target_id=task_id or "",
        )

    def create_accept_action(self, proposal_id: str) -> Action:
        """Create an interaction acceptance action."""
        return Action(
            action_type=ActionType.ACCEPT_INTERACTION,
            agent_id=self.agent_id,
            target_id=proposal_id,
        )

    def create_reject_action(self, proposal_id: str) -> Action:
        """Create an interaction rejection action."""
        return Action(
            action_type=ActionType.REJECT_INTERACTION,
            agent_id=self.agent_id,
            target_id=proposal_id,
        )

    def create_claim_task_action(self, task_id: str) -> Action:
        """Create a task claim action."""
        return Action(
            action_type=ActionType.CLAIM_TASK,
            agent_id=self.agent_id,
            target_id=task_id,
        )

    def create_submit_output_action(self, task_id: str, content: str) -> Action:
        """Create a task output submission action."""
        return Action(
            action_type=ActionType.SUBMIT_OUTPUT,
            agent_id=self.agent_id,
            target_id=task_id,
            content=content,
        )

    def create_post_bounty_action(
        self,
        reward_amount: float,
        task_description: str = "",
        min_reputation: float = 0.0,
        deadline_epoch: Optional[int] = None,
    ) -> Action:
        """Create an action to post a bounty."""
        return Action(
            action_type=ActionType.POST_BOUNTY,
            agent_id=self.agent_id,
            content=task_description,
            metadata={
                "reward_amount": reward_amount,
                "min_reputation": min_reputation,
                "deadline_epoch": deadline_epoch,
            },
        )

    def create_place_bid_action(
        self,
        bounty_id: str,
        bid_amount: float,
        message: str = "",
    ) -> Action:
        """Create an action to place a bid on a bounty."""
        return Action(
            action_type=ActionType.PLACE_BID,
            agent_id=self.agent_id,
            target_id=bounty_id,
            content=message,
            metadata={"bid_amount": bid_amount},
        )

    def create_accept_bid_action(self, bounty_id: str, bid_id: str) -> Action:
        """Create an action to accept a bid."""
        return Action(
            action_type=ActionType.ACCEPT_BID,
            agent_id=self.agent_id,
            target_id=bounty_id,
            metadata={"bid_id": bid_id},
        )

    def create_reject_bid_action(self, bid_id: str) -> Action:
        """Create an action to reject a bid."""
        return Action(
            action_type=ActionType.REJECT_BID,
            agent_id=self.agent_id,
            target_id=bid_id,
        )

    def create_withdraw_bid_action(self, bid_id: str) -> Action:
        """Create an action to withdraw a bid."""
        return Action(
            action_type=ActionType.WITHDRAW_BID,
            agent_id=self.agent_id,
            target_id=bid_id,
        )

    def create_file_dispute_action(self, escrow_id: str, reason: str = "") -> Action:
        """Create an action to file a dispute."""
        return Action(
            action_type=ActionType.FILE_DISPUTE,
            agent_id=self.agent_id,
            target_id=escrow_id,
            content=reason,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type.value})"
