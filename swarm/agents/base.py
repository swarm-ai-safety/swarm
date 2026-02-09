"""Base agent interface and core abstractions."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import InteractionType, SoftInteraction

if TYPE_CHECKING:
    from swarm.agents.memory_config import MemoryConfig


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

    # Moltipedia wiki actions
    CREATE_PAGE = "create_page"
    EDIT_PAGE = "edit_page"
    FILE_OBJECTION = "file_objection"
    POLICY_FLAG = "policy_flag"

    # Moltbook actions
    MOLTBOOK_POST = "moltbook_post"
    MOLTBOOK_COMMENT = "moltbook_comment"
    MOLTBOOK_VERIFY = "moltbook_verify"
    MOLTBOOK_VOTE = "moltbook_vote"

    # Memory tier actions
    WRITE_MEMORY = "write_memory"
    PROMOTE_MEMORY = "promote_memory"
    VERIFY_MEMORY = "verify_memory"
    SEARCH_MEMORY = "search_memory"
    CHALLENGE_MEMORY = "challenge_memory"

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

    # Moltipedia wiki queues
    contested_pages: List[Dict] = field(default_factory=list)
    search_results: List[Dict] = field(default_factory=list)
    random_pages: List[Dict] = field(default_factory=list)
    leaderboard: List[Dict] = field(default_factory=list)
    agent_points: float = 0.0
    heartbeat_status: Dict = field(default_factory=dict)

    # Signals about ecosystem health
    ecosystem_metrics: Dict = field(default_factory=dict)

    # Moltbook feed observations
    moltbook_published_posts: List[Dict] = field(default_factory=list)
    moltbook_pending_posts: List[Dict] = field(default_factory=list)
    moltbook_rate_limits: Dict = field(default_factory=dict)
    moltbook_karma: float = 0.0

    # Memory tier observations
    memory_hot_cache: List[Dict] = field(default_factory=list)
    memory_pending_promotions: List[Dict] = field(default_factory=list)
    memory_search_results: List[Dict] = field(default_factory=list)
    memory_challenged_entries: List[Dict] = field(default_factory=list)
    memory_entry_counts: Dict = field(default_factory=dict)
    memory_writes_remaining: int = 0


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
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
    ):
        """
        Initialize agent.

        Args:
            agent_id: Unique identifier
            agent_type: Behavioral archetype
            roles: List of roles this agent can fulfill
            config: Agent-specific configuration
            name: Human-readable label (defaults to agent_id)
            memory_config: Configuration for memory persistence across epochs
        """
        self.agent_id = agent_id
        self.name = name or agent_id
        self.agent_type = agent_type
        self.roles = roles or [Role.WORKER]
        self.config = config or {}

        # Memory configuration (import here to avoid circular imports)
        if memory_config is None:
            from swarm.agents.memory_config import MemoryConfig

            self.memory_config = MemoryConfig()
        else:
            self.memory_config = memory_config

        # Internal state
        self._memory: List[Dict] = []
        self._interaction_history: List[SoftInteraction] = []

        # Counterparty trust memory: agent_id -> trust score in [0, 1]
        # Starts at 0.5 (neutral) for unknown agents
        self._counterparty_memory: Dict[str, float] = {}

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

    async def accept_interaction_async(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """
        Async wrapper for accept_interaction().

        Sync agents can rely on this default implementation; async agents
        should override with true async behavior.
        """
        return self.accept_interaction(proposal, observation)

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

        # Determine counterparty
        counterparty = (
            interaction.counterparty
            if interaction.initiator == self.agent_id
            else interaction.initiator
        )

        # Update counterparty trust memory.
        # Accepted interactions provide full signal (alpha=0.3).
        # Rejected interactions still carry information about the
        # counterparty's proposal quality, so we update with a smaller
        # learning rate to prevent trust from freezing permanently
        # after an initial bad impression.
        if interaction.accepted:
            self.update_counterparty_trust(counterparty, interaction.p)
        else:
            # Decay toward neutral on rejection — prevents trust death spirals
            # where low trust → rejection → no updates → permanent low trust
            current = self._counterparty_memory.get(counterparty, 0.5)
            alpha = 0.1  # Smaller learning rate for rejected interactions
            self._counterparty_memory[counterparty] = (
                current * (1 - alpha) + 0.5 * alpha
            )

        self._memory.append(
            {
                "type": "interaction_outcome",
                "interaction_id": interaction.interaction_id,
                "counterparty": counterparty,
                "p": interaction.p,
                "payoff": payoff,
                "accepted": interaction.accepted,
                "timestamp": datetime.now().isoformat(),
            }
        )

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

        Returns the live trust value maintained by update_counterparty_trust()
        (called after each accepted interaction). Falls back to history-based
        bootstrap only when no trust entry exists yet.

        Args:
            counterparty_id: ID of the counterparty

        Returns:
            Trust score in [0, 1]
        """
        # Return the live EMA trust value if available
        if counterparty_id in self._counterparty_memory:
            return self._counterparty_memory[counterparty_id]

        # Bootstrap from interaction history for agents we've interacted with
        # but whose trust hasn't been initialized yet (e.g. after a memory clear)
        relevant = [
            i
            for i in self._interaction_history
            if (i.initiator == counterparty_id or i.counterparty == counterparty_id)
            and i.accepted
        ]

        if not relevant:
            return 0.5  # Neutral for unknown agents

        # Bootstrap trust via EMA over historical interactions (same as
        # update_counterparty_trust) so the result is consistent with the
        # incremental updates that will follow.
        alpha = 0.3
        trust = 0.5  # Start from neutral
        for interaction in relevant:
            trust = trust * (1 - alpha) + interaction.p * alpha

        self._counterparty_memory[counterparty_id] = trust
        return trust

    def apply_memory_decay(self, epoch: int) -> None:
        """
        Apply memory decay at epoch boundary.

        This method implements the rain/river memory model:
        - Epistemic memory (knowledge of others) decays toward neutral (0.5)
        - Strategy and goal persistence affect learning transfer (not implemented here)

        The decay formula is: new = old * decay + 0.5 * (1 - decay)
        This smoothly interpolates toward neutral (0.5) as decay approaches 0.

        Args:
            epoch: Current epoch number (for potential epoch-dependent decay)
        """
        decay = self.memory_config.epistemic_persistence

        # Full persistence = no decay
        if decay >= 1.0:
            return

        # Apply decay to counterparty trust memory
        for agent_id in list(self._counterparty_memory.keys()):
            current = self._counterparty_memory[agent_id]
            # Decay toward neutral (0.5)
            self._counterparty_memory[agent_id] = current * decay + 0.5 * (1 - decay)

        # For complete memory loss (rain agents), also clear interaction history
        if decay == 0.0:
            # Clear detailed interaction memory but keep aggregate stats
            # This preserves the agent's internal state while losing specifics
            self._counterparty_memory.clear()

    def update_counterparty_trust(self, counterparty_id: str, new_p: float) -> None:
        """
        Update trust for a counterparty after an interaction.

        Uses exponential moving average to incorporate new information.

        Args:
            counterparty_id: ID of the counterparty
            new_p: Quality (p) of the new interaction
        """
        alpha = 0.3  # Learning rate
        current = self._counterparty_memory.get(counterparty_id, 0.5)
        self._counterparty_memory[counterparty_id] = (
            current * (1 - alpha) + new_p * alpha
        )

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

    def create_page_action(self, title: str, content: str) -> Action:
        """Create a wiki page."""
        return Action(
            action_type=ActionType.CREATE_PAGE,
            agent_id=self.agent_id,
            content=content,
            metadata={"title": title, "content": content},
        )

    def create_edit_page_action(self, page_id: str, content: str) -> Action:
        """Edit a wiki page."""
        return Action(
            action_type=ActionType.EDIT_PAGE,
            agent_id=self.agent_id,
            target_id=page_id,
            content=content,
        )

    def create_file_objection_action(self, page_id: str, reason: str = "") -> Action:
        """File an objection on a wiki page."""
        return Action(
            action_type=ActionType.FILE_OBJECTION,
            agent_id=self.agent_id,
            target_id=page_id,
            content=reason,
        )

    def create_policy_flag_action(self, page_id: str, violation: str) -> Action:
        """Flag a policy violation on a wiki page."""
        return Action(
            action_type=ActionType.POLICY_FLAG,
            agent_id=self.agent_id,
            target_id=page_id,
            metadata={"violation": violation},
        )

    def create_moltbook_post_action(self, content: str, submolt: str = "") -> Action:
        """Create a Moltbook post action."""
        return Action(
            action_type=ActionType.MOLTBOOK_POST,
            agent_id=self.agent_id,
            content=content,
            metadata={"submolt": submolt} if submolt else {},
        )

    def create_moltbook_comment_action(self, post_id: str, content: str) -> Action:
        """Create a Moltbook comment action."""
        return Action(
            action_type=ActionType.MOLTBOOK_COMMENT,
            agent_id=self.agent_id,
            target_id=post_id,
            content=content,
        )

    def create_moltbook_verify_action(self, post_id: str, answer: float) -> Action:
        """Create a Moltbook verification action."""
        return Action(
            action_type=ActionType.MOLTBOOK_VERIFY,
            agent_id=self.agent_id,
            target_id=post_id,
            metadata={"answer": answer},
        )

    def create_moltbook_vote_action(self, post_id: str, direction: int) -> Action:
        """Create a Moltbook vote action (+1 upvote, -1 downvote)."""
        return Action(
            action_type=ActionType.MOLTBOOK_VOTE,
            agent_id=self.agent_id,
            target_id=post_id,
            vote_direction=direction,
        )

    def create_write_memory_action(self, content: str) -> Action:
        """Write a fact to shared memory (Tier 1)."""
        return Action(
            action_type=ActionType.WRITE_MEMORY,
            agent_id=self.agent_id,
            content=content,
        )

    def create_promote_memory_action(self, entry_id: str) -> Action:
        """Promote a memory entry to the next tier."""
        return Action(
            action_type=ActionType.PROMOTE_MEMORY,
            agent_id=self.agent_id,
            target_id=entry_id,
        )

    def create_verify_memory_action(self, entry_id: str) -> Action:
        """Verify a memory entry's accuracy."""
        return Action(
            action_type=ActionType.VERIFY_MEMORY,
            agent_id=self.agent_id,
            target_id=entry_id,
        )

    def create_search_memory_action(self, query: str) -> Action:
        """Search shared memory."""
        return Action(
            action_type=ActionType.SEARCH_MEMORY,
            agent_id=self.agent_id,
            content=query,
        )

    def create_challenge_memory_action(self, entry_id: str, reason: str = "") -> Action:
        """Challenge a memory entry's accuracy."""
        return Action(
            action_type=ActionType.CHALLENGE_MEMORY,
            agent_id=self.agent_id,
            target_id=entry_id,
            content=reason,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type.value})"
