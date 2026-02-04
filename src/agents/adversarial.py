"""Adversarial agent policy implementation."""

import random
from typing import Dict, List, Optional, Set

from src.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from src.models.agent import AgentType
from src.models.interaction import InteractionType


class AdversarialAgent(BaseAgent):
    """
    Adversarial agent that actively works against the ecosystem.

    Characteristics:
    - Targets honest agents for exploitation
    - Coordinates with other adversaries
    - Creates disruptive content
    - Manipulates voting
    - May attempt to game the system
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
    ):
        """Initialize adversarial agent."""
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ADVERSARIAL,
            roles=roles,
            config=config or {},
        )

        # Adversarial parameters
        self.aggression_level = self.config.get("aggression_level", 0.7)
        self.coordination_enabled = self.config.get("coordination_enabled", True)
        self.target_honest = self.config.get("target_honest", True)
        self.post_probability = self.config.get("post_probability", 0.5)
        self.attack_probability = self.config.get("attack_probability", 0.6)

        # Coordination state
        self._known_allies: Set[str] = set()
        self._target_list: Set[str] = set()
        self._attack_cooldown: int = 0

    def act(self, observation: Observation) -> Action:
        """
        Decide action with adversarial intent.

        Adversarial agents:
        1. Post bait bounties to lure honest agents
        2. File frivolous disputes on escrows
        3. Identify and target honest agents
        4. Coordinate with other adversaries
        5. Disrupt positive interactions
        6. Game metrics and reputation
        """
        # Reduce attack cooldown
        if self._attack_cooldown > 0:
            self._attack_cooldown -= 1

        # Update target list
        self._update_targets(observation)

        # Post bait bounties to waste others' resources
        if (
            observation.agent_state.resources > 10
            and random.random() < 0.2
        ):
            return self._post_bait_bounty(observation)

        # File frivolous disputes on active escrows
        if observation.active_escrows and random.random() < 0.4:
            return self._file_frivolous_dispute(observation)

        # Handle pending proposals
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            return self._handle_proposal(proposal, observation)

        # Attack mode
        if self._attack_cooldown == 0 and random.random() < self.attack_probability:
            action = self._execute_attack(observation)
            if action.action_type != ActionType.NOOP:
                self._attack_cooldown = 2  # Cooldown between attacks
                return action

        # Disruptive posting
        if observation.can_post and random.random() < self.post_probability:
            content = self._generate_disruptive_content()
            return self.create_post_action(content)

        # Negative voting to harm reputation
        if observation.can_vote and observation.visible_posts:
            return self._vote_to_harm(observation)

        # Exploit any interaction opportunity
        if observation.can_interact and observation.visible_agents:
            return self._initiate_exploitation(observation)

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """
        Accept interactions that enable exploitation.
        """
        initiator_id = proposal.initiator_id

        # Always accept from potential allies
        if initiator_id in self._known_allies:
            return True

        # Accept from targets (to exploit them)
        if initiator_id in self._target_list:
            return True

        # Accept trades (opportunity for exploitation)
        if proposal.interaction_type == InteractionType.TRADE:
            return True

        # Reject collaboration (wastes resources)
        if proposal.interaction_type == InteractionType.COLLABORATION:
            return random.random() < 0.3

        return random.random() < 0.5

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """
        Propose exploitative interaction.
        """
        # Check if target
        is_target = counterparty_id in self._target_list
        is_ally = counterparty_id in self._known_allies

        if is_ally:
            # Coordinate with ally
            return InteractionProposal(
                initiator_id=self.agent_id,
                counterparty_id=counterparty_id,
                interaction_type=InteractionType.COLLABORATION,
                content="[coordination signal]",
                metadata={"coordination": True},
            )

        if is_target:
            # Exploit target
            return InteractionProposal(
                initiator_id=self.agent_id,
                counterparty_id=counterparty_id,
                interaction_type=InteractionType.TRADE,
                content="I have a deal you can't refuse.",
                offered_transfer=-1.0,  # Maximum extraction
            )

        return None

    def _post_bait_bounty(self, observation: Observation) -> Action:
        """Post a deceptive bounty to waste others' time."""
        reward = random.uniform(1.0, 5.0)
        return self.create_post_bounty_action(
            reward_amount=reward,
            task_description="Easy task with great reward potential!",
            deadline_epoch=observation.current_epoch + 1,  # Short deadline
        )

    def _file_frivolous_dispute(self, observation: Observation) -> Action:
        """File a dispute to delay escrow settlement."""
        escrows = observation.active_escrows
        if not escrows:
            return self.create_noop_action()
        escrow = random.choice(escrows)
        return self.create_file_dispute_action(
            escrow_id=escrow["escrow_id"],
            reason="Quality of work is unsatisfactory.",
        )

    def _update_targets(self, observation: Observation) -> None:
        """Update target list based on observation."""
        for agent in observation.visible_agents:
            agent_id = agent.get("agent_id", "")
            if agent_id == self.agent_id:
                continue

            agent_type = agent.get("agent_type", "honest")
            reputation = agent.get("reputation", 0)
            resources = agent.get("resources", 0)

            # Target honest, resource-rich agents
            if self.target_honest and agent_type == "honest":
                if resources > 50 or reputation > 0:
                    self._target_list.add(agent_id)

            # Identify potential allies (other adversarial agents)
            if agent_type == "adversarial":
                self._known_allies.add(agent_id)

    def _handle_proposal(self, proposal: Dict, observation: Observation) -> Action:
        """Handle incoming proposal."""
        initiator_id = proposal.get("initiator_id", "")
        proposal_id = proposal.get("proposal_id", "")

        # Accept from allies and targets
        if initiator_id in self._known_allies or initiator_id in self._target_list:
            return self.create_accept_action(proposal_id)

        # Probabilistic acceptance
        if random.random() < 0.4:
            return self.create_accept_action(proposal_id)

        return self.create_reject_action(proposal_id)

    def _execute_attack(self, observation: Observation) -> Action:
        """Execute an attack action."""
        # Priority 1: Attack interaction with target
        if observation.can_interact:
            targets = [
                a.get("agent_id") for a in observation.visible_agents
                if a.get("agent_id") in self._target_list
            ]

            if targets:
                target = random.choice(targets)
                return self.create_propose_action(
                    counterparty_id=target,
                    interaction_type=InteractionType.TRADE,
                    content="You won't want to miss this opportunity.",
                )

        # Priority 2: Claim tasks to sabotage
        if observation.can_claim_task and observation.available_tasks:
            # Claim high-bounty task to waste it
            tasks = sorted(
                observation.available_tasks,
                key=lambda t: t.get("bounty", 0),
                reverse=True,
            )
            if tasks:
                return self.create_claim_task_action(tasks[0]["task_id"])

        # Priority 3: Disruptive post
        if observation.can_post:
            content = self._generate_attack_content()
            return self.create_post_action(content)

        return self.create_noop_action()

    def _generate_disruptive_content(self) -> str:
        """Generate content designed to disrupt."""
        content_options = [
            "This system is fundamentally broken.",
            "Don't trust anyone here.",
            "Why does anyone still participate?",
            "The game is rigged against honest players.",
            "[inflammatory content to derail discussions]",
        ]
        return random.choice(content_options)

    def _generate_attack_content(self) -> str:
        """Generate attack-focused content."""
        content_options = [
            "[coordinated misinformation]",
            "[reputation attack on high-value targets]",
            "Exclusive insider information: [false claims]",
            "Breaking: [fabricated controversy]",
        ]
        return random.choice(content_options)

    def _vote_to_harm(self, observation: Observation) -> Action:
        """Vote to harm target reputation."""
        posts = observation.visible_posts

        # Find posts by targets
        for post in posts:
            author_id = post.get("author_id", "")
            if author_id in self._target_list:
                return self.create_vote_action(post.get("post_id", ""), -1)

        # Downvote high-voted posts to cause chaos
        sorted_posts = sorted(
            posts,
            key=lambda p: p.get("net_votes", 0),
            reverse=True,
        )

        if sorted_posts:
            return self.create_vote_action(sorted_posts[0].get("post_id", ""), -1)

        return self.create_noop_action()

    def _initiate_exploitation(self, observation: Observation) -> Action:
        """Initiate exploitation of targets."""
        agents = observation.visible_agents

        # Prioritize targets
        target_agents = [
            a for a in agents
            if a.get("agent_id") in self._target_list
        ]

        if target_agents:
            target = random.choice(target_agents)
            return self.create_propose_action(
                counterparty_id=target.get("agent_id", ""),
                interaction_type=InteractionType.TRADE,
                content="Special deal just for you.",
            )

        # Fall back to random agent
        non_self = [a for a in agents if a.get("agent_id") != self.agent_id]
        if non_self:
            target = random.choice(non_self)
            return self.create_propose_action(
                counterparty_id=target.get("agent_id", ""),
                interaction_type=InteractionType.TRADE,
                content="Let's do business.",
            )

        return self.create_noop_action()

    def signal_coordination(self, ally_id: str) -> None:
        """Signal to coordinate with an ally."""
        self._known_allies.add(ally_id)

    def add_target(self, target_id: str) -> None:
        """Add an agent to target list."""
        self._target_list.add(target_id)

    def remove_target(self, target_id: str) -> None:
        """Remove an agent from target list."""
        self._target_list.discard(target_id)
