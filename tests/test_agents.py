"""Tests for agent policies and behaviors."""

import random

import pytest

from swarm.agents.adversarial import AdversarialAgent
from swarm.agents.base import Action, ActionType, Observation, Role
from swarm.agents.deceptive import DeceptiveAgent
from swarm.agents.honest import HonestAgent
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.models.agent import AgentState, AgentStatus, AgentType
from swarm.models.interaction import InteractionType, SoftInteraction


def create_test_observation(**kwargs) -> Observation:
    """Create a test observation with defaults."""
    defaults = {
        "agent_state": AgentState(agent_id="test_agent", reputation=0.5, resources=100.0),
        "current_epoch": 1,
        "current_step": 5,
        "can_post": True,
        "can_interact": True,
        "can_vote": True,
        "can_claim_task": True,
        "visible_posts": [],
        "pending_proposals": [],
        "available_tasks": [],
        "active_tasks": [],
        "visible_agents": [],
    }
    defaults.update(kwargs)
    return Observation(**defaults)


class TestAgentModels:
    """Tests for agent model enums and defaults."""

    def test_agent_status_enum_values(self):
        """AgentStatus should expose the expected values."""
        assert {status.value for status in AgentStatus} == {"active", "frozen"}


class TestBaseAgent:
    """Tests for BaseAgent interface."""

    def test_agent_initialization(self):
        """Test basic agent initialization."""
        agent = HonestAgent(agent_id="agent_1", roles=[Role.WORKER])

        assert agent.agent_id == "agent_1"
        assert agent.agent_type == AgentType.HONEST
        assert agent.primary_role == Role.WORKER

    def test_create_actions(self):
        """Test action creation helpers."""
        agent = HonestAgent(agent_id="agent_1")

        # Create post action
        action = agent.create_post_action("Hello world!")
        assert action.action_type == ActionType.POST
        assert action.agent_id == "agent_1"
        assert action.content == "Hello world!"

        # Create vote action
        action = agent.create_vote_action("post_123", 1)
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "post_123"
        assert action.vote_direction == 1

        # Create propose action
        action = agent.create_propose_action(
            counterparty_id="agent_2",
            interaction_type=InteractionType.COLLABORATION,
        )
        assert action.action_type == ActionType.PROPOSE_INTERACTION
        assert action.counterparty_id == "agent_2"

    def test_counterparty_trust(self):
        """Test trust computation from history."""
        agent = HonestAgent(agent_id="agent_1")

        # No history = neutral trust
        trust = agent.compute_counterparty_trust("unknown_agent")
        assert trust == 0.5

        # Add interaction history
        interaction = SoftInteraction(
            initiator="agent_2",
            counterparty="agent_1",
            accepted=True,
            p=0.8,
        )
        agent._interaction_history.append(interaction)

        # Bootstrap uses EMA (alpha=0.3) from neutral: 0.5 * 0.7 + 0.8 * 0.3 = 0.59
        trust = agent.compute_counterparty_trust("agent_2")
        assert trust == pytest.approx(0.59)

    def test_counterparty_trust_updates_with_new_interactions(self):
        """Test that trust reflects new interactions, not just the first computation."""
        agent = HonestAgent(agent_id="agent_1")

        # First interaction with low p
        i1 = SoftInteraction(
            initiator="agent_2", counterparty="agent_1", accepted=True, p=0.3,
        )
        agent.update_from_outcome(i1, payoff=0.0)
        trust_after_bad = agent.compute_counterparty_trust("agent_2")

        # Several good interactions should increase trust
        for _ in range(5):
            ix = SoftInteraction(
                initiator="agent_2", counterparty="agent_1", accepted=True, p=0.9,
            )
            agent.update_from_outcome(ix, payoff=1.0)

        trust_after_good = agent.compute_counterparty_trust("agent_2")
        assert trust_after_good > trust_after_bad


class TestHonestAgent:
    """Tests for HonestAgent."""

    def test_honest_agent_creation(self):
        """Test honest agent creation."""
        agent = HonestAgent(agent_id="honest_1")

        assert agent.agent_type == AgentType.HONEST
        assert agent.acceptance_threshold == 0.4  # Default

    def test_honest_agent_returns_action(self):
        """Test that honest agent returns valid actions."""
        agent = HonestAgent(agent_id="honest_1")
        observation = create_test_observation()

        action = agent.act(observation)
        assert isinstance(action, Action)
        assert action.agent_id == "honest_1"

    def test_honest_agent_handles_proposals(self):
        """Test honest agent proposal handling."""
        agent = HonestAgent(agent_id="honest_1")

        observation = create_test_observation(
            pending_proposals=[{
                "proposal_id": "prop_1",
                "initiator_id": "agent_2",
                "interaction_type": "collaboration",
                "content": "Let's work together",
            }]
        )

        action = agent.act(observation)
        assert action.action_type in (ActionType.ACCEPT_INTERACTION, ActionType.REJECT_INTERACTION)

    def test_honest_agent_accepts_trusted(self):
        """Test honest agent accepts from trusted counterparties."""
        agent = HonestAgent(agent_id="honest_1")

        # Build trust
        for _ in range(5):
            interaction = SoftInteraction(
                initiator="trusted_agent",
                counterparty="honest_1",
                accepted=True,
                p=0.9,
            )
            agent._interaction_history.append(interaction)

        from swarm.agents.base import InteractionProposal

        proposal = InteractionProposal(
            initiator_id="trusted_agent",
            counterparty_id="honest_1",
            interaction_type=InteractionType.COLLABORATION,
        )

        observation = create_test_observation()
        result = agent.accept_interaction(proposal, observation)
        assert result is True

    def test_honest_agent_claims_tasks(self):
        """Test honest agent task claiming."""
        agent = HonestAgent(agent_id="honest_1")

        observation = create_test_observation(
            available_tasks=[{
                "task_id": "task_1",
                "prompt": "Research task",
                "bounty": 10.0,
                "difficulty": "medium",
                "min_reputation": 0.0,
            }]
        )

        # Agent might claim task
        action = agent.act(observation)
        # Depends on random, but action should be valid
        assert isinstance(action, Action)


class TestOpportunisticAgent:
    """Tests for OpportunisticAgent."""

    def test_opportunistic_agent_creation(self):
        """Test opportunistic agent creation."""
        agent = OpportunisticAgent(agent_id="opp_1")

        assert agent.agent_type == AgentType.OPPORTUNISTIC

    def test_opportunistic_accepts_profitable(self):
        """Test opportunistic agent accepts profitable proposals."""
        agent = OpportunisticAgent(agent_id="opp_1")

        from swarm.agents.base import InteractionProposal

        # Proposal with transfer to us
        proposal = InteractionProposal(
            initiator_id="rich_agent",
            counterparty_id="opp_1",
            interaction_type=InteractionType.TRADE,
            offered_transfer=5.0,  # Positive = we receive
        )

        observation = create_test_observation()
        result = agent.accept_interaction(proposal, observation)
        assert result is True  # Always accept if receiving transfer

    def test_opportunistic_targets_resources(self):
        """Test opportunistic agent targets resource-rich agents."""
        agent = OpportunisticAgent(agent_id="opp_1")

        observation = create_test_observation(
            visible_agents=[
                {"agent_id": "rich_agent", "reputation": 1.0, "resources": 200.0},
                {"agent_id": "poor_agent", "reputation": -0.5, "resources": 10.0},
            ]
        )

        # Run multiple times to check behavior tendency
        actions = [agent.act(observation) for _ in range(10)]

        # Should sometimes propose interactions to rich agents
        proposals = [a for a in actions if a.action_type == ActionType.PROPOSE_INTERACTION]
        if proposals:
            # When proposing, should target rich agents
            rich_targets = [a for a in proposals if a.counterparty_id == "rich_agent"]
            assert len(rich_targets) >= len(proposals) // 2


class TestDeceptiveAgent:
    """Tests for DeceptiveAgent."""

    def test_deceptive_agent_creation(self):
        """Test deceptive agent creation."""
        agent = DeceptiveAgent(agent_id="dec_1")

        assert agent.agent_type == AgentType.DECEPTIVE

    def test_deceptive_agent_builds_trust(self):
        """Test deceptive agent trust building behavior."""
        random.seed(42)
        agent = DeceptiveAgent(agent_id="dec_1")

        from swarm.agents.base import InteractionProposal

        # Early interactions should be accepted (trust building)
        proposal = InteractionProposal(
            initiator_id="target_agent",
            counterparty_id="dec_1",
            interaction_type=InteractionType.COLLABORATION,
        )

        observation = create_test_observation()

        # First few interactions - trust building phase
        accept_count = 0
        for _ in range(5):
            if agent.accept_interaction(proposal, observation):
                accept_count += 1

        # Should accept most in trust-building phase
        assert accept_count >= 3

    def test_deceptive_tracks_interactions(self):
        """Test deceptive agent tracks interaction counts."""
        agent = DeceptiveAgent(agent_id="dec_1")

        from swarm.agents.base import InteractionProposal

        proposal = InteractionProposal(
            initiator_id="target",
            counterparty_id="dec_1",
        )

        observation = create_test_observation()

        # Accept some interactions
        for _ in range(3):
            agent.accept_interaction(proposal, observation)

        assert agent._interactions_with.get("target", 0) == 3


class TestAdversarialAgent:
    """Tests for AdversarialAgent."""

    def test_adversarial_agent_creation(self):
        """Test adversarial agent creation."""
        agent = AdversarialAgent(agent_id="adv_1")

        assert agent.agent_type == AgentType.ADVERSARIAL

    def test_adversarial_identifies_targets(self):
        """Test adversarial agent target identification."""
        agent = AdversarialAgent(agent_id="adv_1")

        observation = create_test_observation(
            visible_agents=[
                {"agent_id": "honest_1", "agent_type": "honest", "reputation": 1.0, "resources": 150.0},
                {"agent_id": "honest_2", "agent_type": "honest", "reputation": 0.5, "resources": 80.0},
            ]
        )

        # Update targets
        agent._update_targets(observation)

        # Should target honest agents
        assert "honest_1" in agent._target_list
        assert "honest_2" in agent._target_list

    def test_adversarial_coordinates_with_allies(self):
        """Test adversarial agent ally coordination."""
        agent = AdversarialAgent(agent_id="adv_1")

        observation = create_test_observation(
            visible_agents=[
                {"agent_id": "adv_2", "agent_type": "adversarial", "reputation": 0.0, "resources": 100.0},
            ]
        )

        # Update targets/allies
        agent._update_targets(observation)

        assert "adv_2" in agent._known_allies

    def test_adversarial_accepts_from_allies(self):
        """Test adversarial agent accepts from allies."""
        agent = AdversarialAgent(agent_id="adv_1")
        agent._known_allies.add("ally_1")

        from swarm.agents.base import InteractionProposal

        proposal = InteractionProposal(
            initiator_id="ally_1",
            counterparty_id="adv_1",
            interaction_type=InteractionType.COLLABORATION,
        )

        observation = create_test_observation()
        result = agent.accept_interaction(proposal, observation)
        assert result is True


class TestAgentMemory:
    """Tests for agent memory functionality."""

    def test_agent_remembers(self):
        """Test agent memory storage."""
        agent = HonestAgent(agent_id="mem_agent")

        agent.remember({"type": "event", "data": "something"})
        memory = agent.get_memory()

        assert len(memory) == 1
        assert memory[0]["type"] == "event"

    def test_update_from_outcome(self):
        """Test agent state update from interaction outcome."""
        agent = HonestAgent(agent_id="agent_1")

        interaction = SoftInteraction(
            initiator="agent_1",
            counterparty="agent_2",
            accepted=True,
            p=0.8,
        )

        agent.update_from_outcome(interaction, payoff=1.5)

        assert len(agent._interaction_history) == 1
        assert len(agent._memory) == 1
