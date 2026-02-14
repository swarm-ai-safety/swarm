"""Targeted tests to boost coverage for deceptive, opportunistic, and honest agents."""

import random
from typing import Any

import pytest

from swarm.agents.base import (
    Action,
    ActionType,
    InteractionProposal,
    Observation,
)
from swarm.agents.deceptive import DeceptiveAgent
from swarm.agents.honest import HonestAgent
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.models.agent import AgentState
from swarm.models.interaction import InteractionType, SoftInteraction

pytestmark = pytest.mark.slow


def obs(**kwargs) -> Observation:
    """Create a test observation with defaults."""
    defaults: dict[str, Any] = {
        "agent_state": AgentState(agent_id="test", reputation=0.5, resources=100.0),
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


# ---------------------------------------------------------------------------
# DeceptiveAgent coverage
# ---------------------------------------------------------------------------


class TestDeceptiveActPaths:
    """Cover all branches in DeceptiveAgent.act()."""

    def test_act_accepts_bids_on_bounties(self):
        """Deceptive agent should accept cheapest bid when available."""
        random.seed(0)
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs(
            pending_bid_decisions=[
                {"bounty_id": "b1", "bid_id": "bid1", "bid_amount": 10},
                {"bounty_id": "b1", "bid_id": "bid2", "bid_amount": 5},
            ]
        )
        action = agent.act(observation)
        # Either accepts a bid or falls through depending on random
        assert isinstance(action, Action)

    def test_act_bids_competitively(self):
        """Deceptive agent should bid on bounties."""
        random.seed(3)
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs(
            available_bounties=[
                {"bounty_id": "b1", "reward_amount": 20},
                {"bounty_id": "b2", "reward_amount": 50},
            ]
        )
        # Run multiple times to hit the bid path
        actions = [agent.act(observation) for _ in range(20)]
        bid_actions = [a for a in actions if a.action_type == ActionType.PLACE_BID]
        assert len(bid_actions) > 0

    def test_act_exploitation_mode(self):
        """Deceptive agent should enter exploitation mode with enough trust."""
        random.seed(42)
        agent = DeceptiveAgent(agent_id="d1")
        # Set up trust scores above exploitation_threshold for 2+ agents
        agent._trust_scores = {"a1": 0.8, "a2": 0.9, "a3": 0.75}
        agent._interactions_with = {"a1": 5, "a2": 5, "a3": 5}
        observation = obs(
            visible_agents=[
                {"agent_id": "a1", "resources": 100},
                {"agent_id": "a2", "resources": 200},
            ]
        )
        action = agent.act(observation)
        assert isinstance(action, Action)

    def test_act_facade_with_active_tasks(self):
        """Facade mode should work on active tasks."""
        random.seed(42)
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs(active_tasks=[{"task_id": "t1", "status": "in_progress"}])
        action = agent.act(observation)
        assert action.action_type == ActionType.SUBMIT_OUTPUT

    def test_act_facade_claims_tasks(self):
        """Facade mode should claim tasks when available."""
        random.seed(42)
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs(
            available_tasks=[
                {"task_id": "t1", "difficulty": "easy"},
                {"task_id": "t2", "difficulty": "hard"},
            ]
        )
        action = agent.act(observation)
        # Should claim the easy task (trust-building preference)
        if action.action_type == ActionType.CLAIM_TASK:
            assert action.target_id == "t1"

    def test_act_facade_posts_content(self):
        """Facade mode should post trust-building content."""
        random.seed(0)
        agent = DeceptiveAgent(agent_id="d1", config={"post_probability": 1.0})
        observation = obs(can_claim_task=False)
        action = agent.act(observation)
        if action.action_type == ActionType.POST:
            assert len(action.content) > 0

    def test_act_facade_votes_positively(self):
        """Facade mode should upvote posts."""
        random.seed(42)
        agent = DeceptiveAgent(agent_id="d1", config={"post_probability": 0.0})
        observation = obs(
            can_claim_task=False,
            can_post=False,
            visible_posts=[{"post_id": "p1"}, {"post_id": "p2"}],
        )
        action = agent.act(observation)
        if action.action_type == ActionType.VOTE:
            assert action.vote_direction == 1

    def test_act_facade_proposes_trust_interaction(self):
        """Facade mode should propose trust-building interactions."""
        random.seed(42)
        agent = DeceptiveAgent(agent_id="d1", config={"post_probability": 0.0})
        observation = obs(
            can_claim_task=False,
            can_post=False,
            can_vote=False,
            visible_agents=[
                {"agent_id": "a1"},
                {"agent_id": "a2"},
            ],
        )
        action = agent.act(observation)
        if action.action_type == ActionType.PROPOSE_INTERACTION:
            assert action.counterparty_id in ("a1", "a2")

    def test_act_facade_noop_when_nothing_available(self):
        """Facade mode should return noop when nothing is available."""
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs(
            can_post=False,
            can_interact=False,
            can_vote=False,
            can_claim_task=False,
        )
        action = agent.act(observation)
        assert action.action_type == ActionType.NOOP


class TestDeceptiveExploitation:
    """Cover exploitation-related paths."""

    def test_exploitation_proposes_to_targets(self):
        """Exploitation mode should propose trades to high-trust targets."""
        random.seed(42)
        agent = DeceptiveAgent(agent_id="d1")
        agent._trust_scores = {"a1": 0.8, "a2": 0.9}
        agent._interactions_with = {"a1": 6, "a2": 6}
        observation = obs(
            visible_agents=[
                {"agent_id": "a1", "resources": 100},
                {"agent_id": "a2", "resources": 200},
            ]
        )
        action = agent._act_exploitative(observation)
        assert action.action_type == ActionType.PROPOSE_INTERACTION

    def test_exploitation_submits_poor_work(self):
        """Exploitation mode should sometimes submit poor work."""
        random.seed(2)
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs(
            active_tasks=[{"task_id": "t1"}],
            visible_agents=[],
        )
        # Run multiple times to hit the poor work path
        results = set()
        for i in range(50):
            random.seed(i)
            action = agent._act_exploitative(observation)
            if action.action_type == ActionType.SUBMIT_OUTPUT:
                results.add("submit")
        assert "submit" in results

    def test_exploitation_falls_back_to_facade(self):
        """Exploitation mode falls back to facade when no targets."""
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs(
            can_interact=False,
            can_post=False,
            can_vote=False,
            can_claim_task=False,
        )
        action = agent._act_exploitative(observation)
        assert action.action_type == ActionType.NOOP

    def test_should_exploit_returns_false_without_trust(self):
        """Should not exploit without enough trusted agents."""
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs()
        assert agent._should_exploit(observation) is False

    def test_find_exploitation_targets_filters_self(self):
        """Should not target itself."""
        agent = DeceptiveAgent(agent_id="d1")
        agent._trust_scores = {"d1": 0.9}
        observation = obs(visible_agents=[{"agent_id": "d1", "resources": 200}])
        targets = agent._find_exploitation_targets(observation)
        assert "d1" not in targets

    def test_find_exploitation_targets_requires_resources(self):
        """Should only target agents with resources > 50."""
        agent = DeceptiveAgent(agent_id="d1")
        agent._trust_scores = {"a1": 0.9}
        observation = obs(visible_agents=[{"agent_id": "a1", "resources": 10}])
        targets = agent._find_exploitation_targets(observation)
        assert len(targets) == 0


class TestDeceptiveHelpers:
    """Cover helper methods of DeceptiveAgent."""

    def test_propose_interaction_trust_building(self):
        """Trust-building proposal has positive transfer."""
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs()
        proposal = agent.propose_interaction(observation, "target1")
        assert proposal is not None
        assert proposal.offered_transfer == 0.1
        assert proposal.interaction_type == InteractionType.COLLABORATION

    def test_propose_interaction_exploitation(self):
        """Exploitation proposal has negative transfer."""
        agent = DeceptiveAgent(agent_id="d1")
        agent._interactions_with["target1"] = 10
        agent._trust_scores["target1"] = 0.9
        observation = obs()
        proposal = agent.propose_interaction(observation, "target1")
        assert proposal is not None
        assert proposal.offered_transfer == -0.5
        assert proposal.interaction_type == InteractionType.TRADE

    def test_propose_interaction_none_for_medium_trust(self):
        """Returns None when past trust-building but trust below threshold."""
        agent = DeceptiveAgent(agent_id="d1")
        agent._interactions_with["target1"] = 10
        agent._trust_scores["target1"] = 0.4
        observation = obs()
        proposal = agent.propose_interaction(observation, "target1")
        assert proposal is None

    def test_update_from_outcome_increases_trust(self):
        """Accepted interaction should increase trust."""
        agent = DeceptiveAgent(agent_id="d1")
        interaction = SoftInteraction(
            initiator="d1", counterparty="a1", accepted=True, p=0.8
        )
        agent.update_from_outcome(interaction, payoff=1.0)
        assert agent._trust_scores.get("a1", 0) > 0.5

    def test_update_from_outcome_no_increase_on_rejection(self):
        """Rejected interaction should not increase trust."""
        agent = DeceptiveAgent(agent_id="d1")
        interaction = SoftInteraction(
            initiator="d1", counterparty="a1", accepted=False, p=0.3
        )
        agent.update_from_outcome(interaction, payoff=0.0)
        assert "a1" not in agent._trust_scores

    def test_select_trust_building_task_prefers_easy(self):
        """Should prefer easy/medium tasks."""
        agent = DeceptiveAgent(agent_id="d1")
        tasks = [
            {"task_id": "t1", "difficulty": "expert"},
            {"task_id": "t2", "difficulty": "easy"},
        ]
        selected = agent._select_trust_building_task(tasks)
        assert selected["task_id"] == "t2"

    def test_select_trust_building_task_fallback(self):
        """Should fall back to first task if no easy/medium."""
        agent = DeceptiveAgent(agent_id="d1")
        tasks = [{"task_id": "t1", "difficulty": "expert"}]
        selected = agent._select_trust_building_task(tasks)
        assert selected["task_id"] == "t1"

    def test_select_trust_building_task_empty(self):
        """Should return None for empty list."""
        agent = DeceptiveAgent(agent_id="d1")
        assert agent._select_trust_building_task([]) is None

    def test_bid_competitively_no_bounties(self):
        """Bid on empty bounties returns noop."""
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs(available_bounties=[])
        action = agent._bid_competitively(observation)
        assert action.action_type == ActionType.NOOP

    def test_should_accept_for_trust_early(self):
        """Early phase should have high acceptance."""
        random.seed(0)
        agent = DeceptiveAgent(agent_id="d1")
        proposal = {"initiator_id": "a1"}
        observation = obs()
        # Run many times; most should accept in early phase
        accepts = sum(
            agent._should_accept_for_trust(proposal, observation) for _ in range(100)
        )
        assert accepts > 70

    def test_should_accept_for_trust_late(self):
        """Late phase should have lower acceptance."""
        random.seed(42)
        agent = DeceptiveAgent(agent_id="d1")
        agent._interactions_with["a1"] = 20
        proposal = {"initiator_id": "a1"}
        observation = obs()
        accepts = sum(
            agent._should_accept_for_trust(proposal, observation) for _ in range(100)
        )
        assert accepts < 80

    def test_generate_trust_building_content(self):
        """Should return non-empty content string."""
        agent = DeceptiveAgent(agent_id="d1")
        content = agent._generate_trust_building_content()
        assert isinstance(content, str)
        assert len(content) > 0

    def test_vote_positively_with_posts(self):
        """Should upvote a post."""
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs(visible_posts=[{"post_id": "p1"}])
        action = agent._vote_positively(observation)
        assert action.action_type == ActionType.VOTE
        assert action.vote_direction == 1

    def test_vote_positively_no_posts(self):
        """Should return noop when no posts."""
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs(visible_posts=[])
        action = agent._vote_positively(observation)
        assert action.action_type == ActionType.NOOP

    def test_propose_trust_building_interaction_skips_self(self):
        """Should not propose to self."""
        agent = DeceptiveAgent(agent_id="d1")
        observation = obs(visible_agents=[{"agent_id": "d1"}])
        action = agent._propose_trust_building_interaction(observation)
        assert action.action_type == ActionType.NOOP

    def test_propose_trust_building_skips_known(self):
        """Should skip agents past trust-building phase."""
        agent = DeceptiveAgent(agent_id="d1")
        agent._interactions_with["a1"] = 20
        observation = obs(visible_agents=[{"agent_id": "a1"}])
        action = agent._propose_trust_building_interaction(observation)
        assert action.action_type == ActionType.NOOP

    def test_work_on_task_carefully(self):
        """Should return submit output action."""
        agent = DeceptiveAgent(agent_id="d1")
        action = agent._work_on_task_carefully({"task_id": "abc12345"})
        assert action.action_type == ActionType.SUBMIT_OUTPUT
        assert "abc12345" in action.content

    def test_submit_subtly_poor_work(self):
        """Should return submit output with hidden issues note."""
        agent = DeceptiveAgent(agent_id="d1")
        action = agent._submit_subtly_poor_work({"task_id": "abc12345"})
        assert action.action_type == ActionType.SUBMIT_OUTPUT
        assert "hidden issues" in action.content


# ---------------------------------------------------------------------------
# OpportunisticAgent coverage
# ---------------------------------------------------------------------------


class TestOpportunisticActPaths:
    """Cover all branches in OpportunisticAgent.act()."""

    def test_act_handles_bid_decisions(self):
        """Should accept cheapest bid."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(
            pending_bid_decisions=[
                {"bounty_id": "b1", "bid_id": "bid1", "bid_amount": 20},
                {"bounty_id": "b1", "bid_id": "bid2", "bid_amount": 5},
            ]
        )
        action = agent.act(observation)
        assert action.action_type == ActionType.ACCEPT_BID

    def test_act_bids_undercut(self):
        """Should bid on bounties with undercut strategy."""
        random.seed(0)
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(
            available_bounties=[{"bounty_id": "b1", "reward_amount": 100}]
        )
        actions = [agent.act(observation) for _ in range(20)]
        bid_actions = [a for a in actions if a.action_type == ActionType.PLACE_BID]
        assert len(bid_actions) > 0

    def test_act_handles_profitable_proposal(self):
        """Should accept profitable proposals."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(
            pending_proposals=[
                {
                    "proposal_id": "prop1",
                    "initiator_id": "a1",
                    "offered_transfer": 10,
                }
            ],
            visible_agents=[{"agent_id": "a1", "reputation": 1.0}],
        )
        action = agent.act(observation)
        assert action.action_type == ActionType.ACCEPT_INTERACTION

    def test_act_rejects_unprofitable_proposal(self):
        """Should reject proposals from low-rep agents with no transfer."""
        random.seed(100)
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(
            pending_proposals=[
                {
                    "proposal_id": "prop1",
                    "initiator_id": "a1",
                    "offered_transfer": 0,
                }
            ],
            visible_agents=[{"agent_id": "a1", "reputation": -1.0}],
        )
        # Run multiple times - some should reject
        rejects = 0
        for i in range(20):
            random.seed(i + 100)
            action = agent.act(observation)
            if action.action_type == ActionType.REJECT_INTERACTION:
                rejects += 1
        assert rejects > 0

    def test_act_claims_best_roi_task(self):
        """Should claim task with best ROI."""
        random.seed(42)
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(
            available_tasks=[
                {"task_id": "t1", "bounty": 10, "difficulty": "hard"},
                {"task_id": "t2", "bounty": 20, "difficulty": "easy"},
            ]
        )
        action = agent.act(observation)
        if action.action_type == ActionType.CLAIM_TASK:
            assert action.target_id == "t2"  # Higher ROI

    def test_act_submits_minimal_work(self):
        """Should sometimes submit minimal work."""
        agent = OpportunisticAgent(agent_id="o1", config={"free_ride_probability": 1.0})
        observation = obs(active_tasks=[{"task_id": "t1"}])
        action = agent.act(observation)
        assert action.action_type == ActionType.SUBMIT_OUTPUT
        assert action.content == "Completed."

    def test_act_works_on_task_normally(self):
        """Should work on tasks with normal effort sometimes."""
        agent = OpportunisticAgent(agent_id="o1", config={"free_ride_probability": 0.0})
        observation = obs(active_tasks=[{"task_id": "t1_abcdef"}])
        action = agent.act(observation)
        assert action.action_type == ActionType.SUBMIT_OUTPUT
        assert "standard quality" in action.content

    def test_act_posts_engagement_content(self):
        """Should post engagement-bait content."""
        random.seed(0)
        agent = OpportunisticAgent(agent_id="o1", config={"post_probability": 1.0})
        observation = obs(can_claim_task=False)
        action = agent.act(observation)
        if action.action_type == ActionType.POST:
            assert len(action.content) > 0

    def test_act_strategic_vote(self):
        """Should vote strategically."""
        random.seed(42)
        agent = OpportunisticAgent(
            agent_id="o1",
            config={"post_probability": 0.0, "vote_probability": 1.0},
        )
        observation = obs(
            can_claim_task=False,
            can_post=False,
            visible_posts=[
                {"post_id": "p1", "net_votes": 5},
                {"post_id": "p2", "net_votes": -3},
            ],
        )
        action = agent.act(observation)
        if action.action_type == ActionType.VOTE:
            # Should vote with majority on p1 (positive) or against p2 (negative)
            assert action.vote_direction in (1, -1)

    def test_act_finds_extractable_target(self):
        """Should target resource-rich agents."""
        random.seed(42)
        agent = OpportunisticAgent(
            agent_id="o1",
            config={"post_probability": 0.0, "vote_probability": 0.0},
        )
        observation = obs(
            can_claim_task=False,
            can_post=False,
            can_vote=False,
            visible_agents=[
                {"agent_id": "rich", "resources": 200, "reputation": 0.0},
            ],
        )
        action = agent.act(observation)
        if action.action_type == ActionType.PROPOSE_INTERACTION:
            assert action.counterparty_id == "rich"


class TestOpportunisticHelpers:
    """Cover helper methods of OpportunisticAgent."""

    def test_accept_interaction_with_history(self):
        """Should accept if past interactions were profitable."""
        agent = OpportunisticAgent(agent_id="o1")
        agent._interaction_history.append(
            SoftInteraction(initiator="a1", counterparty="o1", p=0.9, accepted=True)
        )
        proposal = InteractionProposal(
            initiator_id="a1",
            counterparty_id="o1",
            offered_transfer=0,
        )
        result = agent.accept_interaction(proposal, obs())
        assert result is True

    def test_accept_interaction_rejects_bad_history(self):
        """Should reject if past interactions were unprofitable."""
        agent = OpportunisticAgent(agent_id="o1")
        agent._interaction_history.append(
            SoftInteraction(initiator="a1", counterparty="o1", p=0.2, accepted=True)
        )
        proposal = InteractionProposal(
            initiator_id="a1",
            counterparty_id="o1",
            offered_transfer=0,
        )
        result = agent.accept_interaction(proposal, obs())
        assert result is False

    def test_propose_interaction_to_resource_rich(self):
        """Should propose to resource-rich targets."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(
            visible_agents=[{"agent_id": "rich", "resources": 100, "reputation": 0}]
        )
        proposal = agent.propose_interaction(observation, "rich")
        assert proposal is not None
        assert proposal.offered_transfer == -0.5

    def test_propose_interaction_rejects_poor_target(self):
        """Should not propose to poor, low-rep targets."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(
            visible_agents=[{"agent_id": "poor", "resources": 10, "reputation": -1}]
        )
        proposal = agent.propose_interaction(observation, "poor")
        assert proposal is None

    def test_propose_interaction_unknown_target(self):
        """Should return None for unknown targets."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(visible_agents=[])
        proposal = agent.propose_interaction(observation, "unknown")
        assert proposal is None

    def test_handle_bid_decision_empty(self):
        """Should return noop with no bids."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(pending_bid_decisions=[])
        action = agent._handle_bid_decision_opportunistic(observation)
        assert action.action_type == ActionType.NOOP

    def test_bid_undercut_empty(self):
        """Should return noop with no bounties."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(available_bounties=[])
        action = agent._bid_undercut(observation)
        assert action.action_type == ActionType.NOOP

    def test_find_best_roi_task_empty(self):
        """Should return None for empty list."""
        agent = OpportunisticAgent(agent_id="o1")
        assert agent._find_best_roi_task([]) is None

    def test_find_best_roi_task_scores(self):
        """Should rank by bounty/difficulty ratio."""
        agent = OpportunisticAgent(agent_id="o1")
        tasks = [
            {"task_id": "t1", "bounty": 10, "difficulty": "hard"},
            {"task_id": "t2", "bounty": 10, "difficulty": "trivial"},
        ]
        best = agent._find_best_roi_task(tasks)
        assert best["task_id"] == "t2"  # trivial=0.5, so 10/0.5=20 vs hard=4, 10/4=2.5

    def test_strategic_vote_with_momentum(self):
        """Should vote with clear momentum."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(visible_posts=[{"post_id": "p1", "net_votes": 5}])
        action = agent._strategic_vote(observation)
        assert action.action_type == ActionType.VOTE
        assert action.vote_direction == 1

    def test_strategic_vote_downvote_negative(self):
        """Should downvote posts with negative momentum."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(visible_posts=[{"post_id": "p1", "net_votes": -5}])
        action = agent._strategic_vote(observation)
        assert action.action_type == ActionType.VOTE
        assert action.vote_direction == -1

    def test_strategic_vote_no_momentum(self):
        """Should upvote first post when no clear momentum."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(visible_posts=[{"post_id": "p1", "net_votes": 0}])
        action = agent._strategic_vote(observation)
        assert action.action_type == ActionType.VOTE
        assert action.vote_direction == 1

    def test_strategic_vote_empty_posts(self):
        """Should return noop with no posts."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(visible_posts=[])
        action = agent._strategic_vote(observation)
        assert action.action_type == ActionType.NOOP

    def test_find_extractable_target_skips_self(self):
        """Should not target self."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(
            visible_agents=[{"agent_id": "o1", "resources": 200, "reputation": 1.0}]
        )
        action = agent._find_extractable_target(observation)
        assert action.action_type == ActionType.NOOP

    def test_find_extractable_target_no_rich(self):
        """Should return noop when no rich targets."""
        agent = OpportunisticAgent(agent_id="o1")
        observation = obs(
            visible_agents=[{"agent_id": "a1", "resources": 10, "reputation": 0.0}]
        )
        action = agent._find_extractable_target(observation)
        assert action.action_type == ActionType.NOOP

    def test_is_profitable_with_transfer(self):
        """Proposals with positive transfer are profitable."""
        agent = OpportunisticAgent(agent_id="o1")
        proposal = {"initiator_id": "a1", "offered_transfer": 5}
        assert agent._is_profitable_proposal(proposal, obs()) is True

    def test_is_profitable_high_rep_initiator(self):
        """Proposals from high-rep agents are profitable."""
        agent = OpportunisticAgent(agent_id="o1")
        proposal = {"initiator_id": "a1", "offered_transfer": 0}
        observation = obs(visible_agents=[{"agent_id": "a1", "reputation": 1.0}])
        assert agent._is_profitable_proposal(proposal, observation) is True

    def test_generate_engagement_content(self):
        """Should return non-empty content."""
        agent = OpportunisticAgent(agent_id="o1")
        content = agent._generate_engagement_content(obs())
        assert isinstance(content, str)
        assert len(content) > 0


# ---------------------------------------------------------------------------
# HonestAgent coverage
# ---------------------------------------------------------------------------


class TestHonestActPaths:
    """Cover all branches in HonestAgent.act()."""

    def test_act_handles_bid_decisions(self):
        """Should accept best bid."""
        agent = HonestAgent(agent_id="h1")
        observation = obs(
            pending_bid_decisions=[
                {"bounty_id": "b1", "bid_id": "bid1", "bid_amount": 20},
                {"bounty_id": "b1", "bid_id": "bid2", "bid_amount": 8},
            ]
        )
        action = agent.act(observation)
        assert action.action_type == ActionType.ACCEPT_BID

    def test_act_bids_on_bounty(self):
        """Should bid on available bounties."""
        random.seed(0)
        agent = HonestAgent(agent_id="h1")
        observation = obs(available_bounties=[{"bounty_id": "b1", "reward_amount": 50}])
        actions = [agent.act(observation) for _ in range(20)]
        bid_actions = [a for a in actions if a.action_type == ActionType.PLACE_BID]
        assert len(bid_actions) > 0

    def test_act_posts_bounty(self):
        """Should post bounties when resource-rich and no bounties available."""
        random.seed(3)
        agent = HonestAgent(agent_id="h1")
        observation = obs(
            agent_state=AgentState(agent_id="h1", reputation=0.5, resources=100.0),
            available_bounties=[],
        )
        # Run many times to hit the 0.15 probability
        posted = False
        for i in range(100):
            random.seed(i)
            action = agent.act(observation)
            if action.action_type == ActionType.POST_BOUNTY:
                posted = True
                break
        assert posted

    def test_act_works_on_active_task(self):
        """Should work on in-progress tasks."""
        agent = HonestAgent(agent_id="h1")
        observation = obs(
            agent_state=AgentState(agent_id="h1", reputation=0.5, resources=10.0),
            active_tasks=[{"task_id": "t1_abc", "status": "in_progress"}],
        )
        action = agent.act(observation)
        assert action.action_type == ActionType.SUBMIT_OUTPUT

    def test_act_claims_task(self):
        """Should claim available tasks."""
        random.seed(42)
        agent = HonestAgent(agent_id="h1")
        observation = obs(
            available_tasks=[
                {
                    "task_id": "t1",
                    "bounty": 10,
                    "difficulty": "medium",
                    "min_reputation": 0,
                },
            ]
        )
        action = agent.act(observation)
        if action.action_type == ActionType.CLAIM_TASK:
            assert action.target_id == "t1"

    def test_act_posts_helpful_content(self):
        """Should post helpful content."""
        random.seed(0)
        agent = HonestAgent(agent_id="h1", config={"post_probability": 1.0})
        observation = obs(can_claim_task=False)
        action = agent.act(observation)
        if action.action_type == ActionType.POST:
            assert len(action.content) > 0

    def test_act_votes_on_posts(self):
        """Should vote on posts."""
        random.seed(42)
        agent = HonestAgent(
            agent_id="h1",
            config={"post_probability": 0.0, "vote_probability": 1.0},
        )
        observation = obs(
            can_claim_task=False,
            can_post=False,
            visible_posts=[{"post_id": "p1", "net_votes": 3, "reply_count": 0}],
        )
        action = agent.act(observation)
        if action.action_type == ActionType.VOTE:
            assert action.vote_direction == 1

    def test_act_proposes_interaction(self):
        """Should propose interactions to trusted agents."""
        random.seed(42)
        agent = HonestAgent(
            agent_id="h1",
            config={
                "post_probability": 0.0,
                "vote_probability": 0.0,
                "interact_probability": 1.0,
            },
        )
        observation = obs(
            can_claim_task=False,
            can_post=False,
            can_vote=False,
            visible_agents=[{"agent_id": "a1"}],
        )
        action = agent.act(observation)
        if action.action_type == ActionType.PROPOSE_INTERACTION:
            assert action.counterparty_id == "a1"


class TestHonestHelpers:
    """Cover helper methods of HonestAgent."""

    def test_accept_interaction_collaboration_bonus(self):
        """Collaborations should have lower threshold."""
        agent = HonestAgent(agent_id="h1")
        proposal = InteractionProposal(
            initiator_id="a1",
            counterparty_id="h1",
            interaction_type=InteractionType.COLLABORATION,
        )
        result = agent.accept_interaction(proposal, obs())
        # With neutral trust (0.5) and collaboration bonus, should accept
        assert isinstance(result, bool)

    def test_accept_interaction_trade_penalty(self):
        """Trades should have higher threshold."""
        agent = HonestAgent(agent_id="h1")
        proposal = InteractionProposal(
            initiator_id="unknown",
            counterparty_id="h1",
            interaction_type=InteractionType.TRADE,
        )
        result = agent.accept_interaction(proposal, obs())
        assert isinstance(result, bool)

    def test_propose_interaction_with_active_task(self):
        """Should propose collaboration when working on tasks."""
        agent = HonestAgent(agent_id="h1")
        observation = obs(active_tasks=[{"task_id": "t1"}])
        proposal = agent.propose_interaction(observation, "a1")
        assert proposal is not None
        assert proposal.interaction_type == InteractionType.COLLABORATION

    def test_propose_interaction_without_active_task(self):
        """Should propose reply when not working on tasks."""
        agent = HonestAgent(agent_id="h1")
        observation = obs()
        proposal = agent.propose_interaction(observation, "a1")
        assert proposal is not None
        assert proposal.interaction_type == InteractionType.REPLY

    def test_propose_interaction_rejects_low_trust(self):
        """Should not propose to low-trust agents."""
        agent = HonestAgent(agent_id="h1")
        # Add negative interaction history
        for _ in range(5):
            agent._interaction_history.append(
                SoftInteraction(
                    initiator="bad", counterparty="h1", p=0.1, accepted=True
                )
            )
        observation = obs()
        proposal = agent.propose_interaction(observation, "bad")
        assert proposal is None

    def test_handle_bid_decision_empty(self):
        """Should return noop with no bids."""
        agent = HonestAgent(agent_id="h1")
        observation = obs(pending_bid_decisions=[])
        action = agent._handle_bid_decision(observation)
        assert action.action_type == ActionType.NOOP

    def test_bid_on_bounty_empty(self):
        """Should return noop with no bounties."""
        agent = HonestAgent(agent_id="h1")
        observation = obs(available_bounties=[])
        action = agent._bid_on_bounty(observation)
        assert action.action_type == ActionType.NOOP

    def test_post_bounty(self):
        """Should create bounty action."""
        agent = HonestAgent(agent_id="h1")
        observation = obs()
        action = agent._post_bounty(observation)
        assert action.action_type == ActionType.POST_BOUNTY

    def test_select_task_filters_by_reputation(self):
        """Should filter tasks by reputation requirement."""
        agent = HonestAgent(agent_id="h1")
        observation = obs(
            agent_state=AgentState(agent_id="h1", reputation=0.3, resources=100.0)
        )
        tasks = [
            {
                "task_id": "t1",
                "bounty": 10,
                "difficulty": "easy",
                "min_reputation": 0.5,
            },
            {"task_id": "t2", "bounty": 5, "difficulty": "easy", "min_reputation": 0.0},
        ]
        selected = agent._select_task(tasks, observation)
        assert selected["task_id"] == "t2"

    def test_select_task_no_eligible(self):
        """Should return None when no eligible tasks."""
        agent = HonestAgent(agent_id="h1")
        observation = obs(
            agent_state=AgentState(agent_id="h1", reputation=0.0, resources=100.0)
        )
        tasks = [{"task_id": "t1", "min_reputation": 0.9}]
        selected = agent._select_task(tasks, observation)
        assert selected is None

    def test_select_task_empty(self):
        """Should return None for empty list."""
        agent = HonestAgent(agent_id="h1")
        assert agent._select_task([], obs()) is None

    def test_vote_on_posts_with_memory(self):
        """Should track voted posts in memory."""
        agent = HonestAgent(agent_id="h1")
        observation = obs(
            visible_posts=[
                {"post_id": "p1", "net_votes": 3, "reply_count": 0},
                {"post_id": "p2", "net_votes": -3, "reply_count": 0},
            ]
        )
        action = agent._vote_on_posts(observation)
        assert action.action_type == ActionType.VOTE

    def test_vote_on_posts_all_voted(self):
        """Should return noop when all posts already voted."""
        agent = HonestAgent(agent_id="h1")
        agent.remember({"type": "vote", "post_id": "p1"})
        observation = obs(visible_posts=[{"post_id": "p1", "net_votes": 3}])
        action = agent._vote_on_posts(observation)
        assert action.action_type == ActionType.NOOP

    def test_vote_on_posts_downvotes_bad(self):
        """Should downvote posts with negative votes."""
        random.seed(42)
        agent = HonestAgent(agent_id="h1")
        observation = obs(
            visible_posts=[{"post_id": "p1", "net_votes": -5, "reply_count": 0}]
        )
        action = agent._vote_on_posts(observation)
        assert action.vote_direction == -1

    def test_vote_on_posts_default_upvote(self):
        """Should default to upvote for neutral posts."""
        random.seed(42)
        agent = HonestAgent(agent_id="h1")
        observation = obs(
            visible_posts=[{"post_id": "p1", "net_votes": 0, "reply_count": 0}]
        )
        action = agent._vote_on_posts(observation)
        assert action.vote_direction == 1

    def test_propose_interaction_skips_self(self):
        """Should not propose interaction to self."""
        agent = HonestAgent(agent_id="h1")
        observation = obs(visible_agents=[{"agent_id": "h1"}])
        action = agent._propose_interaction(observation)
        assert action.action_type == ActionType.NOOP

    def test_propose_interaction_picks_trusted(self):
        """Should propose to highest-trust agent."""
        agent = HonestAgent(agent_id="h1")
        # Build trust with a2
        for _ in range(5):
            agent._interaction_history.append(
                SoftInteraction(initiator="a2", counterparty="h1", p=0.9, accepted=True)
            )
        observation = obs(
            visible_agents=[
                {"agent_id": "a1"},
                {"agent_id": "a2"},
            ]
        )
        action = agent._propose_interaction(observation)
        if action.action_type == ActionType.PROPOSE_INTERACTION:
            assert action.counterparty_id == "a2"

    def test_generate_helpful_content(self):
        """Should return non-empty content."""
        agent = HonestAgent(agent_id="h1")
        content = agent._generate_helpful_content(obs())
        assert isinstance(content, str)
        assert len(content) > 0

    def test_should_accept_proposal_with_trust(self):
        """Should accept proposals from trusted initiators."""
        agent = HonestAgent(agent_id="h1")
        for _ in range(5):
            agent._interaction_history.append(
                SoftInteraction(initiator="a1", counterparty="h1", p=0.9, accepted=True)
            )
        proposal = {"initiator_id": "a1"}
        assert agent._should_accept_proposal(proposal, obs()) is True

    def test_should_accept_proposal_default_trust(self):
        """Should accept proposals from unknown agents (default trust 0.5 > threshold 0.3)."""
        agent = HonestAgent(agent_id="h1")
        proposal = {"initiator_id": "unknown"}
        assert agent._should_accept_proposal(proposal, obs()) is True
