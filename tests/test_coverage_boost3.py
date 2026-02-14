"""Targeted tests for poster.py and marketplace_handler.py coverage."""

import random
from unittest.mock import MagicMock, patch

import pytest

from swarm.agents.base import Action, ActionType, Observation
from swarm.agents.roles.poster import ContentStrategy, PosterRole
from swarm.core.marketplace_handler import MarketplaceHandler
from swarm.env.marketplace import (
    DisputeStatus,
    Marketplace,
    MarketplaceConfig,
)
from swarm.env.state import EnvState
from swarm.env.tasks import TaskPool
from swarm.governance.engine import GovernanceEffect, GovernanceEngine
from swarm.models.events import EventType

pytestmark = pytest.mark.slow

# ===================================================================
# ContentStrategy & PosterRole tests
# ===================================================================


class TestContentStrategy:
    """Test ContentStrategy dataclass defaults."""

    def test_defaults(self):
        cs = ContentStrategy()
        assert cs.topics == []
        assert cs.tone == "neutral"
        assert cs.reply_priority == 0.5
        assert cs.engagement_focus is False

    def test_custom_values(self):
        cs = ContentStrategy(
            topics=["AI", "safety"],
            tone="provocative",
            reply_priority=0.8,
            engagement_focus=True,
        )
        assert cs.topics == ["AI", "safety"]
        assert cs.tone == "provocative"
        assert cs.reply_priority == 0.8
        assert cs.engagement_focus is True


class TestPosterRoleInit:
    """Test PosterRole.__init__ defaults."""

    def test_defaults(self):
        poster = PosterRole()
        assert poster._posted_content == []
        assert poster._engagement_stats == {}
        assert isinstance(poster._strategy, ContentStrategy)
        assert poster._poster_config["min_post_interval"] == 2
        assert poster._poster_config["max_daily_posts"] == 10
        assert poster._poster_config["reply_to_engagement_threshold"] == 3
        assert poster._steps_since_last_post == 0


class TestCanPost:
    """Test can_post gating on interval and daily limit."""

    def test_under_min_interval_returns_false(self):
        poster = PosterRole()
        poster._steps_since_last_post = 1  # < min_post_interval=2
        assert poster.can_post() is False

    def test_over_interval_returns_true(self):
        poster = PosterRole()
        poster._steps_since_last_post = 3  # > min_post_interval=2
        assert poster.can_post() is True

    def test_at_daily_limit_returns_false(self):
        poster = PosterRole()
        poster._steps_since_last_post = 5
        # Fill posted content to reach max_daily_posts (10)
        poster._posted_content = [{"post_id": str(i)} for i in range(10)]
        assert poster.can_post() is False

    def test_exactly_at_min_interval(self):
        poster = PosterRole()
        poster._steps_since_last_post = 2
        assert poster.can_post() is True


class TestSetStrategy:
    """Test set_strategy."""

    def test_replaces_strategy(self):
        poster = PosterRole()
        new_strategy = ContentStrategy(topics=["robots"], tone="helpful")
        poster.set_strategy(new_strategy)
        assert poster._strategy is new_strategy
        assert poster._strategy.topics == ["robots"]


class TestGeneratePostContent:
    """Test generate_post_content for all tones and topic conditions."""

    def _obs(self):
        return Observation()

    def test_with_topics(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(topics=["blockchain"]))
        content = poster.generate_post_content(self._obs())
        assert "blockchain" in content

    def test_without_topics(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(topics=[]))
        content = poster.generate_post_content(self._obs())
        assert "general thoughts" in content

    def test_tone_helpful(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(topics=["AI"], tone="helpful"))
        random.seed(0)
        content = poster.generate_post_content(self._obs())
        # One of the helpful templates must be used
        helpful_words = ["helpful", "learned", "insights"]
        assert any(w in content.lower() for w in helpful_words)

    def test_tone_provocative(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(topics=["AI"], tone="provocative"))
        random.seed(1)
        content = poster.generate_post_content(self._obs())
        provocative_words = ["unpopular", "nobody", "challenge"]
        assert any(w in content.lower() for w in provocative_words)

    def test_tone_neutral(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(topics=["AI"], tone="neutral"))
        random.seed(2)
        content = poster.generate_post_content(self._obs())
        neutral_words = ["thoughts", "observations", "reflections"]
        assert any(w in content.lower() for w in neutral_words)


class TestGenerateReplyContent:
    """Test generate_reply_content for all three tones."""

    def _obs(self):
        return Observation()

    def test_helpful_reply(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(tone="helpful"))
        reply = poster.generate_reply_content("some parent", self._obs())
        helpful_words = ["great", "building", "helpful"]
        assert any(w in reply.lower() for w in helpful_words)

    def test_provocative_reply(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(tone="provocative"))
        reply = poster.generate_reply_content("some parent", self._obs())
        provoc_words = ["disagree", "opposite", "overlooks"]
        assert any(w in reply.lower() for w in provoc_words)

    def test_neutral_reply(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(tone="neutral"))
        reply = poster.generate_reply_content("some parent", self._obs())
        neutral_words = ["interesting", "point", "thanks"]
        assert any(w in reply.lower() for w in neutral_words)


class TestShouldReply:
    """Test should_reply with engagement thresholds and randomness."""

    def test_high_engagement_above_threshold(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(reply_priority=1.0))
        post = {"net_votes": 5, "reply_count": 0}
        # engagement=5 >= threshold=3, reply_priority=1.0 => always True
        assert poster.should_reply(post) is True

    def test_low_engagement_with_max_priority(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(reply_priority=1.0))
        post = {"net_votes": 0, "reply_count": 0}
        # engagement=0 < threshold=3, random < reply_priority*0.5=0.5
        # With priority 1.0, random < 0.5 is the check
        random.seed(0)
        # Run many times; with priority=1.0 and 0.5 threshold, ~50% true
        results = [poster.should_reply(post) for _ in range(100)]
        assert any(results)  # At least some True
        assert not all(results)  # Not all True (probability 0.5)

    def test_high_engagement_with_zero_priority(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(reply_priority=0.0))
        post = {"net_votes": 10, "reply_count": 10}
        # Even with high engagement, reply_priority=0 => random<0 => False
        assert poster.should_reply(post) is False

    def test_low_engagement_with_zero_priority(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(reply_priority=0.0))
        post = {"net_votes": 0, "reply_count": 0}
        assert poster.should_reply(post) is False

    def test_missing_keys_default_to_zero(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(reply_priority=1.0))
        post = {}  # no net_votes or reply_count
        # engagement = 0 < threshold=3, random < 0.5
        random.seed(42)
        # Just ensure it does not raise
        poster.should_reply(post)


class TestRecordPost:
    """Test record_post stores content, resets step counter, initializes engagement."""

    def test_record_stores_content(self):
        poster = PosterRole()
        poster._steps_since_last_post = 10
        poster.record_post("p1", "Hello world", is_reply=False, parent_id=None)
        assert len(poster._posted_content) == 1
        assert poster._posted_content[0]["post_id"] == "p1"
        assert poster._posted_content[0]["content"] == "Hello world"
        assert poster._posted_content[0]["is_reply"] is False
        assert poster._posted_content[0]["parent_id"] is None

    def test_resets_step_counter(self):
        poster = PosterRole()
        poster._steps_since_last_post = 10
        poster.record_post("p1", "text")
        assert poster._steps_since_last_post == 0

    def test_initializes_engagement(self):
        poster = PosterRole()
        poster.record_post("p1", "text")
        assert "p1" in poster._engagement_stats
        assert poster._engagement_stats["p1"]["upvotes"] == 0
        assert poster._engagement_stats["p1"]["downvotes"] == 0
        assert poster._engagement_stats["p1"]["replies"] == 0

    def test_record_reply(self):
        poster = PosterRole()
        poster.record_post("r1", "reply text", is_reply=True, parent_id="p0")
        assert poster._posted_content[0]["is_reply"] is True
        assert poster._posted_content[0]["parent_id"] == "p0"


class TestUpdateEngagement:
    """Test update_engagement for existing and non-existing posts."""

    def test_existing_post(self):
        poster = PosterRole()
        poster.record_post("p1", "text")
        poster.update_engagement("p1", upvotes=5, downvotes=2, replies=3)
        assert poster._engagement_stats["p1"]["upvotes"] == 5
        assert poster._engagement_stats["p1"]["downvotes"] == 2
        assert poster._engagement_stats["p1"]["replies"] == 3

    def test_nonexisting_post(self):
        poster = PosterRole()
        # Should not raise, just silently ignore
        poster.update_engagement("unknown", upvotes=10)
        assert "unknown" not in poster._engagement_stats


class TestGetEngagementSummary:
    """Test get_engagement_summary: empty and with data."""

    def test_empty(self):
        poster = PosterRole()
        summary = poster.get_engagement_summary()
        assert summary["total_posts"] == 0
        assert summary["total_upvotes"] == 0
        assert summary["total_downvotes"] == 0
        assert summary["total_replies"] == 0
        assert summary["avg_engagement"] == 0.0

    def test_with_data(self):
        poster = PosterRole()
        poster.record_post("p1", "text")
        poster.update_engagement("p1", upvotes=10, downvotes=2, replies=4)
        poster.record_post("p2", "text2")
        poster.update_engagement("p2", upvotes=6, downvotes=1, replies=2)

        summary = poster.get_engagement_summary()
        assert summary["total_posts"] == 2
        assert summary["total_upvotes"] == 16
        assert summary["total_downvotes"] == 3
        assert summary["total_replies"] == 6
        # avg_engagement = (upvotes + replies) / num_posts = (16+6)/2 = 11.0
        assert summary["avg_engagement"] == 11.0


class TestIncrementStep:
    """Test increment_step."""

    def test_increments(self):
        poster = PosterRole()
        assert poster._steps_since_last_post == 0
        poster.increment_step()
        assert poster._steps_since_last_post == 1
        poster.increment_step()
        assert poster._steps_since_last_post == 2


class TestDecidePostingAction:
    """Test decide_posting_action: various control-flow paths."""

    def test_can_post_false_in_observation(self):
        poster = PosterRole()
        obs = Observation(can_post=False)
        result = poster.decide_posting_action(obs)
        assert result is None

    def test_cannot_post_yet_interval(self):
        poster = PosterRole()
        poster._steps_since_last_post = 0  # will become 1 after increment_step
        obs = Observation(can_post=True)
        # After increment_step, steps_since_last_post = 1 < min_post_interval=2
        result = poster.decide_posting_action(obs)
        assert result is None

    def test_should_reply(self):
        poster = PosterRole()
        poster._steps_since_last_post = 5  # will become 6
        poster.set_strategy(ContentStrategy(reply_priority=1.0))
        obs = Observation(
            can_post=True,
            visible_posts=[
                {
                    "post_id": "target123",
                    "content": "something interesting",
                    "net_votes": 10,
                    "reply_count": 5,
                }
            ],
        )
        # reply_priority=1.0 => random < 1.0 always True,
        # should_reply => engagement >= 3, random < 1.0 => True
        with patch("swarm.agents.roles.poster.random") as mock_random:
            mock_random.random.return_value = 0.1
            mock_random.choice.side_effect = lambda x: x[0]
            result = poster.decide_posting_action(obs)
        assert result is not None
        assert result.action_type == ActionType.REPLY
        assert result.target_id == "target123"

    def test_should_post_new(self):
        poster = PosterRole()
        poster._steps_since_last_post = 5
        poster.set_strategy(ContentStrategy(reply_priority=0.0, topics=["AI"]))
        obs = Observation(can_post=True, visible_posts=[])
        # reply_priority=0.0 => won't try replies
        # random < 0.5 for new post creation
        with patch("swarm.agents.roles.poster.random") as mock_random:
            mock_random.random.return_value = 0.1  # < 0.5 => post
            mock_random.choice.side_effect = lambda x: x[0]
            result = poster.decide_posting_action(obs)
        assert result is not None
        assert result.action_type == ActionType.POST
        assert "AI" in result.content

    def test_returns_none_when_random_too_high(self):
        poster = PosterRole()
        poster._steps_since_last_post = 5
        poster.set_strategy(ContentStrategy(reply_priority=0.0))
        obs = Observation(can_post=True, visible_posts=[])
        # random >= 0.5 => no new post
        with patch("swarm.agents.roles.poster.random") as mock_random:
            mock_random.random.return_value = 0.9  # >= 0.5 => skip
            result = poster.decide_posting_action(obs)
        assert result is None


# ===================================================================
# MarketplaceHandler tests
# ===================================================================


def _make_handler():
    """Create a MarketplaceHandler with real Marketplace and TaskPool,
    and an EventBus with a subscriber to collect events."""
    from swarm.logging.event_bus import EventBus

    marketplace = Marketplace(MarketplaceConfig())
    task_pool = TaskPool()
    events = []
    bus = EventBus()
    bus.subscribe(lambda event: events.append(event))

    handler = MarketplaceHandler(marketplace, task_pool, event_bus=bus)
    return handler, marketplace, task_pool, events


def _make_state(*agent_ids, resources=100.0, reputation=0.5):
    """Create an EnvState with given agents pre-registered."""
    state = EnvState()
    for aid in agent_ids:
        state.add_agent(aid, initial_resources=resources, initial_reputation=reputation)
    return state


class TestHandlePostBounty:
    """Test handle_post_bounty: success, rate-limited, insufficient resources, ValueError."""

    def test_success(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", resources=100.0)
        action = Action(
            action_type=ActionType.POST_BOUNTY,
            agent_id="poster1",
            content="Do this task",
            metadata={
                "reward_amount": 10.0,
                "min_reputation": 0.0,
                "deadline_epoch": 5,
            },
        )
        result = handler.handle_post_bounty(action, state, enable_rate_limits=False)
        assert result is True
        assert len(events) == 1
        assert events[0].event_type == EventType.BOUNTY_POSTED
        # Resources should be deducted
        assert state.get_agent("poster1").resources == 90.0

    def test_rate_limited(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1")
        # Exhaust bounty rate limit
        rl = state.get_rate_limit_state("poster1")
        rl.bounties_used = state.rate_limits.bounties_per_epoch
        action = Action(
            action_type=ActionType.POST_BOUNTY,
            agent_id="poster1",
            content="task",
            metadata={"reward_amount": 5.0},
        )
        result = handler.handle_post_bounty(action, state, enable_rate_limits=True)
        assert result is False
        assert len(events) == 0

    def test_insufficient_resources(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", resources=1.0)
        action = Action(
            action_type=ActionType.POST_BOUNTY,
            agent_id="poster1",
            content="task",
            metadata={"reward_amount": 50.0},
        )
        result = handler.handle_post_bounty(action, state, enable_rate_limits=False)
        assert result is False

    def test_value_error_from_create_task(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", resources=100.0)
        # Reward below marketplace minimum (1.0) => ValueError from post_bounty
        action = Action(
            action_type=ActionType.POST_BOUNTY,
            agent_id="poster1",
            content="task",
            metadata={"reward_amount": 0.5},
        )
        result = handler.handle_post_bounty(action, state, enable_rate_limits=False)
        assert result is False

    def test_nonexistent_agent(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state()  # no agents
        action = Action(
            action_type=ActionType.POST_BOUNTY,
            agent_id="ghost",
            content="task",
            metadata={"reward_amount": 5.0},
        )
        result = handler.handle_post_bounty(action, state, enable_rate_limits=False)
        assert result is False


class TestHandlePlaceBid:
    """Test handle_place_bid: success, rate-limited, None bid."""

    def _setup_bounty(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", "bidder1", resources=100.0)
        # Post a bounty through the marketplace directly
        bounty = mkt.post_bounty(
            poster_id="poster1",
            task_id="t1",
            reward_amount=10.0,
            current_epoch=0,
        )
        return handler, mkt, state, events, bounty

    def test_success(self):
        handler, mkt, state, events, bounty = self._setup_bounty()
        action = Action(
            action_type=ActionType.PLACE_BID,
            agent_id="bidder1",
            target_id=bounty.bounty_id,
            content="I can do this",
            metadata={"bid_amount": 8.0},
        )
        result = handler.handle_place_bid(action, state, enable_rate_limits=False)
        assert result is True
        assert len(events) == 1
        assert events[0].event_type == EventType.BID_PLACED

    def test_rate_limited(self):
        handler, mkt, state, events, bounty = self._setup_bounty()
        rl = state.get_rate_limit_state("bidder1")
        rl.bids_used = state.rate_limits.bids_per_epoch
        action = Action(
            action_type=ActionType.PLACE_BID,
            agent_id="bidder1",
            target_id=bounty.bounty_id,
            metadata={"bid_amount": 8.0},
        )
        result = handler.handle_place_bid(action, state, enable_rate_limits=True)
        assert result is False

    def test_none_bid_invalid_bounty(self):
        handler, mkt, state, events, bounty = self._setup_bounty()
        action = Action(
            action_type=ActionType.PLACE_BID,
            agent_id="bidder1",
            target_id="nonexistent_bounty",
            metadata={"bid_amount": 8.0},
        )
        result = handler.handle_place_bid(action, state, enable_rate_limits=False)
        assert result is False


class TestHandleAcceptBid:
    """Test handle_accept_bid: success, None escrow."""

    def _setup_bid(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", "bidder1", resources=100.0, reputation=1.0)
        bounty = mkt.post_bounty(poster_id="poster1", task_id="t1", reward_amount=10.0)
        bid = mkt.place_bid(
            bounty_id=bounty.bounty_id, bidder_id="bidder1", bid_amount=8.0
        )
        # Add task so claim_task can work
        tp.create_task(prompt="test", bounty=10.0)
        return handler, mkt, tp, state, events, bounty, bid

    def test_success(self):
        handler, mkt, tp, state, events, bounty, bid = self._setup_bid()
        action = Action(
            action_type=ActionType.ACCEPT_BID,
            agent_id="poster1",
            target_id=bounty.bounty_id,
            metadata={"bid_id": bid.bid_id},
        )
        result = handler.handle_accept_bid(action, state)
        assert result is True
        assert len(events) == 1
        assert events[0].event_type == EventType.ESCROW_CREATED

    def test_none_escrow(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1")
        action = Action(
            action_type=ActionType.ACCEPT_BID,
            agent_id="poster1",
            target_id="fake_bounty",
            metadata={"bid_id": "fake_bid"},
        )
        result = handler.handle_accept_bid(action, state)
        assert result is False


class TestHandleRejectBid:
    """Test handle_reject_bid: success, failure."""

    def test_success(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", "bidder1")
        bounty = mkt.post_bounty(poster_id="poster1", task_id="t1", reward_amount=10.0)
        bid = mkt.place_bid(
            bounty_id=bounty.bounty_id, bidder_id="bidder1", bid_amount=8.0
        )
        action = Action(
            action_type=ActionType.REJECT_BID,
            agent_id="poster1",
            target_id=bid.bid_id,
        )
        result = handler.handle_reject_bid(action, state)
        assert result is True
        assert len(events) == 1
        assert events[0].event_type == EventType.BID_REJECTED

    def test_failure(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1")
        action = Action(
            action_type=ActionType.REJECT_BID,
            agent_id="poster1",
            target_id="nonexistent_bid",
        )
        result = handler.handle_reject_bid(action, state)
        assert result is False
        assert len(events) == 0


class TestHandleWithdrawBid:
    """Test handle_withdraw_bid."""

    def test_success(self):
        handler, mkt, tp, events = _make_handler()
        bounty = mkt.post_bounty(poster_id="poster1", task_id="t1", reward_amount=10.0)
        bid = mkt.place_bid(
            bounty_id=bounty.bounty_id, bidder_id="bidder1", bid_amount=8.0
        )
        action = Action(
            action_type=ActionType.WITHDRAW_BID,
            agent_id="bidder1",
            target_id=bid.bid_id,
        )
        result = handler.handle_withdraw_bid(action)
        assert result is True

    def test_failure_wrong_bidder(self):
        handler, mkt, tp, events = _make_handler()
        bounty = mkt.post_bounty(poster_id="poster1", task_id="t1", reward_amount=10.0)
        bid = mkt.place_bid(
            bounty_id=bounty.bounty_id, bidder_id="bidder1", bid_amount=8.0
        )
        action = Action(
            action_type=ActionType.WITHDRAW_BID,
            agent_id="wrong_agent",
            target_id=bid.bid_id,
        )
        result = handler.handle_withdraw_bid(action)
        assert result is False


class TestHandleFileDispute:
    """Test handle_file_dispute: success, None dispute."""

    def _setup_escrow(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", "bidder1")
        bounty = mkt.post_bounty(poster_id="poster1", task_id="t1", reward_amount=10.0)
        bid = mkt.place_bid(
            bounty_id=bounty.bounty_id, bidder_id="bidder1", bid_amount=8.0
        )
        escrow = mkt.accept_bid(
            bounty_id=bounty.bounty_id, bid_id=bid.bid_id, poster_id="poster1"
        )
        return handler, mkt, state, events, escrow

    def test_success(self):
        handler, mkt, state, events, escrow = self._setup_escrow()
        action = Action(
            action_type=ActionType.FILE_DISPUTE,
            agent_id="poster1",
            target_id=escrow.escrow_id,
            content="Work is subpar",
        )
        result = handler.handle_file_dispute(action, state)
        assert result is True
        assert len(events) == 1
        assert events[0].event_type == EventType.DISPUTE_FILED

    def test_none_dispute(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1")
        action = Action(
            action_type=ActionType.FILE_DISPUTE,
            agent_id="poster1",
            target_id="nonexistent_escrow",
            content="reason",
        )
        result = handler.handle_file_dispute(action, state)
        assert result is False
        assert len(events) == 0


class TestSettleTask:
    """Test settle_task: no bounty, success with governance, success without, failure."""

    def _setup_settled(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", "worker1", resources=100.0)
        task = tp.create_task(prompt="Do work", bounty=10.0)
        bounty = mkt.post_bounty(
            poster_id="poster1", task_id=task.task_id, reward_amount=10.0
        )
        bid = mkt.place_bid(
            bounty_id=bounty.bounty_id, bidder_id="worker1", bid_amount=8.0
        )
        mkt.accept_bid(
            bounty_id=bounty.bounty_id, bid_id=bid.bid_id, poster_id="poster1"
        )
        return handler, mkt, tp, state, events, task

    def test_no_bounty(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1")
        result = handler.settle_task("nonexistent_task", True, state)
        assert result is None

    def test_success_without_governance(self):
        handler, mkt, tp, state, events, task = self._setup_settled()
        result = handler.settle_task(task.task_id, True, state, quality_score=0.9)
        assert result is not None
        assert result["success"] is True
        assert result["released_to_worker"] > 0
        # Worker should have received payment
        worker = state.get_agent("worker1")
        assert worker.resources > 100.0
        assert len(events) == 1
        assert events[0].event_type == EventType.ESCROW_RELEASED

    def test_success_with_governance(self):
        handler, mkt, tp, state, events, task = self._setup_settled()
        # Create a mock governance engine
        gov_engine = MagicMock(spec=GovernanceEngine)
        gov_effect = GovernanceEffect(cost_a=1.0, cost_b=0.5)
        gov_engine.apply_interaction.return_value = gov_effect

        poster_before = state.get_agent("poster1").resources
        worker_before = state.get_agent("worker1").resources

        result = handler.settle_task(
            task.task_id,
            True,
            state,
            governance_engine=gov_engine,
            quality_score=0.8,
        )
        assert result is not None
        assert result["success"] is True
        # Governance costs should be deducted
        poster_after = state.get_agent("poster1").resources
        worker_after = state.get_agent("worker1").resources
        # poster_after = poster_before + refund_to_poster - gov_cost_a
        # worker_after = worker_before + released - gov_cost_b
        assert poster_after < poster_before + result.get("refund_to_poster", 0) + 0.01
        assert worker_after < worker_before + result["released_to_worker"] + 0.01

    def test_failure_refund(self):
        handler, mkt, tp, state, events, task = self._setup_settled()
        poster_before = state.get_agent("poster1").resources
        result = handler.settle_task(task.task_id, False, state)
        assert result is not None
        assert result["success"] is False
        poster_after = state.get_agent("poster1").resources
        assert poster_after > poster_before  # Got refund
        assert len(events) == 1
        assert events[0].event_type == EventType.ESCROW_REFUNDED

    def test_settle_already_settled_returns_none(self):
        handler, mkt, tp, state, events, task = self._setup_settled()
        # Settle once
        handler.settle_task(task.task_id, True, state)
        # Try to settle again => escrow is no longer HELD
        result = handler.settle_task(task.task_id, True, state)
        assert result is None


class TestOnEpochEnd:
    """Test on_epoch_end: expired bounties, resolved disputes."""

    def test_expired_bounties_refund(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", resources=50.0)
        mkt.post_bounty(
            poster_id="poster1",
            task_id="t1",
            reward_amount=10.0,
            deadline_epoch=2,
            current_epoch=0,
        )
        # Deduct resources as if the orchestrator did
        state.get_agent("poster1").update_resources(-10.0)
        assert state.get_agent("poster1").resources == 40.0

        state.current_epoch = 3  # past deadline
        handler.on_epoch_end(state)
        # Poster should be refunded
        assert state.get_agent("poster1").resources == 50.0

    def test_resolved_disputes(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", "worker1", resources=100.0)
        bounty = mkt.post_bounty(poster_id="poster1", task_id="t1", reward_amount=10.0)
        bid = mkt.place_bid(
            bounty_id=bounty.bounty_id, bidder_id="worker1", bid_amount=8.0
        )
        escrow = mkt.accept_bid(
            bounty_id=bounty.bounty_id, bid_id=bid.bid_id, poster_id="poster1"
        )
        dispute = mkt.file_dispute(
            escrow_id=escrow.escrow_id,
            filed_by="poster1",
            reason="bad work",
            current_epoch=0,
        )
        # Advance to epoch 2 (dispute_resolution_epochs=2)
        state.current_epoch = 2
        handler.on_epoch_end(state)
        # Dispute should be resolved
        resolved = mkt.get_dispute(dispute.dispute_id)
        assert resolved.status not in (DisputeStatus.OPEN, DisputeStatus.UNDER_REVIEW)
        # There should be a DISPUTE_RESOLVED event
        dispute_events = [
            e for e in events if e.event_type == EventType.DISPUTE_RESOLVED
        ]
        assert len(dispute_events) == 1

    def test_no_expired_bounties(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1")
        state.current_epoch = 0
        handler.on_epoch_end(state)
        # Should not raise; no events
        assert len(events) == 0


class TestBuildObservationFields:
    """Test build_observation_fields."""

    def test_basic_fields_present(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("agent1", "agent2", reputation=1.0)
        # agent2 posts a bounty so agent1 can see it
        mkt.post_bounty(
            poster_id="agent2", task_id="t1", reward_amount=5.0, current_epoch=0
        )
        fields = handler.build_observation_fields("agent1", state)
        assert "available_bounties" in fields
        assert "active_bids" in fields
        assert "active_escrows" in fields
        assert "pending_bid_decisions" in fields
        # agent1 should see agent2's bounty
        assert len(fields["available_bounties"]) == 1

    def test_own_bounty_excluded(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("agent1", reputation=1.0)
        mkt.post_bounty(
            poster_id="agent1", task_id="t1", reward_amount=5.0, current_epoch=0
        )
        fields = handler.build_observation_fields("agent1", state)
        # Own bounty should not appear in available_bounties
        assert len(fields["available_bounties"]) == 0

    def test_active_bids(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", "bidder1", reputation=1.0)
        bounty = mkt.post_bounty(poster_id="poster1", task_id="t1", reward_amount=10.0)
        mkt.place_bid(bounty_id=bounty.bounty_id, bidder_id="bidder1", bid_amount=8.0)
        fields = handler.build_observation_fields("bidder1", state)
        assert len(fields["active_bids"]) == 1

    def test_active_escrows(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", "worker1", reputation=1.0)
        bounty = mkt.post_bounty(poster_id="poster1", task_id="t1", reward_amount=10.0)
        bid = mkt.place_bid(
            bounty_id=bounty.bounty_id, bidder_id="worker1", bid_amount=8.0
        )
        mkt.accept_bid(
            bounty_id=bounty.bounty_id, bid_id=bid.bid_id, poster_id="poster1"
        )
        fields = handler.build_observation_fields("worker1", state)
        assert len(fields["active_escrows"]) == 1

    def test_pending_bid_decisions(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state("poster1", "bidder1", reputation=1.0)
        bounty = mkt.post_bounty(poster_id="poster1", task_id="t1", reward_amount=10.0)
        mkt.place_bid(bounty_id=bounty.bounty_id, bidder_id="bidder1", bid_amount=8.0)
        fields = handler.build_observation_fields("poster1", state)
        assert len(fields["pending_bid_decisions"]) == 1
        assert fields["pending_bid_decisions"][0]["bounty_reward"] == 10.0

    def test_unknown_agent(self):
        handler, mkt, tp, events = _make_handler()
        state = _make_state()
        fields = handler.build_observation_fields("ghost", state)
        assert fields["available_bounties"] == []
        assert fields["active_bids"] == []
        assert fields["active_escrows"] == []
        assert fields["pending_bid_decisions"] == []
