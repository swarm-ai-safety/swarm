"""Tests for environment modules (state, feed, tasks)."""

import pytest
from datetime import datetime, timedelta

from src.env.state import EnvState, RateLimits, RateLimitState, InteractionProposal
from src.env.feed import Feed, Post, Vote, VoteType
from src.env.tasks import Task, TaskPool, TaskStatus, TaskDifficulty
from src.models.agent import AgentType


class TestEnvState:
    """Tests for EnvState."""

    def test_create_state(self):
        """Test state creation with defaults."""
        state = EnvState()
        assert state.current_epoch == 0
        assert state.current_step == 0
        assert len(state.agents) == 0
        assert not state.is_paused

    def test_add_agent(self):
        """Test adding agents."""
        state = EnvState()
        agent_state = state.add_agent(
            agent_id="agent_1",
            agent_type=AgentType.HONEST,
            initial_reputation=1.0,
            initial_resources=200.0,
        )

        assert agent_state.agent_id == "agent_1"
        assert agent_state.agent_type == AgentType.HONEST
        assert agent_state.reputation == 1.0
        assert agent_state.resources == 200.0
        assert "agent_1" in state.agents

    def test_add_duplicate_agent_raises(self):
        """Test that adding duplicate agent raises."""
        state = EnvState()
        state.add_agent("agent_1")

        with pytest.raises(ValueError, match="already exists"):
            state.add_agent("agent_1")

    def test_freeze_unfreeze_agent(self):
        """Test agent freezing."""
        state = EnvState()
        state.add_agent("agent_1")

        assert not state.is_agent_frozen("agent_1")

        state.freeze_agent("agent_1")
        assert state.is_agent_frozen("agent_1")
        assert not state.can_agent_act("agent_1")

        state.unfreeze_agent("agent_1")
        assert not state.is_agent_frozen("agent_1")
        assert state.can_agent_act("agent_1")

    def test_pause_resume(self):
        """Test simulation pause/resume."""
        state = EnvState()
        state.add_agent("agent_1")

        assert state.can_agent_act("agent_1")

        state.pause()
        assert state.is_paused
        assert not state.can_agent_act("agent_1")

        state.resume()
        assert not state.is_paused
        assert state.can_agent_act("agent_1")

    def test_advance_step(self):
        """Test step advancement."""
        state = EnvState(steps_per_epoch=10)
        assert state.current_step == 0

        state.advance_step()
        assert state.current_step == 1

    def test_advance_epoch(self):
        """Test epoch advancement."""
        state = EnvState(steps_per_epoch=10)
        state.current_step = 5

        state.advance_epoch()
        assert state.current_epoch == 1
        assert state.current_step == 0

    def test_proposal_management(self):
        """Test interaction proposal management."""
        state = EnvState()

        proposal = InteractionProposal(
            initiator_id="agent_1",
            counterparty_id="agent_2",
        )

        state.add_proposal(proposal)
        assert proposal.proposal_id in state.pending_proposals

        # Get proposals for counterparty
        proposals = state.get_proposals_for_agent("agent_2")
        assert len(proposals) == 1

        # Remove proposal
        removed = state.remove_proposal(proposal.proposal_id)
        assert removed is not None
        assert proposal.proposal_id not in state.pending_proposals


class TestRateLimits:
    """Tests for rate limit tracking."""

    def test_rate_limit_state(self):
        """Test rate limit state tracking."""
        limits = RateLimits(posts_per_epoch=5, votes_per_epoch=10)
        state = RateLimitState()

        assert state.can_post(limits)

        for _ in range(5):
            state.record_post()

        assert not state.can_post(limits)

    def test_rate_limit_reset(self):
        """Test rate limit reset on epoch change."""
        limits = RateLimits(posts_per_epoch=5)
        state = RateLimitState()

        for _ in range(5):
            state.record_post()

        assert not state.can_post(limits)

        state.reset()
        assert state.can_post(limits)


class TestFeed:
    """Tests for Feed."""

    def test_create_post(self):
        """Test post creation."""
        feed = Feed()
        post = feed.create_post(
            author_id="author_1",
            content="Hello world!",
        )

        assert post.author_id == "author_1"
        assert post.content == "Hello world!"
        assert not post.is_reply

    def test_create_reply(self):
        """Test reply creation."""
        feed = Feed()
        parent = feed.create_post("author_1", "Parent post")
        reply = feed.create_post("author_2", "Reply!", parent_id=parent.post_id)

        assert reply.is_reply
        assert reply.parent_id == parent.post_id
        assert parent.reply_count == 1

    def test_reply_to_nonexistent_raises(self):
        """Test that replying to nonexistent post raises."""
        feed = Feed()

        with pytest.raises(ValueError, match="does not exist"):
            feed.create_post("author_1", "Reply!", parent_id="nonexistent")

    def test_content_length_limit(self):
        """Test content length validation."""
        feed = Feed(max_content_length=100)

        with pytest.raises(ValueError, match="exceeds maximum"):
            feed.create_post("author_1", "x" * 101)

    def test_voting(self):
        """Test voting on posts."""
        feed = Feed()
        post = feed.create_post("author_1", "Content")

        vote = feed.vote(post.post_id, "voter_1", VoteType.UPVOTE)
        assert vote is not None
        assert post.upvotes == 1

        # Change vote
        vote2 = feed.vote(post.post_id, "voter_1", VoteType.DOWNVOTE)
        assert post.upvotes == 0
        assert post.downvotes == 1

    def test_remove_vote(self):
        """Test vote removal."""
        feed = Feed()
        post = feed.create_post("author_1", "Content")

        feed.vote(post.post_id, "voter_1", VoteType.UPVOTE)
        assert post.upvotes == 1

        feed.remove_vote(post.post_id, "voter_1")
        assert post.upvotes == 0

    def test_visibility_ranking(self):
        """Test visibility score computation."""
        feed = Feed(reply_weight=0.5, age_decay=0.01)
        post = feed.create_post("author_1", "Content")

        score = feed.compute_visibility_score(post)
        assert score is not None

        # Upvotes increase score
        feed.vote(post.post_id, "voter_1", VoteType.UPVOTE)
        new_score = feed.compute_visibility_score(post)
        assert new_score > score

    def test_get_ranked_posts(self):
        """Test getting ranked posts."""
        feed = Feed()

        post1 = feed.create_post("author_1", "Low engagement")
        post2 = feed.create_post("author_2", "High engagement")

        # Add votes to post2
        feed.vote(post2.post_id, "voter_1", VoteType.UPVOTE)
        feed.vote(post2.post_id, "voter_2", VoteType.UPVOTE)

        ranked = feed.get_ranked_posts(limit=10)
        assert len(ranked) == 2
        assert ranked[0].post_id == post2.post_id

    def test_hide_post(self):
        """Test hiding posts."""
        feed = Feed()
        post = feed.create_post("author_1", "Content")

        feed.hide_post(post.post_id, "spam")
        assert post.is_hidden
        assert post.hidden_reason == "spam"

        # Hidden posts have negative infinity score
        assert feed.compute_visibility_score(post) == float("-inf")

    def test_get_thread(self):
        """Test getting a thread."""
        feed = Feed()
        root = feed.create_post("author_1", "Root")
        reply1 = feed.create_post("author_2", "Reply 1", parent_id=root.post_id)
        reply2 = feed.create_post("author_3", "Reply 2", parent_id=root.post_id)

        thread = feed.get_thread(root.post_id)
        assert len(thread) == 3
        assert thread[0].post_id == root.post_id


class TestTasks:
    """Tests for Task and TaskPool."""

    def test_create_task(self):
        """Test task creation."""
        task = Task(
            prompt="Research topic X",
            description="Detailed research",
            required_outputs=["notes", "summary"],
            difficulty=TaskDifficulty.MEDIUM,
            bounty=10.0,
        )

        assert task.status == TaskStatus.OPEN
        assert task.bounty == 10.0

    def test_claim_task(self):
        """Test claiming a task."""
        task = Task(prompt="Test task", bounty=5.0)

        assert task.is_available(current_epoch=0)

        result = task.claim("agent_1")
        assert result
        assert task.status == TaskStatus.CLAIMED
        assert task.claimed_by == "agent_1"

    def test_task_workflow(self):
        """Test full task workflow."""
        task = Task(prompt="Test task")

        # Claim
        task.claim("agent_1")
        assert task.status == TaskStatus.CLAIMED

        # Start
        task.start()
        assert task.status == TaskStatus.IN_PROGRESS

        # Submit
        output = task.submit_output("agent_1", "My output")
        assert task.status == TaskStatus.SUBMITTED
        assert len(task.outputs) == 1

        # Accept
        task.accept_output(output.output_id, quality_score=0.9)
        assert task.status == TaskStatus.COMPLETED
        assert task.final_quality == 0.9

    def test_task_rejection(self):
        """Test task output rejection."""
        task = Task(prompt="Test task")
        task.claim("agent_1")
        task.start()

        output = task.submit_output("agent_1", "Poor output")
        task.reject_output(output.output_id, reason="Too short")

        assert task.status == TaskStatus.IN_PROGRESS
        assert output.is_accepted is False
        assert output.rejection_reason == "Too short"

    def test_task_expiration(self):
        """Test task deadline."""
        task = Task(prompt="Test", deadline_epoch=5)

        assert task.is_available(current_epoch=4)
        assert not task.is_available(current_epoch=5)

    def test_task_pool(self):
        """Test TaskPool operations."""
        pool = TaskPool()

        task = pool.create_task(
            prompt="Research task",
            bounty=10.0,
            min_reputation=0.0,
        )

        assert task.task_id in [t.task_id for t in pool.get_open_tasks(0)]

        # Claim
        pool.claim_task(task.task_id, "agent_1", agent_reputation=0.5)
        assert task.status == TaskStatus.CLAIMED

        # No longer open
        assert task.task_id not in [t.task_id for t in pool.get_open_tasks(0)]

    def test_task_pool_reputation_filter(self):
        """Test task filtering by reputation."""
        pool = TaskPool()

        pool.create_task(prompt="Easy task", min_reputation=0.0)
        pool.create_task(prompt="Hard task", min_reputation=5.0)

        claimable = pool.get_claimable_tasks(agent_reputation=1.0, current_epoch=0)
        assert len(claimable) == 1
        assert claimable[0].prompt == "Easy task"
