"""Tests for agent role mixin classes."""

import random

import pytest

from swarm.agents.base import Action, ActionType, Observation
from swarm.agents.roles.moderator import ModerationAction, ModeratorRole
from swarm.agents.roles.planner import Plan, PlannerRole
from swarm.agents.roles.poster import ContentStrategy, PosterRole
from swarm.agents.roles.verifier import VerificationResult, VerifierRole
from swarm.agents.roles.worker import WorkerRole, WorkProgress

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs(**kwargs) -> Observation:
    """Convenience builder for Observation with sensible defaults."""
    defaults = {
        "visible_posts": [],
        "active_tasks": [],
    }
    defaults.update(kwargs)
    return Observation(**defaults)


# ===========================================================================
# 1. ModeratorRole
# ===========================================================================


class TestModeratorInit:
    """Tests for ModeratorRole.__init__."""

    def test_init_sets_moderation_history(self):
        mod = ModeratorRole()
        assert mod._moderation_history == []

    def test_init_sets_flagged_content(self):
        mod = ModeratorRole()
        assert mod._flagged_content == []

    def test_init_sets_warned_agents(self):
        mod = ModeratorRole()
        assert mod._warned_agents == {}

    def test_init_sets_frozen_agents(self):
        mod = ModeratorRole()
        assert mod._frozen_agents == set()


class TestModeratorCanModerate:
    def test_can_moderate_returns_true(self):
        mod = ModeratorRole()
        assert mod.can_moderate() is True


class TestModeratorFlagContent:
    def test_flag_content_adds_entry(self):
        mod = ModeratorRole()
        mod.flag_content("post_1", "offensive", "reporter_1")
        assert len(mod._flagged_content) == 1
        entry = mod._flagged_content[0]
        assert entry["post_id"] == "post_1"
        assert entry["reason"] == "offensive"
        assert entry["reporter_id"] == "reporter_1"
        assert entry["flag_count"] == 1

    def test_flag_content_multiple_flags(self):
        mod = ModeratorRole()
        mod.flag_content("p1", "spam", "r1")
        mod.flag_content("p2", "scam", "r2")
        assert len(mod._flagged_content) == 2


class TestModeratorReviewContent:
    def test_review_clean_content_returns_none(self):
        mod = ModeratorRole()
        result = mod.review_content(
            "post_1", "This is a perfectly nice message.", "author_1"
        )
        assert result is None

    def test_review_toxic_content_returns_moderation_action(self):
        mod = ModeratorRole()
        result = mod.review_content(
            "post_1", "I will attack and destroy you!!!!", "author_1"
        )
        assert isinstance(result, ModerationAction)
        assert result.severity > 0

    def test_review_medium_severity_warns(self):
        mod = ModeratorRole()
        # "spam" triggers +1 keyword, content >= 5 chars, no excess !, so severity = 1 (low => flag)
        # Need severity >= 2 for warn. Use "spam" + short content (<5 chars) to get severity 2
        result = mod.review_content("post_1", "spam", "author_1")
        # "spam" keyword=1, short(<5)=1 => severity=2 => medium => warn
        assert result is not None
        assert result.action_type == "warn"

    def test_review_high_severity_hides(self):
        mod = ModeratorRole()
        # "attack" keyword=1, "attack" pattern=2 => severity=3 => high => hide
        result = mod.review_content(
            "post_1", "I will attack you right now do it", "author_1"
        )
        assert result is not None
        assert result.action_type == "hide"


class TestModeratorAssessSeverity:
    def test_clean_content_returns_zero(self):
        mod = ModeratorRole()
        assert mod._assess_severity("Hello, how are you today?") == 0

    def test_toxic_keyword_attack(self):
        mod = ModeratorRole()
        # "attack" as keyword (+1) and as attack pattern (+2) = 3
        severity = mod._assess_severity("attack the system now please")
        assert severity >= 1

    def test_toxic_keyword_spam(self):
        mod = ModeratorRole()
        severity = mod._assess_severity("this is pure spam content here")
        assert severity >= 1

    def test_toxic_keyword_scam(self):
        mod = ModeratorRole()
        severity = mod._assess_severity("this is a total scam operation here")
        assert severity >= 1

    def test_toxic_keyword_exploit(self):
        mod = ModeratorRole()
        severity = mod._assess_severity("lets exploit this vulnerability here")
        assert severity >= 1

    def test_excessive_exclamation_marks(self):
        mod = ModeratorRole()
        severity = mod._assess_severity("wow this is great!!!!")
        assert severity >= 1

    def test_short_content(self):
        mod = ModeratorRole()
        severity = mod._assess_severity("hi")
        assert severity >= 1

    def test_attack_pattern_i_will(self):
        mod = ModeratorRole()
        severity = mod._assess_severity("i will do something terrible to this")
        assert severity >= 2

    def test_attack_pattern_destroy(self):
        mod = ModeratorRole()
        severity = mod._assess_severity("lets destroy everything in sight now")
        assert severity >= 2

    def test_severity_capped_at_five(self):
        mod = ModeratorRole()
        # Pack many triggers: keywords (attack, spam, scam, exploit = 4),
        # excessive ! (+1), short (<5) won't apply since long,
        # attack patterns: "attack" (+2), "i will" (+2), "destroy" (+2) = 6
        # total = 4 + 1 + 6 = 11, capped at 5
        severity = mod._assess_severity("I will attack spam scam exploit destroy!!!!")
        assert severity == 5


class TestModeratorFreezeUnfreeze:
    def test_freeze_agent_adds_to_frozen(self):
        mod = ModeratorRole()
        mod._freeze_agent("agent_1")
        assert "agent_1" in mod._frozen_agents

    def test_unfreeze_agent_removes_from_frozen(self):
        mod = ModeratorRole()
        mod._freeze_agent("agent_1")
        result = mod.unfreeze_agent("agent_1")
        assert result is True
        assert "agent_1" not in mod._frozen_agents

    def test_unfreeze_agent_not_frozen_returns_false(self):
        mod = ModeratorRole()
        result = mod.unfreeze_agent("agent_never_frozen")
        assert result is False

    def test_is_frozen_true(self):
        mod = ModeratorRole()
        mod._freeze_agent("agent_1")
        assert mod.is_frozen("agent_1") is True

    def test_is_frozen_false(self):
        mod = ModeratorRole()
        assert mod.is_frozen("agent_1") is False

    def test_unfreeze_resets_warnings(self):
        mod = ModeratorRole()
        mod._warned_agents["agent_1"] = 5
        mod._freeze_agent("agent_1")
        mod.unfreeze_agent("agent_1")
        assert mod.get_warning_count("agent_1") == 0


class TestModeratorWarningCount:
    def test_get_warning_count_default_zero(self):
        mod = ModeratorRole()
        assert mod.get_warning_count("unknown_agent") == 0

    def test_get_warning_count_after_warn(self):
        mod = ModeratorRole()
        # Trigger a warn action: "spam" (keyword+1) + short (<5 chars) => severity 2 => warn
        mod.review_content("p1", "spam", "agent_1")
        assert mod.get_warning_count("agent_1") == 1


class TestModeratorWarnThenFreeze:
    def test_warn_then_freeze_flow(self):
        mod = ModeratorRole()
        # First warning (severity 2 = medium = warn)
        mod.review_content("p1", "spam", "agent_1")
        assert mod.get_warning_count("agent_1") == 1
        assert mod.is_frozen("agent_1") is False

        # Second warning triggers freeze (warn_before_freeze = 2)
        mod.review_content("p2", "scam", "agent_1")
        assert mod.get_warning_count("agent_1") == 2
        assert mod.is_frozen("agent_1") is True


class TestModeratorProcessFlaggedContent:
    def test_auto_hides_content_at_threshold(self):
        mod = ModeratorRole()
        # Manually add flagged content with flag_count >= 3
        mod._flagged_content.append(
            {
                "post_id": "post_1",
                "reason": "offensive",
                "reporter_id": "r1",
                "flag_count": 3,
            }
        )
        actions = mod.process_flagged_content()
        assert len(actions) == 1
        assert actions[0].action_type == "hide"
        assert actions[0].target_id == "post_1"

    def test_does_not_hide_below_threshold(self):
        mod = ModeratorRole()
        mod._flagged_content.append(
            {
                "post_id": "post_2",
                "reason": "minor",
                "reporter_id": "r1",
                "flag_count": 2,
            }
        )
        actions = mod.process_flagged_content()
        assert len(actions) == 0

    def test_clears_flagged_content_after_processing(self):
        mod = ModeratorRole()
        mod._flagged_content.append(
            {
                "post_id": "post_3",
                "reason": "test",
                "reporter_id": "r1",
                "flag_count": 1,
            }
        )
        mod.process_flagged_content()
        assert len(mod._flagged_content) == 0


class TestModeratorStats:
    def test_get_moderation_stats_empty(self):
        mod = ModeratorRole()
        stats = mod.get_moderation_stats()
        assert stats["total_actions"] == 0
        assert stats["action_counts"] == {}
        assert stats["warned_agents"] == 0
        assert stats["frozen_agents"] == 0
        assert stats["pending_flags"] == 0

    def test_get_moderation_stats_after_actions(self):
        mod = ModeratorRole()
        # Generate a hide action
        mod.review_content("p1", "I will attack and destroy you!!!!", "a1")
        # Generate a warn action
        mod.review_content("p2", "spam", "a2")
        stats = mod.get_moderation_stats()
        assert stats["total_actions"] == 2
        assert stats["warned_agents"] == 1  # a2 got warned

    def test_get_moderation_stats_pending_flags(self):
        mod = ModeratorRole()
        mod.flag_content("p1", "bad", "r1")
        stats = mod.get_moderation_stats()
        assert stats["pending_flags"] == 1


class TestModeratorDecideAction:
    def test_decide_moderation_action_clean_posts(self):
        mod = ModeratorRole()
        obs = _obs(
            visible_posts=[
                {
                    "post_id": "p1",
                    "content": "Hello world, nice day!",
                    "author_id": "a1",
                },
            ]
        )
        result = mod.decide_moderation_action(obs)
        assert result is None

    def test_decide_moderation_action_toxic_post_returns_action(self):
        mod = ModeratorRole()
        obs = _obs(
            visible_posts=[
                {
                    "post_id": "p1",
                    "content": "I will attack and destroy you!!!!",
                    "author_id": "a1",
                },
            ]
        )
        result = mod.decide_moderation_action(obs)
        assert isinstance(result, Action)
        assert result.target_id == "p1"
        assert result.metadata["moderation_action"] == "hide"

    def test_decide_moderation_action_processes_flagged_first(self):
        mod = ModeratorRole()
        mod._flagged_content.append(
            {
                "post_id": "p_flag",
                "reason": "bad",
                "reporter_id": "r1",
                "flag_count": 5,
            }
        )
        obs = _obs(visible_posts=[])
        result = mod.decide_moderation_action(obs)
        # Flagged content processed, returns None to signal actions taken
        assert result is None
        # Flagged content should be cleared
        assert len(mod._flagged_content) == 0


# ===========================================================================
# 2. PlannerRole
# ===========================================================================


class TestPlannerInit:
    def test_can_plan_returns_true(self):
        planner = PlannerRole()
        assert planner.can_plan() is True


class TestPlannerCreatePlan:
    def test_create_plan_returns_plan(self):
        planner = PlannerRole()
        plan = planner.create_plan("task_1", "research something", ["a1", "a2"])
        assert isinstance(plan, Plan)
        assert plan.plan_id == "plan_task_1"
        assert plan.task_id == "task_1"
        assert len(plan.steps) > 0
        assert plan.estimated_steps == len(plan.steps)
        assert plan.current_step == 0

    def test_create_plan_stores_in_active_plans(self):
        planner = PlannerRole()
        planner.create_plan("task_1", "research", ["a1"])
        assert "plan_task_1" in planner._active_plans

    def test_create_plan_assigns_agents(self):
        planner = PlannerRole()
        plan = planner.create_plan("task_1", "research", ["a1", "a2"])
        assert len(plan.assigned_agents) > 0


class TestPlannerDecomposeTask:
    def test_decompose_research(self):
        planner = PlannerRole()
        steps = planner._decompose_task("research the topic")
        assert "Gather initial sources" in steps
        assert len(steps) == 5

    def test_decompose_plan(self):
        planner = PlannerRole()
        steps = planner._decompose_task("plan the project")
        assert "Define objectives" in steps
        assert len(steps) == 5

    def test_decompose_optimize(self):
        planner = PlannerRole()
        steps = planner._decompose_task("optimize performance")
        assert "Analyze current state" in steps
        assert len(steps) == 5

    def test_decompose_default(self):
        planner = PlannerRole()
        steps = planner._decompose_task("do something generic")
        assert "Understand requirements" in steps
        assert len(steps) == 4


class TestPlannerAssignWork:
    def test_round_robin_assignment(self):
        planner = PlannerRole()
        steps = ["step1", "step2", "step3", "step4"]
        assignments = planner._assign_work(steps, ["a1", "a2"])
        # a1 gets steps 0, 2; a2 gets steps 1, 3
        assert "0" in assignments["a1"]
        assert "2" in assignments["a1"]
        assert "1" in assignments["a2"]
        assert "3" in assignments["a2"]

    def test_assign_work_empty_collaborators(self):
        planner = PlannerRole()
        assignments = planner._assign_work(["s1", "s2"], [])
        assert assignments == {}

    def test_assign_work_single_collaborator(self):
        planner = PlannerRole()
        steps = ["s1", "s2", "s3"]
        assignments = planner._assign_work(steps, ["a1"])
        assert "a1" in assignments
        assert len(assignments["a1"]) == 3


class TestPlannerGetNextStep:
    def test_get_next_step_returns_first_step(self):
        planner = PlannerRole()
        plan = planner.create_plan("t1", "research topic", ["a1"])
        step = planner.get_next_step(plan.plan_id)
        assert step == plan.steps[0]

    def test_get_next_step_nonexistent_plan(self):
        planner = PlannerRole()
        assert planner.get_next_step("nonexistent") is None

    def test_get_next_step_completed_plan(self):
        planner = PlannerRole()
        plan = planner.create_plan("t1", "do something", ["a1"])
        # Advance past all steps
        for _ in range(len(plan.steps)):
            planner.advance_plan(plan.plan_id)
        assert planner.get_next_step(plan.plan_id) is None


class TestPlannerAdvancePlan:
    def test_advance_plan_increments_step(self):
        planner = PlannerRole()
        plan = planner.create_plan("t1", "do something", ["a1"])
        result = planner.advance_plan(plan.plan_id)
        assert result is True  # More steps remain
        assert plan.current_step == 1

    def test_advance_plan_returns_false_at_end(self):
        planner = PlannerRole()
        plan = planner.create_plan("t1", "do something", ["a1"])
        for _ in range(len(plan.steps) - 1):
            planner.advance_plan(plan.plan_id)
        result = planner.advance_plan(plan.plan_id)
        assert result is False  # No more steps

    def test_advance_plan_nonexistent(self):
        planner = PlannerRole()
        assert planner.advance_plan("nonexistent") is False


class TestPlannerIsPlanComplete:
    def test_is_plan_complete_initially_false(self):
        planner = PlannerRole()
        plan = planner.create_plan("t1", "do something", ["a1"])
        assert planner.is_plan_complete(plan.plan_id) is False

    def test_is_plan_complete_after_all_steps(self):
        planner = PlannerRole()
        plan = planner.create_plan("t1", "do something", ["a1"])
        for _ in range(len(plan.steps)):
            planner.advance_plan(plan.plan_id)
        assert planner.is_plan_complete(plan.plan_id) is True

    def test_is_plan_complete_nonexistent_returns_true(self):
        planner = PlannerRole()
        assert planner.is_plan_complete("nonexistent") is True


class TestPlannerGetPlanProgress:
    def test_progress_starts_at_zero(self):
        planner = PlannerRole()
        plan = planner.create_plan("t1", "do something", ["a1"])
        assert planner.get_plan_progress(plan.plan_id) == 0.0

    def test_progress_after_one_step(self):
        planner = PlannerRole()
        plan = planner.create_plan("t1", "do something", ["a1"])
        planner.advance_plan(plan.plan_id)
        expected = 1 / len(plan.steps)
        assert planner.get_plan_progress(plan.plan_id) == pytest.approx(expected)

    def test_progress_complete_is_one(self):
        planner = PlannerRole()
        plan = planner.create_plan("t1", "do something", ["a1"])
        for _ in range(len(plan.steps)):
            planner.advance_plan(plan.plan_id)
        assert planner.get_plan_progress(plan.plan_id) == pytest.approx(1.0)

    def test_progress_nonexistent_returns_zero(self):
        planner = PlannerRole()
        assert planner.get_plan_progress("nonexistent") == 0.0


class TestPlannerEdgeCases:
    def test_create_plan_empty_collaborators(self):
        planner = PlannerRole()
        plan = planner.create_plan("t1", "do something", [])
        assert plan.assigned_agents == {}
        assert len(plan.steps) > 0


# ===========================================================================
# 3. PosterRole
# ===========================================================================


class TestPosterCanPost:
    def test_can_post_false_when_interval_not_met(self):
        poster = PosterRole()
        # steps_since_last_post starts at 0, min_post_interval is 2
        assert poster.can_post() is False

    def test_can_post_true_after_interval(self):
        poster = PosterRole()
        poster.increment_step()
        poster.increment_step()
        assert poster.can_post() is True


class TestPosterSetStrategy:
    def test_set_strategy_updates_strategy(self):
        poster = PosterRole()
        strategy = ContentStrategy(topics=["AI", "safety"], tone="helpful")
        poster.set_strategy(strategy)
        assert poster._strategy.topics == ["AI", "safety"]
        assert poster._strategy.tone == "helpful"


class TestPosterGeneratePostContent:
    def test_generate_post_uses_strategy_topic(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(topics=["safety"], tone="neutral"))
        obs = _obs()
        random.seed(42)
        content = poster.generate_post_content(obs)
        assert "safety" in content

    def test_generate_post_helpful_tone(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(topics=["AI"], tone="helpful"))
        obs = _obs()
        random.seed(42)
        content = poster.generate_post_content(obs)
        # One of the helpful templates should be used
        helpful_phrases = [
            "helpful tip",
            "learned something useful",
            "share my insights",
        ]
        assert any(phrase in content for phrase in helpful_phrases)

    def test_generate_post_provocative_tone(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(topics=["AI"], tone="provocative"))
        obs = _obs()
        random.seed(42)
        content = poster.generate_post_content(obs)
        provocative_phrases = [
            "Unpopular opinion",
            "nobody's talking about",
            "change my mind",
        ]
        assert any(phrase in content for phrase in provocative_phrases)

    def test_generate_post_neutral_tone(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(topics=["data"], tone="neutral"))
        obs = _obs()
        random.seed(42)
        content = poster.generate_post_content(obs)
        neutral_phrases = ["thoughts on", "Observations about", "Reflections on"]
        assert any(phrase in content for phrase in neutral_phrases)

    def test_generate_post_no_topics_uses_default(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(topics=[], tone="neutral"))
        obs = _obs()
        content = poster.generate_post_content(obs)
        assert "general thoughts" in content


class TestPosterGenerateReplyContent:
    def test_generate_reply_helpful_tone(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(tone="helpful"))
        obs = _obs()
        random.seed(42)
        content = poster.generate_reply_content("parent text", obs)
        helpful_replies = ["Great point!", "Building on this", "This is helpful"]
        assert any(phrase in content for phrase in helpful_replies)

    def test_generate_reply_provocative_tone(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(tone="provocative"))
        obs = _obs()
        random.seed(42)
        content = poster.generate_reply_content("parent text", obs)
        provocative_replies = ["disagree", "considered the opposite", "overlooks"]
        assert any(phrase in content for phrase in provocative_replies)

    def test_generate_reply_neutral_tone(self):
        poster = PosterRole()
        poster.set_strategy(ContentStrategy(tone="neutral"))
        obs = _obs()
        random.seed(42)
        content = poster.generate_reply_content("parent text", obs)
        neutral_replies = [
            "Interesting perspective",
            "I see your point",
            "Thanks for sharing",
        ]
        assert any(phrase in content for phrase in neutral_replies)


class TestPosterRecordPost:
    def test_record_post_adds_to_history(self):
        poster = PosterRole()
        poster.record_post("post_1", "Hello world")
        assert len(poster._posted_content) == 1
        assert poster._posted_content[0]["post_id"] == "post_1"
        assert poster._posted_content[0]["content"] == "Hello world"

    def test_record_post_resets_step_counter(self):
        poster = PosterRole()
        poster.increment_step()
        poster.increment_step()
        poster.increment_step()
        assert poster._steps_since_last_post == 3
        poster.record_post("post_1", "content")
        assert poster._steps_since_last_post == 0

    def test_record_post_initializes_engagement_stats(self):
        poster = PosterRole()
        poster.record_post("post_1", "content")
        assert "post_1" in poster._engagement_stats
        assert poster._engagement_stats["post_1"]["upvotes"] == 0
        assert poster._engagement_stats["post_1"]["downvotes"] == 0
        assert poster._engagement_stats["post_1"]["replies"] == 0


class TestPosterUpdateEngagement:
    def test_update_engagement_updates_stats(self):
        poster = PosterRole()
        poster.record_post("post_1", "content")
        poster.update_engagement("post_1", upvotes=5, downvotes=2, replies=3)
        assert poster._engagement_stats["post_1"]["upvotes"] == 5
        assert poster._engagement_stats["post_1"]["downvotes"] == 2
        assert poster._engagement_stats["post_1"]["replies"] == 3

    def test_update_engagement_unknown_post_ignored(self):
        poster = PosterRole()
        poster.update_engagement("unknown_post", upvotes=5)
        assert "unknown_post" not in poster._engagement_stats


class TestPosterGetEngagementSummary:
    def test_engagement_summary_empty(self):
        poster = PosterRole()
        summary = poster.get_engagement_summary()
        assert summary["total_posts"] == 0
        assert summary["total_upvotes"] == 0
        assert summary["total_downvotes"] == 0
        assert summary["total_replies"] == 0
        assert summary["avg_engagement"] == 0.0

    def test_engagement_summary_correct_totals(self):
        poster = PosterRole()
        poster.record_post("p1", "content1")
        poster.record_post("p2", "content2")
        poster.update_engagement("p1", upvotes=10, downvotes=2, replies=3)
        poster.update_engagement("p2", upvotes=4, downvotes=1, replies=5)

        summary = poster.get_engagement_summary()
        assert summary["total_posts"] == 2
        assert summary["total_upvotes"] == 14
        assert summary["total_downvotes"] == 3
        assert summary["total_replies"] == 8
        # avg_engagement = (total_upvotes + total_replies) / total_posts = (14 + 8) / 2 = 11.0
        assert summary["avg_engagement"] == pytest.approx(11.0)


class TestPosterIncrementStep:
    def test_increment_step_increases_counter(self):
        poster = PosterRole()
        assert poster._steps_since_last_post == 0
        poster.increment_step()
        assert poster._steps_since_last_post == 1
        poster.increment_step()
        assert poster._steps_since_last_post == 2


# ===========================================================================
# 4. VerifierRole
# ===========================================================================


class TestVerifierCanVerify:
    def test_can_verify_true_when_below_max(self):
        verifier = VerifierRole()
        assert verifier.can_verify() is True

    def test_can_verify_false_when_at_max(self):
        verifier = VerifierRole()
        # max_verifications_per_step = 3
        for i in range(3):
            verifier.queue_verification(f"o{i}", f"t{i}", "content", "submitter")
        assert verifier.can_verify() is False


class TestVerifierQueueVerification:
    def test_queue_verification_adds_to_pending(self):
        verifier = VerifierRole()
        verifier.queue_verification("out_1", "task_1", "some content", "sub_1")
        assert len(verifier._pending_verifications) == 1
        pending = verifier._pending_verifications[0]
        assert pending["output_id"] == "out_1"
        assert pending["task_id"] == "task_1"
        assert pending["content"] == "some content"
        assert pending["submitter_id"] == "sub_1"


class TestVerifierVerifyOutput:
    def test_short_content_gets_issues(self):
        verifier = VerifierRole()
        result = verifier.verify_output("o1", "short", ["req1"])
        assert any("too short" in issue for issue in result.issues)

    def test_placeholder_content_gets_issue(self):
        verifier = VerifierRole()
        long_content = (
            "This is a TODO item that needs to be completed with more detail later on "
            * 5
        )
        result = verifier.verify_output("o1", long_content, [])
        assert any(
            "placeholder" in issue.lower() or "TODO" in issue for issue in result.issues
        )

    def test_good_content_can_be_approved(self):
        verifier = VerifierRole()
        # Set strictness low to increase approval chance
        verifier.set_strictness(0.0)
        # Long content with matching requirements, no placeholders
        content = (
            "This comprehensive analysis covers performance metrics and scalability. "
            "The system demonstrates strong throughput under load testing conditions. "
            "Results indicate significant improvements in response time and reliability. "
            "We recommend proceeding with the proposed architecture changes for production. "
            "The benchmarks confirm that the optimization targets have been met successfully."
        )
        requirements = ["performance", "scalability"]
        # Use fixed seed for reproducibility of the random variance
        random.seed(0)
        result = verifier.verify_output("o1", content, requirements)
        # With no issues and good quality, should be approved
        assert result.is_approved is True
        assert result.quality_score > 0.0

    def test_requirements_not_met_gets_issue(self):
        verifier = VerifierRole()
        content = (
            "A generic long content piece that does not address the specific topics "
            * 5
        )
        result = verifier.verify_output("o1", content, ["quantum", "entanglement"])
        assert any("requirements" in issue.lower() for issue in result.issues)


class TestVerifierApprovalRate:
    def test_approval_rate_empty(self):
        verifier = VerifierRole()
        assert verifier.get_approval_rate() == 0.0

    def test_approval_rate_correct(self):
        verifier = VerifierRole()
        # Manually add results
        verifier._verification_history.append(
            VerificationResult(output_id="o1", is_approved=True, quality_score=0.9)
        )
        verifier._verification_history.append(
            VerificationResult(output_id="o2", is_approved=False, quality_score=0.3)
        )
        verifier._verification_history.append(
            VerificationResult(output_id="o3", is_approved=True, quality_score=0.8)
        )
        assert verifier.get_approval_rate() == pytest.approx(2 / 3)


class TestVerifierAverageQuality:
    def test_average_quality_empty(self):
        verifier = VerifierRole()
        assert verifier.get_average_quality() == 0.0

    def test_average_quality_correct(self):
        verifier = VerifierRole()
        verifier._verification_history.append(
            VerificationResult(output_id="o1", quality_score=0.8)
        )
        verifier._verification_history.append(
            VerificationResult(output_id="o2", quality_score=0.6)
        )
        assert verifier.get_average_quality() == pytest.approx(0.7)


class TestVerifierProcessPending:
    def test_process_pending_processes_all(self):
        verifier = VerifierRole()
        verifier.queue_verification(
            "o1", "t1", "some content here that is decent", "s1"
        )
        verifier.queue_verification(
            "o2", "t2", "another piece of content to verify", "s2"
        )
        results = verifier.process_pending()
        assert len(results) == 2
        assert all(isinstance(r, VerificationResult) for r in results)
        # Pending should be cleared
        assert len(verifier._pending_verifications) == 0

    def test_process_pending_sets_task_id(self):
        verifier = VerifierRole()
        verifier.queue_verification("o1", "task_abc", "content here", "s1")
        results = verifier.process_pending()
        assert results[0].task_id == "task_abc"


class TestVerifierSetStrictness:
    def test_set_strictness_normal(self):
        verifier = VerifierRole()
        verifier.set_strictness(0.8)
        assert verifier._verifier_config["strictness"] == 0.8

    def test_set_strictness_clamps_high(self):
        verifier = VerifierRole()
        verifier.set_strictness(1.5)
        assert verifier._verifier_config["strictness"] == 1.0

    def test_set_strictness_clamps_low(self):
        verifier = VerifierRole()
        verifier.set_strictness(-0.5)
        assert verifier._verifier_config["strictness"] == 0.0


class TestVerifierDecideAction:
    def test_decide_verification_action_with_pending(self):
        verifier = VerifierRole()
        verifier.queue_verification("o1", "t1", "content to verify here", "s1")
        obs = _obs()
        result = verifier.decide_verification_action(obs)
        assert isinstance(result, Action)
        assert result.action_type == ActionType.VERIFY_OUTPUT
        assert result.target_id == "o1"
        assert "is_approved" in result.metadata
        assert "quality_score" in result.metadata
        assert "issues" in result.metadata

    def test_decide_verification_action_no_pending(self):
        verifier = VerifierRole()
        obs = _obs()
        result = verifier.decide_verification_action(obs)
        assert result is None


# ===========================================================================
# 5. WorkerRole
# ===========================================================================


class TestWorkerCanWork:
    def test_can_work_true_when_below_max(self):
        worker = WorkerRole()
        assert worker.can_work() is True

    def test_can_work_false_when_at_max(self):
        worker = WorkerRole()
        # max_concurrent_tasks = 2
        worker.accept_work("t1", ["s1"])
        worker.accept_work("t2", ["s1"])
        assert worker.can_work() is False


class TestWorkerAcceptWork:
    def test_accept_work_returns_true(self):
        worker = WorkerRole()
        assert worker.accept_work("t1", ["step1", "step2"]) is True
        assert "t1" in worker._work_queue

    def test_accept_work_rejects_when_full(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["s1"])
        worker.accept_work("t2", ["s1"])
        assert worker.accept_work("t3", ["s1"]) is False
        assert "t3" not in worker._work_queue

    def test_accept_work_creates_work_progress(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["step_a", "step_b"])
        progress = worker._work_queue["t1"]
        assert isinstance(progress, WorkProgress)
        assert progress.task_id == "t1"
        assert progress.assigned_steps == ["step_a", "step_b"]
        assert progress.completed_steps == []
        assert progress.current_step_index == 0
        assert progress.quality_score == 1.0
        assert progress.rework_count == 0


class TestWorkerExecuteStep:
    def test_execute_step_returns_output(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["analyze data", "write report"])
        output = worker.execute_step("t1")
        assert output == "Completed: analyze data"

    def test_execute_step_advances_step(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["step1", "step2"])
        worker.execute_step("t1")
        progress = worker._work_queue["t1"]
        assert progress.current_step_index == 1
        assert len(progress.completed_steps) == 1

    def test_execute_step_nonexistent_task(self):
        worker = WorkerRole()
        assert worker.execute_step("nonexistent") is None

    def test_execute_step_all_steps_done(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["only_step"])
        worker.execute_step("t1")
        assert worker.execute_step("t1") is None


class TestWorkerRecordRework:
    def test_record_rework_increments_count(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["s1"])
        count = worker.record_rework("t1")
        assert count == 1

    def test_record_rework_reduces_quality(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["s1"])
        worker.record_rework("t1")
        progress = worker._work_queue["t1"]
        assert progress.quality_score == pytest.approx(0.9)

    def test_record_rework_cumulative(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["s1"])
        worker.record_rework("t1")
        worker.record_rework("t1")
        progress = worker._work_queue["t1"]
        assert progress.rework_count == 2
        assert progress.quality_score == pytest.approx(0.9 * 0.9)

    def test_record_rework_nonexistent_returns_zero(self):
        worker = WorkerRole()
        assert worker.record_rework("nonexistent") == 0


class TestWorkerNeedsHelp:
    def test_needs_help_false_initially(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["s1"])
        assert worker.needs_help("t1") is False

    def test_needs_help_true_after_three_reworks(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["s1"])
        worker.record_rework("t1")
        worker.record_rework("t1")
        worker.record_rework("t1")
        assert worker.needs_help("t1") is True

    def test_needs_help_false_below_threshold(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["s1"])
        worker.record_rework("t1")
        worker.record_rework("t1")
        assert worker.needs_help("t1") is False

    def test_needs_help_nonexistent_task(self):
        worker = WorkerRole()
        assert worker.needs_help("nonexistent") is False


class TestWorkerCompleteTask:
    def test_complete_task_removes_from_queue(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["s1"])
        progress = worker.complete_task("t1")
        assert isinstance(progress, WorkProgress)
        assert "t1" not in worker._work_queue

    def test_complete_task_increments_total_completed(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["s1"])
        worker.complete_task("t1")
        assert worker._total_completed == 1

    def test_complete_task_nonexistent_returns_none(self):
        worker = WorkerRole()
        assert worker.complete_task("nonexistent") is None

    def test_complete_task_returns_work_progress(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["s1", "s2"])
        worker.execute_step("t1")
        result = worker.complete_task("t1")
        assert result.task_id == "t1"
        assert len(result.completed_steps) == 1


class TestWorkerGetWorkStatus:
    def test_get_work_status_found(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["step_a", "step_b", "step_c"])
        worker.execute_step("t1")
        status = worker.get_work_status("t1")
        assert status["status"] == "in_progress"
        assert status["task_id"] == "t1"
        assert status["total_steps"] == 3
        assert status["completed_steps"] == 1
        assert status["current_step"] == "step_b"
        assert status["quality_score"] == 1.0
        assert status["rework_count"] == 0

    def test_get_work_status_not_found(self):
        worker = WorkerRole()
        status = worker.get_work_status("nonexistent")
        assert status == {"status": "not_found"}

    def test_get_work_status_all_steps_done_current_is_none(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["only_step"])
        worker.execute_step("t1")
        status = worker.get_work_status("t1")
        assert status["current_step"] is None


class TestWorkerGetCompletionRate:
    def test_completion_rate_zero_when_empty(self):
        worker = WorkerRole()
        assert worker.get_completion_rate() == 0.0

    def test_completion_rate_correct(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["s1"])
        worker.accept_work("t2", ["s1"])
        worker.complete_task("t1")
        # 1 completed out of 2 total (1 completed + 1 in queue)
        assert worker.get_completion_rate() == pytest.approx(0.5)

    def test_completion_rate_all_complete(self):
        worker = WorkerRole()
        worker.accept_work("t1", ["s1"])
        worker.complete_task("t1")
        # 1 completed, 0 in queue
        assert worker.get_completion_rate() == pytest.approx(1.0)
