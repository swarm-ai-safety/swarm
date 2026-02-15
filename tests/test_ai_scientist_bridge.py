"""Tests for the AI-Scientist bridge.

Covers event roundtrip, mapper correctness, policy decisions,
and end-to-end bridge ingestion.
"""

from __future__ import annotations

import json
import os
import tempfile

from swarm.bridges.ai_scientist.bridge import AIScientistBridge
from swarm.bridges.ai_scientist.client import AIScientistClient
from swarm.bridges.ai_scientist.config import AIScientistConfig
from swarm.bridges.ai_scientist.events import (
    AIScientistEvent,
    AIScientistEventType,
    ExperimentRunEvent,
    IdeaEvent,
    ReviewEvent,
    WriteupEvent,
)
from swarm.bridges.ai_scientist.mapper import AIScientistMapper
from swarm.bridges.ai_scientist.policy import (
    AIScientistPolicy,
    PolicyDecision,
)

# ---------------------------------------------------------------------------
# Event roundtrip tests
# ---------------------------------------------------------------------------


class TestEventRoundtrip:
    def test_ai_scientist_event_roundtrip(self):
        event = AIScientistEvent(
            event_type=AIScientistEventType.IDEA_GENERATED,
            idea_name="test_idea",
            phase="idea",
            step=0,
            payload={"key": "value"},
        )
        d = event.to_dict()
        restored = AIScientistEvent.from_dict(d)
        assert restored.event_type == event.event_type
        assert restored.idea_name == event.idea_name
        assert restored.phase == event.phase
        assert restored.payload == event.payload

    def test_idea_event_roundtrip(self):
        idea = IdeaEvent(
            idea_name="test",
            interestingness=7.0,
            feasibility=8.0,
            novelty_score=6.0,
            novel=True,
        )
        d = idea.to_dict()
        restored = IdeaEvent.from_dict(d)
        assert restored.interestingness == 7.0
        assert restored.feasibility == 8.0
        assert restored.novelty_score == 6.0
        assert restored.novel is True

    def test_experiment_run_event_roundtrip(self):
        run = ExperimentRunEvent(
            run_index=2,
            success=True,
            metrics={"loss": 0.5},
            cost_usd=1.23,
        )
        d = run.to_dict()
        restored = ExperimentRunEvent.from_dict(d)
        assert restored.run_index == 2
        assert restored.success is True
        assert restored.metrics == {"loss": 0.5}

    def test_writeup_event_roundtrip(self):
        w = WriteupEvent(section="method", compiled=True, citation_count=5)
        d = w.to_dict()
        restored = WriteupEvent.from_dict(d)
        assert restored.section == "method"
        assert restored.compiled is True
        assert restored.citation_count == 5

    def test_review_event_roundtrip(self):
        r = ReviewEvent(
            overall_score=7.0,
            decision="Accept",
            confidence=4.0,
            soundness=3.0,
            strengths=["good"],
            weaknesses=["bad"],
        )
        d = r.to_dict()
        restored = ReviewEvent.from_dict(d)
        assert restored.overall_score == 7.0
        assert restored.decision == "Accept"
        assert restored.strengths == ["good"]


# ---------------------------------------------------------------------------
# Mapper tests
# ---------------------------------------------------------------------------


class TestMapper:
    def setup_method(self):
        self.mapper = AIScientistMapper()

    def test_map_idea_p_in_bounds(self):
        idea = IdeaEvent(
            idea_name="test",
            interestingness=8.0,
            feasibility=7.0,
            novelty_score=9.0,
        )
        event = AIScientistEvent(
            event_type=AIScientistEventType.IDEA_GENERATED,
            idea_name="test",
            phase="idea",
            payload=idea.to_dict(),
        )
        interaction = self.mapper.map_idea(event, idea)
        assert 0.0 <= interaction.p <= 1.0
        assert -1.0 <= interaction.v_hat <= 1.0
        assert interaction.metadata["bridge"] == "ai_scientist"

    def test_map_idea_low_scores(self):
        idea = IdeaEvent(
            idea_name="bad",
            interestingness=1.0,
            feasibility=1.0,
            novelty_score=1.0,
        )
        event = AIScientistEvent(
            event_type=AIScientistEventType.IDEA_GENERATED,
            payload=idea.to_dict(),
        )
        interaction = self.mapper.map_idea(event, idea)
        # Low scores produce negative task_progress but other observables
        # are zero, so p stays close to 0.5 due to proxy weight mix
        assert interaction.p < 0.6

    def test_map_novelty_check_passed(self):
        idea = IdeaEvent(idea_name="novel_idea", novel=True)
        event = AIScientistEvent(
            event_type=AIScientistEventType.NOVELTY_CHECK_PASSED,
            payload=idea.to_dict(),
        )
        interaction = self.mapper.map_novelty_check(event, idea)
        assert interaction.accepted is True
        assert interaction.verifier_rejections == 0

    def test_map_novelty_check_failed(self):
        idea = IdeaEvent(idea_name="old_idea", novel=False)
        event = AIScientistEvent(
            event_type=AIScientistEventType.NOVELTY_CHECK_FAILED,
            payload=idea.to_dict(),
        )
        interaction = self.mapper.map_novelty_check(event, idea)
        assert interaction.accepted is False
        assert interaction.verifier_rejections == 1

    def test_map_experiment_run_success(self):
        run = ExperimentRunEvent(run_index=0, success=True)
        event = AIScientistEvent(
            event_type=AIScientistEventType.EXPERIMENT_RUN_COMPLETED,
            payload=run.to_dict(),
        )
        interaction = self.mapper.map_experiment_run(event, run)
        assert interaction.accepted is True
        assert interaction.task_progress_delta == 0.4

    def test_map_experiment_run_failure(self):
        run = ExperimentRunEvent(
            run_index=1,
            success=False,
            execution_error="OOM",
            retry_count=2,
        )
        event = AIScientistEvent(
            event_type=AIScientistEventType.EXPERIMENT_RUN_FAILED,
            payload=run.to_dict(),
        )
        interaction = self.mapper.map_experiment_run(event, run)
        assert interaction.accepted is False
        assert interaction.verifier_rejections == 1
        assert interaction.rework_count == 2

    def test_map_writeup_compiled(self):
        writeup = WriteupEvent(compiled=True, citation_count=10)
        event = AIScientistEvent(
            event_type=AIScientistEventType.WRITEUP_COMPILED,
            payload=writeup.to_dict(),
        )
        interaction = self.mapper.map_writeup(event, writeup)
        assert interaction.accepted is True
        assert interaction.counterparty_engagement_delta == 0.5  # 10/20

    def test_map_writeup_failed(self):
        writeup = WriteupEvent(
            compiled=False,
            compilation_error="LaTeX error",
        )
        event = AIScientistEvent(
            event_type=AIScientistEventType.WRITEUP_FAILED,
            payload=writeup.to_dict(),
        )
        interaction = self.mapper.map_writeup(event, writeup)
        assert interaction.accepted is False
        assert interaction.verifier_rejections == 1

    def test_map_review_accept(self):
        review = ReviewEvent(
            overall_score=7.0,
            decision="Accept",
            confidence=4.0,
        )
        event = AIScientistEvent(
            event_type=AIScientistEventType.REVIEW_SUBMITTED,
            payload=review.to_dict(),
        )
        interaction = self.mapper.map_review(event, review)
        assert interaction.accepted is True
        assert interaction.verifier_rejections == 0
        assert 0.0 <= interaction.p <= 1.0

    def test_map_review_reject(self):
        review = ReviewEvent(
            overall_score=3.0,
            decision="Reject",
            confidence=3.0,
        )
        event = AIScientistEvent(
            event_type=AIScientistEventType.REVIEW_SUBMITTED,
            payload=review.to_dict(),
        )
        interaction = self.mapper.map_review(event, review)
        assert interaction.accepted is False
        assert interaction.verifier_rejections == 1

    def test_map_event_returns_none_for_unhandled(self):
        event = AIScientistEvent(
            event_type=AIScientistEventType.EXPERIMENT_STARTED,
        )
        assert self.mapper.map_event(event) is None

    def test_map_event_routes_correctly(self):
        idea = IdeaEvent(idea_name="x", interestingness=5, feasibility=5, novelty_score=5)
        event = AIScientistEvent(
            event_type=AIScientistEventType.IDEA_GENERATED,
            payload=idea.to_dict(),
        )
        interaction = self.mapper.map_event(event)
        assert interaction is not None
        assert interaction.metadata["phase"] == "idea"


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------


class TestPolicy:
    def setup_method(self):
        self.config = AIScientistConfig(
            experiment_circuit_breaker_max_failures=3,
            cost_budget_usd=10.0,
            review_accept_threshold=5.0,
            max_improvement_rounds=2,
        )
        self.policy = AIScientistPolicy(self.config)

    def test_novelty_gate_approve(self):
        result = self.policy.evaluate_novelty_gate(novel=True)
        assert result.decision == PolicyDecision.APPROVE

    def test_novelty_gate_deny(self):
        result = self.policy.evaluate_novelty_gate(novel=False)
        assert result.decision == PolicyDecision.DENY

    def test_novelty_gate_disabled(self):
        config = AIScientistConfig(novelty_gate_enabled=False)
        policy = AIScientistPolicy(config)
        result = policy.evaluate_novelty_gate(novel=False)
        assert result.decision == PolicyDecision.APPROVE

    def test_experiment_circuit_breaker(self):
        # First 2 failures -> WARN
        for _i in range(2):
            result = self.policy.evaluate_experiment_run(success=False)
            assert result.decision == PolicyDecision.WARN

        # 3rd failure -> DENY + circuit broken
        result = self.policy.evaluate_experiment_run(success=False)
        assert result.decision == PolicyDecision.DENY
        assert self.policy.should_circuit_break()

        # Subsequent calls denied
        result = self.policy.evaluate_experiment_run(success=True)
        assert result.decision == PolicyDecision.DENY

    def test_experiment_success_resets_counter(self):
        self.policy.evaluate_experiment_run(success=False)
        self.policy.evaluate_experiment_run(success=False)
        # Success resets
        result = self.policy.evaluate_experiment_run(success=True)
        assert result.decision == PolicyDecision.APPROVE
        assert not self.policy.should_circuit_break()
        # Need 3 more failures to trip
        self.policy.evaluate_experiment_run(success=False)
        self.policy.evaluate_experiment_run(success=False)
        result = self.policy.evaluate_experiment_run(success=False)
        assert result.decision == PolicyDecision.DENY

    def test_cost_budget_approve(self):
        result = self.policy.evaluate_cost(5.0)
        assert result.decision == PolicyDecision.APPROVE

    def test_cost_budget_warn_at_80pct(self):
        result = self.policy.evaluate_cost(8.5)
        assert result.decision == PolicyDecision.WARN

    def test_cost_budget_deny_at_100pct(self):
        self.policy.evaluate_cost(8.0)
        result = self.policy.evaluate_cost(3.0)
        assert result.decision == PolicyDecision.DENY

    def test_review_approve(self):
        result = self.policy.evaluate_review(7.0)
        assert result.decision == PolicyDecision.APPROVE

    def test_review_warn_then_deny(self):
        # First low review -> WARN (improvement round 1/2)
        result = self.policy.evaluate_review(3.0)
        assert result.decision == PolicyDecision.WARN

        # Second low review -> DENY (improvement round 2/2 = max)
        result = self.policy.evaluate_review(3.0)
        assert result.decision == PolicyDecision.DENY

    def test_phase_gate_approve(self):
        from swarm.models.interaction import InteractionType, SoftInteraction

        interactions = [
            SoftInteraction(
                initiator="a",
                counterparty="b",
                interaction_type=InteractionType.COLLABORATION,
                accepted=True,
                task_progress_delta=0.5,
                rework_count=0,
                verifier_rejections=0,
                tool_misuse_flags=0,
                counterparty_engagement_delta=0.0,
                v_hat=0.5,
                p=0.7,
            )
        ]
        result = self.policy.evaluate_phase_gate(interactions)
        assert result.decision == PolicyDecision.APPROVE

    def test_phase_gate_deny(self):
        from swarm.models.interaction import InteractionType, SoftInteraction

        interactions = [
            SoftInteraction(
                initiator="a",
                counterparty="b",
                interaction_type=InteractionType.COLLABORATION,
                accepted=True,
                task_progress_delta=-0.5,
                rework_count=0,
                verifier_rejections=0,
                tool_misuse_flags=0,
                counterparty_engagement_delta=0.0,
                v_hat=-0.5,
                p=0.2,
            )
        ]
        result = self.policy.evaluate_phase_gate(interactions)
        assert result.decision == PolicyDecision.DENY

    def test_reset(self):
        self.policy.evaluate_experiment_run(success=False)
        self.policy.evaluate_experiment_run(success=False)
        self.policy.evaluate_experiment_run(success=False)
        assert self.policy.should_circuit_break()
        self.policy.reset()
        assert not self.policy.should_circuit_break()


# ---------------------------------------------------------------------------
# Client tests
# ---------------------------------------------------------------------------


class TestClient:
    def setup_method(self):
        self.client = AIScientistClient()

    def test_parse_idea_from_dict(self):
        ideas_data = [
            {
                "Name": "adaptive_lr",
                "Interestingness": 7,
                "Feasibility": 8,
                "Novelty": 6,
                "novel": True,
            },
            {
                "Name": "other_idea",
                "Interestingness": 5,
                "Feasibility": 5,
                "Novelty": 5,
                "novel": False,
            },
        ]
        events = self.client.parse_idea(ideas_data, "adaptive_lr")
        assert len(events) == 2  # IDEA_GENERATED + NOVELTY_CHECK_PASSED
        assert events[0].event_type == AIScientistEventType.IDEA_GENERATED
        assert events[1].event_type == AIScientistEventType.NOVELTY_CHECK_PASSED

    def test_parse_idea_non_novel(self):
        ideas_data = [
            {
                "Name": "old_idea",
                "Interestingness": 3,
                "Feasibility": 4,
                "Novelty": 2,
                "novel": False,
            },
        ]
        events = self.client.parse_idea(ideas_data, "old_idea")
        assert len(events) == 2
        assert events[1].event_type == AIScientistEventType.NOVELTY_CHECK_FAILED

    def test_parse_experiment_runs_with_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idea_dir = os.path.join(tmpdir, "test_idea")
            os.makedirs(idea_dir)

            # Create 3 run dirs, 2 success, 1 failure
            for i in range(3):
                run_dir = os.path.join(idea_dir, f"run_{i}")
                os.makedirs(run_dir)
                if i < 2:
                    with open(os.path.join(run_dir, "final_info.json"), "w") as f:
                        json.dump({"loss": 0.5 - i * 0.1, "accuracy": 0.8 + i * 0.05}, f)
                # run_2 has no final_info.json -> failure

            events = self.client.parse_experiment_runs(idea_dir)

            # EXPERIMENT_STARTED + 2 completed + 1 failed + EXPERIMENT_COMPLETED
            assert any(
                e.event_type == AIScientistEventType.EXPERIMENT_STARTED for e in events
            )
            completed = [
                e
                for e in events
                if e.event_type == AIScientistEventType.EXPERIMENT_RUN_COMPLETED
            ]
            failed = [
                e
                for e in events
                if e.event_type == AIScientistEventType.EXPERIMENT_RUN_FAILED
            ]
            assert len(completed) == 2
            assert len(failed) == 1
            assert any(
                e.event_type == AIScientistEventType.EXPERIMENT_COMPLETED for e in events
            )

    def test_parse_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            review_path = os.path.join(tmpdir, "review.txt")
            review_data = {
                "Overall": 6,
                "Decision": "Accept",
                "Confidence": 4,
                "Soundness": 3,
                "Presentation": 3,
                "Contribution": 3,
                "Strengths": ["Clear writing"],
                "Weaknesses": ["Limited scope"],
            }
            with open(review_path, "w") as f:
                json.dump(review_data, f)

            events = self.client.parse_review(review_path, "test_idea")
            assert len(events) == 1
            assert events[0].event_type == AIScientistEventType.REVIEW_SUBMITTED
            review = ReviewEvent.from_dict(events[0].payload)
            assert review.overall_score == 6.0
            assert review.decision == "Accept"


# ---------------------------------------------------------------------------
# Bridge integration tests
# ---------------------------------------------------------------------------


class TestBridge:
    def test_ingest_events_end_to_end(self):
        bridge = AIScientistBridge()

        events = [
            AIScientistEvent(
                event_type=AIScientistEventType.IDEA_GENERATED,
                idea_name="test_idea",
                phase="idea",
                payload=IdeaEvent(
                    idea_name="test_idea",
                    interestingness=7.0,
                    feasibility=8.0,
                    novelty_score=6.0,
                    novel=True,
                ).to_dict(),
            ),
            AIScientistEvent(
                event_type=AIScientistEventType.NOVELTY_CHECK_PASSED,
                idea_name="test_idea",
                phase="idea",
                payload=IdeaEvent(
                    idea_name="test_idea",
                    novel=True,
                ).to_dict(),
            ),
            AIScientistEvent(
                event_type=AIScientistEventType.EXPERIMENT_STARTED,
                idea_name="test_idea",
                phase="experiment",
            ),
            AIScientistEvent(
                event_type=AIScientistEventType.EXPERIMENT_RUN_COMPLETED,
                idea_name="test_idea",
                phase="experiment",
                step=0,
                payload=ExperimentRunEvent(
                    run_index=0,
                    success=True,
                    metrics={"loss": 0.3},
                ).to_dict(),
            ),
            AIScientistEvent(
                event_type=AIScientistEventType.WRITEUP_COMPILED,
                idea_name="test_idea",
                phase="writeup",
                payload=WriteupEvent(
                    compiled=True,
                    citation_count=8,
                ).to_dict(),
            ),
            AIScientistEvent(
                event_type=AIScientistEventType.REVIEW_SUBMITTED,
                idea_name="test_idea",
                phase="review",
                payload=ReviewEvent(
                    overall_score=7.0,
                    decision="Accept",
                    confidence=4.0,
                ).to_dict(),
            ),
        ]

        interactions = bridge.ingest_events(events)

        # IDEA_GENERATED, NOVELTY_CHECK_PASSED, EXPERIMENT_RUN_COMPLETED,
        # WRITEUP_COMPILED, REVIEW_SUBMITTED = 5 interactions
        # EXPERIMENT_STARTED produces no interaction
        assert len(interactions) == 5

        for ix in interactions:
            assert 0.0 <= ix.p <= 1.0
            assert -1.0 <= ix.v_hat <= 1.0

    def test_circuit_breaker_halts_processing(self):
        config = AIScientistConfig(
            experiment_circuit_breaker_max_failures=2,
        )
        bridge = AIScientistBridge(config=config)

        # 2 failures should trip the circuit breaker
        failure_events = [
            AIScientistEvent(
                event_type=AIScientistEventType.EXPERIMENT_RUN_FAILED,
                idea_name="test",
                phase="experiment",
                step=i,
                payload=ExperimentRunEvent(
                    run_index=i,
                    success=False,
                    execution_error="OOM",
                ).to_dict(),
            )
            for i in range(3)
        ]

        # Add an event after the CB should trip
        failure_events.append(
            AIScientistEvent(
                event_type=AIScientistEventType.EXPERIMENT_RUN_COMPLETED,
                idea_name="test",
                phase="experiment",
                step=3,
                payload=ExperimentRunEvent(
                    run_index=3,
                    success=True,
                ).to_dict(),
            )
        )

        interactions = bridge.ingest_events(failure_events)

        # CB trips after 2 failures, so only events 0 and 1 produce interactions
        # Event 2 still gets recorded but CB is now active
        # Event 3 is skipped entirely due to CB
        assert len(interactions) == 2
        assert bridge.policy.should_circuit_break()

        # All events should still be recorded
        assert len(bridge.get_events()) == 4

    def test_novelty_gate_blocks_non_novel(self):
        bridge = AIScientistBridge()

        events = [
            AIScientistEvent(
                event_type=AIScientistEventType.NOVELTY_CHECK_FAILED,
                idea_name="old_idea",
                phase="idea",
                payload=IdeaEvent(
                    idea_name="old_idea",
                    novel=False,
                ).to_dict(),
            ),
        ]

        interactions = bridge.ingest_events(events)
        assert len(interactions) == 1
        # The interaction should be created but marked as not accepted
        assert interactions[0].accepted is False

    def test_phase_interactions_tracked(self):
        bridge = AIScientistBridge()

        events = [
            AIScientistEvent(
                event_type=AIScientistEventType.IDEA_GENERATED,
                idea_name="x",
                phase="idea",
                payload=IdeaEvent(
                    idea_name="x",
                    interestingness=7,
                    feasibility=7,
                    novelty_score=7,
                ).to_dict(),
            ),
            AIScientistEvent(
                event_type=AIScientistEventType.EXPERIMENT_RUN_COMPLETED,
                idea_name="x",
                phase="experiment",
                payload=ExperimentRunEvent(
                    run_index=0,
                    success=True,
                ).to_dict(),
            ),
        ]

        bridge.ingest_events(events)
        assert len(bridge.get_phase_interactions("idea")) == 1
        assert len(bridge.get_phase_interactions("experiment")) == 1
        assert len(bridge.get_phase_interactions("writeup")) == 0

    def test_memory_cap_enforced(self):
        config = AIScientistConfig(max_interactions=3, max_events=5)
        bridge = AIScientistBridge(config=config)

        events = [
            AIScientistEvent(
                event_type=AIScientistEventType.EXPERIMENT_RUN_COMPLETED,
                idea_name="x",
                phase="experiment",
                step=i,
                payload=ExperimentRunEvent(
                    run_index=i,
                    success=True,
                ).to_dict(),
            )
            for i in range(10)
        ]

        bridge.ingest_events(events)
        assert len(bridge.get_interactions()) <= 3
        assert len(bridge.get_events()) <= 5
