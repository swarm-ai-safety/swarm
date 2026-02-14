"""Tests for the SWARM-AgentLaboratory bridge.

Uses dict fixtures — no pickle or AgentLab dependency required.
"""

import pickle

import pytest

from swarm.bridges.agent_lab.bridge import AgentLabBridge
from swarm.bridges.agent_lab.client import restricted_loads
from swarm.bridges.agent_lab.config import AgentLabConfig
from swarm.bridges.agent_lab.events import (
    AgentLabEvent,
    AgentLabEventType,
    DialogueEvent,
    ReviewEvent,
    SolverIterationEvent,
)
from swarm.bridges.agent_lab.mapper import AgentLabMapper
from swarm.bridges.agent_lab.policy import AgentLabPolicy, PolicyDecision

# ---------------------------------------------------------------------------
# Event parsing tests
# ---------------------------------------------------------------------------


class TestEvents:
    def test_event_from_dict_falls_back_to_generic(self) -> None:
        event = AgentLabEvent.from_dict(
            {"event_type": "unknown:type", "agent_role": "PhDStudentAgent"}
        )
        assert event.event_type == AgentLabEventType.GENERIC
        assert event.agent_role == "PhDStudentAgent"

    def test_event_from_dict_parses_known_type(self) -> None:
        event = AgentLabEvent.from_dict(
            {
                "event_type": "solver:iteration",
                "agent_role": "MLEngineerAgent",
                "phase": "experiment",
                "step": 3,
            }
        )
        assert event.event_type == AgentLabEventType.SOLVER_ITERATION
        assert event.phase == "experiment"
        assert event.step == 3

    def test_event_roundtrip(self) -> None:
        event = AgentLabEvent(
            event_type=AgentLabEventType.REVIEW_SUBMITTED,
            agent_role="ReviewersAgent_0",
            phase="review",
        )
        d = event.to_dict()
        restored = AgentLabEvent.from_dict(d)
        assert restored.event_type == event.event_type
        assert restored.agent_role == event.agent_role

    def test_solver_iteration_roundtrip(self) -> None:
        si = SolverIterationEvent(
            solver_type="mle",
            iteration_index=2,
            score=0.75,
            repair_attempts=1,
            execution_error=None,
            cost_usd=0.03,
        )
        d = si.to_dict()
        restored = SolverIterationEvent.from_dict(d)
        assert restored.score == 0.75
        assert restored.repair_attempts == 1
        assert restored.execution_error is None

    def test_review_event_roundtrip(self) -> None:
        rev = ReviewEvent(
            reviewer_index=1,
            overall_score=7.0,
            soundness=3.0,
            contribution=2.5,
            presentation=3.0,
            decision="weak_accept",
            confidence=4.0,
        )
        d = rev.to_dict()
        restored = ReviewEvent.from_dict(d)
        assert restored.overall_score == 7.0
        assert restored.decision == "weak_accept"


# ---------------------------------------------------------------------------
# Mapper tests
# ---------------------------------------------------------------------------


class TestMapper:
    def test_solver_iteration_maps_to_collaboration(self) -> None:
        mapper = AgentLabMapper()
        event = AgentLabEvent(
            event_type=AgentLabEventType.SOLVER_ITERATION,
            agent_role="MLEngineerAgent",
            phase="experiment",
        )
        solver = SolverIterationEvent(
            solver_type="mle", iteration_index=0, score=0.8
        )
        interaction = mapper.map_solver_iteration(event, solver)

        assert interaction.metadata["bridge"] == "agent_lab"
        assert interaction.counterparty == "agent_lab_mle"
        assert interaction.task_progress_delta > 0  # score 0.8 -> positive
        assert 0.0 <= interaction.p <= 1.0

    def test_solver_with_zero_score_maps_negative(self) -> None:
        mapper = AgentLabMapper()
        event = AgentLabEvent(
            event_type=AgentLabEventType.SOLVER_ITERATION,
            agent_role="MLEngineerAgent",
        )
        solver = SolverIterationEvent(
            solver_type="mle", score=0.0, execution_error="RuntimeError"
        )
        interaction = mapper.map_solver_iteration(event, solver)

        assert interaction.task_progress_delta < 0
        assert interaction.verifier_rejections == 1
        assert not interaction.accepted

    def test_dialogue_maps_to_collaboration(self) -> None:
        mapper = AgentLabMapper()
        event = AgentLabEvent(
            event_type=AgentLabEventType.DIALOGUE_EXCHANGE,
            agent_role="PostdocAgent",
        )
        dialogue = DialogueEvent(
            speaker_role="PostdocAgent",
            listener_role="PhDStudentAgent",
            phase="plan",
            has_submission=True,
        )
        interaction = mapper.map_dialogue(event, dialogue)

        assert interaction.initiator == "agent_lab_postdoc"
        assert interaction.counterparty == "agent_lab_phd"
        assert interaction.accepted is True
        assert interaction.metadata["has_submission"] is True

    def test_review_maps_to_vote(self) -> None:
        mapper = AgentLabMapper()
        event = AgentLabEvent(
            event_type=AgentLabEventType.REVIEW_SUBMITTED,
            agent_role="ReviewersAgent_0",
        )
        review = ReviewEvent(
            reviewer_index=0,
            overall_score=7.0,
            decision="weak_accept",
            confidence=4.0,
        )
        interaction = mapper.map_review(event, review)

        assert interaction.initiator == "agent_lab_reviewer_1"
        assert interaction.interaction_type.value == "vote"
        assert interaction.task_progress_delta > 0  # 7/5 - 1 = 0.4

    def test_review_rejection_maps_negative(self) -> None:
        mapper = AgentLabMapper()
        event = AgentLabEvent(
            event_type=AgentLabEventType.REVIEW_SUBMITTED,
        )
        review = ReviewEvent(
            reviewer_index=2,
            overall_score=2.0,
            decision="reject",
            confidence=5.0,
        )
        interaction = mapper.map_review(event, review)

        assert interaction.task_progress_delta < 0
        assert interaction.verifier_rejections == 1
        assert not interaction.accepted

    def test_reviewers_mapped_to_distinct_agents(self) -> None:
        mapper = AgentLabMapper()
        base_event = AgentLabEvent(
            event_type=AgentLabEventType.REVIEW_SUBMITTED,
        )

        ids = set()
        for idx in range(3):
            review = ReviewEvent(reviewer_index=idx, overall_score=5.0, decision="weak_accept")
            interaction = mapper.map_review(base_event, review)
            ids.add(interaction.initiator)

        assert len(ids) == 3, "Each reviewer should map to a distinct SWARM agent"

    def test_phase_completion_maps_success(self) -> None:
        mapper = AgentLabMapper()
        event = AgentLabEvent(
            event_type=AgentLabEventType.PHASE_COMPLETED,
            agent_role="PhDStudentAgent",
            phase="literature_review",
        )
        interaction = mapper.map_phase_completion(event, success=True, phase_score=0.7)

        assert interaction.accepted is True
        assert interaction.task_progress_delta > 0
        assert interaction.metadata["phase"] == "literature_review"

    def test_code_event_failure_maps_negative(self) -> None:
        mapper = AgentLabMapper()
        event = AgentLabEvent(
            event_type=AgentLabEventType.CODE_FAILED,
            agent_role="MLEngineerAgent",
        )
        interaction = mapper.map_code_event(event, success=False)

        assert interaction.task_progress_delta < 0
        assert interaction.verifier_rejections == 1
        assert interaction.tool_misuse_flags == 1


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------


class TestPolicy:
    def test_phase_gate_passes_with_high_p(self) -> None:
        policy = AgentLabPolicy()
        from swarm.models.interaction import SoftInteraction

        interactions = [
            SoftInteraction(p=0.7), SoftInteraction(p=0.8), SoftInteraction(p=0.6)
        ]
        result = policy.evaluate_phase_gate(interactions)
        assert result.decision == PolicyDecision.APPROVE

    def test_phase_gate_denies_with_low_p(self) -> None:
        policy = AgentLabPolicy()
        from swarm.models.interaction import SoftInteraction

        interactions = [
            SoftInteraction(p=0.1), SoftInteraction(p=0.2), SoftInteraction(p=0.3)
        ]
        result = policy.evaluate_phase_gate(interactions)
        assert result.decision == PolicyDecision.DENY

    def test_code_circuit_breaker_triggers(self) -> None:
        config = AgentLabConfig(code_circuit_breaker_max_failures=3)
        policy = AgentLabPolicy(config=config)

        # 2 failures -> still ok
        policy.evaluate_code_execution(False)
        policy.evaluate_code_execution(False)
        assert not policy.should_circuit_break()

        # 3rd failure -> circuit break
        result = policy.evaluate_code_execution(False)
        assert result.decision == PolicyDecision.DENY
        assert policy.should_circuit_break()

    def test_code_success_resets_counter(self) -> None:
        config = AgentLabConfig(code_circuit_breaker_max_failures=3)
        policy = AgentLabPolicy(config=config)

        policy.evaluate_code_execution(False)
        policy.evaluate_code_execution(False)
        policy.evaluate_code_execution(True)  # reset
        assert policy.consecutive_code_failures == 0
        assert not policy.should_circuit_break()

    def test_cost_budget_enforcement(self) -> None:
        config = AgentLabConfig(cost_budget_usd=1.0)
        policy = AgentLabPolicy(config=config)

        result = policy.evaluate_cost(0.5)
        assert result.decision == PolicyDecision.APPROVE

        result = policy.evaluate_cost(0.4)  # total 0.9 -> 90% -> warn
        assert result.decision == PolicyDecision.WARN

        result = policy.evaluate_cost(0.2)  # total 1.1 -> over budget
        assert result.decision == PolicyDecision.DENY
        assert policy.should_circuit_break()

    def test_review_loop_limiter(self) -> None:
        config = AgentLabConfig(max_review_rounds=2, review_score_threshold=4.0)
        policy = AgentLabPolicy(config=config)

        policy.evaluate_review_round(3.0)  # low
        result = policy.evaluate_review_round(3.0)  # 2nd low -> deny
        assert result.decision == PolicyDecision.DENY

    def test_review_good_score_resets(self) -> None:
        config = AgentLabConfig(max_review_rounds=2, review_score_threshold=4.0)
        policy = AgentLabPolicy(config=config)

        policy.evaluate_review_round(3.0)
        policy.evaluate_review_round(5.0)  # good -> resets
        result = policy.evaluate_review_round(3.0)  # only 1st low
        assert result.decision == PolicyDecision.APPROVE

    def test_policy_reset(self) -> None:
        policy = AgentLabPolicy()
        policy.evaluate_code_execution(False)
        policy.evaluate_cost(10.0)
        policy.reset()
        assert policy.consecutive_code_failures == 0
        assert policy.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Bridge integration tests
# ---------------------------------------------------------------------------


class TestBridge:
    def test_ingest_events_produces_interactions(self) -> None:
        bridge = AgentLabBridge()
        events = [
            AgentLabEvent(
                event_type=AgentLabEventType.SOLVER_ITERATION,
                agent_role="MLEngineerAgent",
                phase="experiment",
                payload=SolverIterationEvent(
                    solver_type="mle", score=0.8, cost_usd=0.02
                ).to_dict(),
            ),
            AgentLabEvent(
                event_type=AgentLabEventType.DIALOGUE_EXCHANGE,
                agent_role="PostdocAgent",
                phase="plan",
                payload=DialogueEvent(
                    speaker_role="PostdocAgent",
                    listener_role="PhDStudentAgent",
                    phase="plan",
                    has_submission=True,
                ).to_dict(),
            ),
            AgentLabEvent(
                event_type=AgentLabEventType.REVIEW_SUBMITTED,
                agent_role="ReviewersAgent_0",
                phase="review",
                payload=ReviewEvent(
                    reviewer_index=0,
                    overall_score=7.0,
                    decision="weak_accept",
                    confidence=4.0,
                ).to_dict(),
            ),
        ]

        interactions = bridge.ingest_events(events)
        assert len(interactions) == 3
        assert all(0.0 <= i.p <= 1.0 for i in interactions)
        assert all(i.metadata["bridge"] == "agent_lab" for i in interactions)

    def test_bridge_tracks_events_and_interactions(self) -> None:
        bridge = AgentLabBridge()
        events = [
            AgentLabEvent(
                event_type=AgentLabEventType.SOLVER_ITERATION,
                agent_role="MLEngineerAgent",
                payload=SolverIterationEvent(
                    solver_type="mle", score=0.5
                ).to_dict(),
            ),
        ]
        bridge.ingest_events(events)

        assert len(bridge.get_events()) == 1
        assert len(bridge.get_interactions()) == 1

    def test_circuit_breaker_stops_processing(self) -> None:
        config = AgentLabConfig(code_circuit_breaker_max_failures=2)
        bridge = AgentLabBridge(config=config)

        # Create 5 code failure events; only first few should process
        events = [
            AgentLabEvent(
                event_type=AgentLabEventType.CODE_FAILED,
                agent_role="MLEngineerAgent",
                phase="experiment",
                payload={},
            )
            for _ in range(5)
        ]

        interactions = bridge.ingest_events(events)
        # Circuit breaker fires after 2 failures, processing stops
        assert len(interactions) < 5

    def test_phase_interactions_tracked(self) -> None:
        bridge = AgentLabBridge()
        events = [
            AgentLabEvent(
                event_type=AgentLabEventType.SOLVER_ITERATION,
                agent_role="MLEngineerAgent",
                phase="experiment",
                payload=SolverIterationEvent(
                    solver_type="mle", score=0.6
                ).to_dict(),
            ),
            AgentLabEvent(
                event_type=AgentLabEventType.SOLVER_ITERATION,
                agent_role="MLEngineerAgent",
                phase="experiment",
                payload=SolverIterationEvent(
                    solver_type="mle", score=0.7
                ).to_dict(),
            ),
        ]
        bridge.ingest_events(events)

        phase_ints = bridge.get_phase_interactions("experiment")
        assert len(phase_ints) == 2

    def test_cost_events_tracked(self) -> None:
        config = AgentLabConfig(cost_budget_usd=0.10)
        bridge = AgentLabBridge(config=config)

        events = [
            AgentLabEvent(
                event_type=AgentLabEventType.SOLVER_ITERATION,
                agent_role="MLEngineerAgent",
                payload=SolverIterationEvent(
                    solver_type="mle", score=0.5, cost_usd=0.05
                ).to_dict(),
            ),
            AgentLabEvent(
                event_type=AgentLabEventType.SOLVER_ITERATION,
                agent_role="MLEngineerAgent",
                payload=SolverIterationEvent(
                    solver_type="mle", score=0.5, cost_usd=0.06
                ).to_dict(),
            ),
        ]
        bridge.ingest_events(events)

        # Cost should have been tracked via policy
        assert bridge.policy.total_cost_usd > 0

    def test_event_log_receives_entries(self, tmp_path) -> None:
        from swarm.logging.event_log import EventLog

        log_path = tmp_path / "test_events.jsonl"
        event_log = EventLog(log_path)

        bridge = AgentLabBridge(event_log=event_log)
        events = [
            AgentLabEvent(
                event_type=AgentLabEventType.DIALOGUE_EXCHANGE,
                agent_role="PostdocAgent",
                payload=DialogueEvent(
                    speaker_role="PostdocAgent",
                    listener_role="PhDStudentAgent",
                    has_submission=True,
                ).to_dict(),
            ),
        ]
        bridge.ingest_events(events)

        logged = list(event_log.replay())
        assert len(logged) == 1
        assert logged[0].payload["bridge"] == "agent_lab"


# ---------------------------------------------------------------------------
# Client fixture tests
# ---------------------------------------------------------------------------


class TestClient:
    def test_extract_solver_history(self) -> None:
        from swarm.bridges.agent_lab.client import AgentLabClient

        client = AgentLabClient()
        solver_state = {
            "solver_type": "mle",
            "scores": [0.2, 0.5, 0.8],
            "repair_counts": [0, 1, 0],
            "errors": [None, "RuntimeError", None],
            "costs": [0.01, 0.02, 0.01],
        }
        iterations = client.extract_solver_history(solver_state)

        assert len(iterations) == 3
        assert iterations[0].score == 0.2
        assert iterations[1].execution_error == "RuntimeError"
        assert iterations[2].cost_usd == 0.01

    def test_extract_dialogue_events(self) -> None:
        from swarm.bridges.agent_lab.client import AgentLabClient

        client = AgentLabClient()
        history = [
            {
                "speaker_role": "PostdocAgent",
                "listener_role": "PhDStudentAgent",
                "phase": "plan",
                "command_type": "plan",
                "has_submission": True,
            },
            {
                "speaker_role": "PhDStudentAgent",
                "listener_role": "PostdocAgent",
                "phase": "plan",
                "command_type": "respond",
                "has_submission": False,
            },
        ]
        dialogues = client.extract_dialogue_events(history)

        assert len(dialogues) == 2
        assert dialogues[0].speaker_role == "PostdocAgent"
        assert dialogues[1].has_submission is False

    def test_extract_review_events(self) -> None:
        from swarm.bridges.agent_lab.client import AgentLabClient

        client = AgentLabClient()
        reviews = [
            {
                "reviewer_index": 0,
                "overall_score": 7.0,
                "decision": "weak_accept",
                "confidence": 4.0,
            },
            {
                "reviewer_index": 1,
                "overall_score": 3.0,
                "decision": "reject",
                "confidence": 5.0,
            },
        ]
        review_events = client.extract_review_events(reviews)

        assert len(review_events) == 2
        assert review_events[0].overall_score == 7.0
        assert review_events[1].decision == "reject"


# ---------------------------------------------------------------------------
# Restricted unpickler tests
# ---------------------------------------------------------------------------


class TestRestrictedUnpickler:
    """Verify the restricted unpickler blocks dangerous payloads."""

    def test_allows_safe_builtins(self) -> None:
        """Dicts, lists, strings, ints, etc. should deserialize fine."""
        data = {"key": "value", "num": 42, "nested": [1, 2.0, True, None]}
        raw = pickle.dumps(data)
        result = restricted_loads(raw)
        assert result == data

    def test_allows_tuples_and_sets(self) -> None:
        data = (1, 2, 3)
        raw = pickle.dumps(data)
        assert restricted_loads(raw) == data

        data_set = {1, 2, 3}
        raw_set = pickle.dumps(data_set)
        assert restricted_loads(raw_set) == data_set

    def test_allows_bytes(self) -> None:
        data = b"hello"
        raw = pickle.dumps(data)
        assert restricted_loads(raw) == data

    def test_blocks_os_system(self) -> None:
        """os.system is a classic RCE payload — must be blocked."""
        import os

        class Exploit:
            def __reduce__(self):
                return (os.system, ("echo pwned",))

        raw = pickle.dumps(Exploit())
        with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe class"):
            restricted_loads(raw)

    def test_blocks_subprocess(self) -> None:
        """subprocess.Popen must be blocked."""
        import subprocess

        class Exploit:
            def __reduce__(self):
                return (subprocess.Popen, (["echo", "pwned"],))

        raw = pickle.dumps(Exploit())
        with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe class"):
            restricted_loads(raw)

    def test_blocks_eval(self) -> None:
        """builtins.eval must be blocked."""

        class Exploit:
            def __reduce__(self):
                return (eval, ("__import__('os').system('echo pwned')",))

        raw = pickle.dumps(Exploit())
        with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe class"):
            restricted_loads(raw)

    def test_blocks_exec(self) -> None:
        """builtins.exec must be blocked."""

        class Exploit:
            def __reduce__(self):
                return (exec, ("import os; os.system('echo pwned')",))

        raw = pickle.dumps(Exploit())
        with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe class"):
            restricted_loads(raw)

    def test_blocks_arbitrary_class(self) -> None:
        """Classes not on the allowlist must be blocked."""
        # Craft a pickle that tries to instantiate collections.Counter
        # (a real class that can be pickled, but is not allowlisted)
        import collections

        raw = pickle.dumps(collections.Counter({"a": 1}))
        with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe class"):
            restricted_loads(raw)

    def test_extra_allowed_extends_allowlist(self) -> None:
        """The extra_allowed parameter should permit additional classes."""
        import collections

        raw = pickle.dumps(collections.Counter({"a": 1}))

        # Without extra_allowed, this should fail
        with pytest.raises(pickle.UnpicklingError):
            restricted_loads(raw)

        # With extra_allowed, it should succeed
        result = restricted_loads(
            raw,
            extra_allowed={("collections", "Counter")},
        )
        assert result == collections.Counter({"a": 1})

    def test_blocks_pickle_reduce_to_getattr(self) -> None:
        """Crafted __reduce__ using getattr chain must be blocked."""

        class Exploit:
            def __reduce__(self):
                return (getattr, (__builtins__, "__import__"))

        raw = pickle.dumps(Exploit())
        with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe class"):
            restricted_loads(raw)

    def test_error_message_includes_class_name(self) -> None:
        """The error message should identify the blocked class."""
        import os

        class Exploit:
            def __reduce__(self):
                return (os.system, ("echo pwned",))

        raw = pickle.dumps(Exploit())
        with pytest.raises(pickle.UnpicklingError, match="posix.system|nt.system|os.system"):
            restricted_loads(raw)
