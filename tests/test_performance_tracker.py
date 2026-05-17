"""Tests for the PerformanceTracker module."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from swarm.agents.performance_tracker import (
    PerformanceEvent,
    PerformanceTracker,
    TrackerEventType,
)


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "perf.jsonl"


@pytest.fixture
def tracker(log_path: Path) -> PerformanceTracker:
    return PerformanceTracker("agent-1", log_path)


class TestPerformanceEvent:
    def test_roundtrip(self) -> None:
        ev = PerformanceEvent(
            agent_id="a1",
            event_type=TrackerEventType.TASK_COMPLETED,
            timestamp="2026-03-28T12:00:00+00:00",
            metadata={"task_id": "SWA-1"},
        )
        d = ev.to_dict()
        restored = PerformanceEvent.from_dict(d)
        assert restored.agent_id == ev.agent_id
        assert restored.event_type == ev.event_type
        assert restored.timestamp == ev.timestamp
        assert restored.metadata == ev.metadata

    def test_from_dict_missing_metadata(self) -> None:
        ev = PerformanceEvent.from_dict({
            "agent_id": "a1",
            "event_type": "heartbeat",
            "timestamp": "2026-03-28T12:00:00+00:00",
        })
        assert ev.metadata == {}


class TestPerformanceTracker:
    def test_append_creates_file(
        self, tracker: PerformanceTracker, log_path: Path,
    ) -> None:
        tracker.record_heartbeat()
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["agent_id"] == "agent-1"
        assert data["event_type"] == TrackerEventType.HEARTBEAT

    def test_multiple_events_append(self, tracker: PerformanceTracker, log_path: Path) -> None:
        tracker.record_task_started("T-1")
        tracker.record_task_completed("T-1")
        tracker.record_heartbeat()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_replay_yields_events(self, tracker: PerformanceTracker) -> None:
        tracker.record_task_started("T-1")
        tracker.record_task_blocked("T-1", reason="dep missing")
        events = list(tracker.replay())
        assert len(events) == 2
        assert events[0].event_type == TrackerEventType.TASK_STARTED
        assert events[1].event_type == TrackerEventType.TASK_BLOCKED
        assert events[1].metadata["reason"] == "dep missing"

    def test_replay_empty_file(self, log_path: Path) -> None:
        tracker = PerformanceTracker("agent-1", log_path)
        assert list(tracker.replay()) == []

    def test_events_for_agent_filters(self, log_path: Path) -> None:
        t1 = PerformanceTracker("agent-1", log_path)
        t2 = PerformanceTracker("agent-2", log_path)
        t1.record_heartbeat()
        t2.record_heartbeat()
        t1.record_heartbeat()

        assert len(t1.events_for_agent("agent-1")) == 2
        assert len(t1.events_for_agent("agent-2")) == 1

    def test_record_review_events(self, tracker: PerformanceTracker) -> None:
        tracker.record_review_passed("T-1")
        tracker.record_review_failed("T-2", reason="tests failing")
        events = list(tracker.replay())
        assert events[0].event_type == TrackerEventType.REVIEW_PASSED
        assert events[1].event_type == TrackerEventType.REVIEW_FAILED
        assert events[1].metadata["reason"] == "tests failing"

    def test_extra_metadata(self, tracker: PerformanceTracker) -> None:
        tracker.record_task_started("T-1", priority="high")
        events = list(tracker.replay())
        assert events[0].metadata["priority"] == "high"


class TestPerformanceSummary:
    def test_empty_summary(self, tracker: PerformanceTracker) -> None:
        summary = tracker.summarize()
        assert summary.agent_id == "agent-1"
        assert summary.total_tasks_started == 0
        assert summary.completion_rate == 0.0
        assert summary.avg_time_to_close_seconds is None

    def test_completion_rate(self, tracker: PerformanceTracker) -> None:
        tracker.record_task_started("T-1")
        tracker.record_task_started("T-2")
        tracker.record_task_completed("T-1")
        summary = tracker.summarize()
        assert summary.total_tasks_started == 2
        assert summary.total_tasks_completed == 1
        assert summary.completion_rate == pytest.approx(0.5)

    def test_review_pass_rate(self, tracker: PerformanceTracker) -> None:
        tracker.record_review_passed("T-1")
        tracker.record_review_passed("T-2")
        tracker.record_review_failed("T-3")
        summary = tracker.summarize()
        assert summary.review_pass_rate == pytest.approx(2.0 / 3.0)

    def test_blocker_frequency(self, tracker: PerformanceTracker) -> None:
        tracker.record_task_started("T-1")
        tracker.record_task_started("T-2")
        tracker.record_task_started("T-3")
        tracker.record_task_started("T-4")
        tracker.record_task_blocked("T-2")
        summary = tracker.summarize()
        assert summary.blocker_frequency == pytest.approx(0.25)

    def test_heartbeat_count(self, tracker: PerformanceTracker) -> None:
        for _ in range(5):
            tracker.record_heartbeat()
        summary = tracker.summarize()
        assert summary.total_heartbeats == 5

    def test_time_to_close(self, log_path: Path) -> None:
        """Time-to-close is computed from start/complete timestamp pairs."""
        tracker = PerformanceTracker("agent-1", log_path)
        # Manually write events with known timestamps for determinism
        t0 = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=120)
        t2 = t0 + timedelta(seconds=10)
        t3 = t0 + timedelta(seconds=70)

        events = [
            PerformanceEvent("agent-1", TrackerEventType.TASK_STARTED, t0.isoformat(), {"task_id": "T-1"}),
            PerformanceEvent("agent-1", TrackerEventType.TASK_STARTED, t2.isoformat(), {"task_id": "T-2"}),
            PerformanceEvent("agent-1", TrackerEventType.TASK_COMPLETED, t3.isoformat(), {"task_id": "T-2"}),
            PerformanceEvent("agent-1", TrackerEventType.TASK_COMPLETED, t1.isoformat(), {"task_id": "T-1"}),
        ]
        with open(log_path, "w") as f:
            for ev in events:
                f.write(json.dumps(ev.to_dict()) + "\n")

        summary = tracker.summarize()
        # T-1: 120s, T-2: 60s => avg 90s
        assert summary.avg_time_to_close_seconds == pytest.approx(90.0)

    def test_summary_to_dict(self, tracker: PerformanceTracker) -> None:
        tracker.record_heartbeat()
        summary = tracker.summarize()
        d = summary.to_dict()
        assert isinstance(d, dict)
        assert d["agent_id"] == "agent-1"
        assert "completion_rate" in d
