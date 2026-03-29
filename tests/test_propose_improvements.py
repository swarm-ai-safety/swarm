"""Tests for scripts/propose_improvements.py — proposal generator."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.propose_improvements import (
    Proposal,
    compute_metrics,
    count_heartbeats,
    generate_proposals,
    load_tracker_events,
)


def _make_event(event_type: str, task_id: str = "", reason: str = "", ts: str = "") -> dict:
    """Helper to create a tracker event dict."""
    if not ts:
        ts = datetime.now(timezone.utc).isoformat()
    metadata: dict = {}
    if task_id:
        metadata["task_id"] = task_id
    if reason:
        metadata["reason"] = reason
    return {
        "agent_id": "test-agent",
        "event_type": event_type,
        "timestamp": ts,
        "metadata": metadata,
    }


def _write_events(events: list[dict]) -> Path:
    """Write events to a temp JSONL file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for ev in events:
        json.dump(ev, f)
        f.write("\n")
    f.close()
    return Path(f.name)


class TestLoadTrackerEvents:
    def test_missing_file(self):
        assert load_tracker_events(Path("/nonexistent.jsonl")) == []

    def test_valid_events(self):
        events = [_make_event("heartbeat") for _ in range(3)]
        path = _write_events(events)
        try:
            loaded = load_tracker_events(path)
            assert len(loaded) == 3
        finally:
            path.unlink()

    def test_malformed_lines_skipped(self):
        f_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f_tmp.close()
        path = Path(f_tmp.name)
        with open(path, "w") as f:
            json.dump(_make_event("heartbeat"), f)
            f.write("\n")
            f.write("not json\n")
            json.dump(_make_event("heartbeat"), f)
            f.write("\n")
        try:
            loaded = load_tracker_events(path)
            assert len(loaded) == 2
        finally:
            path.unlink()


class TestCountHeartbeats:
    def test_counts_only_heartbeats(self):
        events = [
            _make_event("heartbeat"),
            _make_event("task_started", task_id="t1"),
            _make_event("heartbeat"),
            _make_event("task_completed", task_id="t1"),
        ]
        assert count_heartbeats(events) == 2


class TestComputeMetrics:
    def test_basic_metrics(self):
        events = [
            _make_event("task_started", task_id="t1", ts="2026-01-01T10:00:00+00:00"),
            _make_event("task_completed", task_id="t1", ts="2026-01-01T11:00:00+00:00"),
            _make_event("task_started", task_id="t2", ts="2026-01-01T12:00:00+00:00"),
            _make_event("task_blocked", task_id="t2", reason="waiting on dep"),
            _make_event("heartbeat"),
        ]
        m = compute_metrics(events)
        assert m["tasks_started"] == 2
        assert m["tasks_completed"] == 1
        assert m["tasks_blocked"] == 1
        assert m["completion_rate"] == 0.5
        assert m["blocker_freq"] == 0.5
        assert m["heartbeats"] == 1
        assert m["blocker_reasons"]["waiting on dep"] == 1

    def test_empty_events(self):
        m = compute_metrics([])
        assert m["completion_rate"] == 0.0
        assert m["blocker_freq"] == 0.0

    def test_time_to_close(self):
        events = [
            _make_event("task_started", task_id="t1", ts="2026-01-01T10:00:00+00:00"),
            _make_event("task_completed", task_id="t1", ts="2026-01-01T11:00:00+00:00"),
        ]
        m = compute_metrics(events)
        assert m["avg_time_to_close_s"] == 3600.0

    def test_review_fail_rate(self):
        events = [
            _make_event("review_passed", task_id="t1"),
            _make_event("review_failed", task_id="t2", reason="missing tests"),
            _make_event("review_failed", task_id="t3", reason="wrong approach"),
        ]
        m = compute_metrics(events)
        assert abs(m["review_fail_rate"] - 2 / 3) < 0.01


class TestGenerateProposals:
    def test_high_blocker_freq(self):
        metrics = {
            "tasks_started": 10,
            "tasks_completed": 7,
            "tasks_blocked": 4,
            "heartbeats": 25,
            "completion_rate": 0.7,
            "blocker_freq": 0.4,
            "review_fail_rate": 0.0,
            "ttc_trend_pct": 0.0,
            "blocker_reasons": {"waiting on dep": 3, "unclear spec": 1},
            "avg_time_to_close_s": 3600,
        }
        proposals = generate_proposals("test-agent", metrics)
        titles = [p.title for p in proposals]
        assert "Reduce blocker frequency" in titles

    def test_low_completion_rate(self):
        metrics = {
            "tasks_started": 10,
            "tasks_completed": 4,
            "tasks_blocked": 1,
            "heartbeats": 25,
            "completion_rate": 0.4,
            "blocker_freq": 0.1,
            "review_fail_rate": 0.0,
            "ttc_trend_pct": 0.0,
            "blocker_reasons": {},
            "avg_time_to_close_s": 3600,
        }
        proposals = generate_proposals("test-agent", metrics)
        titles = [p.title for p in proposals]
        assert "Improve task completion rate" in titles

    def test_healthy_metrics(self):
        metrics = {
            "tasks_started": 10,
            "tasks_completed": 9,
            "tasks_blocked": 1,
            "heartbeats": 25,
            "completion_rate": 0.9,
            "blocker_freq": 0.1,
            "review_fail_rate": 0.1,
            "ttc_trend_pct": 5.0,
            "blocker_reasons": {},
            "avg_time_to_close_s": 3600,
        }
        proposals = generate_proposals("test-agent", metrics)
        assert len(proposals) == 1
        assert "healthy" in proposals[0].title.lower()

    def test_multiple_issues(self):
        metrics = {
            "tasks_started": 10,
            "tasks_completed": 4,
            "tasks_blocked": 4,
            "heartbeats": 25,
            "completion_rate": 0.4,
            "blocker_freq": 0.4,
            "review_fail_rate": 0.5,
            "ttc_trend_pct": 30.0,
            "blocker_reasons": {"deps": 4},
            "avg_time_to_close_s": 7200,
        }
        proposals = generate_proposals("test-agent", metrics)
        assert len(proposals) == 4  # blockers, completion, review, slowdown


class TestProposal:
    def test_yaml_output(self):
        prop = Proposal(
            proposal_id="prop-test",
            agent_id="ceo",
            category="prompt",
            title="Test proposal",
            rationale="Test rationale",
            suggested_change="Test change",
            evidence={"metric": 0.5},
        )
        yaml_str = prop.to_yaml_str()
        assert "proposal_id: prop-test" in yaml_str
        assert "category: prompt" in yaml_str
        assert "status: pending" in yaml_str
        assert "reviewed_by: null" in yaml_str

    def test_auto_timestamp(self):
        prop = Proposal(
            proposal_id="p1",
            agent_id="a1",
            category="tool",
            title="t",
            rationale="r",
            suggested_change="c",
        )
        assert prop.created_at  # should be auto-filled


class TestProposalWriteIntegration:
    def test_write_to_directory(self, tmp_path):
        """Test that proposals are written to the correct directory."""
        proposals_dir = tmp_path / "proposals"
        proposals_dir.mkdir()

        prop = Proposal(
            proposal_id="prop-integration-test",
            agent_id="test-agent",
            category="workflow",
            title="Integration test",
            rationale="Testing file write",
            suggested_change="No change",
        )
        filepath = proposals_dir / f"{prop.proposal_id}.yaml"
        filepath.write_text(prop.to_yaml_str())

        assert filepath.exists()
        content = filepath.read_text()
        assert "proposal_id: prop-integration-test" in content
        assert "status: pending" in content
