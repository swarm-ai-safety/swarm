"""Tests for scripts/self_eval.py — self-evaluation summary generator."""

import json
import tempfile
from pathlib import Path

from scripts.self_eval import compute_summary, load_tracker_log


class TestComputeSummary:
    def test_empty_data(self):
        summary = compute_summary([], [], [], [], 10)
        assert "Completed:** 0 tasks" in summary
        assert "blocked:** none" in summary
        assert "PerformanceTracker log not yet available" in summary

    def test_done_issues(self):
        done = [
            {
                "identifier": "SWA-1",
                "title": "Task A",
                "priority": "high",
                "startedAt": "2026-03-28T10:00:00Z",
                "completedAt": "2026-03-28T11:00:00Z",
            },
            {
                "identifier": "SWA-2",
                "title": "Task B",
                "priority": "medium",
                "startedAt": "2026-03-28T12:00:00Z",
                "completedAt": "2026-03-28T14:00:00Z",
            },
        ]
        summary = compute_summary(done, [], [], [], 10)
        assert "Completed:** 2 tasks" in summary
        assert "Avg time-to-close:** 1.5h" in summary
        assert "high: 1" in summary
        assert "medium: 1" in summary

    def test_blocked_issues(self):
        blocked = [
            {"identifier": "SWA-3", "title": "Blocked thing"},
        ]
        summary = compute_summary([], blocked, [], [], 10)
        assert "blocked:** 1 task(s)" in summary
        assert "SWA-3" in summary

    def test_in_progress_issues(self):
        in_prog = [
            {"identifier": "SWA-4", "title": "WIP task"},
        ]
        summary = compute_summary([], [], in_prog, [], 10)
        assert "In progress:** 1 task(s)" in summary
        assert "SWA-4" in summary

    def test_with_tracker_entries(self):
        entries = [
            {"metrics": {"completion_rate": 0.7, "blocker_freq": 0.2}},
            {"metrics": {"completion_rate": 0.8, "blocker_freq": 0.1}},
            {"metrics": {"completion_rate": 0.85, "blocker_freq": 0.05}},
            {"metrics": {"completion_rate": 0.9, "blocker_freq": 0.03}},
        ]
        summary = compute_summary([], [], [], entries, 10)
        assert "PerformanceTracker Trends" in summary
        assert "completion_rate" in summary
        assert "blocker_freq" in summary


class TestLoadTrackerLog:
    def test_missing_file(self):
        result = load_tracker_log(Path("/nonexistent/path.jsonl"), 10)
        assert result == []

    def test_valid_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                json.dump({"metrics": {"score": i}}, f)
                f.write("\n")
            path = Path(f.name)
        try:
            result = load_tracker_log(path, 3)
            assert len(result) == 3
            assert result[0]["metrics"]["score"] == 2
        finally:
            path.unlink()

    def test_malformed_lines_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"metrics": {"a": 1}}\n')
            f.write("not json\n")
            f.write('{"metrics": {"a": 2}}\n')
            path = Path(f.name)
        try:
            result = load_tracker_log(path, 10)
            assert len(result) == 2
        finally:
            path.unlink()
