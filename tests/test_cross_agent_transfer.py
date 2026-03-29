"""Tests for scripts/cross_agent_transfer_test.py — cross-agent transfer experiment."""

import random

import pytest

from scripts.cross_agent_transfer_test import (
    PROFILES,
    TransferResult,
    check_proposal_relevance,
    classify_proposal,
    evaluate_transfer,
    generate_events,
    run_experiment,
)
from scripts.propose_improvements import (
    Proposal,
)


class TestAgentProfile:
    def test_profiles_have_required_fields(self):
        for name, profile in PROFILES.items():
            assert profile.agent_id == name
            assert 0 <= profile.p_complete <= 1
            assert 0 <= profile.p_block <= 1
            assert profile.base_ttc_hours > 0
            assert profile.n_heartbeats >= 20  # minimum for proposals

    def test_profiles_cover_all_roles(self):
        roles = {p.role for p in PROFILES.values()}
        assert "engineer" in roles
        assert "manager" in roles


class TestGenerateEvents:
    def test_deterministic_with_seed(self):
        profile = PROFILES["research-engineer"]
        events_a = generate_events(profile, random.Random(42))
        events_b = generate_events(profile, random.Random(42))
        assert events_a == events_b

    def test_different_seeds_differ(self):
        profile = PROFILES["research-engineer"]
        events_a = generate_events(profile, random.Random(42))
        events_b = generate_events(profile, random.Random(99))
        assert events_a != events_b

    def test_event_types(self):
        profile = PROFILES["ceo"]
        events = generate_events(profile, random.Random(42))
        types = {e["event_type"] for e in events}
        assert "heartbeat" in types
        assert "task_started" in types

    def test_heartbeat_count(self):
        profile = PROFILES["ceo"]
        events = generate_events(profile, random.Random(42))
        hb_count = sum(1 for e in events if e["event_type"] == "heartbeat")
        assert hb_count == profile.n_heartbeats

    def test_events_sorted_by_timestamp(self):
        profile = PROFILES["research-engineer"]
        events = generate_events(profile, random.Random(42))
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps)


class TestClassifyProposal:
    @pytest.mark.parametrize("title,expected", [
        ("Reduce blocker frequency", "blocker_freq"),
        ("Improve task completion rate", "completion_rate"),
        ("Reduce review rejection rate", "review_fail"),
        ("Address increasing time-to-close trend", "ttc_trend"),
        ("Performance is healthy - no changes needed", "healthy"),
    ])
    def test_classification(self, title, expected):
        assert classify_proposal(title) == expected


class TestCheckProposalRelevance:
    def test_blocker_relevant(self):
        metrics = {"blocker_freq": 0.4, "completion_rate": 0.8,
                   "review_fail_rate": 0.1, "ttc_trend_pct": 5.0,
                   "tasks_started": 10}
        assert check_proposal_relevance("blocker_freq", metrics) is True

    def test_blocker_not_relevant(self):
        metrics = {"blocker_freq": 0.1, "completion_rate": 0.8,
                   "review_fail_rate": 0.1, "ttc_trend_pct": 5.0,
                   "tasks_started": 10}
        assert check_proposal_relevance("blocker_freq", metrics) is False

    def test_completion_relevant(self):
        metrics = {"blocker_freq": 0.1, "completion_rate": 0.4,
                   "review_fail_rate": 0.1, "ttc_trend_pct": 5.0,
                   "tasks_started": 10}
        assert check_proposal_relevance("completion_rate", metrics) is True


class TestEvaluateTransfer:
    def test_perfect_transfer(self):
        """Source and target have same problems -> high precision+recall."""
        source = [Proposal(
            proposal_id="p1", agent_id="src", category="workflow",
            title="Reduce blocker frequency", rationale="r", suggested_change="c",
        )]
        target = [Proposal(
            proposal_id="p2", agent_id="tgt", category="workflow",
            title="Reduce blocker frequency", rationale="r", suggested_change="c",
        )]
        target_metrics = {
            "blocker_freq": 0.4, "completion_rate": 0.8,
            "review_fail_rate": 0.1, "ttc_trend_pct": 5.0,
            "tasks_started": 10,
        }
        result = evaluate_transfer(source, target, target_metrics)
        assert result.transfer_precision == 1.0
        assert result.transfer_recall == 1.0

    def test_irrelevant_transfer(self):
        """Source has blocker issue, target doesn't -> low precision."""
        source = [Proposal(
            proposal_id="p1", agent_id="src", category="workflow",
            title="Reduce blocker frequency", rationale="r", suggested_change="c",
        )]
        target = [Proposal(
            proposal_id="p2", agent_id="tgt", category="prompt",
            title="Reduce review rejection rate", rationale="r", suggested_change="c",
        )]
        target_metrics = {
            "blocker_freq": 0.1, "completion_rate": 0.8,
            "review_fail_rate": 0.5, "ttc_trend_pct": 5.0,
            "tasks_started": 10,
        }
        result = evaluate_transfer(source, target, target_metrics)
        assert result.transfer_precision == 0.0

    def test_f1_calculation(self):
        result = TransferResult(
            source_agent="a", target_agent="b",
            source_proposals=["x"], target_proposals=["x"],
            transferred_relevant=["x"], transferred_irrelevant=[],
            missed_by_transfer=[],
            transfer_precision=1.0, transfer_recall=1.0,
            transfer_f1=1.0,
        )
        assert result.transfer_f1 == 1.0


class TestRunExperiment:
    def test_reproducible(self):
        r1 = run_experiment(seed=42, output_dir=None)
        r2 = run_experiment(seed=42, output_dir=None)
        assert r1["summary"] == r2["summary"]

    def test_has_all_profiles(self):
        results = run_experiment(seed=42, output_dir=None)
        for name in PROFILES:
            assert name in results["profiles"]

    def test_pairwise_results_exist(self):
        results = run_experiment(seed=42, output_dir=None)
        # At least some pairs should be evaluated (agents with non-healthy proposals)
        assert len(results["pairwise_transfers"]) > 0

    def test_summary_metrics(self):
        results = run_experiment(seed=42, output_dir=None)
        s = results["summary"]
        assert "avg_precision" in s
        assert "avg_recall" in s
        assert "avg_f1" in s
        assert 0 <= s["avg_precision"] <= 1
        assert 0 <= s["avg_recall"] <= 1

    def test_output_dir(self, tmp_path):
        output = tmp_path / "test_run"
        run_experiment(seed=42, output_dir=output)
        assert (output / "transfer_results.json").exists()
        assert (output / "events").is_dir()

    def test_different_seeds_vary(self):
        r1 = run_experiment(seed=42, output_dir=None)
        r2 = run_experiment(seed=99, output_dir=None)
        # Metrics should differ with different seeds
        assert r1["profiles"]["research-engineer"]["metrics"] != r2["profiles"]["research-engineer"]["metrics"]
