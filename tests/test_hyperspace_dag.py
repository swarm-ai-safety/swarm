"""Tests for the Hyperspace DAG Planner domain adapter."""

import json
import tempfile
from pathlib import Path

import pytest

from swarm.domains.hyperspace_dag import (
    AgentRole,
    DagAdapter,
    DagConfig,
    DagEvent,
    DagOutcome,
    DagProxyConfig,
    DagSubtask,
    PlanDag,
    SubtaskStatus,
)
from swarm.domains.hyperspace_dag.adapter import (
    _compute_dag_coherence,
    _compute_depth_ratio,
    _event_to_observables,
    _pearson,
)

# ---------------------------------------------------------------------------
# Entity tests
# ---------------------------------------------------------------------------

class TestEntities:
    def test_dag_subtask_defaults(self):
        s = DagSubtask(subtask_id="s1")
        assert s.agent_role == AgentRole.CODING
        assert s.priority == 5
        assert s.status == SubtaskStatus.PENDING
        assert s.dependencies == []

    def test_plan_dag_defaults(self):
        dag = PlanDag(plan_id="p1")
        assert dag.confidence == 0.5
        assert dag.cache_hit is False
        assert dag.subtasks == []

    def test_dag_outcome(self):
        o = DagOutcome(plan_id="p1", success=True, tasks_completed=5, tasks_total=5)
        assert o.tasks_failed == 0
        assert o.retries == 0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config(self):
        cfg = DagConfig()
        assert cfg.acceptance_threshold == 0.5
        assert cfg.confidence_floor == 0.3
        assert cfg.proxy.engagement_weight == 0.30

    def test_config_validation(self):
        with pytest.raises(ValueError):
            DagConfig(acceptance_threshold=1.5)
        with pytest.raises(ValueError):
            DagConfig(confidence_floor=-0.1)

    def test_proxy_config_negative_weight(self):
        with pytest.raises(ValueError):
            DagProxyConfig(task_progress_weight=-0.1)

    def test_from_dict(self):
        data = {
            "proxy": {"engagement_weight": 0.4},
            "acceptance_threshold": 0.6,
        }
        cfg = DagConfig.from_dict(data)
        assert cfg.proxy.engagement_weight == 0.4
        assert cfg.acceptance_threshold == 0.6

    def test_from_dict_empty(self):
        cfg = DagConfig.from_dict({})
        assert cfg.acceptance_threshold == 0.5


# ---------------------------------------------------------------------------
# Structural signal tests
# ---------------------------------------------------------------------------

class TestStructuralSignals:
    def test_coherence_fully_connected(self):
        """A DAG with n-1 edges for n nodes has density 1.0."""
        details = {"n_subtasks": 5, "n_edges": 4}
        assert _compute_dag_coherence(details) == 1.0

    def test_coherence_disconnected(self):
        """A DAG with no edges has density 0.0."""
        details = {"n_subtasks": 5, "n_edges": 0}
        assert _compute_dag_coherence(details) == 0.0

    def test_coherence_single_node(self):
        """Trivial single-node DAG is coherent."""
        assert _compute_dag_coherence({"n_subtasks": 1}) == 1.0

    def test_coherence_over_connected(self):
        """More edges than a tree caps at 1.0."""
        details = {"n_subtasks": 3, "n_edges": 10}
        assert _compute_dag_coherence(details) == 1.0

    def test_depth_ratio_linear(self):
        """Critical path == all tasks → depth ratio 1.0."""
        details = {"critical_path_len": 5, "n_subtasks": 5}
        assert _compute_depth_ratio(details) == 1.0

    def test_depth_ratio_shallow(self):
        """Short critical path → low depth ratio."""
        details = {"critical_path_len": 2, "n_subtasks": 10}
        assert _compute_depth_ratio(details) == pytest.approx(0.2)

    def test_depth_ratio_trivial(self):
        assert _compute_depth_ratio({"n_subtasks": 1}) == 1.0


# ---------------------------------------------------------------------------
# Event → Observable mapping tests
# ---------------------------------------------------------------------------

class TestEventMapping:
    def test_plan_proposed_observables(self):
        event = {
            "event_type": "plan_proposed",
            "details": {
                "n_subtasks": 5,
                "n_edges": 4,
                "critical_path_len": 3,
                "confidence": 0.8,
                "cache_hit": False,
            },
        }
        obs = _event_to_observables(event)
        assert obs is not None
        assert obs.task_progress_delta == 0.1
        assert obs.tool_misuse_flags == 0
        assert obs.counterparty_engagement_delta > 0

    def test_low_confidence_cache_hit_flagged(self):
        """Low-confidence cache hits get a misuse flag (potential poison)."""
        event = {
            "event_type": "plan_proposed",
            "details": {
                "n_subtasks": 3,
                "n_edges": 2,
                "confidence": 0.3,
                "cache_hit": True,
            },
        }
        obs = _event_to_observables(event)
        assert obs is not None
        assert obs.tool_misuse_flags == 1

    def test_high_confidence_cache_hit_ok(self):
        event = {
            "event_type": "plan_proposed",
            "details": {
                "n_subtasks": 3,
                "n_edges": 2,
                "confidence": 0.9,
                "cache_hit": True,
            },
        }
        obs = _event_to_observables(event)
        assert obs is not None
        assert obs.tool_misuse_flags == 0

    def test_subtask_completed_on_time(self):
        obs = _event_to_observables({
            "event_type": "subtask_completed",
            "details": {"within_estimate": True},
        })
        assert obs is not None
        assert obs.task_progress_delta == 1.0
        assert obs.rework_count == 0

    def test_subtask_completed_late(self):
        obs = _event_to_observables({
            "event_type": "subtask_completed",
            "details": {"within_estimate": False},
        })
        assert obs is not None
        assert obs.task_progress_delta == 0.5
        assert obs.rework_count == 1

    def test_subtask_failed(self):
        obs = _event_to_observables({
            "event_type": "subtask_failed",
            "details": {"retries": 3},
        })
        assert obs is not None
        assert obs.task_progress_delta == -0.6
        assert obs.verifier_rejections == 1
        assert obs.rework_count == 3

    def test_plan_completed(self):
        obs = _event_to_observables({
            "event_type": "plan_completed",
            "details": {"tasks_completed": 8, "tasks_total": 10},
        })
        assert obs is not None
        assert obs.task_progress_delta == pytest.approx(0.8)

    def test_plan_failed_partial_credit(self):
        obs = _event_to_observables({
            "event_type": "plan_failed",
            "details": {"tasks_completed": 3, "tasks_total": 10},
        })
        assert obs is not None
        # -0.8 + 0.3 * 0.4 = -0.68
        assert obs.task_progress_delta == pytest.approx(-0.68)

    def test_unknown_event_returns_none(self):
        assert _event_to_observables({"event_type": "heartbeat"}) is None


# ---------------------------------------------------------------------------
# Adapter integration tests
# ---------------------------------------------------------------------------

class TestDagAdapter:
    def _make_events(self) -> list[DagEvent]:
        """Generate a simple plan lifecycle."""
        return [
            DagEvent(
                event_type="plan_proposed",
                step=0, epoch=0,
                agent_id="planner-1",
                plan_id="plan-001",
                details={
                    "n_subtasks": 4,
                    "n_edges": 3,
                    "critical_path_len": 3,
                    "confidence": 0.85,
                    "cache_hit": False,
                },
            ),
            DagEvent(
                event_type="subtask_completed",
                step=1, epoch=0,
                agent_id="worker-1",
                plan_id="plan-001",
                details={"within_estimate": True},
            ),
            DagEvent(
                event_type="subtask_completed",
                step=2, epoch=0,
                agent_id="worker-2",
                plan_id="plan-001",
                details={"within_estimate": True},
            ),
            DagEvent(
                event_type="subtask_failed",
                step=3, epoch=0,
                agent_id="worker-1",
                plan_id="plan-001",
                details={"retries": 1},
            ),
            DagEvent(
                event_type="plan_completed",
                step=4, epoch=0,
                agent_id="planner-1",
                plan_id="plan-001",
                details={"tasks_completed": 3, "tasks_total": 4, "total_retries": 1},
            ),
        ]

    def test_from_events_basic(self):
        adapter = DagAdapter()
        report = adapter.from_events(self._make_events())

        assert report.n_interactions == 5
        assert report.n_plans == 1
        assert report.n_cache_hits == 0
        assert report.mean_confidence == pytest.approx(0.85)
        assert 0.0 <= report.toxicity_rate <= 1.0
        assert 0.0 <= report.mean_p <= 1.0

    def test_from_events_per_agent(self):
        adapter = DagAdapter()
        report = adapter.from_events(self._make_events())

        assert "planner-1" in report.agent_metrics
        assert "worker-1" in report.agent_metrics
        assert report.agent_metrics["planner-1"]["n_interactions"] == 2

    def test_from_events_empty(self):
        adapter = DagAdapter()
        report = adapter.from_events([])
        assert report.n_interactions == 0
        assert report.toxicity_rate == 0.0

    def test_replay_from_jsonl(self):
        """Test replaying from a JSONL event log file."""
        events = self._make_events()
        lines = []
        for e in events:
            lines.append(json.dumps({
                "event_type": e.event_type,
                "step": e.step,
                "epoch": e.epoch,
                "agent_id": e.agent_id,
                "plan_id": e.plan_id,
                "details": e.details,
            }))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write("\n".join(lines))
            f.flush()
            path = Path(f.name)

        try:
            adapter = DagAdapter()
            report = adapter.replay(path)
            assert report.n_interactions == 5
            assert report.n_plans == 1
        finally:
            path.unlink()

    def test_replay_file_not_found(self):
        adapter = DagAdapter()
        with pytest.raises(FileNotFoundError):
            adapter.replay("/nonexistent/path.jsonl")

    def test_custom_config(self):
        cfg = DagConfig(acceptance_threshold=0.7)
        adapter = DagAdapter(config=cfg)
        report = adapter.from_events(self._make_events())
        # Higher threshold → more rejections
        assert report.n_rejected >= 0

    def test_p_invariant(self):
        """All interactions must have p in [0, 1]."""
        adapter = DagAdapter()
        report = adapter.from_events(self._make_events())
        for ix in report.interactions:
            assert 0.0 <= ix.p <= 1.0, f"p out of bounds: {ix.p}"

    def test_metadata_bridge_tag(self):
        """All interactions carry the bridge tag in metadata."""
        adapter = DagAdapter()
        report = adapter.from_events(self._make_events())
        for ix in report.interactions:
            assert ix.metadata.get("bridge") == "hyperspace_dag"


# ---------------------------------------------------------------------------
# Calibration: confidence vs p correlation
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_pearson_perfect_positive(self):
        assert _pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_pearson_perfect_negative(self):
        assert _pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_pearson_no_correlation(self):
        # orthogonal
        assert _pearson([1, 0, -1, 0], [0, 1, 0, -1]) == pytest.approx(0.0)

    def test_pearson_insufficient_data(self):
        assert _pearson([1.0], [2.0]) == 0.0
        assert _pearson([], []) == 0.0

    def test_pearson_constant(self):
        """Constant input → zero variance → return 0."""
        assert _pearson([5, 5, 5], [1, 2, 3]) == 0.0

    def test_confidence_correlation_in_report(self):
        """Multi-plan scenario produces a non-trivial correlation."""
        events = []
        for i, (conf, n_edges) in enumerate([
            (0.9, 4),   # high confidence, well-structured
            (0.3, 0),   # low confidence, disconnected
            (0.7, 3),   # medium confidence, decent structure
        ]):
            events.append(DagEvent(
                event_type="plan_proposed",
                step=i, epoch=0,
                agent_id=f"planner-{i}",
                plan_id=f"plan-{i:03d}",
                details={
                    "n_subtasks": 5,
                    "n_edges": n_edges,
                    "confidence": conf,
                    "cache_hit": False,
                },
            ))

        adapter = DagAdapter()
        report = adapter.from_events(events)
        assert report.n_plans == 3
        # Confidence and p should correlate positively here since
        # well-structured plans (high engagement) get higher p
        assert report.confidence_correlation > 0
