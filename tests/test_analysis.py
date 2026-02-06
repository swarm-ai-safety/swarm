"""Tests for src/analysis/aggregation.py and src/analysis/export.py."""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from swarm.analysis.aggregation import (
    AgentSnapshot,
    EpochSnapshot,
    MetricsAggregator,
    SimulationHistory,
    TimeSeriesPoint,
    aggregate_incoherence_scaling,
    build_scaling_curve_points,
    compute_rolling_average,
    compute_trend,
)
from swarm.analysis.export import (
    export_to_json,
    generate_summary_report,
    history_to_agent_records,
    history_to_epoch_records,
    load_from_json,
)
from swarm.analysis.plots import create_incoherence_scaling_data
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history_with_two_epochs() -> SimulationHistory:
    """Build a small SimulationHistory with two epochs and two agents."""
    history = SimulationHistory(
        simulation_id="test-sim",
        started_at=datetime(2025, 1, 1, 0, 0, 0),
        ended_at=datetime(2025, 1, 1, 1, 0, 0),
        n_epochs=2,
        steps_per_epoch=10,
        n_agents=2,
        seed=42,
    )
    for epoch in range(2):
        snapshot = EpochSnapshot(
            epoch=epoch,
            timestamp=datetime(2025, 1, 1, 0, epoch, 0),
            total_interactions=10 + epoch * 5,
            accepted_interactions=8 + epoch * 3,
            rejected_interactions=2 + epoch * 2,
            toxicity_rate=0.1 + epoch * 0.05,
            quality_gap=0.3 - epoch * 0.1,
            avg_p=0.8 - epoch * 0.05,
            total_welfare=100.0 + epoch * 50,
            avg_payoff=10.0 + epoch * 5,
            n_agents=2,
            avg_reputation=0.5 + epoch * 0.1,
        )
        history.add_epoch_snapshot(snapshot)

    for epoch in range(2):
        for agent_id in ["agent_a", "agent_b"]:
            snap = AgentSnapshot(
                agent_id=agent_id,
                epoch=epoch,
                reputation=0.5 + epoch * 0.1,
                resources=100.0 + epoch * 10,
                interactions_initiated=3 + epoch,
                interactions_received=2 + epoch,
                total_payoff=5.0 + epoch * 2,
            )
            history.add_agent_snapshot(snap)

    return history


# ---------------------------------------------------------------------------
# Tests for TimeSeriesPoint
# ---------------------------------------------------------------------------

class TestTimeSeriesPoint:

    def test_fields_stored(self):
        ts = datetime(2025, 6, 1)
        pt = TimeSeriesPoint(epoch=3, value=1.5, timestamp=ts)
        assert pt.epoch == 3
        assert pt.value == 1.5
        assert pt.timestamp == ts

    def test_timestamp_defaults_to_none(self):
        pt = TimeSeriesPoint(epoch=0, value=0.0)
        assert pt.timestamp is None


# ---------------------------------------------------------------------------
# Tests for AgentSnapshot
# ---------------------------------------------------------------------------

class TestAgentSnapshot:

    def test_defaults(self):
        snap = AgentSnapshot(agent_id="a1", epoch=0)
        assert snap.reputation == 0.0
        assert snap.resources == 100.0
        assert snap.interactions_initiated == 0
        assert snap.interactions_received == 0
        assert snap.avg_p_initiated == 0.5
        assert snap.avg_p_received == 0.5
        assert snap.total_payoff == 0.0
        assert snap.is_frozen is False
        assert snap.is_quarantined is False

    def test_custom_values(self):
        snap = AgentSnapshot(
            agent_id="x",
            epoch=5,
            reputation=0.9,
            resources=200.0,
            is_frozen=True,
        )
        assert snap.agent_id == "x"
        assert snap.epoch == 5
        assert snap.reputation == 0.9
        assert snap.resources == 200.0
        assert snap.is_frozen is True


# ---------------------------------------------------------------------------
# Tests for EpochSnapshot
# ---------------------------------------------------------------------------

class TestEpochSnapshot:

    def test_defaults(self):
        snap = EpochSnapshot(epoch=0)
        assert snap.total_interactions == 0
        assert snap.toxicity_rate == 0.0
        assert snap.avg_p == 0.5
        assert snap.n_agents == 0
        assert snap.n_components == 1

    def test_custom_values(self):
        snap = EpochSnapshot(epoch=7, total_interactions=50, toxicity_rate=0.2)
        assert snap.epoch == 7
        assert snap.total_interactions == 50
        assert snap.toxicity_rate == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Tests for SimulationHistory
# ---------------------------------------------------------------------------

class TestSimulationHistory:

    def test_add_epoch_snapshot(self):
        h = SimulationHistory()
        s1 = EpochSnapshot(epoch=0, total_welfare=10.0)
        s2 = EpochSnapshot(epoch=1, total_welfare=20.0)
        h.add_epoch_snapshot(s1)
        h.add_epoch_snapshot(s2)
        assert len(h.epoch_snapshots) == 2
        assert h.epoch_snapshots[0].total_welfare == 10.0
        assert h.epoch_snapshots[1].total_welfare == 20.0

    def test_add_agent_snapshot_groups_by_agent_id(self):
        h = SimulationHistory()
        h.add_agent_snapshot(AgentSnapshot(agent_id="a", epoch=0))
        h.add_agent_snapshot(AgentSnapshot(agent_id="a", epoch=1))
        h.add_agent_snapshot(AgentSnapshot(agent_id="b", epoch=0))
        assert len(h.agent_snapshots) == 2
        assert len(h.agent_snapshots["a"]) == 2
        assert len(h.agent_snapshots["b"]) == 1

    def test_get_time_series(self):
        h = SimulationHistory()
        h.add_epoch_snapshot(EpochSnapshot(epoch=0, toxicity_rate=0.1))
        h.add_epoch_snapshot(EpochSnapshot(epoch=1, toxicity_rate=0.2))
        h.add_epoch_snapshot(EpochSnapshot(epoch=2, toxicity_rate=0.3))
        series = h.get_time_series("toxicity_rate")
        assert len(series) == 3
        assert series[0].value == pytest.approx(0.1)
        assert series[1].value == pytest.approx(0.2)
        assert series[2].value == pytest.approx(0.3)
        assert series[0].epoch == 0

    def test_get_time_series_invalid_metric_returns_empty(self):
        h = SimulationHistory()
        h.add_epoch_snapshot(EpochSnapshot(epoch=0))
        series = h.get_time_series("nonexistent_metric")
        assert series == []

    def test_get_agent_time_series(self):
        h = SimulationHistory()
        h.add_agent_snapshot(AgentSnapshot(agent_id="a", epoch=0, reputation=0.1))
        h.add_agent_snapshot(AgentSnapshot(agent_id="a", epoch=1, reputation=0.4))
        series = h.get_agent_time_series("a", "reputation")
        assert len(series) == 2
        assert series[0].value == pytest.approx(0.1)
        assert series[1].value == pytest.approx(0.4)

    def test_get_agent_time_series_unknown_agent_returns_empty(self):
        h = SimulationHistory()
        assert h.get_agent_time_series("nobody", "reputation") == []

    def test_get_final_agent_states(self):
        h = SimulationHistory()
        h.add_agent_snapshot(AgentSnapshot(agent_id="a", epoch=0, reputation=0.1))
        h.add_agent_snapshot(AgentSnapshot(agent_id="a", epoch=1, reputation=0.9))
        h.add_agent_snapshot(AgentSnapshot(agent_id="b", epoch=0, reputation=0.5))
        final = h.get_final_agent_states()
        assert set(final.keys()) == {"a", "b"}
        assert final["a"].reputation == pytest.approx(0.9)
        assert final["a"].epoch == 1
        assert final["b"].epoch == 0

    def test_get_final_agent_states_empty(self):
        h = SimulationHistory()
        assert h.get_final_agent_states() == {}


# ---------------------------------------------------------------------------
# Tests for MetricsAggregator
# ---------------------------------------------------------------------------

class TestMetricsAggregator:

    def test_full_lifecycle(self):
        """start -> record interactions -> record payoffs -> finalize epoch -> end."""
        agg = MetricsAggregator()
        agg.start_simulation(
            simulation_id="lifecycle-test",
            n_epochs=1,
            steps_per_epoch=5,
            n_agents=2,
            seed=99,
        )

        # Record interactions
        i1 = SoftInteraction(initiator="agent_a", counterparty="agent_b", p=0.9, accepted=True)
        i2 = SoftInteraction(initiator="agent_b", counterparty="agent_a", p=0.3, accepted=False)
        agg.record_interaction(i1)
        agg.record_interaction(i2)

        # Record payoffs
        agg.record_payoff("agent_a", 10.0)
        agg.record_payoff("agent_b", -2.0)

        # Finalize epoch
        agent_states = {
            "agent_a": SimpleNamespace(reputation=1.0, resources=110.0),
            "agent_b": SimpleNamespace(reputation=0.5, resources=95.0),
        }
        snapshot = agg.finalize_epoch(epoch=0, agent_states=agent_states)

        assert snapshot.epoch == 0
        assert snapshot.total_interactions == 2
        assert snapshot.accepted_interactions == 1
        assert snapshot.rejected_interactions == 1
        assert snapshot.avg_p == pytest.approx((0.9 + 0.3) / 2)
        # toxicity = 1 - mean(accepted_p) = 1 - 0.9 = 0.1
        assert snapshot.toxicity_rate == pytest.approx(0.1)
        # quality_gap = mean(accepted_p) - mean(rejected_p) = 0.9 - 0.3 = 0.6
        assert snapshot.quality_gap == pytest.approx(0.6)
        assert snapshot.total_welfare == pytest.approx(10.0 + (-2.0))
        assert snapshot.n_agents == 2

        # End simulation
        history = agg.end_simulation()
        assert history.simulation_id == "lifecycle-test"
        assert history.ended_at is not None
        assert len(history.epoch_snapshots) == 1
        assert "agent_a" in history.agent_snapshots
        assert "agent_b" in history.agent_snapshots

    def test_start_resets_state(self):
        agg = MetricsAggregator()
        agg.start_simulation("sim1", 1, 1, 1)
        agg.record_payoff("x", 5.0)

        # Starting a new simulation should reset
        agg.start_simulation("sim2", 2, 2, 2)
        h = agg.get_history()
        assert h.simulation_id == "sim2"
        assert len(h.epoch_snapshots) == 0

    def test_get_history(self):
        agg = MetricsAggregator()
        agg.start_simulation("h-test", 1, 1, 1)
        h = agg.get_history()
        assert h.simulation_id == "h-test"

    def test_finalize_epoch_with_frozen_and_quarantined(self):
        agg = MetricsAggregator()
        agg.start_simulation("fq-test", 1, 5, 2)

        i = SoftInteraction(initiator="a", counterparty="b", p=0.7, accepted=True)
        agg.record_interaction(i)

        agent_states = {
            "a": SimpleNamespace(reputation=0.2, resources=50.0),
            "b": SimpleNamespace(reputation=0.8, resources=150.0),
        }
        snapshot = agg.finalize_epoch(
            epoch=0,
            agent_states=agent_states,
            frozen_agents={"a"},
            quarantined_agents={"b"},
        )
        assert snapshot.n_frozen == 1
        assert snapshot.n_quarantined == 1

        # Check agent snapshots reflect frozen/quarantined status
        h = agg.get_history()
        a_snap = h.agent_snapshots["a"][-1]
        b_snap = h.agent_snapshots["b"][-1]
        assert a_snap.is_frozen is True
        assert a_snap.is_quarantined is False
        assert b_snap.is_frozen is False
        assert b_snap.is_quarantined is True

    def test_finalize_epoch_with_network_metrics(self):
        agg = MetricsAggregator()
        agg.start_simulation("net-test", 1, 1, 1)
        agent_states = {"a": SimpleNamespace(reputation=0.5, resources=100.0)}
        snapshot = agg.finalize_epoch(
            epoch=0,
            agent_states=agent_states,
            network_metrics={
                "n_edges": 10,
                "avg_degree": 2.5,
                "avg_clustering": 0.3,
                "n_components": 2,
            },
        )
        assert snapshot.n_edges == 10
        assert snapshot.avg_degree == pytest.approx(2.5)
        assert snapshot.avg_clustering == pytest.approx(0.3)
        assert snapshot.n_components == 2

    def test_finalize_epoch_no_interactions(self):
        agg = MetricsAggregator()
        agg.start_simulation("empty-test", 1, 1, 1)
        agent_states = {"a": SimpleNamespace(reputation=0.0, resources=100.0)}
        snapshot = agg.finalize_epoch(epoch=0, agent_states=agent_states)
        assert snapshot.total_interactions == 0
        assert snapshot.avg_p == 0.5  # default when no interactions
        assert snapshot.total_welfare == 0.0

    def test_multiple_epochs(self):
        agg = MetricsAggregator()
        agg.start_simulation("multi", 2, 5, 1)

        for ep in range(2):
            i = SoftInteraction(
                initiator="a", counterparty="b", p=0.5 + ep * 0.2, accepted=True,
            )
            agg.record_interaction(i)
            agg.record_payoff("a", 5.0 + ep)
            agent_states = {
                "a": SimpleNamespace(reputation=0.5 + ep * 0.1, resources=100.0),
                "b": SimpleNamespace(reputation=0.5, resources=100.0),
            }
            agg.finalize_epoch(epoch=ep, agent_states=agent_states)

        h = agg.get_history()
        assert len(h.epoch_snapshots) == 2
        assert h.epoch_snapshots[0].avg_p == pytest.approx(0.5)
        assert h.epoch_snapshots[1].avg_p == pytest.approx(0.7)

    def test_agent_epoch_data_tracks_initiator_and_counterparty(self):
        agg = MetricsAggregator()
        agg.start_simulation("track", 1, 5, 2)

        agg.record_interaction(
            SoftInteraction(initiator="a", counterparty="b", p=0.8, accepted=True)
        )
        agg.record_interaction(
            SoftInteraction(initiator="a", counterparty="b", p=0.6, accepted=True)
        )

        agent_states = {
            "a": SimpleNamespace(reputation=0.5, resources=100.0),
            "b": SimpleNamespace(reputation=0.5, resources=100.0),
        }
        agg.finalize_epoch(epoch=0, agent_states=agent_states)

        h = agg.get_history()
        a_snap = h.agent_snapshots["a"][-1]
        b_snap = h.agent_snapshots["b"][-1]
        assert a_snap.interactions_initiated == 2
        assert b_snap.interactions_received == 2
        assert a_snap.avg_p_initiated == pytest.approx(0.7)
        assert b_snap.avg_p_received == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Tests for Gini coefficient
# ---------------------------------------------------------------------------

class TestGini:

    def test_equal_values_give_zero(self):
        agg = MetricsAggregator()
        result = agg._compute_gini([10.0, 10.0, 10.0, 10.0])
        assert result == pytest.approx(0.0)

    def test_unequal_values_give_positive(self):
        agg = MetricsAggregator()
        result = agg._compute_gini([0.0, 0.0, 0.0, 100.0])
        assert result > 0.0

    def test_empty_returns_zero(self):
        agg = MetricsAggregator()
        assert agg._compute_gini([]) == 0.0

    def test_single_value_returns_zero(self):
        agg = MetricsAggregator()
        assert agg._compute_gini([42.0]) == 0.0

    def test_two_different_values(self):
        agg = MetricsAggregator()
        result = agg._compute_gini([0.0, 100.0])
        assert result > 0.0
        assert result <= 1.0


# ---------------------------------------------------------------------------
# Tests for compute_rolling_average
# ---------------------------------------------------------------------------

class TestComputeRollingAverage:

    def test_returns_smoothed_values(self):
        points = [TimeSeriesPoint(epoch=i, value=float(i)) for i in range(10)]
        smoothed = compute_rolling_average(points, window=3)
        assert len(smoothed) == 10
        # First point: only itself -> 0.0
        assert smoothed[0].value == pytest.approx(0.0)
        # Second point: mean(0, 1) = 0.5
        assert smoothed[1].value == pytest.approx(0.5)
        # Third point: mean(0, 1, 2) = 1.0
        assert smoothed[2].value == pytest.approx(1.0)
        # Fourth point: mean(1, 2, 3) = 2.0
        assert smoothed[3].value == pytest.approx(2.0)

    def test_short_series_returned_as_is(self):
        points = [TimeSeriesPoint(epoch=0, value=5.0), TimeSeriesPoint(epoch=1, value=10.0)]
        result = compute_rolling_average(points, window=5)
        assert len(result) == 2
        assert result[0].value == pytest.approx(5.0)
        assert result[1].value == pytest.approx(10.0)

    def test_preserves_epochs(self):
        points = [TimeSeriesPoint(epoch=i * 10, value=1.0) for i in range(6)]
        smoothed = compute_rolling_average(points, window=3)
        for orig, sm in zip(points, smoothed, strict=False):
            assert sm.epoch == orig.epoch

    def test_constant_series_unchanged(self):
        points = [TimeSeriesPoint(epoch=i, value=7.0) for i in range(10)]
        smoothed = compute_rolling_average(points, window=5)
        for pt in smoothed:
            assert pt.value == pytest.approx(7.0)

    def test_empty_series(self):
        result = compute_rolling_average([], window=5)
        assert result == []


# ---------------------------------------------------------------------------
# Tests for compute_trend
# ---------------------------------------------------------------------------

class TestComputeTrend:

    def test_positive_slope_for_increasing(self):
        points = [TimeSeriesPoint(epoch=i, value=float(i) * 2) for i in range(10)]
        slope, r_sq = compute_trend(points)
        assert slope > 0.0
        assert r_sq == pytest.approx(1.0, abs=1e-6)

    def test_zero_slope_for_constant(self):
        points = [TimeSeriesPoint(epoch=i, value=5.0) for i in range(10)]
        slope, r_sq = compute_trend(points)
        assert slope == pytest.approx(0.0)
        # r_squared should be 1.0 (perfect fit for constant)
        assert r_sq == pytest.approx(1.0)

    def test_negative_slope_for_decreasing(self):
        points = [TimeSeriesPoint(epoch=i, value=100.0 - i * 3) for i in range(10)]
        slope, r_sq = compute_trend(points)
        assert slope < 0.0
        assert r_sq == pytest.approx(1.0, abs=1e-6)

    def test_single_point_returns_zero(self):
        points = [TimeSeriesPoint(epoch=0, value=5.0)]
        slope, r_sq = compute_trend(points)
        assert slope == 0.0
        assert r_sq == 0.0

    def test_empty_returns_zero(self):
        slope, r_sq = compute_trend([])
        assert slope == 0.0
        assert r_sq == 0.0

    def test_two_points_linear(self):
        points = [TimeSeriesPoint(epoch=0, value=0.0), TimeSeriesPoint(epoch=1, value=10.0)]
        slope, r_sq = compute_trend(points)
        assert slope == pytest.approx(10.0)
        assert r_sq == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests for export: history_to_epoch_records / history_to_agent_records
# ---------------------------------------------------------------------------

class TestHistoryToRecords:

    def test_epoch_records_structure(self):
        h = _make_history_with_two_epochs()
        records = history_to_epoch_records(h)
        assert len(records) == 2
        r0 = records[0]
        assert r0["simulation_id"] == "test-sim"
        assert r0["epoch"] == 0
        assert r0["total_interactions"] == 10
        assert r0["accepted_interactions"] == 8
        assert "toxicity_rate" in r0
        assert "avg_p" in r0

    def test_agent_records_structure(self):
        h = _make_history_with_two_epochs()
        records = history_to_agent_records(h)
        # 2 agents * 2 epochs = 4 records
        assert len(records) == 4
        r = records[0]
        assert "agent_id" in r
        assert "epoch" in r
        assert "reputation" in r
        assert "resources" in r
        assert "total_payoff" in r

    def test_empty_history_gives_empty_records(self):
        h = SimulationHistory()
        assert history_to_epoch_records(h) == []
        assert history_to_agent_records(h) == []


# ---------------------------------------------------------------------------
# Tests for JSON export/import round-trip
# ---------------------------------------------------------------------------

class TestJsonExportImport:

    def test_round_trip(self, tmp_path):
        h = _make_history_with_two_epochs()
        out_path = tmp_path / "history.json"
        returned_path = export_to_json(h, out_path)
        assert returned_path.exists()

        loaded = load_from_json(out_path)
        assert loaded.simulation_id == h.simulation_id
        assert loaded.n_epochs == h.n_epochs
        assert loaded.steps_per_epoch == h.steps_per_epoch
        assert loaded.n_agents == h.n_agents
        assert loaded.seed == h.seed

        # Epoch snapshots preserved
        assert len(loaded.epoch_snapshots) == len(h.epoch_snapshots)
        for orig, loaded_snap in zip(h.epoch_snapshots, loaded.epoch_snapshots, strict=False):
            assert loaded_snap.epoch == orig.epoch
            assert loaded_snap.total_interactions == orig.total_interactions
            assert loaded_snap.toxicity_rate == pytest.approx(orig.toxicity_rate)
            assert loaded_snap.avg_p == pytest.approx(orig.avg_p)
            assert loaded_snap.total_welfare == pytest.approx(orig.total_welfare)

        # Agent snapshots preserved
        assert set(loaded.agent_snapshots.keys()) == set(h.agent_snapshots.keys())
        for agent_id in h.agent_snapshots:
            assert len(loaded.agent_snapshots[agent_id]) == len(h.agent_snapshots[agent_id])

    def test_json_file_is_valid_json(self, tmp_path):
        h = _make_history_with_two_epochs()
        out_path = tmp_path / "output.json"
        export_to_json(h, out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "simulation_id" in data
        assert "epoch_snapshots" in data
        assert "agent_snapshots" in data

    def test_round_trip_preserves_timestamps(self, tmp_path):
        h = _make_history_with_two_epochs()
        out_path = tmp_path / "ts_test.json"
        export_to_json(h, out_path)
        loaded = load_from_json(out_path)
        assert loaded.started_at == h.started_at
        assert loaded.ended_at == h.ended_at

    def test_export_creates_parent_dirs(self, tmp_path):
        h = _make_history_with_two_epochs()
        out_path = tmp_path / "sub" / "dir" / "history.json"
        export_to_json(h, out_path)
        assert out_path.exists()

    def test_custom_indent(self, tmp_path):
        h = _make_history_with_two_epochs()
        out_path = tmp_path / "indent4.json"
        export_to_json(h, out_path, indent=4)
        content = out_path.read_text()
        # With indent=4, lines should start with 4-space indentation
        assert "    " in content

    def test_load_empty_history_from_json(self, tmp_path):
        h = SimulationHistory(simulation_id="empty")
        out_path = tmp_path / "empty.json"
        export_to_json(h, out_path)
        loaded = load_from_json(out_path)
        assert loaded.simulation_id == "empty"
        assert len(loaded.epoch_snapshots) == 0
        assert len(loaded.agent_snapshots) == 0


# ---------------------------------------------------------------------------
# Tests for CSV export/import (requires pandas)
# ---------------------------------------------------------------------------

class TestCsvExportImport:

    def test_csv_round_trip(self, tmp_path):
        pytest.importorskip("pandas")
        from swarm.analysis.export import export_to_csv, load_from_csv

        h = _make_history_with_two_epochs()
        paths = export_to_csv(h, tmp_path, prefix="test")

        assert "epochs" in paths
        assert "agents" in paths
        assert paths["epochs"].exists()
        assert paths["agents"].exists()

        loaded = load_from_csv(paths["epochs"], paths["agents"])
        assert len(loaded.epoch_snapshots) == 2
        assert loaded.simulation_id == "test-sim"

        # Check agent snapshots loaded
        assert len(loaded.agent_snapshots) > 0

    def test_csv_epochs_only(self, tmp_path):
        pytest.importorskip("pandas")
        from swarm.analysis.export import export_to_csv, load_from_csv

        h = _make_history_with_two_epochs()
        paths = export_to_csv(h, tmp_path, prefix="ep_only")

        loaded = load_from_csv(paths["epochs"])
        assert len(loaded.epoch_snapshots) == 2
        assert len(loaded.agent_snapshots) == 0

    def test_csv_creates_output_dir(self, tmp_path):
        pytest.importorskip("pandas")
        from swarm.analysis.export import export_to_csv

        h = _make_history_with_two_epochs()
        output_dir = tmp_path / "new_sub_dir"
        paths = export_to_csv(h, output_dir, prefix="sub")
        assert output_dir.exists()
        assert paths["epochs"].exists()

    def test_csv_empty_history(self, tmp_path):
        pytest.importorskip("pandas")
        from swarm.analysis.export import export_to_csv

        h = SimulationHistory()
        paths = export_to_csv(h, tmp_path, prefix="empty")
        # No records -> no files created
        assert len(paths) == 0


# ---------------------------------------------------------------------------
# Tests for generate_summary_report
# ---------------------------------------------------------------------------

class TestGenerateSummaryReport:

    def test_report_contains_key_strings(self):
        h = _make_history_with_two_epochs()
        report = generate_summary_report(h)
        assert "SIMULATION SUMMARY REPORT" in report
        assert "test-sim" in report
        assert "Epochs:" in report
        assert "Toxicity Rate:" in report
        assert "Quality Gap:" in report
        assert "Total Welfare:" in report
        assert "Gini Coefficient:" in report
        assert "Agent Summary:" in report
        assert "agent_a" in report
        assert "agent_b" in report

    def test_report_for_empty_history(self):
        h = SimulationHistory(simulation_id="empty-sim")
        report = generate_summary_report(h)
        assert "SIMULATION SUMMARY REPORT" in report
        assert "empty-sim" in report

    def test_report_shows_agent_status(self):
        h = SimulationHistory(
            simulation_id="status-test",
            started_at=datetime(2025, 1, 1),
            ended_at=datetime(2025, 1, 2),
        )
        h.add_epoch_snapshot(EpochSnapshot(
            epoch=0,
            timestamp=datetime(2025, 1, 1),
            total_interactions=5,
            accepted_interactions=5,
        ))
        h.add_agent_snapshot(AgentSnapshot(
            agent_id="frozen_agent", epoch=0, is_frozen=True,
        ))
        h.add_agent_snapshot(AgentSnapshot(
            agent_id="normal_agent", epoch=0, is_frozen=False, is_quarantined=False,
        ))
        report = generate_summary_report(h)
        assert "FROZEN" in report
        assert "active" in report

    def test_report_returns_string(self):
        h = _make_history_with_two_epochs()
        report = generate_summary_report(h)
        assert isinstance(report, str)
        assert len(report) > 0


# ---------------------------------------------------------------------------
# Tests for MetricsAggregator with security/collusion/capability reports
# ---------------------------------------------------------------------------

class TestMetricsAggregatorOptionalReports:

    def test_security_report(self):
        agg = MetricsAggregator()
        agg.start_simulation("sec-test", 1, 1, 1)
        agent_states = {"a": SimpleNamespace(reputation=0.5, resources=100.0)}
        sec = SimpleNamespace(
            ecosystem_threat_level=0.7,
            active_threat_count=3,
            contagion_depth=2,
        )
        snapshot = agg.finalize_epoch(epoch=0, agent_states=agent_states, security_report=sec)
        assert snapshot.ecosystem_threat_level == pytest.approx(0.7)
        assert snapshot.active_threats == 3
        assert snapshot.contagion_depth == 2

    def test_collusion_report(self):
        agg = MetricsAggregator()
        agg.start_simulation("col-test", 1, 1, 1)
        agent_states = {"a": SimpleNamespace(reputation=0.5, resources=100.0)}
        col = SimpleNamespace(
            ecosystem_collusion_risk=0.4,
            n_flagged_pairs=5,
        )
        snapshot = agg.finalize_epoch(epoch=0, agent_states=agent_states, collusion_report=col)
        assert snapshot.ecosystem_collusion_risk == pytest.approx(0.4)
        assert snapshot.n_flagged_pairs == 5

    def test_capability_metrics(self):
        agg = MetricsAggregator()
        agg.start_simulation("cap-test", 1, 1, 1)
        agent_states = {"a": SimpleNamespace(reputation=0.5, resources=100.0)}
        cap = SimpleNamespace(
            avg_coordination_score=0.8,
            avg_synergy_score=0.6,
            tasks_completed=12,
        )
        snapshot = agg.finalize_epoch(epoch=0, agent_states=agent_states, capability_metrics=cap)
        assert snapshot.avg_coordination_score == pytest.approx(0.8)
        assert snapshot.avg_synergy_score == pytest.approx(0.6)
        assert snapshot.tasks_completed == 12


class TestIncoherenceScalingAggregation:

    def test_aggregate_incoherence_scaling_groups_rows(self):
        rows = [
            {
                "horizon_tier": "short",
                "branching_tier": "low",
                "incoherence_index": 0.2,
                "error_rate": 0.3,
                "disagreement_rate": 0.1,
            },
            {
                "horizon_tier": "short",
                "branching_tier": "low",
                "incoherence_index": 0.4,
                "error_rate": 0.5,
                "disagreement_rate": 0.3,
            },
        ]
        summary = aggregate_incoherence_scaling(rows)
        assert len(summary) == 1
        assert summary[0]["n_runs"] == 2
        assert summary[0]["mean_incoherence_index"] == pytest.approx(0.3)

    def test_build_scaling_curve_points_filters_and_orders(self):
        aggregated_rows = [
            {
                "horizon_tier": "medium",
                "branching_tier": "low",
                "mean_incoherence_index": 0.4,
                "mean_error_rate": 0.5,
                "mean_disagreement_rate": 0.2,
            },
            {
                "horizon_tier": "short",
                "branching_tier": "low",
                "mean_incoherence_index": 0.2,
                "mean_error_rate": 0.3,
                "mean_disagreement_rate": 0.1,
            },
        ]
        points = build_scaling_curve_points(aggregated_rows, x_axis="horizon", fixed_tier="low")
        assert points["x_labels"] == ["medium", "short"] or points["x_labels"] == ["short", "medium"]
        assert len(points["incoherence_index"]) == 2

    def test_create_incoherence_scaling_data(self):
        rows = [
            {
                "horizon_tier": "short",
                "branching_tier": "low",
                "mean_incoherence_index": 0.2,
                "mean_error_rate": 0.3,
                "mean_disagreement_rate": 0.1,
            }
        ]
        plot_data = create_incoherence_scaling_data(rows)
        assert plot_data["labels"] == ["short|low"]
        assert plot_data["incoherence_index"] == [0.2]
