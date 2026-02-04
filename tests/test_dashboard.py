"""Tests for dashboard and visualization components."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.analysis.aggregation import (
    AgentSnapshot,
    EpochSnapshot,
    MetricsAggregator,
    SimulationHistory,
    TimeSeriesPoint,
    compute_rolling_average,
    compute_trend,
)
from src.analysis.export import (
    export_to_csv,
    export_to_json,
    generate_summary_report,
    history_to_agent_records,
    history_to_epoch_records,
    load_from_csv,
    load_from_json,
)
from src.analysis.plots import (
    create_agent_comparison_data,
    create_agent_trajectory_data,
    create_distribution_data,
    create_heatmap_data,
    create_network_graph_data,
    create_scatter_data,
    create_time_series_data,
)
from src.models.interaction import InteractionType, SoftInteraction


# =============================================================================
# Test Fixtures
# =============================================================================


def create_test_history(n_epochs: int = 5, n_agents: int = 3) -> SimulationHistory:
    """Create a test simulation history."""
    history = SimulationHistory(
        simulation_id="test_sim_001",
        started_at=datetime.now(),
        n_epochs=n_epochs,
        steps_per_epoch=10,
        n_agents=n_agents,
        seed=42,
    )

    # Create epoch snapshots with varying metrics
    for epoch in range(n_epochs):
        toxicity = 0.2 + 0.05 * epoch  # Increasing
        welfare = 100 - 5 * epoch  # Decreasing
        reputation = 5 + epoch  # Increasing

        snapshot = EpochSnapshot(
            epoch=epoch,
            timestamp=datetime.now() + timedelta(seconds=epoch * 10),
            total_interactions=10 + epoch,
            accepted_interactions=7 + epoch,
            rejected_interactions=3,
            toxicity_rate=toxicity,
            quality_gap=0.1 - 0.02 * epoch,
            avg_p=0.6 - 0.02 * epoch,
            total_welfare=welfare,
            avg_payoff=welfare / n_agents,
            payoff_std=5.0,
            gini_coefficient=0.3,
            n_agents=n_agents,
            n_frozen=epoch // 3,
            avg_reputation=reputation,
            n_edges=n_agents * (n_agents - 1) // 2,
            avg_degree=n_agents - 1,
            ecosystem_threat_level=0.1 + 0.05 * epoch,
            ecosystem_collusion_risk=0.05 * epoch,
        )
        history.add_epoch_snapshot(snapshot)

    # Create agent snapshots
    for agent_idx in range(n_agents):
        agent_id = f"agent_{agent_idx}"
        for epoch in range(n_epochs):
            snapshot = AgentSnapshot(
                agent_id=agent_id,
                epoch=epoch,
                reputation=5.0 + agent_idx + epoch * 0.5,
                resources=100.0 - agent_idx * 10,
                interactions_initiated=5 + agent_idx,
                interactions_received=3 + agent_idx,
                avg_p_initiated=0.6 + 0.05 * agent_idx,
                avg_p_received=0.5 + 0.05 * agent_idx,
                total_payoff=20.0 + agent_idx * 5 - epoch,
                is_frozen=False,
                is_quarantined=False,
            )
            history.add_agent_snapshot(snapshot)

    history.ended_at = datetime.now() + timedelta(seconds=n_epochs * 10)
    return history


def create_test_interaction(
    initiator: str,
    counterparty: str,
    p: float = 0.5,
    accepted: bool = True,
) -> SoftInteraction:
    """Create a test interaction."""
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        interaction_type=InteractionType.COLLABORATION,
        accepted=accepted,
        p=p,
    )


# =============================================================================
# TimeSeriesPoint Tests
# =============================================================================


class TestTimeSeriesPoint:
    """Tests for TimeSeriesPoint dataclass."""

    def test_create_point(self):
        """Test creating a time series point."""
        point = TimeSeriesPoint(epoch=5, value=0.75)
        assert point.epoch == 5
        assert point.value == 0.75
        assert point.timestamp is None

    def test_create_with_timestamp(self):
        """Test creating with timestamp."""
        ts = datetime.now()
        point = TimeSeriesPoint(epoch=3, value=0.5, timestamp=ts)
        assert point.timestamp == ts


# =============================================================================
# AgentSnapshot Tests
# =============================================================================


class TestAgentSnapshot:
    """Tests for AgentSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating an agent snapshot."""
        snapshot = AgentSnapshot(
            agent_id="agent_1",
            epoch=5,
            reputation=10.5,
            resources=95.0,
        )
        assert snapshot.agent_id == "agent_1"
        assert snapshot.epoch == 5
        assert snapshot.reputation == 10.5
        assert snapshot.resources == 95.0

    def test_default_values(self):
        """Test default values."""
        snapshot = AgentSnapshot(agent_id="test", epoch=0)
        assert snapshot.reputation == 0.0
        assert snapshot.resources == 100.0
        assert snapshot.interactions_initiated == 0
        assert not snapshot.is_frozen
        assert not snapshot.is_quarantined


# =============================================================================
# EpochSnapshot Tests
# =============================================================================


class TestEpochSnapshot:
    """Tests for EpochSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating an epoch snapshot."""
        snapshot = EpochSnapshot(
            epoch=10,
            total_interactions=50,
            toxicity_rate=0.3,
        )
        assert snapshot.epoch == 10
        assert snapshot.total_interactions == 50
        assert snapshot.toxicity_rate == 0.3

    def test_default_values(self):
        """Test default values."""
        snapshot = EpochSnapshot(epoch=0)
        assert snapshot.accepted_interactions == 0
        assert snapshot.quality_gap == 0.0
        assert snapshot.avg_p == 0.5
        assert snapshot.n_components == 1


# =============================================================================
# SimulationHistory Tests
# =============================================================================


class TestSimulationHistory:
    """Tests for SimulationHistory class."""

    def test_add_epoch_snapshot(self):
        """Test adding epoch snapshots."""
        history = SimulationHistory()
        snapshot = EpochSnapshot(epoch=0, toxicity_rate=0.2)
        history.add_epoch_snapshot(snapshot)

        assert len(history.epoch_snapshots) == 1
        assert history.epoch_snapshots[0].toxicity_rate == 0.2

    def test_add_agent_snapshot(self):
        """Test adding agent snapshots."""
        history = SimulationHistory()
        snapshot = AgentSnapshot(agent_id="agent_1", epoch=0, reputation=5.0)
        history.add_agent_snapshot(snapshot)

        assert "agent_1" in history.agent_snapshots
        assert len(history.agent_snapshots["agent_1"]) == 1

    def test_get_time_series(self):
        """Test extracting time series."""
        history = create_test_history(n_epochs=5)
        series = history.get_time_series("toxicity_rate")

        assert len(series) == 5
        assert series[0].epoch == 0
        assert series[-1].epoch == 4

    def test_get_agent_time_series(self):
        """Test extracting agent time series."""
        history = create_test_history(n_epochs=5, n_agents=3)
        series = history.get_agent_time_series("agent_0", "reputation")

        assert len(series) == 5

    def test_get_final_agent_states(self):
        """Test getting final agent states."""
        history = create_test_history(n_epochs=5, n_agents=3)
        final = history.get_final_agent_states()

        assert len(final) == 3
        for agent_id, state in final.items():
            assert state.epoch == 4  # Last epoch


# =============================================================================
# MetricsAggregator Tests
# =============================================================================


class TestMetricsAggregator:
    """Tests for MetricsAggregator class."""

    def test_start_simulation(self):
        """Test starting simulation tracking."""
        aggregator = MetricsAggregator()
        aggregator.start_simulation(
            simulation_id="test_001",
            n_epochs=10,
            steps_per_epoch=5,
            n_agents=3,
            seed=42,
        )

        history = aggregator.get_history()
        assert history.simulation_id == "test_001"
        assert history.n_epochs == 10
        assert history.seed == 42

    def test_record_interaction(self):
        """Test recording interactions."""
        aggregator = MetricsAggregator()
        aggregator.start_simulation("test", 10, 5, 2)

        interaction = create_test_interaction("a", "b", p=0.7)
        aggregator.record_interaction(interaction)

        # Check internal tracking
        assert "a" in aggregator._agent_epoch_data
        assert aggregator._agent_epoch_data["a"]["interactions_initiated"] == 1

    def test_record_payoff(self):
        """Test recording payoffs."""
        aggregator = MetricsAggregator()
        aggregator.start_simulation("test", 10, 5, 2)

        aggregator.record_payoff("agent_1", 15.0)
        aggregator.record_payoff("agent_1", 10.0)

        assert len(aggregator._agent_epoch_data["agent_1"]["payoffs"]) == 2

    def test_finalize_epoch(self):
        """Test finalizing an epoch."""
        from src.models.agent import AgentState

        aggregator = MetricsAggregator()
        aggregator.start_simulation("test", 10, 5, 2, seed=42)

        # Record some interactions
        for i in range(5):
            interaction = create_test_interaction("a", "b", p=0.6 + i * 0.05, accepted=True)
            aggregator.record_interaction(interaction)
            aggregator.record_payoff("a", 10.0)
            aggregator.record_payoff("b", 8.0)

        # Create mock agent states
        agent_states = {
            "a": AgentState(agent_id="a", reputation=5.0, resources=100.0),
            "b": AgentState(agent_id="b", reputation=3.0, resources=90.0),
        }

        snapshot = aggregator.finalize_epoch(
            epoch=0,
            agent_states=agent_states,
        )

        assert snapshot.epoch == 0
        assert snapshot.total_interactions == 5
        assert snapshot.accepted_interactions == 5
        assert snapshot.n_agents == 2

    def test_end_simulation(self):
        """Test ending simulation."""
        aggregator = MetricsAggregator()
        aggregator.start_simulation("test", 10, 5, 2)

        history = aggregator.end_simulation()

        assert history.ended_at is not None


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestComputeRollingAverage:
    """Tests for compute_rolling_average function."""

    def test_rolling_average(self):
        """Test rolling average computation."""
        points = [
            TimeSeriesPoint(epoch=i, value=float(i))
            for i in range(10)
        ]

        smoothed = compute_rolling_average(points, window=3)

        assert len(smoothed) == 10
        # First point unchanged
        assert smoothed[0].value == 0.0
        # Third point is average of 0, 1, 2
        assert abs(smoothed[2].value - 1.0) < 0.01

    def test_small_input(self):
        """Test with input smaller than window."""
        points = [TimeSeriesPoint(epoch=0, value=5.0)]
        smoothed = compute_rolling_average(points, window=5)

        assert smoothed == points


class TestComputeTrend:
    """Tests for compute_trend function."""

    def test_increasing_trend(self):
        """Test detecting increasing trend."""
        points = [
            TimeSeriesPoint(epoch=i, value=float(i * 2))
            for i in range(10)
        ]

        slope, r_squared = compute_trend(points)

        assert slope > 0
        assert r_squared > 0.9

    def test_decreasing_trend(self):
        """Test detecting decreasing trend."""
        points = [
            TimeSeriesPoint(epoch=i, value=10.0 - i)
            for i in range(10)
        ]

        slope, r_squared = compute_trend(points)

        assert slope < 0

    def test_constant_values(self):
        """Test with constant values."""
        points = [
            TimeSeriesPoint(epoch=i, value=5.0)
            for i in range(10)
        ]

        slope, r_squared = compute_trend(points)

        assert slope == 0.0
        assert r_squared == 1.0


# =============================================================================
# Plot Data Creation Tests
# =============================================================================


class TestCreateTimeSeriesData:
    """Tests for create_time_series_data function."""

    def test_create_data(self):
        """Test creating time series data."""
        history = create_test_history(n_epochs=5)
        data = create_time_series_data(history, ["toxicity_rate", "total_welfare"])

        assert "epochs" in data
        assert "toxicity_rate" in data
        assert "total_welfare" in data
        assert len(data["epochs"]) == 5

    def test_empty_history(self):
        """Test with empty history."""
        history = SimulationHistory()
        data = create_time_series_data(history, ["toxicity_rate"])

        assert len(data["epochs"]) == 0


class TestCreateAgentComparisonData:
    """Tests for create_agent_comparison_data function."""

    def test_create_data(self):
        """Test creating agent comparison data."""
        history = create_test_history(n_epochs=5, n_agents=3)
        data = create_agent_comparison_data(history, "total_payoff")

        assert "agent_ids" in data
        assert "values" in data
        assert len(data["agent_ids"]) == 3


class TestCreateAgentTrajectoryData:
    """Tests for create_agent_trajectory_data function."""

    def test_create_data(self):
        """Test creating agent trajectory data."""
        history = create_test_history(n_epochs=5, n_agents=3)
        data = create_agent_trajectory_data(
            history,
            ["agent_0", "agent_1"],
            "reputation",
        )

        assert "agent_0" in data
        assert "agent_1" in data


class TestCreateDistributionData:
    """Tests for create_distribution_data function."""

    def test_create_data(self):
        """Test creating distribution data."""
        history = create_test_history(n_epochs=10)
        data = create_distribution_data(history.epoch_snapshots, "avg_p")

        assert "values" in data
        assert len(data["values"]) == 10


class TestCreateHeatmapData:
    """Tests for create_heatmap_data function."""

    def test_create_data(self):
        """Test creating heatmap data."""
        history = create_test_history(n_epochs=5, n_agents=3)
        data = create_heatmap_data(history, "reputation", col_agents=True)

        assert "matrix" in data
        assert "row_labels" in data
        assert "col_labels" in data
        assert len(data["matrix"]) == 5  # epochs
        assert len(data["col_labels"]) == 3  # agents


class TestCreateNetworkGraphData:
    """Tests for create_network_graph_data function."""

    def test_create_data(self):
        """Test creating network graph data."""
        edges = [
            ("a", "b", 1.0),
            ("b", "c", 0.5),
            ("a", "c", 0.8),
        ]
        data = create_network_graph_data(edges)

        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 3

    def test_with_attributes(self):
        """Test with node attributes."""
        edges = [("a", "b", 1.0)]
        attrs = {"a": {"reputation": 5.0}, "b": {"reputation": 3.0}}
        data = create_network_graph_data(edges, attrs)

        node_a = next(n for n in data["nodes"] if n["id"] == "a")
        assert node_a["reputation"] == 5.0


class TestCreateScatterData:
    """Tests for create_scatter_data function."""

    def test_create_data(self):
        """Test creating scatter data."""
        history = create_test_history(n_epochs=10)
        data = create_scatter_data(history, "toxicity_rate", "total_welfare")

        assert "x" in data
        assert "y" in data
        assert len(data["x"]) == 10

    def test_with_color(self):
        """Test with color metric."""
        history = create_test_history(n_epochs=10)
        data = create_scatter_data(
            history,
            "toxicity_rate",
            "total_welfare",
            "ecosystem_threat_level",
        )

        assert "colors" in data


# =============================================================================
# Export Tests
# =============================================================================


class TestHistoryToRecords:
    """Tests for history to records conversion."""

    def test_epoch_records(self):
        """Test converting to epoch records."""
        history = create_test_history(n_epochs=5)
        records = history_to_epoch_records(history)

        assert len(records) == 5
        assert "toxicity_rate" in records[0]
        assert "total_welfare" in records[0]

    def test_agent_records(self):
        """Test converting to agent records."""
        history = create_test_history(n_epochs=5, n_agents=3)
        records = history_to_agent_records(history)

        # 5 epochs * 3 agents = 15 records
        assert len(records) == 15
        assert "agent_id" in records[0]
        assert "reputation" in records[0]


class TestExportToCsv:
    """Tests for CSV export."""

    def test_export(self):
        """Test exporting to CSV."""
        pytest.importorskip("pandas")

        history = create_test_history(n_epochs=5, n_agents=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_to_csv(history, tmpdir, "test")

            assert "epochs" in paths
            assert "agents" in paths
            assert Path(paths["epochs"]).exists()
            assert Path(paths["agents"]).exists()


class TestExportToJson:
    """Tests for JSON export."""

    def test_export(self):
        """Test exporting to JSON."""
        history = create_test_history(n_epochs=5, n_agents=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "history.json"
            export_to_json(history, output_path)

            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            assert data["simulation_id"] == "test_sim_001"
            assert len(data["epoch_snapshots"]) == 5

    def test_roundtrip(self):
        """Test JSON export and import roundtrip."""
        history = create_test_history(n_epochs=5, n_agents=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "history.json"
            export_to_json(history, output_path)

            loaded = load_from_json(output_path)

            assert loaded.simulation_id == history.simulation_id
            assert len(loaded.epoch_snapshots) == len(history.epoch_snapshots)


class TestLoadFromCsv:
    """Tests for CSV import."""

    def test_load(self):
        """Test loading from CSV."""
        pytest.importorskip("pandas")

        history = create_test_history(n_epochs=5, n_agents=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_to_csv(history, tmpdir, "test")

            loaded = load_from_csv(paths["epochs"], paths["agents"])

            assert len(loaded.epoch_snapshots) == 5


class TestGenerateSummaryReport:
    """Tests for summary report generation."""

    def test_generate_report(self):
        """Test generating summary report."""
        history = create_test_history(n_epochs=5, n_agents=3)
        report = generate_summary_report(history)

        assert "SIMULATION SUMMARY REPORT" in report
        assert "test_sim_001" in report
        assert "Toxicity Rate" in report
        assert "agent_0" in report

    def test_empty_history(self):
        """Test with empty history."""
        history = SimulationHistory()
        report = generate_summary_report(history)

        assert "SIMULATION SUMMARY REPORT" in report


# =============================================================================
# Plotly Chart Tests (skip if plotly not installed)
# =============================================================================


class TestPlotlyCharts:
    """Tests for Plotly chart generators."""

    @pytest.fixture
    def plotly(self):
        """Skip if plotly not installed."""
        return pytest.importorskip("plotly")

    def test_plotly_time_series(self, plotly):
        """Test time series chart."""
        from src.analysis.plots import plotly_time_series

        history = create_test_history(n_epochs=5)
        data = create_time_series_data(history, ["toxicity_rate"])
        fig = plotly_time_series(data, "Test Chart")

        assert fig is not None

    def test_plotly_bar_chart(self, plotly):
        """Test bar chart."""
        from src.analysis.plots import plotly_bar_chart

        history = create_test_history(n_epochs=5, n_agents=3)
        data = create_agent_comparison_data(history, "total_payoff")
        fig = plotly_bar_chart(data, "Agent Payoffs")

        assert fig is not None

    def test_plotly_gauge(self, plotly):
        """Test gauge chart."""
        from src.analysis.plots import plotly_gauge

        fig = plotly_gauge(0.65, "Threat Level")
        assert fig is not None

    def test_plotly_network(self, plotly):
        """Test network chart."""
        from src.analysis.plots import plotly_network

        edges = [("a", "b", 1.0), ("b", "c", 0.5)]
        data = create_network_graph_data(edges)
        fig = plotly_network(data, "Network")

        assert fig is not None
