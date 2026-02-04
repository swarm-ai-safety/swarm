"""Tests for the dashboard module."""

import pytest
import json
from pathlib import Path
import tempfile

from src.analysis.dashboard import (
    DashboardConfig,
    DashboardState,
    MetricSnapshot,
    AgentSnapshot,
    extract_metrics_from_orchestrator,
    extract_agent_snapshots,
    create_dashboard_file,
)


class TestDashboardConfig:
    """Tests for DashboardConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DashboardConfig()

        assert config.title == "Distributional AGI Safety Sandbox"
        assert config.refresh_rate_ms == 1000
        assert config.show_agent_details is True
        assert config.max_history_points == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = DashboardConfig(
            title="Custom Dashboard",
            refresh_rate_ms=500,
            max_history_points=50,
        )

        assert config.title == "Custom Dashboard"
        assert config.refresh_rate_ms == 500
        assert config.max_history_points == 50


class TestMetricSnapshot:
    """Tests for MetricSnapshot."""

    def test_create_snapshot(self):
        """Test creating a metric snapshot."""
        snapshot = MetricSnapshot(
            epoch=5,
            step=3,
            toxicity_rate=0.15,
            quality_gap=-0.02,
            avg_payoff=5.5,
        )

        assert snapshot.epoch == 5
        assert snapshot.step == 3
        assert snapshot.toxicity_rate == 0.15
        assert snapshot.quality_gap == -0.02

    def test_to_dict(self):
        """Test converting to dictionary."""
        snapshot = MetricSnapshot(
            epoch=10,
            step=5,
            toxicity_rate=0.2,
            total_welfare=850.0,
        )

        d = snapshot.to_dict()

        assert d["epoch"] == 10
        assert d["step"] == 5
        assert d["toxicity_rate"] == 0.2
        assert d["total_welfare"] == 850.0


class TestAgentSnapshot:
    """Tests for AgentSnapshot."""

    def test_create_snapshot(self):
        """Test creating an agent snapshot."""
        snapshot = AgentSnapshot(
            agent_id="agent_1",
            agent_type="honest",
            reputation=10.5,
            resources=100.0,
            interactions=25,
        )

        assert snapshot.agent_id == "agent_1"
        assert snapshot.agent_type == "honest"
        assert snapshot.reputation == 10.5

    def test_to_dict(self):
        """Test converting to dictionary."""
        snapshot = AgentSnapshot(
            agent_id="agent_2",
            agent_type="adversarial",
            reputation=3.0,
            is_frozen=True,
        )

        d = snapshot.to_dict()

        assert d["agent_id"] == "agent_2"
        assert d["agent_type"] == "adversarial"
        assert d["is_frozen"] is True


class TestDashboardState:
    """Tests for DashboardState."""

    def test_initial_state(self):
        """Test initial state values."""
        state = DashboardState()

        assert len(state.metric_history) == 0
        assert len(state.agent_snapshots) == 0
        assert state.is_running is False
        assert state.current_epoch == 0

    def test_update_metrics(self):
        """Test updating metrics."""
        state = DashboardState()

        snapshot = MetricSnapshot(epoch=1, step=5, toxicity_rate=0.1)
        state.update_metrics(snapshot)

        assert len(state.metric_history) == 1
        assert state.current_epoch == 1
        assert state.current_step == 5

    def test_metric_history_trimming(self):
        """Test that history is trimmed to max_history_points."""
        config = DashboardConfig(max_history_points=5)
        state = DashboardState(config)

        for i in range(10):
            state.update_metrics(MetricSnapshot(epoch=i, step=0))

        assert len(state.metric_history) == 5
        assert state.metric_history[0].epoch == 5  # First 5 trimmed

    def test_update_agent(self):
        """Test updating agent snapshots."""
        state = DashboardState()

        agent = AgentSnapshot(
            agent_id="agent_1",
            agent_type="honest",
            reputation=5.0,
        )
        state.update_agent(agent)

        assert "agent_1" in state.agent_snapshots
        assert state.agent_snapshots["agent_1"].reputation == 5.0

        # Update same agent
        agent2 = AgentSnapshot(
            agent_id="agent_1",
            agent_type="honest",
            reputation=7.0,
        )
        state.update_agent(agent2)

        assert state.agent_snapshots["agent_1"].reputation == 7.0

    def test_add_event(self):
        """Test adding events."""
        state = DashboardState()

        state.add_event({"type": "interaction", "description": "Test event"})

        assert len(state.events) == 1
        assert state.events[0]["type"] == "interaction"

    def test_get_metric_series(self):
        """Test getting metric time series."""
        state = DashboardState()

        for i in range(5):
            state.update_metrics(MetricSnapshot(
                epoch=i,
                step=0,
                toxicity_rate=i * 0.1,
            ))

        epochs, values = state.get_metric_series("toxicity_rate")

        assert epochs == [0, 1, 2, 3, 4]
        assert values == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])

    def test_get_agent_rankings(self):
        """Test getting agent rankings."""
        state = DashboardState()

        state.update_agent(AgentSnapshot("a1", "honest", reputation=10.0))
        state.update_agent(AgentSnapshot("a2", "honest", reputation=5.0))
        state.update_agent(AgentSnapshot("a3", "honest", reputation=15.0))

        rankings = state.get_agent_rankings(sort_by="reputation")

        assert rankings[0].agent_id == "a3"  # Highest reputation
        assert rankings[1].agent_id == "a1"
        assert rankings[2].agent_id == "a2"

    def test_export_to_json(self):
        """Test exporting state to JSON."""
        state = DashboardState()

        state.update_metrics(MetricSnapshot(epoch=1, step=0, toxicity_rate=0.1))
        state.update_agent(AgentSnapshot("a1", "honest", reputation=10.0))
        state.add_event({"type": "test"})

        json_str = state.export_to_json()
        data = json.loads(json_str)

        assert data["current_epoch"] == 1
        assert len(data["metric_history"]) == 1
        assert len(data["agents"]) == 1
        assert len(data["recent_events"]) == 1


class TestOrchestratorExtraction:
    """Tests for extracting data from orchestrator."""

    def test_extract_metrics(self):
        """Test extracting metrics from orchestrator."""
        from src.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(n_epochs=2, steps_per_epoch=2)
        orchestrator = Orchestrator(config)

        snapshot = extract_metrics_from_orchestrator(orchestrator)

        assert isinstance(snapshot, MetricSnapshot)
        assert snapshot.epoch == 0
        assert snapshot.step == 0

    def test_extract_agent_snapshots(self):
        """Test extracting agent snapshots."""
        from src.core.orchestrator import Orchestrator, OrchestratorConfig
        from src.agents.honest import HonestAgent

        config = OrchestratorConfig()
        orchestrator = Orchestrator(config)

        agent = HonestAgent(agent_id="test_agent")
        orchestrator.register_agent(agent)

        snapshots = extract_agent_snapshots(orchestrator)

        assert len(snapshots) == 1
        assert snapshots[0].agent_id == "test_agent"
        assert snapshots[0].agent_type == "honest"


class TestDashboardFileCreation:
    """Tests for dashboard file creation."""

    def test_create_dashboard_file(self):
        """Test creating the dashboard file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_dashboard.py")
            result_path = create_dashboard_file(output_path)

            assert Path(result_path).exists()

            with open(result_path) as f:
                content = f.read()

            assert "streamlit" in content
            assert "def main():" in content
            assert "Distributional AGI Safety" in content

    def test_dashboard_file_contains_charts(self):
        """Test that dashboard file contains chart code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_dashboard.py")
            create_dashboard_file(output_path)

            with open(output_path) as f:
                content = f.read()

            assert "plotly" in content
            assert "st.metric" in content
            assert "st.plotly_chart" in content


class TestDashboardIntegration:
    """Integration tests for dashboard with orchestrator."""

    def test_full_integration(self):
        """Test full integration with a running orchestrator."""
        from src.core.orchestrator import Orchestrator, OrchestratorConfig
        from src.agents.honest import HonestAgent
        from src.agents.adversarial import AdversarialAgent

        config = OrchestratorConfig(
            n_epochs=3,
            steps_per_epoch=2,
            enable_boundaries=True,
        )
        orchestrator = Orchestrator(config)

        # Register agents
        orchestrator.register_agent(HonestAgent(agent_id="honest_1"))
        orchestrator.register_agent(HonestAgent(agent_id="honest_2"))
        orchestrator.register_agent(AdversarialAgent(agent_id="adv_1"))

        # Run simulation
        orchestrator.run()

        # Create dashboard state
        state = DashboardState()

        # Extract and update metrics
        snapshot = extract_metrics_from_orchestrator(orchestrator)
        state.update_metrics(snapshot)

        # Extract and update agent snapshots
        agent_snapshots = extract_agent_snapshots(orchestrator)
        for agent_snapshot in agent_snapshots:
            state.update_agent(agent_snapshot)

        # Verify state
        assert len(state.metric_history) == 1
        assert len(state.agent_snapshots) == 3

        # Export should work
        json_str = state.export_to_json()
        data = json.loads(json_str)
        assert "metric_history" in data
        assert "agents" in data
