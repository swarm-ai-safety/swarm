"""Streamlit dashboard for real-time simulation visualization.

Run with: streamlit run src/analysis/dashboard.py

Provides:
- Real-time metrics display during simulation
- Post-simulation analysis and exploration
- Agent state comparison
- Network visualization
- Security and collusion threat monitoring
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

# Dashboard state management
@dataclass
class DashboardState:
    """State container for the dashboard."""

    # Simulation config
    n_epochs: int = 10
    steps_per_epoch: int = 10
    n_honest: int = 2
    n_opportunistic: int = 1
    n_adversarial: int = 1
    seed: int = 42

    # Governance config
    transaction_tax_rate: float = 0.05
    reputation_decay_rate: float = 0.95
    circuit_breaker_enabled: bool = True
    security_enabled: bool = True
    collusion_detection_enabled: bool = True

    # Network config
    network_topology: str = "complete"
    network_dynamic: bool = False

    # Display options
    selected_metrics: List[str] = field(default_factory=lambda: [
        "toxicity_rate",
        "total_welfare",
        "avg_reputation",
    ])
    rolling_window: int = 5
    show_agent_details: bool = True

    # State
    is_running: bool = False
    current_epoch: int = 0


def create_sidebar_config() -> Dict[str, Any]:
    """Create sidebar configuration controls."""
    try:
        import streamlit as st
    except ImportError:
        return {}

    st.sidebar.header("Simulation Settings")

    config = {}

    # Timing
    config["n_epochs"] = st.sidebar.slider(
        "Number of Epochs",
        min_value=5,
        max_value=100,
        value=20,
    )
    config["steps_per_epoch"] = st.sidebar.slider(
        "Steps per Epoch",
        min_value=5,
        max_value=50,
        value=10,
    )
    config["seed"] = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
    )

    # Agents
    st.sidebar.subheader("Agent Configuration")
    config["n_honest"] = st.sidebar.slider("Honest Agents", 0, 5, 2)
    config["n_opportunistic"] = st.sidebar.slider("Opportunistic Agents", 0, 5, 1)
    config["n_adversarial"] = st.sidebar.slider("Adversarial Agents", 0, 5, 1)

    # Governance
    st.sidebar.subheader("Governance")
    config["transaction_tax_rate"] = st.sidebar.slider(
        "Transaction Tax Rate",
        0.0, 0.2, 0.05, 0.01,
    )
    config["reputation_decay_rate"] = st.sidebar.slider(
        "Reputation Decay Rate",
        0.8, 1.0, 0.95, 0.01,
    )
    config["circuit_breaker_enabled"] = st.sidebar.checkbox(
        "Circuit Breaker",
        value=True,
    )
    config["security_enabled"] = st.sidebar.checkbox(
        "Security Detection",
        value=True,
    )
    config["collusion_detection_enabled"] = st.sidebar.checkbox(
        "Collusion Detection",
        value=True,
    )

    # Network
    st.sidebar.subheader("Network")
    config["network_topology"] = st.sidebar.selectbox(
        "Topology",
        ["complete", "ring", "star", "small_world", "scale_free"],
    )
    config["network_dynamic"] = st.sidebar.checkbox(
        "Dynamic Network",
        value=False,
    )

    return config


def render_metrics_overview(
    epoch_snapshot: Any,
) -> None:
    """Render the metrics overview section."""
    try:
        import streamlit as st
    except ImportError:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Toxicity Rate",
            f"{epoch_snapshot.toxicity_rate:.3f}",
            delta=None,
        )
    with col2:
        st.metric(
            "Total Welfare",
            f"{epoch_snapshot.total_welfare:.2f}",
        )
    with col3:
        st.metric(
            "Avg Reputation",
            f"{epoch_snapshot.avg_reputation:.2f}",
        )
    with col4:
        st.metric(
            "Interactions",
            f"{epoch_snapshot.total_interactions}",
        )

    # Second row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Quality Gap",
            f"{epoch_snapshot.quality_gap:.3f}",
        )
    with col2:
        st.metric(
            "Agents Frozen",
            f"{epoch_snapshot.n_frozen}",
        )
    with col3:
        st.metric(
            "Threat Level",
            f"{epoch_snapshot.ecosystem_threat_level:.2f}",
        )
    with col4:
        st.metric(
            "Collusion Risk",
            f"{epoch_snapshot.ecosystem_collusion_risk:.2f}",
        )


def render_time_series_charts(
    history: Any,
    selected_metrics: List[str],
    rolling_window: int = 5,
) -> None:
    """Render time series charts for selected metrics."""
    try:
        import streamlit as st
        from src.analysis.plots import (
            create_time_series_data,
            plotly_time_series,
        )
    except ImportError:
        return

    if not history.epoch_snapshots:
        st.info("No data yet. Run a simulation to see metrics.")
        return

    # Create tabs for different metric categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "Quality Metrics",
        "Economic Metrics",
        "Security Metrics",
        "Network Metrics",
    ])

    with tab1:
        quality_metrics = ["toxicity_rate", "quality_gap", "avg_p"]
        data = create_time_series_data(history, quality_metrics, rolling_window)
        fig = plotly_time_series(data, "Quality Metrics Over Time")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        econ_metrics = ["total_welfare", "avg_payoff", "gini_coefficient"]
        data = create_time_series_data(history, econ_metrics, rolling_window)
        fig = plotly_time_series(data, "Economic Metrics Over Time")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        sec_metrics = ["ecosystem_threat_level", "ecosystem_collusion_risk"]
        data = create_time_series_data(history, sec_metrics, rolling_window)
        fig = plotly_time_series(data, "Security Metrics Over Time")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        net_metrics = ["n_edges", "avg_degree", "avg_clustering"]
        data = create_time_series_data(history, net_metrics, rolling_window)
        fig = plotly_time_series(data, "Network Metrics Over Time")
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def render_agent_table(
    history: Any,
) -> None:
    """Render agent state table."""
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        return

    final_states = history.get_final_agent_states()
    if not final_states:
        st.info("No agent data available.")
        return

    # Convert to dataframe
    rows = []
    for agent_id, snapshot in sorted(final_states.items()):
        rows.append({
            "Agent": agent_id,
            "Reputation": f"{snapshot.reputation:.2f}",
            "Resources": f"{snapshot.resources:.2f}",
            "Interactions": snapshot.interactions_initiated + snapshot.interactions_received,
            "Total Payoff": f"{snapshot.total_payoff:.2f}",
            "Avg P (Init)": f"{snapshot.avg_p_initiated:.2f}",
            "Status": "Frozen" if snapshot.is_frozen else ("Quarantined" if snapshot.is_quarantined else "Active"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


def render_agent_comparison(
    history: Any,
) -> None:
    """Render agent comparison charts."""
    try:
        import streamlit as st
        from src.analysis.plots import (
            create_agent_comparison_data,
            create_agent_trajectory_data,
            plotly_bar_chart,
            plotly_multi_line,
        )
    except ImportError:
        return

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart of total payoffs
        data = create_agent_comparison_data(history, "total_payoff")
        fig = plotly_bar_chart(data, "Total Payoff by Agent")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bar chart of reputations
        data = create_agent_comparison_data(history, "reputation")
        fig = plotly_bar_chart(data, "Final Reputation by Agent")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Reputation trajectories
    agent_ids = list(history.agent_snapshots.keys())
    if agent_ids:
        data = create_agent_trajectory_data(history, agent_ids, "reputation")
        fig = plotly_multi_line(data, "Reputation Over Time")
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def render_security_panel(
    security_report: Any,
) -> None:
    """Render security analysis panel."""
    try:
        import streamlit as st
        from src.analysis.plots import plotly_gauge
    except ImportError:
        return

    if security_report is None:
        st.info("Security detection not enabled or no data available.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        fig = plotly_gauge(
            security_report.ecosystem_threat_level,
            "Ecosystem Threat Level",
            thresholds=[(0.3, "green"), (0.6, "yellow"), (1.0, "red")],
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Active Threats", security_report.active_threat_count)
        st.metric("Contagion Depth", security_report.contagion_depth)
        st.metric("Flagged Agents", len(security_report.agents_flagged))

    with col3:
        st.subheader("Threat Indicators")
        for indicator in security_report.threat_indicators[:5]:
            st.write(
                f"**{indicator.threat_type.value}**: "
                f"{indicator.source_agent} (severity: {indicator.severity:.2f})"
            )


def render_network_panel(
    network: Any,
    agent_states: Dict[str, Any],
) -> None:
    """Render network visualization panel."""
    try:
        import streamlit as st
        from src.analysis.plots import create_network_graph_data, plotly_network
    except ImportError:
        return

    if network is None:
        st.info("Network module not enabled.")
        return

    # Get edges from network
    edges = []
    for src, tgt in network.get_edges():
        weight = network.get_edge_weight(src, tgt)
        edges.append((src, tgt, weight))

    # Get node attributes
    node_attrs = {}
    for agent_id, state in agent_states.items():
        node_attrs[agent_id] = {
            "reputation": state.reputation,
            "resources": state.resources,
        }

    data = create_network_graph_data(edges, node_attrs)
    fig = plotly_network(data, "Agent Network")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Network metrics
    metrics = network.get_metrics()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Edges", metrics.get("n_edges", 0))
    with col2:
        st.metric("Avg Degree", f"{metrics.get('avg_degree', 0):.2f}")
    with col3:
        st.metric("Clustering", f"{metrics.get('avg_clustering', 0):.2f}")
    with col4:
        st.metric("Components", metrics.get("n_components", 1))


def render_event_log(
    events: List[Dict],
    max_events: int = 20,
) -> None:
    """Render recent events log."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.subheader("Recent Events")

    if not events:
        st.info("No events recorded yet.")
        return

    for event in events[-max_events:]:
        event_type = event.get("event_type", "unknown")
        timestamp = event.get("timestamp", "")
        details = event.get("details", {})

        if event_type == "interaction_completed":
            st.write(
                f"**{timestamp}** - Interaction: "
                f"{details.get('initiator', '?')} -> {details.get('counterparty', '?')} "
                f"(p={details.get('p', 0):.2f}, accepted={details.get('accepted', False)})"
            )
        elif event_type == "agent_frozen":
            st.write(f"**{timestamp}** - Agent frozen: {details.get('agent_id', '?')}")
        elif event_type == "epoch_completed":
            st.write(f"**{timestamp}** - Epoch {details.get('epoch', '?')} completed")


def run_dashboard():
    """Main entry point for the Streamlit dashboard."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Run: pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="AGI Safety Sandbox Dashboard",
        page_icon="🔬",
        layout="wide",
    )

    st.title("Distributional AGI Safety Sandbox")
    st.markdown("Real-time visualization of multi-agent simulation metrics")

    # Sidebar configuration
    config = create_sidebar_config()

    # Initialize session state
    if "history" not in st.session_state:
        from src.analysis.aggregation import SimulationHistory
        st.session_state.history = SimulationHistory()
        st.session_state.is_running = False
        st.session_state.orchestrator = None
        st.session_state.security_report = None
        st.session_state.network = None

    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("Run Simulation", type="primary"):
            run_simulation(config)

    with col2:
        if st.button("Reset"):
            from src.analysis.aggregation import SimulationHistory
            st.session_state.history = SimulationHistory()
            st.session_state.security_report = None
            st.session_state.network = None
            st.rerun()

    # Display current state
    history = st.session_state.history

    if history.epoch_snapshots:
        latest = history.epoch_snapshots[-1]

        # Header metrics
        st.header(f"Epoch {latest.epoch}")
        render_metrics_overview(latest)

        st.divider()

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Time Series",
            "Agents",
            "Security",
            "Network",
            "Events",
        ])

        with tab1:
            render_time_series_charts(
                history,
                config.get("selected_metrics", ["toxicity_rate"]),
                rolling_window=5,
            )

        with tab2:
            render_agent_table(history)
            st.divider()
            render_agent_comparison(history)

        with tab3:
            render_security_panel(st.session_state.security_report)

        with tab4:
            if st.session_state.orchestrator:
                render_network_panel(
                    st.session_state.network,
                    st.session_state.orchestrator.state.agents,
                )

        with tab5:
            render_event_log([])  # Would need event log integration

    else:
        st.info("Configure settings in the sidebar and click 'Run Simulation' to start.")


def run_simulation(config: Dict[str, Any]) -> None:
    """Run a simulation with the given configuration."""
    try:
        import streamlit as st
        from src.agents.honest import HonestAgent
        from src.agents.opportunistic import OpportunisticAgent
        from src.agents.adversarial import AdversarialAgent
        from src.core.orchestrator import Orchestrator, OrchestratorConfig
        from src.governance.config import GovernanceConfig
        from src.env.network import NetworkConfig, NetworkTopology
        from src.analysis.aggregation import MetricsAggregator
    except ImportError as e:
        st.error(f"Missing dependency: {e}")
        return

    # Build governance config
    gov_config = GovernanceConfig(
        transaction_tax_rate=config.get("transaction_tax_rate", 0.05),
        reputation_decay_rate=config.get("reputation_decay_rate", 0.95),
        circuit_breaker_enabled=config.get("circuit_breaker_enabled", True),
        freeze_threshold_toxicity=0.7,
        security_enabled=config.get("security_enabled", True),
        security_quarantine_threshold=0.7,
        collusion_detection_enabled=config.get("collusion_detection_enabled", True),
    )

    # Build network config
    topology_map = {
        "complete": NetworkTopology.COMPLETE,
        "ring": NetworkTopology.RING,
        "star": NetworkTopology.STAR,
        "small_world": NetworkTopology.SMALL_WORLD,
        "scale_free": NetworkTopology.SCALE_FREE,
    }
    network_config = NetworkConfig(
        topology=topology_map.get(config.get("network_topology", "complete"), NetworkTopology.COMPLETE),
        dynamic=config.get("network_dynamic", False),
    )

    # Build orchestrator config
    orch_config = OrchestratorConfig(
        n_epochs=config.get("n_epochs", 10),
        steps_per_epoch=config.get("steps_per_epoch", 10),
        seed=config.get("seed", 42),
        governance_config=gov_config,
        network_config=network_config,
    )

    # Create orchestrator
    orchestrator = Orchestrator(config=orch_config)

    # Register agents
    agent_id = 0
    for _ in range(config.get("n_honest", 2)):
        orchestrator.register_agent(HonestAgent(agent_id=f"honest_{agent_id}"))
        agent_id += 1

    for _ in range(config.get("n_opportunistic", 1)):
        orchestrator.register_agent(OpportunisticAgent(agent_id=f"opp_{agent_id}"))
        agent_id += 1

    for _ in range(config.get("n_adversarial", 1)):
        orchestrator.register_agent(AdversarialAgent(agent_id=f"adv_{agent_id}"))
        agent_id += 1

    # Create aggregator
    aggregator = MetricsAggregator()
    aggregator.start_simulation(
        simulation_id=orchestrator.state.simulation_id,
        n_epochs=orch_config.n_epochs,
        steps_per_epoch=orch_config.steps_per_epoch,
        n_agents=len(orchestrator.agents),
        seed=orch_config.seed,
    )

    # Run simulation with progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(orch_config.n_epochs):
        status_text.text(f"Running epoch {epoch + 1}/{orch_config.n_epochs}...")

        # Run one epoch
        orchestrator._run_epoch(epoch)

        # Collect metrics
        for interaction in orchestrator.state.completed_interactions:
            aggregator.record_interaction(interaction)

        # Get reports
        security_report = None
        collusion_report = None
        network_metrics = None

        if orchestrator.governance_engine:
            security_report = orchestrator.governance_engine.get_security_report()
            collusion_report = orchestrator.governance_engine.get_collusion_report()

        if orchestrator.network:
            network_metrics = orchestrator.network.get_metrics()

        # Finalize epoch
        aggregator.finalize_epoch(
            epoch=epoch,
            agent_states=orchestrator.state.agents,
            frozen_agents=orchestrator.state.frozen_agents,
            quarantined_agents=orchestrator.governance_engine.get_quarantined_agents() if orchestrator.governance_engine else set(),
            network_metrics=network_metrics,
            security_report=security_report,
            collusion_report=collusion_report,
        )

        # Advance orchestrator
        orchestrator.state.advance_epoch()

        # Update progress
        progress_bar.progress((epoch + 1) / orch_config.n_epochs)

    # Store results in session state
    st.session_state.history = aggregator.end_simulation()
    st.session_state.orchestrator = orchestrator
    st.session_state.security_report = security_report
    st.session_state.network = orchestrator.network

    progress_bar.empty()
    status_text.empty()
    st.success(f"Simulation completed! {orch_config.n_epochs} epochs run.")
    st.rerun()


# Entry point for streamlit run
if __name__ == "__main__":
    run_dashboard()
