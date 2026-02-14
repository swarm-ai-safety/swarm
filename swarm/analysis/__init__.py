"""Analysis tools for simulation results."""

from swarm.analysis.dashboard import (
    AgentSnapshot,
    DashboardConfig,
    DashboardState,
    MetricSnapshot,
    create_dashboard_file,
    extract_agent_snapshots,
    extract_metrics_from_orchestrator,
    run_dashboard,
)
from swarm.analysis.dolt_export import export_run_summary_to_dolt, export_to_dolt
from swarm.analysis.phylogeny import generate_phylogeny
from swarm.analysis.sweep import (
    SweepConfig,
    SweepParameter,
    SweepResult,
    SweepRunner,
    quick_sweep,
)

__all__ = [
    # Sweep
    "SweepConfig",
    "SweepParameter",
    "SweepResult",
    "SweepRunner",
    "quick_sweep",
    # Dashboard
    "DashboardConfig",
    "DashboardState",
    "MetricSnapshot",
    "AgentSnapshot",
    "extract_metrics_from_orchestrator",
    "extract_agent_snapshots",
    "create_dashboard_file",
    "run_dashboard",
    # Phylogeny
    "generate_phylogeny",
    # Dolt export
    "export_to_dolt",
    "export_run_summary_to_dolt",
]
