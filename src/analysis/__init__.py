"""Analysis tools for simulation results."""

from src.analysis.sweep import (
    SweepConfig,
    SweepParameter,
    SweepResult,
    SweepRunner,
    quick_sweep,
)
from src.analysis.dashboard import (
    DashboardConfig,
    DashboardState,
    MetricSnapshot,
    AgentSnapshot,
    extract_metrics_from_orchestrator,
    extract_agent_snapshots,
    create_dashboard_file,
    run_dashboard,
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
]
