"""Analysis tools for simulation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from swarm.analysis.theme import (
        COLORS as COLORS,
    )
    from swarm.analysis.theme import (
        SWARM_LIGHT_STYLE as SWARM_LIGHT_STYLE,
    )
    from swarm.analysis.theme import (
        SWARM_STYLE as SWARM_STYLE,
    )
    from swarm.analysis.theme import (
        agent_color as agent_color,
    )
    from swarm.analysis.theme import (
        apply_theme as apply_theme,
    )
    from swarm.analysis.theme import (
        metric_color as metric_color,
    )
    from swarm.analysis.theme import (
        swarm_theme as swarm_theme,
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
    # Theme & colors
    "COLORS",
    "SWARM_STYLE",
    "SWARM_LIGHT_STYLE",
    "apply_theme",
    "swarm_theme",
    "agent_color",
    "metric_color",
]


def __getattr__(name: str) -> object:
    """Lazy-load theme symbols so matplotlib is not required at import time."""
    _theme_names = {
        "COLORS", "SWARM_STYLE", "SWARM_LIGHT_STYLE",
        "apply_theme", "swarm_theme", "agent_color", "metric_color",
    }
    if name in _theme_names:
        from swarm.analysis import theme  # noqa: F811

        return getattr(theme, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
