"""Analysis tools for simulation results."""

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
    export_to_parquet,
    generate_summary_report,
    load_from_csv,
    load_from_json,
)
from src.analysis.plots import (
    PlotData,
    create_agent_comparison_data,
    create_agent_trajectory_data,
    create_distribution_data,
    create_heatmap_data,
    create_network_graph_data,
    create_scatter_data,
    create_time_series_data,
    plotly_bar_chart,
    plotly_gauge,
    plotly_heatmap,
    plotly_multi_line,
    plotly_network,
    plotly_scatter,
    plotly_time_series,
)
from src.analysis.sweep import (
    SweepConfig,
    SweepParameter,
    SweepResult,
    SweepRunner,
    quick_sweep,
)

__all__ = [
    # Aggregation
    "AgentSnapshot",
    "EpochSnapshot",
    "MetricsAggregator",
    "SimulationHistory",
    "TimeSeriesPoint",
    "compute_rolling_average",
    "compute_trend",
    # Export
    "export_to_csv",
    "export_to_json",
    "export_to_parquet",
    "generate_summary_report",
    "load_from_csv",
    "load_from_json",
    # Plots
    "PlotData",
    "create_agent_comparison_data",
    "create_agent_trajectory_data",
    "create_distribution_data",
    "create_heatmap_data",
    "create_network_graph_data",
    "create_scatter_data",
    "create_time_series_data",
    "plotly_bar_chart",
    "plotly_gauge",
    "plotly_heatmap",
    "plotly_multi_line",
    "plotly_network",
    "plotly_scatter",
    "plotly_time_series",
    # Sweep
    "SweepConfig",
    "SweepParameter",
    "SweepResult",
    "SweepRunner",
    "quick_sweep",
]
