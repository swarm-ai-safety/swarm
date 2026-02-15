"""Enhanced dashboard with gradient fills, KPI cards, and danger zones.

Upgrades over the basic plot_run.py charts:

- **KPI cards**: top-row summary with delta indicators
- **Gradient fills**: translucent area under each line
- **Glow lines**: double-layer rendering for line pop
- **Min/max annotations**: peak and trough labels on the chart
- **Danger zones**: shaded threshold regions (e.g. toxicity > 0.35)
- **Multi-scenario**: overlay multiple runs for comparison

Usage::

    from swarm.analysis.enhanced_dashboard import (
        plot_enhanced_dashboard,
        plot_multi_scenario_dashboard,
    )

    # Single run from CSV
    fig = plot_enhanced_dashboard("runs/my_run/csv/baseline_epochs.csv")
    fig.savefig("dashboard.png", dpi=200, bbox_inches="tight")

    # Multiple runs
    fig = plot_multi_scenario_dashboard({
        "Baseline": "runs/run1/csv/baseline_epochs.csv",
        "Adversarial": "runs/run2/csv/adversarial_epochs.csv",
    })
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch

from swarm.analysis.theme import (
    COLORS,
    add_danger_zone,
    metric_color,
    swarm_theme,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default metrics to plot in single-run dashboards
_DEFAULT_METRICS: List[str] = [
    "toxicity_rate",
    "total_welfare",
    "avg_p",
    "accepted_interactions",
]

# KPI card definitions: (csv_column, display_label, format, color_attr)
_KPI_DEFS: List[Tuple[str, str, str, str]] = [
    ("toxicity_rate", "Toxicity", ".1%", "TOXICITY"),
    ("total_welfare", "Welfare", ".1f", "WELFARE"),
    ("avg_p", "Avg p", ".3f", "HONEST"),
    ("total_interactions", "Interactions", ".0f", "PRODUCTIVITY"),
]

# Colors for multi-scenario comparison (up to 6)
_SCENARIO_COLORS: List[str] = [
    COLORS.ADVERSARIAL,
    COLORS.WELFARE,
    COLORS.PLANNER,
    COLORS.HONEST,
    COLORS.DECEPTIVE,
    COLORS.EVASION,
]

# Danger thresholds: metric -> (threshold, direction)
_DANGER_THRESHOLDS: Dict[str, Tuple[float, str]] = {
    "toxicity_rate": (0.35, "above"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_df(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Accept a path or DataFrame and return a DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_csv(source)


def _kpi_card(
    ax: matplotlib.axes.Axes,
    value: float,
    label: str,
    color: str,
    delta: float = 0.0,
    fmt: str = ".2f",
) -> None:
    """Render a single KPI card with rounded border and delta indicator."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rect = FancyBboxPatch(
        (0.05, 0.05),
        0.9,
        0.9,
        boxstyle="round,pad=0.05",
        facecolor=to_rgba(color, 0.12),
        edgecolor=to_rgba(color, 0.4),
        linewidth=1.5,
    )
    ax.add_patch(rect)

    # Main value
    value_text = f"{value:{fmt}}"
    ax.text(
        0.5, 0.72, value_text,
        ha="center", va="center", fontsize=22,
        fontweight="bold", color=color,
    )

    # Label
    ax.text(
        0.5, 0.35, label,
        ha="center", va="center", fontsize=10,
        color=COLORS.TEXT_MUTED,
    )

    # Delta indicator
    if delta != 0:
        sign = "+" if delta > 0 else ""
        delta_color = COLORS.PRODUCTIVITY if delta > 0 else COLORS.TOXICITY
        ax.text(
            0.5, 0.12, f"{sign}{delta:.3f}",
            ha="center", va="center", fontsize=8,
            color=delta_color,
        )


def plot_enhanced_line(
    ax: matplotlib.axes.Axes,
    x: pd.Series,
    y: pd.Series,
    color: str,
    *,
    label: Optional[str] = None,
    show_annotations: bool = True,
    threshold: Optional[float] = None,
    threshold_label: Optional[str] = None,
    threshold_direction: str = "above",
) -> None:
    """Plot a line with gradient fill, glow effect, and data annotations.

    Args:
        ax: Matplotlib axes.
        x: X values (typically epoch numbers).
        y: Y values (metric series).
        color: Line/fill color.
        label: Legend label.
        show_annotations: Whether to label min/max/current values.
        threshold: Optional danger-zone threshold.
        threshold_label: Label for the threshold line.
        threshold_direction: ``"above"`` or ``"below"``.
    """
    # Gradient fill
    ax.fill_between(x, y, alpha=0.15, color=color)

    # Glow layer
    ax.plot(x, y, color=to_rgba(color, 0.25), linewidth=4)

    # Main line with markers
    ax.plot(
        x, y, color=color, linewidth=2,
        marker="o", markersize=4,
        markerfacecolor=color,
        markeredgecolor=COLORS.BG_DARK,
        markeredgewidth=1,
        label=label,
    )

    if show_annotations and len(y) > 1:
        yvals = y.values
        ymin_idx = int(np.argmin(yvals))
        ymax_idx = int(np.argmax(yvals))

        # Peak annotation
        ax.annotate(
            f"{yvals[ymax_idx]:.2f}",
            xy=(x.iloc[ymax_idx], yvals[ymax_idx]),
            xytext=(0, 10), textcoords="offset points",
            fontsize=7, color=color, ha="center", fontweight="bold",
        )

        # Trough annotation
        ax.annotate(
            f"{yvals[ymin_idx]:.2f}",
            xy=(x.iloc[ymin_idx], yvals[ymin_idx]),
            xytext=(0, -12), textcoords="offset points",
            fontsize=7, color=COLORS.TEXT_MUTED, ha="center",
        )

        # Current value callout
        ax.annotate(
            f"  {yvals[-1]:.3f}",
            xy=(x.iloc[-1], yvals[-1]),
            fontsize=8, color=color, va="center", fontweight="bold",
        )

    # Danger zone
    if threshold is not None:
        add_danger_zone(
            ax, threshold,
            direction=threshold_direction,
            label=threshold_label,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_enhanced_dashboard(
    source: Union[str, Path, pd.DataFrame],
    *,
    title: Optional[str] = None,
    mode: str = "dark",
    metrics: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """Generate an enhanced single-run dashboard.

    Creates a figure with:

    * Top row: KPI summary cards (toxicity, welfare, avg p, interactions).
    * Middle row: toxicity rate and total welfare time series.
    * Bottom row: avg probability and interaction volume.

    All charts use gradient fills, glow lines, min/max annotations,
    and danger-zone shading where applicable.

    Args:
        source: Path to ``*_epochs.csv`` or a pre-loaded DataFrame.
        title: Figure suptitle (default: ``"SWARM Dashboard"``).
        mode: ``"dark"`` (default) or ``"light"``.
        metrics: Override default metrics to plot in the four chart panels.

    Returns:
        The matplotlib Figure.
    """
    df = _load_df(source)
    title = title or "SWARM Dashboard"
    metrics = list(metrics or _DEFAULT_METRICS)

    with swarm_theme(mode):
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(
            3, 4,
            height_ratios=[0.25, 1, 1],
            hspace=0.35, wspace=0.3,
        )

        # ── KPI cards ──
        for i, (col, label, fmt, color_attr) in enumerate(_KPI_DEFS):
            ax = fig.add_subplot(gs[0, i])
            color = getattr(COLORS, color_attr)
            if col == "total_interactions":
                value = float(df[col].sum())
                delta = 0.0
            else:
                value = float(df[col].iloc[-1])
                delta = value - float(df[col].iloc[0])
            _kpi_card(ax, value, label, color, delta=delta, fmt=fmt)

        # ── Chart panels ──
        panel_positions = [(1, slice(0, 2)), (1, slice(2, 4)),
                           (2, slice(0, 2)), (2, slice(2, 4))]
        panel_titles = {
            "toxicity_rate": "Toxicity Rate",
            "total_welfare": "Total Welfare",
            "avg_p": "Average Probability (p)",
            "accepted_interactions": "Interactions per Epoch",
            "quality_gap": "Quality Gap",
            "gini_coefficient": "Gini Coefficient",
        }

        for idx, metric in enumerate(metrics[:4]):
            row, col_slice = panel_positions[idx]
            ax = fig.add_subplot(gs[row, col_slice])

            color = metric_color(metric)
            threshold_info = _DANGER_THRESHOLDS.get(metric)

            if metric == "accepted_interactions":
                # Stacked area for interactions
                ax.fill_between(
                    df["epoch"], df["accepted_interactions"],
                    alpha=0.2, color=COLORS.PRODUCTIVITY,
                )
                ax.fill_between(
                    df["epoch"], df["total_interactions"],
                    alpha=0.1, color=COLORS.WELFARE,
                )
                ax.plot(
                    df["epoch"], df["total_interactions"],
                    color=COLORS.WELFARE, linewidth=2, label="Total",
                )
                ax.plot(
                    df["epoch"], df["accepted_interactions"],
                    color=COLORS.PRODUCTIVITY, linewidth=2, label="Accepted",
                )
                ax.legend(fontsize=8)
            else:
                plot_enhanced_line(
                    ax, df["epoch"], df[metric], color,
                    threshold=threshold_info[0] if threshold_info else None,
                    threshold_label="danger" if threshold_info else None,
                    threshold_direction=(
                        threshold_info[1] if threshold_info else "above"
                    ),
                )

            ax.set_title(
                panel_titles.get(metric, metric),
                fontsize=11, pad=10,
            )
            ax.set_xlabel("Epoch", fontsize=9)

        fig.suptitle(title, fontsize=15, y=0.98, fontweight="bold")

    return fig


def plot_multi_scenario_dashboard(
    sources: Dict[str, Union[str, Path, pd.DataFrame]],
    *,
    title: Optional[str] = None,
    mode: str = "dark",
) -> plt.Figure:
    """Generate a multi-scenario comparison dashboard.

    Creates a figure with:

    * Top row: one KPI card per scenario (name, toxicity, welfare).
    * Middle row: toxicity rate and welfare overlays across scenarios.
    * Bottom row: acceptance rate and interaction volume bar chart.

    Args:
        sources: Mapping of ``{scenario_name: csv_path_or_df}``.
        title: Figure suptitle (default: ``"SWARM Multi-Scenario Dashboard"``).
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        The matplotlib Figure.
    """
    title = title or "SWARM Multi-Scenario Dashboard"
    dfs: Dict[str, pd.DataFrame] = {
        name: _load_df(src) for name, src in sources.items()
    }
    names = list(dfs.keys())
    n = len(names)
    colors = _SCENARIO_COLORS[:n]

    with swarm_theme(mode):
        fig = plt.figure(figsize=(max(16, 4 * n), 14))
        gs = fig.add_gridspec(
            3, n,
            height_ratios=[0.22, 1, 1],
            hspace=0.35, wspace=0.3,
        )

        # ── KPI cards ──
        for i, name in enumerate(names):
            df = dfs[name]
            ax = fig.add_subplot(gs[0, i])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            color = colors[i]

            rect = FancyBboxPatch(
                (0.05, 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=to_rgba(color, 0.12),
                edgecolor=to_rgba(color, 0.4),
                linewidth=1.5,
            )
            ax.add_patch(rect)

            tox = df["toxicity_rate"].iloc[-1]
            welfare = df["total_welfare"].iloc[-1]
            ax.text(
                0.5, 0.75, name,
                ha="center", va="center", fontsize=11,
                fontweight="bold", color=color,
            )
            ax.text(
                0.5, 0.45, f"Tox: {tox:.1%}",
                ha="center", va="center", fontsize=9,
                color=COLORS.TEXT_PRIMARY,
            )
            ax.text(
                0.5, 0.2, f"Welfare: {welfare:.1f}",
                ha="center", va="center", fontsize=9,
                color=COLORS.TEXT_MUTED,
            )

        # ── Toxicity comparison ──
        ax1 = fig.add_subplot(gs[1, : n // 2 or 1])
        for i, name in enumerate(names):
            df = dfs[name]
            plot_enhanced_line(
                ax1, df["epoch"], df["toxicity_rate"],
                colors[i], label=name, show_annotations=(i == 0),
            )
        add_danger_zone(ax1, 0.35, direction="above", label="danger threshold")
        ax1.set_title("Toxicity Rate Comparison", fontsize=12, pad=10)
        ax1.set_xlabel("Epoch")
        ax1.legend(fontsize=8, loc="upper left")

        # ── Welfare comparison ──
        ax2 = fig.add_subplot(gs[1, n // 2 or 1 :])
        for i, name in enumerate(names):
            df = dfs[name]
            plot_enhanced_line(
                ax2, df["epoch"], df["total_welfare"],
                colors[i], label=name, show_annotations=(i == 0),
            )
        ax2.set_title("Total Welfare Comparison", fontsize=12, pad=10)
        ax2.set_xlabel("Epoch")
        ax2.legend(fontsize=8, loc="upper left")

        # ── Acceptance rate ──
        ax3 = fig.add_subplot(gs[2, : n // 2 or 1])
        for i, name in enumerate(names):
            df = dfs[name]
            total = df["total_interactions"].replace(0, np.nan)
            acc = (df["accepted_interactions"] / total).fillna(0)
            plot_enhanced_line(
                ax3, df["epoch"], acc,
                colors[i], label=name, show_annotations=False,
            )
        ax3.set_ylim(-0.05, 1.1)
        ax3.set_title("Acceptance Rate", fontsize=12, pad=10)
        ax3.set_xlabel("Epoch")
        ax3.legend(fontsize=8, loc="lower left")

        # ── Interaction volume bars ──
        ax4 = fig.add_subplot(gs[2, n // 2 or 1 :])
        bar_width = 0.8 / n
        for i, name in enumerate(names):
            df = dfs[name]
            offset = (i - n / 2 + 0.5) * bar_width
            ax4.bar(
                df["epoch"] + offset,
                df["total_interactions"],
                width=bar_width,
                color=to_rgba(colors[i], 0.7),
                edgecolor=colors[i],
                linewidth=0.5,
                label=name,
            )
        ax4.set_title("Interactions per Epoch", fontsize=12, pad=10)
        ax4.set_xlabel("Epoch")
        ax4.legend(fontsize=8)

        fig.suptitle(title, fontsize=16, y=0.99, fontweight="bold")

    return fig
