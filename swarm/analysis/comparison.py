"""Bar charts, slope charts, and faceted comparisons for the SWARM analysis suite.

Provides reusable matplotlib chart types for comparing agents, metrics,
and experimental conditions across simulation runs.  All functions apply
the SWARM theme via :func:`swarm_theme` and default to dark mode.

Usage::

    from swarm.analysis.comparison import (
        plot_grouped_bar,
        plot_slope_chart,
        plot_faceted_comparison,
        plot_agent_comparison_bar,
    )

    fig, ax = plot_slope_chart(
        data={"honest": [0.8, 0.9], "deceptive": [0.3, 0.2]},
        conditions=["baseline", "governed"],
        metric="Cooperation probability",
    )
    fig.savefig("slope.png")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from swarm.analysis.theme import (
    AGENT_COLOR_MAP,
    COLORS,
    agent_color,
    swarm_theme,
)

__all__ = [
    "plot_grouped_bar",
    "plot_slope_chart",
    "plot_faceted_comparison",
    "plot_agent_comparison_bar",
]

# Default color cycle used when no explicit colors are provided.
_DEFAULT_CYCLE: List[str] = [
    COLORS.HONEST,
    COLORS.DECEPTIVE,
    COLORS.OPPORTUNISTIC,
    COLORS.ADVERSARIAL,
    COLORS.PLANNER,
    COLORS.WELFARE,
    COLORS.PRODUCTIVITY,
]


# ---------------------------------------------------------------------------
# 1. Grouped bar chart
# ---------------------------------------------------------------------------


def plot_grouped_bar(
    ax: matplotlib.axes.Axes,
    categories: Sequence[str],
    groups: Sequence[str],
    values: Sequence[Sequence[float]],
    *,
    colors: Optional[Sequence[str]] = None,
    group_labels: Optional[Sequence[str]] = None,
    ylabel: str = "Value",
    title: str = "",
) -> matplotlib.axes.Axes:
    """Draw a grouped bar chart on an existing axes.

    Args:
        ax: Matplotlib axes to draw on.
        categories: Category labels shown on the x-axis.
        groups: Group names (one bar per group within each category).
        values: 2-D array-like of shape ``(n_categories, n_groups)``.
        colors: Optional list of colors, one per group.  Falls back to
            :data:`AGENT_COLOR_MAP` lookup on group name, then to the
            default theme cycle.
        group_labels: Legend labels (defaults to *groups*).
        ylabel: Label for the y-axis.
        title: Axes title.

    Returns:
        The axes with the chart drawn.

    Example::

        fig, ax = plt.subplots()
        plot_grouped_bar(
            ax,
            categories=["epoch 1", "epoch 5", "epoch 10"],
            groups=["honest", "deceptive"],
            values=[[0.8, 0.3], [0.85, 0.25], [0.9, 0.2]],
            ylabel="Cooperation rate",
            title="Cooperation by epoch",
        )
    """
    vals = np.asarray(values, dtype=float)
    n_cats, n_groups = vals.shape

    if colors is None:
        colors = [
            AGENT_COLOR_MAP.get(g.lower(), _DEFAULT_CYCLE[i % len(_DEFAULT_CYCLE)])
            for i, g in enumerate(groups)
        ]
    if group_labels is None:
        group_labels = list(groups)

    x = np.arange(n_cats)
    bar_width = 0.8 / n_groups

    for j in range(n_groups):
        offset = (j - n_groups / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            vals[:, j],
            width=bar_width,
            color=colors[j],
            label=group_labels[j],
            edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# 2. Slope chart
# ---------------------------------------------------------------------------


def plot_slope_chart(
    data: Dict[str, Sequence[float]],
    *,
    conditions: Sequence[str],
    metric: str = "Value",
    title: str = "Slope chart",
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Slope chart connecting a metric across two or more conditions.

    Each series is drawn as a line from condition to condition, making it
    easy to see which agents improve and which degrade.

    Args:
        data: Mapping of *model_name* to a list of values (one per
            condition).
        conditions: Condition labels for the x-axis.
        metric: Label for the y-axis.
        title: Figure title.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, ax)`` tuple.

    Example::

        fig, ax = plot_slope_chart(
            data={
                "honest": [0.80, 0.92],
                "deceptive": [0.30, 0.15],
            },
            conditions=["baseline", "governed"],
            metric="Cooperation probability",
        )
    """
    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 2.5), 6))

    x = np.arange(len(conditions))

    for idx, (name, series) in enumerate(data.items()):
        color = AGENT_COLOR_MAP.get(
            name.lower(), _DEFAULT_CYCLE[idx % len(_DEFAULT_CYCLE)]
        )
        ax.plot(x, series, marker="o", linewidth=2, color=color, label=name)
        # Annotate start and end values.
        for xi, yi in zip(x, series, strict=False):
            ax.annotate(
                f"{yi:.2f}",
                (float(xi), yi),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Faceted comparison (small multiples)
# ---------------------------------------------------------------------------


def plot_faceted_comparison(
    data: Dict[str, Dict[str, float]],
    *,
    metric: str = "Value",
    facet_by: str = "facet",
    conditions: Optional[Sequence[str]] = None,
    title: str = "Faceted comparison",
    ncols: int = 3,
    mode: str = "dark",
) -> Tuple[plt.Figure, np.ndarray]:
    """Small-multiples bar chart: one subplot per facet value.

    Each subplot shows the *metric* across *conditions* as a simple bar
    chart, making it easy to compare patterns across facets.

    Args:
        data: Mapping of ``facet_value -> {condition: value}``.
        metric: Label for the y-axis in each subplot.
        facet_by: Human-readable name of the faceting dimension (used in
            the super-title).
        conditions: Explicit ordering of condition labels.  If *None*,
            keys are collected from the first facet entry.
        title: Super-title for the figure.
        ncols: Number of columns in the grid (default 3).
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, axes_array)`` where *axes_array* is a flat numpy array of
        all subplot axes (including any unused ones, which are hidden).

    Example::

        fig, axes = plot_faceted_comparison(
            data={
                "honest":       {"baseline": 0.8, "governed": 0.9},
                "deceptive":    {"baseline": 0.3, "governed": 0.15},
                "opportunistic": {"baseline": 0.5, "governed": 0.45},
            },
            metric="Cooperation probability",
            facet_by="agent_type",
            title="Cooperation across governance regimes",
        )
    """
    facets = list(data.keys())
    n_facets = len(facets)

    if conditions is None:
        first = next(iter(data.values()))
        conditions = list(first.keys())

    nrows = max(1, int(np.ceil(n_facets / ncols)))

    with swarm_theme(mode):
        fig, axes_2d = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 3.5 * nrows),
            squeeze=False,
        )

    axes_flat: np.ndarray = axes_2d.ravel()

    x = np.arange(len(conditions))
    bar_colors = [_DEFAULT_CYCLE[i % len(_DEFAULT_CYCLE)] for i in range(len(conditions))]

    for idx, facet_name in enumerate(facets):
        ax = axes_flat[idx]
        facet_vals = data[facet_name]
        heights = [facet_vals.get(c, 0.0) for c in conditions]
        ax.bar(x, heights, color=bar_colors, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(metric, fontsize=8)
        ax.set_title(facet_name, fontsize=9)

    # Hide unused axes.
    for idx in range(n_facets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig, axes_flat


# ---------------------------------------------------------------------------
# 4. Agent comparison bar
# ---------------------------------------------------------------------------


def plot_agent_comparison_bar(
    agent_data: Sequence[Dict[str, Any]],
    *,
    metric: str = "Value",
    title: str = "Agent comparison",
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Bar chart comparing agents on a single metric, colored by agent type.

    Args:
        agent_data: Sequence of dicts, each containing at least
            ``"agent_id"``, ``"agent_type"``, and ``"value"`` keys.
        metric: Label for the y-axis.
        title: Figure title.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, ax)`` tuple.

    Example::

        fig, ax = plot_agent_comparison_bar(
            agent_data=[
                {"agent_id": "a1", "agent_type": "honest",    "value": 0.92},
                {"agent_id": "a2", "agent_type": "deceptive", "value": 0.31},
                {"agent_id": "a3", "agent_type": "honest",    "value": 0.88},
            ],
            metric="Cooperation rate",
            title="Agent cooperation comparison",
        )
    """
    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(max(6, len(agent_data) * 0.8), 5))

    ids = [d["agent_id"] for d in agent_data]
    vals = [float(d["value"]) for d in agent_data]
    bar_colors = [agent_color(d.get("agent_type", "unknown")) for d in agent_data]

    x = np.arange(len(ids))
    ax.bar(x, vals, color=bar_colors, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title)

    # Build a legend showing each unique agent type.
    seen: Dict[str, str] = {}
    for d in agent_data:
        atype = d.get("agent_type", "unknown")
        if atype not in seen:
            seen[atype] = agent_color(atype)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor="none")
        for color in seen.values()
    ]
    ax.legend(legend_handles, list(seen.keys()), fontsize=8, loc="best")

    fig.tight_layout()
    return fig, ax
