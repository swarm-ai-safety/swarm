"""Summary card generator and multi-panel layout composer for SWARM dashboards.

Provides matplotlib-based summary cards and layout composition that work
standalone (without Streamlit).  The existing ``dashboard.py`` handles
Streamlit integration; this module targets static figure export and
notebook-friendly rendering.

Usage::

    from swarm.analysis.dashboard_cards import plot_summary_cards, compose_dashboard

    fig, axes = plot_summary_cards([
        {"value": 0.12, "label": "Toxicity"},
        {"value": 850,  "label": "Welfare", "fmt": ".0f"},
    ])
    fig.savefig("cards.png")
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.axes
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from swarm.analysis.theme import (
    COLORS,
    metric_color,
    swarm_theme,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DELTA_UP = "\u25B2"  # filled up-pointing triangle
_DELTA_DOWN = "\u25BC"  # filled down-pointing triangle

# Metrics rendered by the pre-built overview dashboard
_OVERVIEW_METRICS: List[str] = [
    "toxicity_rate",
    "quality_gap",
    "total_welfare",
    "avg_payoff",
]

# Second-row metrics (displayed as a supplementary card strip)
_OVERVIEW_SECONDARY: List[str] = [
    "gini_coefficient",
    "n_frozen",
    "avg_reputation",
    "ecosystem_threat_level",
]


# ---------------------------------------------------------------------------
# 1. Single metric card
# ---------------------------------------------------------------------------


def plot_metric_card(
    ax: matplotlib.axes.Axes,
    value: float,
    *,
    label: str = "",
    delta: Optional[float] = None,
    delta_label: Optional[str] = None,
    color: Optional[str] = None,
    fmt: str = ".2f",
) -> None:
    """Render a single summary card on *ax*.

    The card shows a large centred number, a label underneath, and an
    optional delta indicator (up/down arrow + percentage change).

    Args:
        ax: Matplotlib axes to draw on.
        value: The main metric value.
        label: Metric name displayed below the number.
        delta: Optional change from a previous period.
        delta_label: Optional annotation for the delta (e.g. "vs prev epoch").
        color: Card accent colour.  Falls back to ``metric_color(label)``.
        fmt: Format string for *value* (default ``".2f"``).
    """
    color = color or metric_color(label)

    # Clear axes chrome
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main value
    value_text = f"{value:{fmt}}"
    ax.text(
        0.5,
        0.58,
        value_text,
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
    )

    # Label
    ax.text(
        0.5,
        0.30,
        label,
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS.TEXT_MUTED,
        transform=ax.transAxes,
    )

    # Delta indicator
    if delta is not None:
        arrow = _DELTA_UP if delta >= 0 else _DELTA_DOWN
        delta_color = COLORS.metric.PRODUCTIVITY if delta >= 0 else COLORS.metric.TOXICITY
        delta_str = f"{arrow} {abs(delta):.2%}"
        if delta_label:
            delta_str = f"{delta_str}  {delta_label}"
        ax.text(
            0.5,
            0.12,
            delta_str,
            ha="center",
            va="center",
            fontsize=8,
            color=delta_color,
            transform=ax.transAxes,
        )


# ---------------------------------------------------------------------------
# 2. Row of summary cards
# ---------------------------------------------------------------------------


def plot_summary_cards(
    metrics: Sequence[Dict[str, Any]],
    *,
    ncols: int = 4,
    title: Optional[str] = None,
    mode: str = "dark",
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a row of summary cards for key metrics.

    Args:
        metrics: Sequence of dicts, each containing:
            - ``"value"`` (float): metric value (required).
            - ``"label"`` (str): metric name (required).
            - ``"delta"`` (float, optional): change from previous period.
            - ``"delta_label"`` (str, optional): annotation for the delta.
            - ``"color"`` (str, optional): accent colour override.
            - ``"fmt"`` (str, optional): format string override.
        ncols: Number of columns in the card row (default 4).
        title: Optional suptitle for the figure.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, axes_array)`` where *axes_array* is a 1-D numpy array of Axes.
    """
    nrows = max(1, int(np.ceil(len(metrics) / ncols)))

    with swarm_theme(mode):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.2 * ncols, 2.2 * nrows),
        )
        axes_flat: np.ndarray = np.atleast_1d(np.asarray(axes)).flatten()

        for idx, card in enumerate(metrics):
            plot_metric_card(
                axes_flat[idx],
                card["value"],
                label=card.get("label", ""),
                delta=card.get("delta"),
                delta_label=card.get("delta_label"),
                color=card.get("color"),
                fmt=card.get("fmt", ".2f"),
            )

        # Hide surplus axes
        for idx in range(len(metrics), len(axes_flat)):
            axes_flat[idx].axis("off")

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        rect = (0, 0, 1, 0.94) if title else (0, 0, 1, 1)
        fig.tight_layout(rect=rect)

    return fig, axes_flat


# ---------------------------------------------------------------------------
# 3. Multi-panel layout composer
# ---------------------------------------------------------------------------


def compose_dashboard(
    panels: Sequence[Dict[str, Any]],
    *,
    layout: Tuple[int, int] = (2, 2),
    title: Optional[str] = None,
    mode: str = "dark",
) -> Tuple[plt.Figure, Dict[str, matplotlib.axes.Axes]]:
    """Arrange multiple plot functions into a flexible grid layout.

    Args:
        panels: Sequence of dicts, each containing:
            - ``"plot_func"`` (callable): function ``(ax) -> None``.
            - ``"title"`` (str): panel title.
            - ``"colspan"`` (int, optional): columns to span (default 1).
            - ``"rowspan"`` (int, optional): rows to span (default 1).
        layout: ``(nrows, ncols)`` grid dimensions.
        title: Optional suptitle for the figure.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, axes_dict)`` where *axes_dict* maps each panel title to its
        Axes instance.
    """
    nrows, ncols = layout

    with swarm_theme(mode):
        fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
        gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.35, wspace=0.30)

        axes_dict: Dict[str, matplotlib.axes.Axes] = {}
        row, col = 0, 0

        for panel in panels:
            colspan: int = panel.get("colspan", 1)
            rowspan: int = panel.get("rowspan", 1)
            panel_title: str = panel.get("title", "")
            plot_func: Callable[..., Any] = panel["plot_func"]

            # Advance to next row if we would overflow
            if col + colspan > ncols:
                row += 1
                col = 0
            if row + rowspan > nrows:
                break  # no more room

            ax = fig.add_subplot(gs[row : row + rowspan, col : col + colspan])
            plot_func(ax)

            if panel_title:
                ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=8)
                axes_dict[panel_title] = ax

            col += colspan
            if col >= ncols:
                col = 0
                row += 1

        if title:
            fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)

    return fig, axes_dict


# ---------------------------------------------------------------------------
# 4. Pre-built overview dashboard
# ---------------------------------------------------------------------------

_METRIC_LABELS: Dict[str, str] = {
    "toxicity_rate": "Toxicity Rate",
    "quality_gap": "Quality Gap",
    "total_welfare": "Total Welfare",
    "avg_payoff": "Avg Payoff",
    "gini_coefficient": "Gini Coefficient",
    "n_frozen": "Frozen Agents",
    "avg_reputation": "Avg Reputation",
    "ecosystem_threat_level": "Threat Level",
}

_METRIC_FMT: Dict[str, str] = {
    "toxicity_rate": ".2%",
    "quality_gap": ".3f",
    "total_welfare": ".1f",
    "avg_payoff": ".2f",
    "gini_coefficient": ".3f",
    "n_frozen": ".0f",
    "avg_reputation": ".2f",
    "ecosystem_threat_level": ".2f",
}


def _build_card_dicts(
    snapshot: Dict[str, Any],
    keys: Sequence[str],
    prev_snapshot: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build card dicts for a list of metric keys from snapshots."""
    cards: List[Dict[str, Any]] = []
    for key in keys:
        value = snapshot.get(key, 0.0)
        card: Dict[str, Any] = {
            "value": float(value),
            "label": _METRIC_LABELS.get(key, key),
            "fmt": _METRIC_FMT.get(key, ".2f"),
            "color": metric_color(key),
        }
        if prev_snapshot is not None and key in prev_snapshot:
            prev_val = float(prev_snapshot[key])
            if prev_val != 0:
                card["delta"] = (float(value) - prev_val) / abs(prev_val)
            else:
                card["delta"] = 0.0
            card["delta_label"] = "vs prev epoch"
        cards.append(card)
    return cards


def plot_overview_dashboard(
    epoch_snapshot: Dict[str, Any],
    *,
    prev_snapshot: Optional[Dict[str, Any]] = None,
    mode: str = "dark",
) -> Tuple[plt.Figure, np.ndarray]:
    """Pre-built overview dashboard showing key SWARM metrics.

    Creates a two-row layout:

    * **Top row** -- four primary summary cards (toxicity, quality gap,
      welfare, average payoff).
    * **Bottom row** -- four secondary cards (Gini, frozen agents,
      reputation, threat level).

    Args:
        epoch_snapshot: Dict with metric values.  Expected keys include
            ``toxicity_rate``, ``quality_gap``, ``total_welfare``,
            ``avg_payoff``, ``gini_coefficient``, ``n_frozen``,
            ``avg_reputation``, and ``ecosystem_threat_level``.
        prev_snapshot: Optional previous-epoch snapshot used to compute
            delta indicators on each card.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, axes)`` -- the figure and a flat array of all card axes.
    """
    primary = _build_card_dicts(epoch_snapshot, _OVERVIEW_METRICS, prev_snapshot)
    secondary = _build_card_dicts(epoch_snapshot, _OVERVIEW_SECONDARY, prev_snapshot)
    all_cards = primary + secondary

    fig, axes = plot_summary_cards(
        all_cards,
        ncols=4,
        title="SWARM Overview",
        mode=mode,
    )

    return fig, axes
