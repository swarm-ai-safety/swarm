"""SWARM visual theme: colors, matplotlib styles, and theme helpers.

Provides a consistent visual identity for all SWARM charts.
Dark-first palette designed for safety-critical system monitoring,
with a light variant for paper/print exports.

Usage::

    from swarm.analysis.theme import apply_theme, COLORS

    apply_theme()          # dark mode (default)
    apply_theme("light")   # paper/print mode

    fig, ax = plt.subplots()
    ax.plot(x, y, color=COLORS.HONEST)
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

import matplotlib as mpl
import numpy as np

# ---------------------------------------------------------------------------
# Semantic color constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _AgentColors:
    """Semantic colors for agent types."""
    HONEST: str = "#3ECFB4"
    DECEPTIVE: str = "#F2994A"
    OPPORTUNISTIC: str = "#828894"
    ADVERSARIAL: str = "#EB5757"
    PLANNER: str = "#BB6BD9"


@dataclass(frozen=True)
class _MetricColors:
    """Semantic colors for metric categories."""
    WELFARE: str = "#2F80ED"
    TOXICITY: str = "#EB5757"
    PRODUCTIVITY: str = "#27AE60"
    REVENUE: str = "#6FCF97"
    EVASION: str = "#F2C94C"


@dataclass(frozen=True)
class _ChromeColors:
    """Background, text, and grid colors."""
    # Dark mode
    BG_DARK: str = "#0D1117"
    BG_PANEL: str = "#161B22"
    TEXT_PRIMARY: str = "#E6EDF3"
    TEXT_MUTED: str = "#7D8590"
    GRID: str = "#21262D"
    ACCENT_BORDER: str = "#30363D"
    # Light mode
    BG_LIGHT: str = "#FFFFFF"
    TEXT_DARK: str = "#1F2328"
    GRID_LIGHT: str = "#D0D7DE"


@dataclass(frozen=True)
class _Colors:
    """All SWARM color constants, grouped by domain."""
    agent: _AgentColors = _AgentColors()
    metric: _MetricColors = _MetricColors()
    chrome: _ChromeColors = _ChromeColors()

    # Convenience aliases (flat access)
    HONEST: str = _AgentColors.HONEST
    DECEPTIVE: str = _AgentColors.DECEPTIVE
    OPPORTUNISTIC: str = _AgentColors.OPPORTUNISTIC
    ADVERSARIAL: str = _AgentColors.ADVERSARIAL
    PLANNER: str = _AgentColors.PLANNER

    WELFARE: str = _MetricColors.WELFARE
    TOXICITY: str = _MetricColors.TOXICITY
    PRODUCTIVITY: str = _MetricColors.PRODUCTIVITY
    REVENUE: str = _MetricColors.REVENUE
    EVASION: str = _MetricColors.EVASION

    BG_DARK: str = _ChromeColors.BG_DARK
    BG_PANEL: str = _ChromeColors.BG_PANEL
    TEXT_PRIMARY: str = _ChromeColors.TEXT_PRIMARY
    TEXT_MUTED: str = _ChromeColors.TEXT_MUTED
    GRID: str = _ChromeColors.GRID
    ACCENT_BORDER: str = _ChromeColors.ACCENT_BORDER
    BG_LIGHT: str = _ChromeColors.BG_LIGHT
    TEXT_DARK: str = _ChromeColors.TEXT_DARK
    GRID_LIGHT: str = _ChromeColors.GRID_LIGHT


COLORS = _Colors()

# Ordered color cycle for multi-series plots
AGENT_CYCLE: List[str] = [
    COLORS.HONEST,
    COLORS.DECEPTIVE,
    COLORS.OPPORTUNISTIC,
    COLORS.ADVERSARIAL,
    COLORS.PLANNER,
]

METRIC_CYCLE: List[str] = [
    COLORS.WELFARE,
    COLORS.TOXICITY,
    COLORS.PRODUCTIVITY,
    COLORS.REVENUE,
    COLORS.EVASION,
]

# Mapping from agent type string → color (case-insensitive lookup)
AGENT_COLOR_MAP: Dict[str, str] = {
    "honest": COLORS.HONEST,
    "deceptive": COLORS.DECEPTIVE,
    "opportunistic": COLORS.OPPORTUNISTIC,
    "adversarial": COLORS.ADVERSARIAL,
    "adaptive_adversary": COLORS.ADVERSARIAL,
    "planner": COLORS.PLANNER,
}

# Mapping from metric name → color
METRIC_COLOR_MAP: Dict[str, str] = {
    "toxicity_rate": COLORS.TOXICITY,
    "toxicity": COLORS.TOXICITY,
    "welfare": COLORS.WELFARE,
    "total_welfare": COLORS.WELFARE,
    "productivity": COLORS.PRODUCTIVITY,
    "revenue": COLORS.REVENUE,
    "evasion": COLORS.EVASION,
    "quality_gap": COLORS.WELFARE,
    "avg_p": COLORS.HONEST,
    "avg_payoff": COLORS.PRODUCTIVITY,
    "gini_coefficient": COLORS.EVASION,
    "ecosystem_threat_level": COLORS.ADVERSARIAL,
    "incoherence_index": COLORS.DECEPTIVE,
}

# ---------------------------------------------------------------------------
# matplotlib style dicts
# ---------------------------------------------------------------------------

SWARM_STYLE: Dict[str, Any] = {
    "figure.facecolor": COLORS.BG_DARK,
    "axes.facecolor": COLORS.BG_PANEL,
    "axes.edgecolor": COLORS.ACCENT_BORDER,
    "axes.labelcolor": COLORS.TEXT_PRIMARY,
    "text.color": COLORS.TEXT_PRIMARY,
    "xtick.color": COLORS.TEXT_MUTED,
    "ytick.color": COLORS.TEXT_MUTED,
    "grid.color": COLORS.GRID,
    "grid.alpha": 0.6,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.facecolor": COLORS.BG_PANEL,
    "legend.edgecolor": COLORS.ACCENT_BORDER,
    "legend.labelcolor": COLORS.TEXT_PRIMARY,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.facecolor": COLORS.BG_DARK,
    "savefig.edgecolor": "none",
    "font.family": "monospace",
    "axes.prop_cycle": mpl.cycler(color=AGENT_CYCLE),
    "figure.figsize": (10, 6),
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "patch.edgecolor": COLORS.ACCENT_BORDER,
}

SWARM_LIGHT_STYLE: Dict[str, Any] = {
    "figure.facecolor": COLORS.BG_LIGHT,
    "axes.facecolor": COLORS.BG_LIGHT,
    "axes.edgecolor": COLORS.GRID_LIGHT,
    "axes.labelcolor": COLORS.TEXT_DARK,
    "text.color": COLORS.TEXT_DARK,
    "xtick.color": COLORS.TEXT_DARK,
    "ytick.color": COLORS.TEXT_DARK,
    "grid.color": COLORS.GRID_LIGHT,
    "grid.alpha": 0.5,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.facecolor": COLORS.BG_LIGHT,
    "legend.edgecolor": COLORS.GRID_LIGHT,
    "legend.labelcolor": COLORS.TEXT_DARK,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.facecolor": COLORS.BG_LIGHT,
    "savefig.edgecolor": "none",
    "font.family": "monospace",
    "axes.prop_cycle": mpl.cycler(color=[
        # Slightly desaturated for print
        "#35B09B", "#D4822F", "#6E7380", "#CC4545", "#9E5ABB",
    ]),
    "figure.figsize": (10, 6),
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "patch.edgecolor": COLORS.GRID_LIGHT,
}

# ---------------------------------------------------------------------------
# Diverging colormap for heatmaps (blue → white → red)
# ---------------------------------------------------------------------------


def _build_diverging_cmap(name: str = "swarm_diverging") -> mpl.colors.LinearSegmentedColormap:
    """Build a blue-white-red diverging colormap for SWARM heatmaps."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        name,
        [COLORS.WELFARE, "#E6EDF3", COLORS.TOXICITY],
        N=256,
    )


def _build_dark_diverging_cmap(name: str = "swarm_diverging_dark") -> mpl.colors.LinearSegmentedColormap:
    """Diverging colormap with dark midpoint for dark-mode heatmaps."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        name,
        [COLORS.WELFARE, COLORS.BG_PANEL, COLORS.TOXICITY],
        N=256,
    )


SWARM_DIVERGING = _build_diverging_cmap()
SWARM_DIVERGING_DARK = _build_dark_diverging_cmap()


# ---------------------------------------------------------------------------
# Theme application helpers
# ---------------------------------------------------------------------------


def apply_theme(mode: str = "dark") -> None:
    """Apply the SWARM matplotlib theme globally.

    Args:
        mode: ``"dark"`` (default) for monitoring dashboards,
              ``"light"`` for paper/print exports.
    """
    style = SWARM_STYLE if mode == "dark" else SWARM_LIGHT_STYLE
    mpl.rcParams.update(style)


@contextlib.contextmanager
def swarm_theme(mode: str = "dark") -> Iterator[None]:
    """Context manager that temporarily applies the SWARM theme.

    Usage::

        with swarm_theme("light"):
            fig, ax = plt.subplots()
            ...  # plots use SWARM light theme
        # original rcParams restored here
    """
    style = SWARM_STYLE if mode == "dark" else SWARM_LIGHT_STYLE
    with mpl.rc_context(style):
        yield


def agent_color(agent_type: str) -> str:
    """Return the canonical color for an agent type string.

    Falls back to ``COLORS.OPPORTUNISTIC`` (slate) for unknown types.
    """
    return AGENT_COLOR_MAP.get(agent_type.lower(), COLORS.OPPORTUNISTIC)


def metric_color(metric_name: str) -> str:
    """Return the canonical color for a metric name.

    Falls back to ``COLORS.WELFARE`` (blue) for unknown metrics.
    """
    return METRIC_COLOR_MAP.get(metric_name.lower(), COLORS.WELFARE)


def color_for_values(
    values: Sequence[float],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[mpl.colors.Colormap] = None,
) -> np.ndarray:
    """Map numeric values to RGBA colors using a SWARM colormap.

    Args:
        values: Numeric sequence to map.
        vmin: Minimum for normalization (defaults to min of *values*).
        vmax: Maximum for normalization (defaults to max of *values*).
        cmap: Colormap to use (defaults to :data:`SWARM_DIVERGING`).

    Returns:
        ``(N, 4)`` RGBA array.
    """
    arr = np.asarray(values, dtype=float)
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))
    if cmap is None:
        cmap = SWARM_DIVERGING
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    return np.asarray(cmap(norm(arr)))


def annotate_events(
    ax: mpl.axes.Axes,
    events: Sequence[Dict[str, Any]],
    y_frac: float = 0.95,
    color: Optional[str] = None,
) -> None:
    """Add vertical annotation lines for governance events.

    Args:
        ax: Matplotlib axes to annotate.
        events: Sequence of dicts with ``"epoch"`` and optional ``"label"`` keys.
        y_frac: Fractional y-position for annotation text (0–1).
        color: Line/text color (defaults to muted text).
    """
    color = color or COLORS.TEXT_MUTED
    for event in events:
        epoch = event.get("epoch")
        if epoch is None:
            continue
        label = event.get("label", "")
        ax.axvline(epoch, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
        if label:
            ax.text(
                epoch,
                y_frac,
                f" {label}",
                transform=ax.get_xaxis_transform(),
                fontsize=7,
                color=color,
                rotation=90,
                verticalalignment="top",
            )


def add_danger_zone(
    ax: mpl.axes.Axes,
    threshold: float,
    direction: str = "above",
    color: Optional[str] = None,
    alpha: float = 0.08,
    label: Optional[str] = None,
) -> None:
    """Shade a danger zone above or below a threshold.

    Args:
        ax: Matplotlib axes.
        threshold: Threshold value on the y-axis.
        direction: ``"above"`` or ``"below"``.
        color: Fill color (defaults to TOXICITY red).
        alpha: Fill opacity.
        label: Optional label to display at the threshold line.
    """
    color = color or COLORS.TOXICITY
    ylim = ax.get_ylim()
    if direction == "above":
        ax.axhspan(threshold, ylim[1], color=color, alpha=alpha)
    else:
        ax.axhspan(ylim[0], threshold, color=color, alpha=alpha)
    ax.axhline(threshold, color=color, linewidth=0.8, linestyle=":", alpha=0.5)
    if label:
        ax.text(
            0.01,
            threshold,
            f" {label}",
            transform=ax.get_yaxis_transform(),
            fontsize=7,
            color=color,
            verticalalignment="bottom" if direction == "above" else "top",
        )
