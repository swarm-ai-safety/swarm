"""Raincloud plots, histograms, and bunching analysis for SWARM.

Provides distribution-focused visualisations used in tax-policy and
reward-fairness analysis (GTB / AI Economist scenarios).  Every
plotting function returns ``(fig, ax)`` and applies the SWARM theme
via :func:`swarm_theme`.

Typical usage::

    from swarm.analysis.distributions import (
        plot_raincloud,
        plot_reward_distribution,
        plot_income_histogram,
        plot_bunching_timeseries,
        compute_bunching_coefficient,
    )

    fig, ax = plot_reward_distribution(data, group_by="persona")
    fig.savefig("reward_dist.png")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from swarm.analysis.theme import (
    COLORS,
    agent_color,
    swarm_theme,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_ORIENT = "h"
_DEFAULT_WIDTH = 0.6
_JITTER_SCALE = 0.04
_RNG = np.random.RandomState(42)


def _resolve_colors(
    n: int,
    colors: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return *n* colours, falling back to ``agent_color`` or the agent cycle."""
    if colors is not None:
        return list(colors[:n])
    if labels is not None:
        return [agent_color(str(lbl)) for lbl in labels]
    from swarm.analysis.theme import AGENT_CYCLE
    return [AGENT_CYCLE[i % len(AGENT_CYCLE)] for i in range(n)]


def _half_violin(
    ax: matplotlib.axes.Axes,
    data: np.ndarray,
    pos: float,
    orient: str,
    color: str,
    width: float,
) -> None:
    """Draw a half-violin (density on one side only).

    Uses ``ax.violinplot`` and clips the polygon vertices so only
    the upper (or right) half remains visible.
    """
    if len(data) < 2:
        return

    if orient == "h":
        parts = ax.violinplot(
            data, positions=[pos], vert=False, showextrema=False, widths=width,
        )
    else:
        parts = ax.violinplot(
            data, positions=[pos], vert=True, showextrema=False, widths=width,
        )

    for body in parts["bodies"]:  # type: ignore[attr-defined]
        vertices = body.get_paths()[0].vertices
        if orient == "h":
            # Keep only the upper half (y >= pos)
            clip_val = pos
            vertices[:, 1] = np.clip(vertices[:, 1], clip_val, None)
        else:
            # Keep only the right half (x >= pos)
            clip_val = pos
            vertices[:, 0] = np.clip(vertices[:, 0], clip_val, None)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)


def _box_whisker(
    ax: matplotlib.axes.Axes,
    data: np.ndarray,
    pos: float,
    orient: str,
    color: str,
    width: float,
) -> None:
    """Draw a compact box-whisker plot on the opposite side of the violin."""
    box_width = width * 0.15
    if orient == "h":
        bp = ax.boxplot(
            data,
            positions=[pos - width * 0.2],
            vert=False,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
        )
    else:
        bp = ax.boxplot(
            data,
            positions=[pos - width * 0.2],
            vert=True,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
        )

    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.6)
        box.set_edgecolor(color)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(color)
            line.set_linewidth(1.2)
    for flier in bp["fliers"]:
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
        flier.set_markersize(3)
        flier.set_alpha(0.5)


def _jitter_strip(
    ax: matplotlib.axes.Axes,
    data: np.ndarray,
    pos: float,
    orient: str,
    color: str,
    width: float,
) -> None:
    """Draw jittered strip of individual data points below the violin."""
    n = len(data)
    jitter = _RNG.uniform(-_JITTER_SCALE * width, _JITTER_SCALE * width, size=n)
    strip_pos = pos - width * 0.35

    if orient == "h":
        ax.scatter(
            data,
            np.full(n, strip_pos) + jitter,
            s=8,
            color=color,
            alpha=0.45,
            edgecolors="none",
            zorder=3,
        )
    else:
        ax.scatter(
            np.full(n, strip_pos) + jitter,
            data,
            s=8,
            color=color,
            alpha=0.45,
            edgecolors="none",
            zorder=3,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_raincloud(
    ax: matplotlib.axes.Axes,
    data_groups: Sequence[Sequence[float]],
    *,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    orient: str = _DEFAULT_ORIENT,
    width: float = _DEFAULT_WIDTH,
) -> None:
    """Raincloud plot (half-violin + jittered strip + box whisker).

    Renders directly onto the supplied *ax* without creating a new figure.

    Args:
        ax: Matplotlib axes to draw on.
        data_groups: List of arrays/lists, one per group.
        labels: Human-readable label for each group.  Also used to
            resolve default colours via :func:`agent_color`.
        colors: Explicit colour per group.  Overrides *labels*-based
            colour resolution when provided.
        orient: ``"h"`` for horizontal (default) or ``"v"`` for vertical.
        width: Visual width of each raincloud element.
    """
    n_groups = len(data_groups)
    resolved_labels = list(labels) if labels else [f"Group {i}" for i in range(n_groups)]
    resolved_colors = _resolve_colors(n_groups, colors=colors, labels=resolved_labels)

    for idx, group_data in enumerate(data_groups):
        arr = np.asarray(group_data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue

        pos = idx
        color = resolved_colors[idx]

        _half_violin(ax, arr, pos, orient, color, width)
        _box_whisker(ax, arr, pos, orient, color, width)
        _jitter_strip(ax, arr, pos, orient, color, width)

    # Labels and ticks
    if orient == "h":
        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(resolved_labels)
        ax.invert_yaxis()
    else:
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(resolved_labels, rotation=45, ha="right")


def plot_reward_distribution(
    data: List[Dict[str, Any]],
    *,
    group_by: str = "group",
    title: Optional[str] = None,
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Full-figure raincloud plot of rewards grouped by persona/agent type.

    Args:
        data: List of dicts, each containing ``"value"`` (numeric reward)
            and a grouping key whose name is given by *group_by*.
        group_by: Key within each dict used to partition groups
            (e.g. ``"persona"``, ``"agent_type"``).
        title: Plot title.  Defaults to ``"Reward Distribution by <group_by>"``.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    # Partition data into groups
    groups: Dict[str, List[float]] = {}
    for record in data:
        key = str(record.get(group_by, "unknown"))
        groups.setdefault(key, []).append(float(record["value"]))

    sorted_keys = sorted(groups.keys())
    data_groups = [groups[k] for k in sorted_keys]
    labels = sorted_keys

    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_keys) * 1.2)))
        plot_raincloud(ax, data_groups, labels=labels, orient="h")

        resolved_title = title or f"Reward Distribution by {group_by.replace('_', ' ').title()}"
        ax.set_title(resolved_title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Reward")
        fig.tight_layout()

    return fig, ax


def plot_income_histogram(
    incomes: Sequence[float],
    *,
    bracket_boundaries: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    highlight_bunching: bool = False,
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Histogram of incomes with tax-bracket boundaries overlaid.

    Designed for the GTB / AI Economist scenario, this chart overlays
    dashed vertical lines at each bracket boundary and optionally
    highlights bins where bunching (excess mass) is visible.

    Args:
        incomes: 1-D array of income values.
        bracket_boundaries: Boundary values drawn as vertical dashed
            lines.  Pass ``None`` to omit.
        title: Plot title.  Defaults to ``"Income Distribution"``.
        highlight_bunching: If ``True``, shade bins within 5 % of each
            bracket boundary to visually flag bunching.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    arr = np.asarray(incomes, dtype=float)
    arr = arr[np.isfinite(arr)]

    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(10, 5))

        # Histogram
        n_bins = min(80, max(20, len(arr) // 20))
        counts, bin_edges, patches = ax.hist(
            arr, bins=n_bins, color=COLORS.WELFARE, alpha=0.75,
            edgecolor=COLORS.WELFARE, linewidth=0.4,
        )

        # Bracket boundaries
        if bracket_boundaries is not None:
            for boundary in bracket_boundaries:
                ax.axvline(
                    boundary,
                    color=COLORS.EVASION,
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.85,
                    label=f"Bracket @ {boundary:,.0f}",
                )

            # Highlight bunching zones
            if highlight_bunching:
                income_range = float(np.ptp(arr)) if len(arr) > 0 else 1.0
                bandwidth = income_range * 0.05  # 5 % of range
                for boundary in bracket_boundaries:
                    ax.axvspan(
                        boundary - bandwidth,
                        boundary + bandwidth,
                        color=COLORS.TOXICITY,
                        alpha=0.10,
                        zorder=0,
                    )

        resolved_title = title or "Income Distribution"
        ax.set_title(resolved_title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Income")
        ax.set_ylabel("Count")

        # Deduplicate legend entries
        if bracket_boundaries:
            handles, lbl_list = ax.get_legend_handles_labels()
            seen: Dict[str, Any] = {}
            unique_handles = []
            unique_labels = []
            for handle, label in zip(handles, lbl_list, strict=False):
                if label not in seen:
                    seen[label] = True
                    unique_handles.append(handle)
                    unique_labels.append(label)
            ax.legend(unique_handles, unique_labels, fontsize=8)

        fig.tight_layout()

    return fig, ax


def plot_bunching_timeseries(
    bunching_coefficients: Sequence[Sequence[float]],
    *,
    epochs: Optional[Sequence[int]] = None,
    bracket_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Line chart of bunching coefficients over epochs per bracket boundary.

    Args:
        bunching_coefficients: 2-D structure shaped
            ``(n_brackets, n_epochs)``.  Each inner sequence is the time
            series for one bracket boundary.
        epochs: Epoch indices for the x-axis.  If ``None``, uses
            ``range(n_epochs)``.
        bracket_labels: Display labels per bracket boundary (e.g.
            ``["$50k", "$100k"]``).
        title: Plot title.  Defaults to
            ``"Bunching Coefficients Over Time"``.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    coefs = [np.asarray(seq, dtype=float) for seq in bunching_coefficients]
    n_brackets = len(coefs)

    if n_brackets == 0:
        with swarm_theme(mode):
            fig, ax = plt.subplots()
            ax.set_title(title or "Bunching Coefficients Over Time")
            return fig, ax

    n_epochs = len(coefs[0])
    x = np.asarray(epochs) if epochs is not None else np.arange(n_epochs)

    resolved_labels = (
        list(bracket_labels)
        if bracket_labels
        else [f"Bracket {i}" for i in range(n_brackets)]
    )

    from swarm.analysis.theme import METRIC_CYCLE

    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(10, 5))

        for idx, series in enumerate(coefs):
            color = METRIC_CYCLE[idx % len(METRIC_CYCLE)]
            ax.plot(
                x,
                series,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=4,
                label=resolved_labels[idx],
                alpha=0.9,
            )

        # Reference line at zero
        ax.axhline(0, color=COLORS.TEXT_MUTED, linewidth=0.6, linestyle=":", alpha=0.5)

        resolved_title = title or "Bunching Coefficients Over Time"
        ax.set_title(resolved_title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Bunching Coefficient")
        ax.legend(fontsize=8)
        fig.tight_layout()

    return fig, ax


def compute_bunching_coefficient(
    incomes: Sequence[float],
    boundary: float,
    bandwidth: float,
) -> float:
    """Compute excess mass (bunching coefficient) near a kink point.

    The bunching coefficient *b* measures how many more observations
    fall within ``[boundary - bandwidth, boundary + bandwidth]``
    relative to a uniform counterfactual density estimated from the
    surrounding region.

    Formally::

        b = (density_near - density_away) / density_away

    where *density_near* is the fraction of observations inside the
    window and *density_away* is the fraction expected under uniform
    density across the broader region
    ``[boundary - 3*bandwidth, boundary + 3*bandwidth]``.

    A value of 0 means no excess bunching; positive values indicate
    agents clustering near the boundary.

    Args:
        incomes: 1-D array of income values.
        boundary: Kink / bracket boundary value.
        bandwidth: Half-width of the window around *boundary*.

    Returns:
        Bunching coefficient (float).  Returns ``0.0`` when there are
        insufficient observations in the broader region.
    """
    arr = np.asarray(incomes, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)

    if n == 0 or bandwidth <= 0:
        return 0.0

    # Near window: [boundary - bandwidth, boundary + bandwidth]
    near_mask = (arr >= boundary - bandwidth) & (arr <= boundary + bandwidth)
    count_near = int(np.sum(near_mask))
    near_width = 2.0 * bandwidth

    # Broader region for counterfactual: [boundary - 3*bw, boundary + 3*bw]
    broad_lo = boundary - 3.0 * bandwidth
    broad_hi = boundary + 3.0 * bandwidth
    broad_mask = (arr >= broad_lo) & (arr <= broad_hi)
    count_broad = int(np.sum(broad_mask))
    broad_width = 6.0 * bandwidth

    if count_broad == 0:
        return 0.0

    # Density = count / (total * bin_width) for each region
    density_near = count_near / (n * near_width)
    density_away = count_broad / (n * broad_width)

    if density_away == 0.0:
        return 0.0

    return float((density_near - density_away) / density_away)
