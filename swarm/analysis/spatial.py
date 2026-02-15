"""Grid world renderer for the AI Economist GTB scenario.

Renders 2-D grid states as static images or animated GIFs, with agent
overlays and resource heatmaps.  Designed for spatial scenarios where
agents move through a discrete grid collecting resources and building
structures.

Usage::

    from swarm.analysis.spatial import render_grid_frame, render_grid_sequence

    fig, ax = render_grid_frame(grid, agent_positions=agents, title="Step 10")
    render_grid_sequence(frames, output_path="runs/animation.gif", fps=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from swarm.analysis.theme import (
    COLORS,
    agent_color,
    swarm_theme,
)

# ---------------------------------------------------------------------------
# Agent marker configuration
# ---------------------------------------------------------------------------

_AGENT_MARKERS: Dict[str, str] = {
    "honest": "o",         # circle
    "deceptive": "^",      # triangle
    "planner": "s",        # square
    "adversarial": "D",    # diamond
}
_DEFAULT_MARKER: str = "o"

# Default cell palette: cell_value -> color
_DEFAULT_CELL_COLORS: Dict[int, str] = {
    0: COLORS.BG_DARK,    # empty
    1: "#8B6914",          # wood / brown
    2: "#A0A0A0",          # stone / gray
    3: COLORS.HONEST,     # house / teal
}

_CELL_LABELS: Dict[int, str] = {
    0: "Empty",
    1: "Wood",
    2: "Stone",
    3: "House",
}


# ---------------------------------------------------------------------------
# Core grid renderer
# ---------------------------------------------------------------------------


def render_grid(
    ax: matplotlib.axes.Axes,
    grid: np.ndarray,
    *,
    agent_positions: Optional[List[Dict[str, Any]]] = None,
    cell_colors: Optional[Dict[int, str]] = None,
    legend: bool = True,
    title: Optional[str] = None,
) -> matplotlib.axes.Axes:
    """Render a grid world onto an existing matplotlib *Axes*.

    Args:
        ax: Target axes.
        grid: 2-D array; cell values map to terrain/resource types.
        agent_positions: List of dicts with ``row``, ``col``, ``type``,
            and optional ``label``.
        cell_colors: Cell value -> color string override.
        legend: Show legend for cell types and agent markers.
        title: Optional axes title.
    """
    if cell_colors is None:
        cell_colors = dict(_DEFAULT_CELL_COLORS)

    grid = np.asarray(grid)
    rows, cols = grid.shape

    # Ensure every unique value has a colour entry
    for v in sorted({int(v) for v in np.unique(grid)}):
        if v not in cell_colors:
            cell_colors[v] = COLORS.BG_DARK

    # Build RGBA image from cell values
    rgba = np.zeros((rows, cols, 4), dtype=float)
    for val, hex_color in cell_colors.items():
        mask = grid == val
        if not np.any(mask):
            continue
        r, g, b = mcolors.to_rgb(hex_color)
        rgba[mask] = [r, g, b, 1.0]

    ax.imshow(rgba, interpolation="nearest", origin="upper", aspect="equal")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color=COLORS.GRID, linewidth=0.5, alpha=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(np.arange(0, cols, max(1, cols // 5)))
    ax.set_yticks(np.arange(0, rows, max(1, rows // 5)))
    ax.tick_params(labelsize=7)

    # Draw agents
    agent_handles: Dict[str, mpatches.Patch] = {}
    if agent_positions:
        for agent in agent_positions:
            r = agent["row"]
            c = agent["col"]
            atype = agent.get("type", "honest")
            label = agent.get("label", "")
            color = agent_color(atype)
            marker = _AGENT_MARKERS.get(atype.lower(), _DEFAULT_MARKER)

            ax.plot(
                c, r,
                marker=marker,
                color=color,
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=5,
            )

            if label:
                ax.annotate(
                    label,
                    (c, r),
                    textcoords="offset points",
                    xytext=(5, -5),
                    fontsize=6,
                    color="white",
                    zorder=6,
                )

            key = atype.lower()
            if key not in agent_handles:
                agent_handles[key] = mpatches.Patch(
                    facecolor=color,
                    edgecolor="white",
                    label=f"{atype.capitalize()} ({marker})",
                )

    # Legend
    if legend:
        handles: List[mpatches.Patch] = []
        for val in sorted(cell_colors.keys()):
            lbl = _CELL_LABELS.get(val, f"Type {val}")
            handles.append(
                mpatches.Patch(
                    facecolor=cell_colors[val],
                    edgecolor=COLORS.ACCENT_BORDER,
                    label=lbl,
                )
            )
        handles.extend(agent_handles.values())
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            framealpha=0.85,
        )

    if title:
        ax.set_title(title, fontsize=10, pad=8)

    return ax


# ---------------------------------------------------------------------------
# Stand-alone frame
# ---------------------------------------------------------------------------


def render_grid_frame(
    grid: np.ndarray,
    *,
    agent_positions: Optional[List[Dict[str, Any]]] = None,
    cell_colors: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
    mode: str = "dark",
    figsize: Tuple[float, float] = (8, 8),
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Create a complete figure from a single grid state.

    Wrapper around :func:`render_grid` that creates its own figure
    with :func:`swarm_theme`.  Returns ``(fig, ax)``.
    """
    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=figsize)
        render_grid(
            ax,
            grid,
            agent_positions=agent_positions,
            cell_colors=cell_colors,
            title=title,
        )
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Animated GIF from a sequence of frames
# ---------------------------------------------------------------------------


def render_grid_sequence(
    frames: List[Dict[str, Any]],
    *,
    output_path: Optional[Path] = None,
    fps: int = 5,
    mode: str = "dark",
) -> Path:
    """Render a sequence of grid states as an animated GIF.

    Each entry in *frames* is a dict with key ``"grid"`` (2-D array)
    and optional ``"agent_positions"`` and ``"title"``.  Frames are
    rendered to in-memory PNG buffers, converted to PIL images, then
    combined via :func:`~swarm.analysis.figure_export.save_gif`.

    Returns the output :class:`~pathlib.Path`.
    """
    import io

    from PIL import Image as PILImage

    from swarm.analysis.figure_export import save_gif

    if output_path is None:
        output_path = Path("grid_animation.gif")
    output_path = Path(output_path)

    pil_frames: List[PILImage.Image] = []

    for frame_data in frames:
        grid = np.asarray(frame_data["grid"])
        agent_positions = frame_data.get("agent_positions")
        title = frame_data.get("title")

        fig, _ax = render_grid_frame(
            grid,
            agent_positions=agent_positions,
            title=title,
            mode=mode,
            figsize=(6, 6),
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)

        pil_img = PILImage.open(buf).convert("RGBA")
        pil_frames.append(pil_img)

    return save_gif(pil_frames, output_path, fps=fps)


# ---------------------------------------------------------------------------
# Resource heatmap across a grid sequence
# ---------------------------------------------------------------------------


def plot_resource_heatmap(
    grid_sequence: Sequence[np.ndarray],
    *,
    resource_type: int = 1,
    title: Optional[str] = None,
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Aggregate resource density heatmap across a sequence of frames.

    Counts how often each cell contains *resource_type* over the full
    sequence and normalises to ``[0, 1]``.  Reveals where resources
    accumulate or deplete over time.  Returns ``(fig, ax)``.
    """
    if not grid_sequence:
        raise ValueError("grid_sequence must contain at least one grid")

    grids = [np.asarray(g) for g in grid_sequence]
    shape = grids[0].shape
    n_frames = len(grids)

    density = np.zeros(shape, dtype=float)
    for g in grids:
        density += (g == resource_type).astype(float)
    density /= max(n_frames, 1)

    if title is None:
        title = f"Resource density (type {resource_type})"

    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(8, 8))

        if mode == "dark":
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "resource_density", [COLORS.BG_DARK, COLORS.HONEST], N=256,
            )
        else:
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "resource_density", ["#FFFFFF", COLORS.HONEST], N=256,
            )

        im = ax.imshow(
            density, cmap=cmap, interpolation="nearest",
            origin="upper", aspect="equal", vmin=0.0, vmax=1.0,
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Fraction of frames present", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        rows, cols = shape
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color=COLORS.GRID, linewidth=0.3, alpha=0.4)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(np.arange(0, cols, max(1, cols // 5)))
        ax.set_yticks(np.arange(0, rows, max(1, rows // 5)))
        ax.tick_params(labelsize=7)

        ax.set_title(title, fontsize=10, pad=8)
        fig.tight_layout()

    return fig, ax
