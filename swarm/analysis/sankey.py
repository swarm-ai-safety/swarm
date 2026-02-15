"""Audit/evasion flow Sankey diagrams for SWARM.

Provides pure-matplotlib Sankey diagram renderers for visualizing
enforcement pipelines: agent populations flowing through audit,
detection, and enforcement stages.  No Plotly dependency required.

Every plotting function returns ``(fig, ax)`` (or ``(fig, axes)``)
and applies the SWARM theme via :func:`swarm_theme`.

Typical usage::

    from swarm.analysis.sankey import (
        plot_sankey,
        plot_audit_evasion_flow,
        plot_enforcement_summary,
    )

    flow_data = {
        "total": 100, "honest": 70, "evaders": 30,
        "audited": 20, "unaudited": 10,
        "caught": 15, "undetected": 5,
        "fined": 8, "frozen": 4, "continues": 3,
    }
    fig, ax = plot_audit_evasion_flow(flow_data)
    fig.savefig("enforcement_sankey.png")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from swarm.analysis.theme import (
    COLORS,
    swarm_theme,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assign_columns(
    flows: List[Dict[str, Any]],
    n_nodes: int,
) -> List[int]:
    """Assign each node to a column using a topological heuristic.

    Nodes only appearing as sources go to column 0 (leftmost).
    Nodes only appearing as targets go to the rightmost column.
    Intermediate nodes are placed proportionally between their
    earliest source and latest target.

    Returns a list of column indices (one per node).
    """
    sources_set: set[int] = set()
    targets_set: set[int] = set()
    for f in flows:
        sources_set.add(f["source"])
        targets_set.add(f["target"])

    # Build adjacency for topological depth computation.
    children: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
    for f in flows:
        children[f["source"]].append(f["target"])

    # BFS depth from roots (nodes that are never targets).
    roots = sources_set - targets_set
    if not roots:
        roots = {0}

    depth = [-1] * n_nodes
    queue: List[int] = []
    for r in roots:
        depth[r] = 0
        queue.append(r)

    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for child in children[node]:
            candidate = depth[node] + 1
            if candidate > depth[child]:
                depth[child] = candidate
                queue.append(child)

    # Any unreached node gets column 0.
    for i in range(n_nodes):
        if depth[i] < 0:
            depth[i] = 0

    return depth


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, ...]:
    """Convert a hex color string to an RGBA tuple with values in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)


def _draw_bezier_flow(
    ax: matplotlib.axes.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float,
    color: str,
    alpha: float,
) -> None:
    """Draw a single flow band as a filled bezier curve between two ports.

    The band extends from ``(x0, y0)`` to ``(x1, y1)`` with the given
    *width* (half above, half below the center line at each end).
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    hw = width / 2.0
    mid_x = (x0 + x1) / 2.0

    # Upper edge: left-top to right-top (cubic bezier).
    # Lower edge: right-bottom back to left-bottom (cubic bezier).
    verts = [
        (x0, y0 - hw),       # start bottom-left
        (mid_x, y0 - hw),    # control 1
        (mid_x, y1 - hw),    # control 2
        (x1, y1 - hw),       # end bottom-right
        (x1, y1 + hw),       # right-top
        (mid_x, y1 + hw),    # control 3
        (mid_x, y0 + hw),    # control 4
        (x0, y0 + hw),       # left-top
        (x0, y0 - hw),       # close
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]

    path = Path(verts, codes)
    rgba = _hex_to_rgba(color, alpha)
    patch = PathPatch(path, facecolor=rgba, edgecolor="none", lw=0)
    ax.add_patch(patch)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_sankey(
    ax: matplotlib.axes.Axes,
    flows: List[Dict[str, Any]],
    *,
    labels: List[str],
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    alpha: float = 0.45,
) -> matplotlib.axes.Axes:
    """Core Sankey renderer using matplotlib patches (no Plotly dependency).

    Draws a horizontal Sankey diagram on the provided *ax*.  Nodes are
    arranged in auto-detected columns (topological depth from flow
    topology), and flow bands are drawn as filled bezier curves whose
    width is proportional to value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    flows : list[dict]
        Each dict has keys ``"source"`` (int index), ``"target"``
        (int index), and ``"value"`` (float >= 0).
    labels : list[str]
        Node label strings, indexed by source/target indices.
    colors : list[str] or None
        Optional list of hex color strings per node.  Falls back to
        a rotating palette from :data:`COLORS`.
    title : str or None
        Optional title rendered above the diagram.
    alpha : float
        Opacity for flow bands (default 0.45).

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the diagram drawn.

    Examples
    --------
    ::

        fig, ax = plt.subplots(figsize=(10, 5))
        flows = [
            {"source": 0, "target": 1, "value": 70},
            {"source": 0, "target": 2, "value": 30},
            {"source": 2, "target": 3, "value": 20},
            {"source": 2, "target": 4, "value": 10},
        ]
        labels = ["Total", "Honest", "Evaders", "Audited", "Unaudited"]
        plot_sankey(ax, flows, labels=labels)
    """
    n_nodes = len(labels)
    if not flows:
        return ax

    # Default colors: cycle through a semantic palette.
    default_palette = [
        COLORS.HONEST,
        COLORS.ADVERSARIAL,
        COLORS.EVASION,
        COLORS.TOXICITY,
        COLORS.OPPORTUNISTIC,
        COLORS.WELFARE,
        COLORS.PRODUCTIVITY,
        COLORS.PLANNER,
        COLORS.DECEPTIVE,
        COLORS.REVENUE,
    ]
    if colors is None:
        colors = [default_palette[i % len(default_palette)] for i in range(n_nodes)]

    # ------------------------------------------------------------------
    # 1. Determine column assignment for each node.
    # ------------------------------------------------------------------
    columns = _assign_columns(flows, n_nodes)
    n_cols = max(columns) + 1

    # ------------------------------------------------------------------
    # 2. Compute total inflow / outflow per node (for sizing).
    # ------------------------------------------------------------------
    total_out = np.zeros(n_nodes)
    total_in = np.zeros(n_nodes)
    for f in flows:
        total_out[f["source"]] += f["value"]
        total_in[f["target"]] += f["value"]
    node_size = np.maximum(total_out, total_in)

    # ------------------------------------------------------------------
    # 3. Stack nodes vertically within each column.
    # ------------------------------------------------------------------
    node_pad = 0.05  # vertical gap between nodes (normalised)
    col_x_positions = np.linspace(0.0, 1.0, max(n_cols, 2))

    # Group nodes by column.
    col_nodes: Dict[int, List[int]] = {c: [] for c in range(n_cols)}
    for i, c in enumerate(columns):
        col_nodes[c].append(i)

    # For each column, determine vertical positions.
    node_x = np.zeros(n_nodes)
    node_y_center = np.zeros(n_nodes)
    node_height = np.zeros(n_nodes)

    global_max = float(np.max(node_size)) if np.max(node_size) > 0 else 1.0

    for c in range(n_cols):
        nodes_in_col = col_nodes[c]
        if not nodes_in_col:
            continue

        sizes = np.array([node_size[n] for n in nodes_in_col])
        # Normalize heights so they fill available vertical space.
        total_size = sizes.sum()
        total_pad = node_pad * (len(nodes_in_col) - 1)
        available = 1.0 - total_pad
        if total_size > 0:
            heights = (sizes / global_max) * available * 0.85
        else:
            heights = np.ones(len(nodes_in_col)) * 0.05

        # Center the column vertically.
        total_column_height = heights.sum() + total_pad
        y_start = 0.5 - total_column_height / 2.0

        y_cursor = y_start
        for idx, node_id in enumerate(nodes_in_col):
            h = heights[idx]
            node_x[node_id] = col_x_positions[c]
            node_y_center[node_id] = y_cursor + h / 2.0
            node_height[node_id] = h
            y_cursor += h + node_pad

    # ------------------------------------------------------------------
    # 4. Draw flow bands.
    # ------------------------------------------------------------------
    node_width = 0.03  # horizontal extent of node rectangle

    # Track port cursors: how much of each node's top/bottom has been
    # used for outgoing / incoming flows respectively.
    out_cursor = np.copy(node_y_center) + node_height / 2.0  # start from top
    in_cursor = np.copy(node_y_center) + node_height / 2.0

    for f in flows:
        src = f["source"]
        tgt = f["target"]
        value = f["value"]
        if value <= 0 or node_size[src] == 0:
            continue

        # Flow width proportional to value relative to node size.
        src_frac = value / node_size[src] if node_size[src] > 0 else 0
        tgt_frac = value / node_size[tgt] if node_size[tgt] > 0 else 0
        flow_h_src = src_frac * node_height[src]
        flow_h_tgt = tgt_frac * node_height[tgt]
        flow_h = max(flow_h_src, flow_h_tgt)

        # Update cursors (stack from top downward).
        out_cursor[src] -= flow_h / 2.0
        y_src = out_cursor[src]
        out_cursor[src] -= flow_h / 2.0

        in_cursor[tgt] -= flow_h / 2.0
        y_tgt = in_cursor[tgt]
        in_cursor[tgt] -= flow_h / 2.0

        x_src = node_x[src] + node_width / 2.0
        x_tgt = node_x[tgt] - node_width / 2.0

        _draw_bezier_flow(
            ax, x_src, y_src, x_tgt, y_tgt,
            width=flow_h, color=colors[src], alpha=alpha,
        )

    # ------------------------------------------------------------------
    # 5. Draw node rectangles and labels.
    # ------------------------------------------------------------------
    for i in range(n_nodes):
        if node_height[i] <= 0:
            continue
        rect = mpatches.FancyBboxPatch(
            (node_x[i] - node_width / 2.0, node_y_center[i] - node_height[i] / 2.0),
            node_width,
            node_height[i],
            boxstyle="round,pad=0.003",
            facecolor=colors[i],
            edgecolor="none",
            zorder=3,
        )
        ax.add_patch(rect)

        # Label positioning: left-column labels go to the left, others
        # to the right of the node rectangle.
        if columns[i] == 0:
            lx = node_x[i] - node_width / 2.0 - 0.01
            ha = "right"
        else:
            lx = node_x[i] + node_width / 2.0 + 0.01
            ha = "left"

        ax.text(
            lx,
            node_y_center[i],
            labels[i],
            ha=ha,
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=4,
        )

    # ------------------------------------------------------------------
    # 6. Axes cosmetics.
    # ------------------------------------------------------------------
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("auto")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    return ax


# ---------------------------------------------------------------------------
# Audit / Evasion convenience wrapper
# ---------------------------------------------------------------------------

# Standard node layout for the enforcement pipeline:
#   0: Total Agents
#   1: Honest Reporters
#   2: Evaders
#   3: Audited
#   4: Unaudited
#   5: Caught
#   6: Undetected
#   7: Fined
#   8: Frozen
#   9: Continues

_ENFORCEMENT_LABELS = [
    "Total Agents",
    "Honest",
    "Evaders",
    "Audited",
    "Unaudited",
    "Caught",
    "Undetected",
    "Fined",
    "Frozen",
    "Continues",
]

_ENFORCEMENT_COLORS = {
    0: COLORS.WELFARE,        # total - neutral blue
    1: COLORS.HONEST,         # honest - teal
    2: COLORS.ADVERSARIAL,    # evaders - red
    3: COLORS.EVASION,        # audited - yellow
    4: COLORS.OPPORTUNISTIC,  # unaudited - slate
    5: COLORS.TOXICITY,       # caught - red
    6: COLORS.DECEPTIVE,      # undetected - orange
    7: COLORS.TOXICITY,       # fined - red
    8: COLORS.ADVERSARIAL,    # frozen - red
    9: COLORS.OPPORTUNISTIC,  # continues - slate
}


def plot_audit_evasion_flow(
    flow_data: Dict[str, Any],
    *,
    title: Optional[str] = "Audit & Evasion Enforcement Flow",
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Convenience wrapper for the standard audit/evasion enforcement pipeline.

    Builds a Sankey diagram from a simple dict of population counts
    through the enforcement stages::

        total_agents --> honest_reporters
                     \\-> evaders --> audited   --> caught   --> fined
                                                             \\-> frozen
                                  \\-> unaudited              \\-> continues
                                                \\-> undetected --> continues

    Parameters
    ----------
    flow_data : dict
        Population counts with keys:

        - ``"total"`` -- total agent population
        - ``"honest"`` -- agents reporting honestly
        - ``"evaders"`` -- agents attempting evasion
        - ``"audited"`` -- evaders selected for audit
        - ``"unaudited"`` -- evaders not audited
        - ``"caught"`` -- audited evaders detected
        - ``"undetected"`` -- audited evaders that slipped through
        - ``"fined"`` -- caught agents receiving fines
        - ``"frozen"`` -- caught agents with frozen accounts
        - ``"continues"`` -- agents continuing to evade

    title : str or None
        Diagram title.
    mode : str
        ``"dark"`` (default) or ``"light"`` theme.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axes.

    Examples
    --------
    ::

        flow_data = {
            "total": 100, "honest": 70, "evaders": 30,
            "audited": 20, "unaudited": 10,
            "caught": 15, "undetected": 5,
            "fined": 8, "frozen": 4, "continues": 3,
        }
        fig, ax = plot_audit_evasion_flow(flow_data)
        fig.savefig("enforcement.png")
    """
    # Extract values with safe defaults.
    honest = float(flow_data.get("honest", 0))
    evaders = float(flow_data.get("evaders", 0))
    audited = float(flow_data.get("audited", 0))
    unaudited = float(flow_data.get("unaudited", 0))
    caught = float(flow_data.get("caught", 0))
    undetected = float(flow_data.get("undetected", 0))
    fined = float(flow_data.get("fined", 0))
    frozen = float(flow_data.get("frozen", 0))
    continues = float(flow_data.get("continues", 0))

    # Build flow list.
    flows: List[Dict[str, Any]] = []

    def _add(src: int, tgt: int, val: float) -> None:
        if val > 0:
            flows.append({"source": src, "target": tgt, "value": val})

    # Stage 1: total --> honest | evaders
    _add(0, 1, honest)
    _add(0, 2, evaders)

    # Stage 2: evaders --> audited | unaudited
    _add(2, 3, audited)
    _add(2, 4, unaudited)

    # Stage 3: audited --> caught | undetected
    _add(3, 5, caught)
    _add(3, 6, undetected)

    # Stage 4: caught --> fined | frozen; undetected + unaudited --> continues
    _add(5, 7, fined)
    _add(5, 8, frozen)

    # Agents that continue evading: undetected + unaudited flow into
    # "continues".  We also allow explicit "continues" to override.
    continues_from_undetected = undetected
    continues_from_unaudited = unaudited
    if continues > 0:
        # Use explicit value, split proportionally.
        if (undetected + unaudited) > 0:
            ratio = undetected / (undetected + unaudited)
            continues_from_undetected = continues * ratio
            continues_from_unaudited = continues * (1.0 - ratio)
        else:
            continues_from_undetected = continues / 2.0
            continues_from_unaudited = continues / 2.0

    _add(6, 9, continues_from_undetected)
    _add(4, 9, continues_from_unaudited)

    # Build color list.
    node_colors = [_ENFORCEMENT_COLORS[i] for i in range(len(_ENFORCEMENT_LABELS))]

    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_sankey(
            ax,
            flows,
            labels=_ENFORCEMENT_LABELS,
            colors=node_colors,
            title=title,
            alpha=0.40,
        )

    return fig, ax


# ---------------------------------------------------------------------------
# Enforcement summary (small multiples)
# ---------------------------------------------------------------------------


def plot_enforcement_summary(
    flow_data_per_epoch: List[Dict[str, Any]],
    *,
    title: Optional[str] = "Enforcement Pipeline Over Time",
    mode: str = "dark",
) -> Tuple[plt.Figure, np.ndarray]:
    """Small-multiples Sankey diagrams showing enforcement changes per epoch.

    Renders one mini-Sankey per epoch so the reader can visually compare
    how the audit/evasion pipeline evolves across simulation epochs.

    Parameters
    ----------
    flow_data_per_epoch : list[dict]
        One ``flow_data`` dict per epoch (same schema as
        :func:`plot_audit_evasion_flow`).
    title : str or None
        Suptitle for the whole figure.
    mode : str
        ``"dark"`` (default) or ``"light"`` theme.

    Returns
    -------
    (fig, axes_array)
        Matplotlib figure and ndarray of axes (one per epoch).

    Examples
    --------
    ::

        epoch_data = [
            {"total": 100, "honest": 80, "evaders": 20,
             "audited": 15, "unaudited": 5,
             "caught": 10, "undetected": 5,
             "fined": 6, "frozen": 2, "continues": 2},
            {"total": 100, "honest": 85, "evaders": 15,
             "audited": 12, "unaudited": 3,
             "caught": 10, "undetected": 2,
             "fined": 7, "frozen": 2, "continues": 1},
        ]
        fig, axes = plot_enforcement_summary(epoch_data)
        fig.savefig("enforcement_over_time.png")
    """
    n_epochs = len(flow_data_per_epoch)
    if n_epochs == 0:
        with swarm_theme(mode):
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "No epoch data", ha="center", va="center")
            ax.axis("off")
            return fig, np.array([ax])

    # Layout: up to 4 columns, as many rows as needed.
    n_cols = min(n_epochs, 4)
    n_rows = int(np.ceil(n_epochs / n_cols))

    node_colors = [_ENFORCEMENT_COLORS[i] for i in range(len(_ENFORCEMENT_LABELS))]

    with swarm_theme(mode):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            squeeze=False,
        )

        for idx, fd in enumerate(flow_data_per_epoch):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Build per-epoch flows (same logic as plot_audit_evasion_flow).
            honest = float(fd.get("honest", 0))
            evaders = float(fd.get("evaders", 0))
            audited = float(fd.get("audited", 0))
            unaudited = float(fd.get("unaudited", 0))
            caught = float(fd.get("caught", 0))
            undetected = float(fd.get("undetected", 0))
            fined = float(fd.get("fined", 0))
            frozen = float(fd.get("frozen", 0))
            continues = float(fd.get("continues", 0))

            epoch_flows: List[Dict[str, Any]] = []

            def _add_e(src: int, tgt: int, val: float, _ef: List[Dict[str, Any]] = epoch_flows) -> None:
                if val > 0:
                    _ef.append(
                        {"source": src, "target": tgt, "value": val}
                    )

            _add_e(0, 1, honest)
            _add_e(0, 2, evaders)
            _add_e(2, 3, audited)
            _add_e(2, 4, unaudited)
            _add_e(3, 5, caught)
            _add_e(3, 6, undetected)
            _add_e(5, 7, fined)
            _add_e(5, 8, frozen)

            continues_from_undetected = undetected
            continues_from_unaudited = unaudited
            if continues > 0:
                denom = undetected + unaudited
                if denom > 0:
                    ratio = undetected / denom
                    continues_from_undetected = continues * ratio
                    continues_from_unaudited = continues * (1.0 - ratio)
                else:
                    continues_from_undetected = continues / 2.0
                    continues_from_unaudited = continues / 2.0

            _add_e(6, 9, continues_from_undetected)
            _add_e(4, 9, continues_from_unaudited)

            plot_sankey(
                ax,
                epoch_flows,
                labels=_ENFORCEMENT_LABELS,
                colors=node_colors,
                title=f"Epoch {idx}",
                alpha=0.40,
            )

        # Hide unused subplot panels.
        for idx in range(n_epochs, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        if title:
            fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

        fig.tight_layout()

    return fig, axes
