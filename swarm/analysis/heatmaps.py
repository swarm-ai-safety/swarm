"""Diverging heatmaps and difference maps for SWARM.

Provides publication-ready heatmap visualizations centered on diverging
colormaps (blue-below, red-above) with automatic seaborn fallback to
matplotlib imshow + manual annotation.

Usage::

    from swarm.analysis.heatmaps import plot_diverging_heatmap

    fig, ax = plot_diverging_heatmap(
        matrix, row_labels=personas, col_labels=mixes,
        title="Mean Toxicity by Persona x Mix",
    )
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from swarm.analysis.theme import (
    COLORS,
    SWARM_DIVERGING,
    SWARM_DIVERGING_DARK,
    swarm_theme,
)

# Optional seaborn import -- fall back gracefully.
try:
    import seaborn as sns  # type: ignore[import-untyped]

    _HAS_SEABORN = True
except ImportError:  # pragma: no cover
    _HAS_SEABORN = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_cmap(
    cmap: Optional[Any],
    mode: str,
) -> Any:
    """Return the colormap to use, respecting *mode* when no override given."""
    if cmap is not None:
        return cmap
    return SWARM_DIVERGING_DARK if mode == "dark" else SWARM_DIVERGING


def _annotation_color(mode: str) -> str:
    """Text color for cell annotations."""
    return COLORS.TEXT_PRIMARY if mode == "dark" else COLORS.TEXT_DARK


def _imshow_heatmap(
    ax: matplotlib.axes.Axes,
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    center: float,
    annotate: bool,
    fmt: str,
    cmap: Any,
    mode: str,
) -> matplotlib.axes.Axes:
    """Render a heatmap on *ax* using ``imshow`` + manual annotation.

    This is the pure-matplotlib path used when seaborn is unavailable.
    """
    vmax_abs = max(
        abs(float(np.nanmax(matrix)) - center),
        abs(float(np.nanmin(matrix)) - center),
    )
    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=center - vmax_abs,
        vmax=center + vmax_abs,
        aspect="auto",
    )
    # Tick labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Cell annotations
    if annotate:
        txt_color = _annotation_color(mode)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, f"{matrix[i, j]:{fmt}}",
                    ha="center", va="center",
                    fontsize=8, color=txt_color,
                )

    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def _sns_heatmap(
    ax: matplotlib.axes.Axes,
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    center: float,
    annotate: bool,
    fmt: str,
    cmap: Any,
    mode: str,
) -> matplotlib.axes.Axes:
    """Render a heatmap on *ax* using seaborn."""
    annot_kws = {"fontsize": 8, "color": _annotation_color(mode)} if annotate else {}
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        center=center,
        annot=annotate,
        fmt=fmt,
        annot_kws=annot_kws if annotate else {},
        xticklabels=list(col_labels),
        yticklabels=list(row_labels),
        linewidths=0.5,
        linecolor=COLORS.GRID if mode == "dark" else COLORS.GRID_LIGHT,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    return ax


def _draw_heatmap(
    ax: matplotlib.axes.Axes,
    matrix: np.ndarray,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """Dispatch to seaborn or imshow backend."""
    if _HAS_SEABORN:
        return _sns_heatmap(ax, matrix, **kwargs)
    return _imshow_heatmap(ax, matrix, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_diverging_heatmap(
    matrix: Any,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    center: Optional[float] = None,
    title: Optional[str] = None,
    annotate: bool = True,
    fmt: str = ".2f",
    cmap: Optional[Any] = None,
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Diverging heatmap centered on a value (grand mean by default).

    Cells below *center* are shaded blue; cells above are shaded red,
    using the SWARM diverging colormap.

    Args:
        matrix: 2-D array-like of shape ``(n_rows, n_cols)``.
        row_labels: Labels for each row.
        col_labels: Labels for each column.
        center: Center value for the diverging colormap.  When *None*
            the grand mean of *matrix* is used.
        title: Optional figure title.
        annotate: If *True*, print cell values inside each tile.
        fmt: Format string for annotations (default ``".2f"``).
        cmap: Matplotlib colormap override.  Defaults to the SWARM
            diverging colormaps (dark or light variant by *mode*).
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    mat = np.asarray(matrix, dtype=float)
    resolved_cmap = _resolve_cmap(cmap, mode)
    if center is None:
        center = float(np.nanmean(mat))

    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 1.0), max(4, len(row_labels) * 0.7)))
        _draw_heatmap(
            ax, mat,
            row_labels=row_labels,
            col_labels=col_labels,
            center=center,
            annotate=annotate,
            fmt=fmt,
            cmap=resolved_cmap,
            mode=mode,
        )
        if title:
            ax.set_title(title, fontsize=11, pad=12)
        fig.tight_layout()

    return fig, ax


def plot_difference_heatmap(
    matrix_a: Any,
    matrix_b: Any,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    label_a: str = "A",
    label_b: str = "B",
    title: Optional[str] = None,
    mode: str = "dark",
) -> Tuple[plt.Figure, np.ndarray]:
    """Side-by-side heatmaps with a difference panel (A - B).

    Creates a 1x3 figure: heatmap A | heatmap B | difference (A - B).

    Args:
        matrix_a: 2-D array-like for the first condition.
        matrix_b: 2-D array-like for the second condition.
        row_labels: Labels for each row.
        col_labels: Labels for each column.
        label_a: Display name for *matrix_a*.
        label_b: Display name for *matrix_b*.
        title: Optional super-title for the figure.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, axes)`` where *axes* is a length-3 ndarray.
    """
    mat_a = np.asarray(matrix_a, dtype=float)
    mat_b = np.asarray(matrix_b, dtype=float)
    diff = mat_a - mat_b
    resolved_cmap = _resolve_cmap(None, mode)

    with swarm_theme(mode):
        fig, axes = plt.subplots(
            1, 3,
            figsize=(max(16, len(col_labels) * 3.0), max(4, len(row_labels) * 0.7)),
        )

        common_kw: Dict[str, Any] = {
            "row_labels": row_labels,
            "col_labels": col_labels,
            "annotate": True,
            "fmt": ".2f",
            "cmap": resolved_cmap,
            "mode": mode,
        }

        # Panel A
        center_a = float(np.nanmean(mat_a))
        _draw_heatmap(axes[0], mat_a, center=center_a, **common_kw)
        axes[0].set_title(label_a, fontsize=10, pad=8)

        # Panel B
        center_b = float(np.nanmean(mat_b))
        _draw_heatmap(axes[1], mat_b, center=center_b, **common_kw)
        axes[1].set_title(label_b, fontsize=10, pad=8)

        # Difference panel (centered on 0)
        _draw_heatmap(axes[2], diff, center=0.0, **common_kw)
        axes[2].set_title(f"{label_a} \u2212 {label_b}", fontsize=10, pad=8)

        if title:
            fig.suptitle(title, fontsize=12, y=1.02)
        fig.tight_layout()

    return fig, axes


def plot_persona_mix_heatmap(
    data: Any,
    *,
    personas: Sequence[str],
    mixes: Sequence[str],
    metric_label: Optional[str] = None,
    title: Optional[str] = None,
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Convenience wrapper for persona x mix heatmaps.

    This is the most common SWARM heatmap layout: rows are agent
    personas and columns are population-mix conditions.

    Args:
        data: 2-D array of shape ``(n_personas, n_mixes)``.
        personas: List of persona labels (rows).
        mixes: List of mix-condition labels (columns).
        metric_label: Optional y-axis / colorbar label describing
            the metric shown (e.g. ``"Mean Toxicity"``).
        title: Optional figure title.  When *None* a default title
            is generated from *metric_label*.
        mode: ``"dark"`` (default) or ``"light"``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if title is None and metric_label is not None:
        title = f"{metric_label} by Persona \u00d7 Mix"

    fig, ax = plot_diverging_heatmap(
        data,
        row_labels=personas,
        col_labels=mixes,
        title=title,
        mode=mode,
    )

    ax.set_ylabel("Persona", fontsize=9)
    ax.set_xlabel("Mix", fontsize=9)

    return fig, ax
