"""Unified figure export for PNG, SVG, PDF, and HTML (Plotly) output.

Companion to :mod:`swarm.analysis.export` (which handles CSV/JSON/Parquet
data export).  This module handles *figure* export -- saving matplotlib
figures and Plotly figures to various raster, vector, and interactive
formats.

Usage::

    import matplotlib.pyplot as plt
    from swarm.analysis.figure_export import save_figure, save_plotly

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [3, 1, 4])
    save_figure(fig, "output/my_chart.png", dpi=300)

    # Multiple formats at once
    save_figure(fig, "output/my_chart", formats=["png", "svg", "pdf"])

    # Plotly interactive HTML
    import plotly.graph_objects as go
    pfig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
    save_plotly(pfig, "output/interactive.html")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Formats that matplotlib can render natively
_MPL_FORMATS = {"png", "svg", "pdf", "eps", "ps", "pgf"}

# Default raster DPI used throughout the SWARM analysis suite
_DEFAULT_DPI = 300


def save_figure(
    fig: Any,
    path: Union[str, Path],
    formats: Optional[Sequence[str]] = None,
    dpi: int = _DEFAULT_DPI,
    *,
    tight: bool = True,
    transparent: bool = False,
) -> List[Path]:
    """Save a matplotlib figure to one or more formats.

    When *formats* is ``None`` the format is inferred from the file
    extension of *path*.  When *formats* is given explicitly, each
    format is saved as a sibling file with the appropriate extension.

    Args:
        fig: A :class:`matplotlib.figure.Figure` instance.
        path: Destination file path (or stem when *formats* is given).
        formats: Optional list of formats, e.g. ``["png", "svg", "pdf"]``.
        dpi: Resolution for raster formats (default 300).
        tight: Use ``bbox_inches="tight"`` to avoid clipping (default True).
        transparent: Transparent background (default False).

    Returns:
        List of :class:`~pathlib.Path` objects for every file written.

    Raises:
        ValueError: If the inferred or requested format is not supported.

    Example::

        from swarm.analysis.figure_export import save_figure
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        # Single file, format from extension
        save_figure(fig, "runs/plots/welfare.png")

        # Multiple formats from a single stem
        paths = save_figure(fig, "runs/plots/welfare", formats=["png", "svg"])
        # -> [Path('runs/plots/welfare.png'), Path('runs/plots/welfare.svg')]
    """
    path = Path(path)

    if formats is None:
        suffix = path.suffix.lstrip(".")
        if not suffix:
            raise ValueError(
                f"Cannot infer format from path without extension: {path}. "
                "Pass formats=['png'] explicitly or add a file extension."
            )
        formats = [suffix]

    unsupported = {f.lower() for f in formats} - _MPL_FORMATS
    if unsupported:
        raise ValueError(
            f"Unsupported matplotlib format(s): {unsupported}. "
            f"Supported: {sorted(_MPL_FORMATS)}"
        )

    stem = path.parent / path.stem
    stem.parent.mkdir(parents=True, exist_ok=True)

    savefig_kwargs: Dict[str, Any] = {"dpi": dpi, "transparent": transparent}
    if tight:
        savefig_kwargs["bbox_inches"] = "tight"

    written: List[Path] = []
    for fmt in formats:
        fmt = fmt.lower()
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(str(out), format=fmt, **savefig_kwargs)
        logger.info("Saved figure: %s", out)
        written.append(out)

    return written


def save_plotly(
    fig: Any,
    path: Union[str, Path],
    format: str = "html",
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 2.0,
) -> Path:
    """Save a Plotly figure to HTML or a static image format.

    For static image export (png, svg, pdf) the ``kaleido`` package must
    be installed.

    Args:
        fig: A :class:`plotly.graph_objects.Figure` instance.
        path: Destination file path.
        format: Output format -- ``"html"``, ``"png"``, ``"svg"``, or ``"pdf"``.
        width: Image width in pixels (static formats only).
        height: Image height in pixels (static formats only).
        scale: Resolution multiplier for static export (default 2.0).

    Returns:
        :class:`~pathlib.Path` of the written file.

    Example::

        import plotly.graph_objects as go
        from swarm.analysis.figure_export import save_plotly

        pfig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        save_plotly(pfig, "runs/plots/interactive.html")
        save_plotly(pfig, "runs/plots/static.png", format="png")
    """
    path = Path(path)
    format = format.lower()

    # Ensure the extension matches the requested format
    expected_suffix = f".{format}"
    if path.suffix.lower() != expected_suffix:
        path = path.with_suffix(expected_suffix)

    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "html":
        fig.write_html(str(path), include_plotlyjs="cdn")
        logger.info("Saved Plotly HTML: %s", path)
    else:
        # Static image export via kaleido
        write_kwargs: Dict[str, Any] = {"format": format, "scale": scale}
        if width is not None:
            write_kwargs["width"] = width
        if height is not None:
            write_kwargs["height"] = height
        fig.write_image(str(path), **write_kwargs)
        logger.info("Saved Plotly image: %s", path)

    return path


def export_figure_set(
    figures: Dict[str, Any],
    output_dir: Union[str, Path],
    prefix: str = "fig",
    formats: Optional[Sequence[str]] = None,
    dpi: int = _DEFAULT_DPI,
) -> Dict[str, List[Path]]:
    """Batch-export a dictionary of named figures.

    Each key in *figures* becomes part of the filename.  The value may be
    a matplotlib ``Figure`` or a Plotly ``Figure``; the function dispatches
    automatically.

    Args:
        figures: Mapping of ``name -> figure`` (matplotlib or Plotly).
        output_dir: Directory to write files into.
        prefix: Filename prefix prepended to each name (default ``"fig"``).
        formats: Formats for matplotlib figures (default ``["png"]``).
            Plotly figures always emit HTML; pass ``"png"`` in *formats*
            to additionally render Plotly as a static image.
        dpi: Resolution for raster formats.

    Returns:
        Mapping of ``name -> [Path, ...]`` for every file written.

    Example::

        import matplotlib.pyplot as plt
        from swarm.analysis.figure_export import export_figure_set

        figs = {}
        fig1, ax1 = plt.subplots()
        ax1.plot([1, 2], [3, 4])
        figs["welfare"] = fig1

        fig2, ax2 = plt.subplots()
        ax2.bar(["a", "b"], [5, 6])
        figs["agents"] = fig2

        paths = export_figure_set(figs, "runs/my_run/plots", prefix="sim")
        # paths["welfare"] -> [Path('.../sim_welfare.png')]
    """
    if formats is None:
        formats = ["png"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, List[Path]] = {}

    for name, fig in figures.items():
        safe_name = name.replace(" ", "_").replace("/", "_")
        stem = output_dir / f"{prefix}_{safe_name}"

        if _is_plotly_figure(fig):
            paths: List[Path] = []
            # Always emit HTML for Plotly figures
            html_path = save_plotly(fig, stem.with_suffix(".html"), format="html")
            paths.append(html_path)
            # Optionally render static formats (requires kaleido)
            for fmt in formats:
                fmt = fmt.lower()
                if fmt in ("png", "svg", "pdf"):
                    try:
                        static_path = save_plotly(fig, stem, format=fmt)
                        paths.append(static_path)
                    except (ValueError, ImportError) as exc:
                        logger.warning(
                            "Skipping Plotly static export (%s) for '%s': %s",
                            fmt,
                            name,
                            exc,
                        )
            result[name] = paths
        else:
            # Matplotlib figure
            result[name] = save_figure(fig, stem, formats=formats, dpi=dpi)

    logger.info(
        "Exported %d figures (%d files total) to %s",
        len(figures),
        sum(len(v) for v in result.values()),
        output_dir,
    )
    return result


def save_gif(
    frames: Sequence[Any],
    path: Union[str, Path],
    fps: int = 5,
    loop: int = 0,
) -> Path:
    """Combine a sequence of PIL images into an animated GIF.

    Args:
        frames: Sequence of :class:`PIL.Image.Image` objects (at least one).
        path: Output ``.gif`` file path.
        fps: Frames per second (default 5).
        loop: Number of loops; 0 means loop forever (default).

    Returns:
        :class:`~pathlib.Path` of the written GIF file.

    Raises:
        ValueError: If *frames* is empty.
        ImportError: If Pillow is not installed.

    Example::

        from PIL import Image
        from swarm.analysis.figure_export import save_gif

        frames = [Image.new("RGB", (100, 100), color) for color in
                  ["red", "green", "blue"]]
        save_gif(frames, "runs/plots/animation.gif", fps=2)
    """
    if not frames:
        raise ValueError("frames must contain at least one image")

    try:
        from PIL import Image as _PIL_Image  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "Pillow is required for GIF export. "
            "Install with: python -m pip install Pillow"
        ) from err

    path = Path(path)
    if path.suffix.lower() != ".gif":
        path = path.with_suffix(".gif")
    path.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = max(1, int(1000 / fps))

    first, *rest = frames
    first.save(
        str(path),
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=loop,
        optimize=True,
    )

    logger.info("Saved GIF (%d frames, %d fps): %s", len(frames), fps, path)
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_plotly_figure(obj: Any) -> bool:
    """Check whether *obj* is a Plotly Figure without importing plotly eagerly."""
    cls_name = type(obj).__name__
    mod = type(obj).__module__ or ""
    return cls_name == "Figure" and "plotly" in mod
