"""Verification tests for swarm-sweep-to-plots task."""

import os

PLOTS_DIR = "/root/output/plots"


def test_plots_dir_exists():
    assert os.path.isdir(PLOTS_DIR), "plots/ directory not found"


def test_at_least_3_pngs():
    pngs = [f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")]
    assert len(pngs) >= 3, f"Expected at least 3 PNG files, found {len(pngs)}"


def test_pngs_not_empty():
    """Each PNG must be >5KB to ensure it's not a placeholder."""
    pngs = [f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")]
    for png in pngs:
        path = os.path.join(PLOTS_DIR, png)
        size = os.path.getsize(path)
        assert size > 5000, f"{png} is too small ({size} bytes), likely empty"


def test_welfare_bar_chart_exists():
    pngs = [f.lower() for f in os.listdir(PLOTS_DIR) if f.endswith(".png")]
    bar_candidates = [f for f in pngs if "bar" in f or "welfare" in f or "config" in f]
    assert len(bar_candidates) > 0, "No welfare bar chart found (expected filename with 'bar', 'welfare', or 'config')"


def test_boxplot_exists():
    pngs = [f.lower() for f in os.listdir(PLOTS_DIR) if f.endswith(".png")]
    box_candidates = [f for f in pngs if "box" in f or "distribution" in f]
    assert len(box_candidates) > 0, "No box plot found (expected filename with 'box' or 'distribution')"
