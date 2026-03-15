"""Verification tests for swarm-end-to-end-study task."""

import json
import os

OUTPUT_DIR = "/root/output"


def test_sweep_dir_exists():
    assert os.path.isdir(os.path.join(OUTPUT_DIR, "sweep"))


def test_sweep_csv_has_9_rows():
    import csv
    path = os.path.join(OUTPUT_DIR, "sweep", "sweep_results.csv")
    assert os.path.isfile(path), "sweep_results.csv not found"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 9, f"Expected 9 rows (3 configs x 3 seeds), got {len(rows)}"


def test_analysis_dir_exists():
    assert os.path.isdir(os.path.join(OUTPUT_DIR, "analysis"))


def test_analysis_summary_has_results():
    path = os.path.join(OUTPUT_DIR, "analysis", "summary.json")
    assert os.path.isfile(path), "summary.json not found"
    with open(path) as f:
        s = json.load(f)
    assert "results" in s
    assert len(s["results"]) > 0, "No statistical results found"


def test_plots_dir_has_pngs():
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    assert os.path.isdir(plots_dir), "plots/ directory not found"
    pngs = [f for f in os.listdir(plots_dir) if f.endswith(".png")]
    assert len(pngs) >= 2, f"Expected at least 2 PNGs, found {len(pngs)}"


def test_paper_exists():
    path = os.path.join(OUTPUT_DIR, "paper", "paper.md")
    assert os.path.isfile(path), "paper.md not found"


def test_paper_has_results_with_data():
    import re
    path = os.path.join(OUTPUT_DIR, "paper", "paper.md")
    with open(path) as f:
        paper = f.read()
    assert "Results" in paper, "Paper missing Results section"
    # Check for numeric data in the results section
    results_section = paper.split("Results")[1] if "Results" in paper else ""
    numbers = re.findall(r"\d+\.\d+", results_section)
    assert len(numbers) >= 3, \
        f"Results section should have numeric values from sweep, found {len(numbers)}"
