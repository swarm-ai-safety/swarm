"""Verification tests for swarm-sweep-analysis task."""

import csv
import json
import os

OUTPUT_DIR = "/root/output"


def test_sweep_csv_exists():
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "sweep_results.csv"))


def test_sweep_csv_has_12_rows():
    with open(os.path.join(OUTPUT_DIR, "sweep_results.csv")) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 12, f"Expected 12 rows (4 configs x 3 seeds), got {len(rows)}"


def test_sweep_csv_has_param_column():
    with open(os.path.join(OUTPUT_DIR, "sweep_results.csv")) as f:
        reader = csv.DictReader(f)
        assert "transaction_tax_rate" in reader.fieldnames


def test_summary_json_exists():
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "summary.json"))


def test_summary_json_has_4_configs():
    with open(os.path.join(OUTPUT_DIR, "summary.json")) as f:
        summary = json.load(f)
    assert "configs" in summary
    assert len(summary["configs"]) == 4, f"Expected 4 configs, got {len(summary['configs'])}"


def test_summary_json_has_mean_welfare():
    with open(os.path.join(OUTPUT_DIR, "summary.json")) as f:
        summary = json.load(f)
    for config in summary["configs"]:
        assert "mean_welfare" in config, f"Config missing mean_welfare: {config}"
        assert isinstance(config["mean_welfare"], (int, float))
