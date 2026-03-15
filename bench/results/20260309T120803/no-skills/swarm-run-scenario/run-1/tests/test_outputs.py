"""Verification tests for swarm-run-scenario task."""

import json
import os

OUTPUT_DIR = "/root/output"


def test_history_json_exists():
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "history.json")), \
        "history.json not found in output directory"


def test_csv_dir_exists():
    csv_dir = os.path.join(OUTPUT_DIR, "csv")
    assert os.path.isdir(csv_dir), "csv/ directory not found"
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    assert len(csv_files) > 0, "No CSV files found in csv/ directory"


def test_history_json_schema():
    with open(os.path.join(OUTPUT_DIR, "history.json")) as f:
        history = json.load(f)

    assert "epoch_snapshots" in history, "history.json missing 'epoch_snapshots' key"
    snapshots = history["epoch_snapshots"]
    assert len(snapshots) == 10, f"Expected 10 epoch snapshots, got {len(snapshots)}"


def test_welfare_in_range():
    with open(os.path.join(OUTPUT_DIR, "history.json")) as f:
        history = json.load(f)

    final = history["epoch_snapshots"][-1]
    assert "welfare" in final, "Final snapshot missing 'welfare'"
    # Welfare should be a reasonable number (not NaN, not extreme)
    welfare = final["welfare"]
    assert isinstance(welfare, (int, float)), f"Welfare is not numeric: {welfare}"
    assert -1000 < welfare < 1000, f"Welfare out of reasonable range: {welfare}"


def test_toxicity_in_range():
    with open(os.path.join(OUTPUT_DIR, "history.json")) as f:
        history = json.load(f)

    final = history["epoch_snapshots"][-1]
    assert "toxicity_rate" in final, "Final snapshot missing 'toxicity_rate'"
    toxicity = final["toxicity_rate"]
    assert isinstance(toxicity, (int, float)), f"Toxicity is not numeric: {toxicity}"
    assert 0.0 <= toxicity <= 1.0, f"Toxicity rate out of [0,1]: {toxicity}"
