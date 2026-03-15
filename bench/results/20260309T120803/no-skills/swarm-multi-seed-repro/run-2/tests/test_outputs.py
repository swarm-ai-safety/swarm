"""Verification tests for swarm-multi-seed-repro task."""

import csv
import json
import os

OUTPUT_DIR = "/root/output"
SEEDS = [42, 7, 123]


def test_seed_directories_exist():
    for seed in SEEDS:
        d = os.path.join(OUTPUT_DIR, f"seed_{seed}")
        assert os.path.isdir(d), f"Directory seed_{seed}/ not found"
        assert os.path.isfile(os.path.join(d, "history.json")), \
            f"history.json not found in seed_{seed}/"


def test_summary_csv_exists():
    path = os.path.join(OUTPUT_DIR, "summary.csv")
    assert os.path.isfile(path), "summary.csv not found"


def test_summary_csv_has_3_rows():
    path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 3, f"Expected 3 data rows in summary.csv, got {len(rows)}"


def test_summary_csv_columns():
    path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
    assert "seed" in fieldnames, "summary.csv missing 'seed' column"
    assert "welfare" in fieldnames, "summary.csv missing 'welfare' column"
    assert "toxicity_rate" in fieldnames, "summary.csv missing 'toxicity_rate' column"


def test_reproducibility():
    """Same seed should produce identical welfare when run twice."""
    # We verify by checking the history.json values match summary.csv
    path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        csv_rows = {int(row["seed"]): float(row["welfare"]) for row in reader}

    for seed in SEEDS:
        with open(os.path.join(OUTPUT_DIR, f"seed_{seed}", "history.json")) as f:
            history = json.load(f)
        json_welfare = history["epoch_snapshots"][-1]["welfare"]
        assert abs(csv_rows[seed] - json_welfare) < 1e-6, \
            f"Welfare mismatch for seed {seed}: CSV={csv_rows[seed]}, JSON={json_welfare}"
