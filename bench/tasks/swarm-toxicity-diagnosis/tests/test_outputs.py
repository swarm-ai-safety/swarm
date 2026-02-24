"""Verification tests for swarm-toxicity-diagnosis task."""

import json
import os

import pandas as pd

OUTPUT_DIR = "/root/output"
DATA_PATH = "/root/data/sweep_results.csv"


def _load_ground_truth():
    """Compute expected answers from the fixture data."""
    df = pd.read_csv(DATA_PATH)
    tox = df.groupby("transaction_tax_rate")["toxicity_rate"].mean()
    return {
        "worst_config": float(tox.idxmax()),
        "best_config": float(tox.idxmin()),
        "worst_toxicity": float(tox.max()),
        "best_toxicity": float(tox.min()),
    }


def test_diagnosis_json_exists():
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "diagnosis.json"))


def test_required_keys():
    with open(os.path.join(OUTPUT_DIR, "diagnosis.json")) as f:
        diag = json.load(f)
    for key in ["worst_config", "best_config", "worst_toxicity", "best_toxicity", "recommendation"]:
        assert key in diag, f"Missing key: {key}"


def test_worst_config_matches():
    with open(os.path.join(OUTPUT_DIR, "diagnosis.json")) as f:
        diag = json.load(f)
    truth = _load_ground_truth()
    assert abs(diag["worst_config"] - truth["worst_config"]) < 0.01, \
        f"worst_config mismatch: got {diag['worst_config']}, expected {truth['worst_config']}"


def test_best_config_matches():
    with open(os.path.join(OUTPUT_DIR, "diagnosis.json")) as f:
        diag = json.load(f)
    truth = _load_ground_truth()
    assert abs(diag["best_config"] - truth["best_config"]) < 0.01, \
        f"best_config mismatch: got {diag['best_config']}, expected {truth['best_config']}"


def test_recommendation_non_empty():
    with open(os.path.join(OUTPUT_DIR, "diagnosis.json")) as f:
        diag = json.load(f)
    assert isinstance(diag["recommendation"], str)
    assert len(diag["recommendation"]) > 10, "Recommendation too short"
