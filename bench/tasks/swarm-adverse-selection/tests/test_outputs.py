"""Verification tests for swarm-adverse-selection task."""

import json
import os

OUTPUT_DIR = "/root/output"


def test_json_exists():
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "adverse_selection.json"))


def test_required_keys():
    with open(os.path.join(OUTPUT_DIR, "adverse_selection.json")) as f:
        report = json.load(f)
    for key in ["quality_gap_value", "adverse_selection_detected",
                 "accepted_mean_p", "rejected_mean_p", "recommendation"]:
        assert key in report, f"Missing key: {key}"


def test_quality_gap_is_float():
    with open(os.path.join(OUTPUT_DIR, "adverse_selection.json")) as f:
        report = json.load(f)
    assert isinstance(report["quality_gap_value"], (int, float))


def test_adverse_selection_matches_gap_sign():
    with open(os.path.join(OUTPUT_DIR, "adverse_selection.json")) as f:
        report = json.load(f)
    expected = report["quality_gap_value"] < 0
    assert report["adverse_selection_detected"] == expected, \
        f"adverse_selection_detected should be {expected} for quality_gap={report['quality_gap_value']}"


def test_mean_p_in_range():
    with open(os.path.join(OUTPUT_DIR, "adverse_selection.json")) as f:
        report = json.load(f)
    for key in ["accepted_mean_p", "rejected_mean_p"]:
        val = report[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} not in [0,1]"


def test_recommendation_non_empty():
    with open(os.path.join(OUTPUT_DIR, "adverse_selection.json")) as f:
        report = json.load(f)
    assert isinstance(report["recommendation"], str)
    assert len(report["recommendation"]) > 20
