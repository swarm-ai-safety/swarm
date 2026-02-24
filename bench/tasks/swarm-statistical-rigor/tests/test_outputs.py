"""Verification tests for swarm-statistical-rigor task."""

import json
import os

OUTPUT_DIR = "/root/output"


def test_summary_json_exists():
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "summary.json"))


def test_results_txt_exists():
    path = os.path.join(OUTPUT_DIR, "results.txt")
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0, "results.txt is empty"


def test_total_hypotheses():
    with open(os.path.join(OUTPUT_DIR, "summary.json")) as f:
        s = json.load(f)
    # 5 groups -> C(5,2) = 10 pairwise comparisons
    assert s["total_hypotheses"] == 10, \
        f"Expected 10 hypotheses (C(5,2)), got {s['total_hypotheses']}"


def test_bonferroni_threshold():
    with open(os.path.join(OUTPUT_DIR, "summary.json")) as f:
        s = json.load(f)
    expected = 0.05 / s["total_hypotheses"]
    assert abs(s["bonferroni_threshold"] - expected) < 1e-6, \
        f"Bonferroni threshold should be 0.05/{s['total_hypotheses']} = {expected}"


def test_results_have_required_fields():
    with open(os.path.join(OUTPUT_DIR, "summary.json")) as f:
        s = json.load(f)
    assert "results" in s
    assert len(s["results"]) == s["total_hypotheses"]
    for r in s["results"]:
        for key in ["group_a", "group_b", "p_value", "cohens_d", "bonferroni_significant"]:
            assert key in r, f"Result missing key: {key}"
        assert 0 <= r["p_value"] <= 1, f"Invalid p-value: {r['p_value']}"


def test_normality_tests_present():
    with open(os.path.join(OUTPUT_DIR, "summary.json")) as f:
        s = json.load(f)
    assert "normality_tests" in s
    assert len(s["normality_tests"]) > 0, "No normality tests found"


def test_n_bonferroni_significant_consistent():
    with open(os.path.join(OUTPUT_DIR, "summary.json")) as f:
        s = json.load(f)
    counted = sum(1 for r in s["results"] if r["bonferroni_significant"])
    assert s["n_bonferroni_significant"] == counted
