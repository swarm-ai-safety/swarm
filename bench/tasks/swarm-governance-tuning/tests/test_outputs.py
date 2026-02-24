"""Verification tests for swarm-governance-tuning task."""

import json
import os

OUTPUT_DIR = "/root/output"
VALID_TAX_RATES = [0.0, 0.05, 0.10, 0.15, 0.20]


def test_optimal_json_exists():
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "optimal.json"))


def test_required_keys():
    with open(os.path.join(OUTPUT_DIR, "optimal.json")) as f:
        opt = json.load(f)
    for key in ["optimal_tax_rate", "mean_welfare", "mean_toxicity",
                 "n_configs_tested", "statistical_confidence"]:
        assert key in opt, f"Missing key: {key}"


def test_optimal_tax_rate_is_valid():
    with open(os.path.join(OUTPUT_DIR, "optimal.json")) as f:
        opt = json.load(f)
    assert opt["optimal_tax_rate"] in VALID_TAX_RATES, \
        f"optimal_tax_rate {opt['optimal_tax_rate']} not in swept values"


def test_toxicity_below_threshold():
    with open(os.path.join(OUTPUT_DIR, "optimal.json")) as f:
        opt = json.load(f)
    # Allow small tolerance for floating point
    assert opt["mean_toxicity"] < 0.16, \
        f"mean_toxicity {opt['mean_toxicity']} should be < 0.15 (with tolerance)"


def test_n_configs_tested():
    with open(os.path.join(OUTPUT_DIR, "optimal.json")) as f:
        opt = json.load(f)
    assert opt["n_configs_tested"] == 5


def test_statistical_confidence_is_valid_pvalue():
    with open(os.path.join(OUTPUT_DIR, "optimal.json")) as f:
        opt = json.load(f)
    p = opt["statistical_confidence"]
    assert isinstance(p, (int, float))
    assert 0.0 <= p <= 1.0, f"p-value {p} not in [0,1]"
