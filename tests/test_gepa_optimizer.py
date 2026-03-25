"""Tests for GEPA optimize_anything integration."""

from pathlib import Path

import pytest
import yaml

from swarm.analysis.evolver import INT_PARAMS, PARAM_RANGES
from swarm.analysis.gepa_optimizer import (
    _params_to_yaml,
    _yaml_to_params,
    make_evaluator,
)
from swarm.scenarios import load_scenario


class TestYamlRoundTrip:
    """Test YAML serialization/deserialization for GEPA candidates."""

    def test_roundtrip_basic(self):
        params = {
            "governance.transaction_tax_rate": 0.05,
            "payoff.s_plus": 3.0,
            "governance.bandwidth_cap": 15,
        }
        yaml_str = _params_to_yaml(params)
        recovered = _yaml_to_params(yaml_str)
        assert recovered == params

    def test_roundtrip_all_params(self):
        """All PARAM_RANGES keys survive a round-trip."""
        params = {}
        for key, (lo, hi) in PARAM_RANGES.items():
            mid = (lo + hi) / 2
            if key in INT_PARAMS:
                mid = int(round(mid))
            params[key] = mid

        yaml_str = _params_to_yaml(params)
        recovered = _yaml_to_params(yaml_str)

        for key in PARAM_RANGES:
            assert key in recovered, f"Missing key: {key}"
            assert recovered[key] == pytest.approx(params[key], abs=0.01)

    def test_clamping(self):
        """Out-of-range values are clamped."""
        yaml_str = yaml.dump({
            "governance": {"transaction_tax_rate": 5.0},  # max is 1.0
            "payoff": {"s_plus": -3.0},  # min is 0.0
        })
        params = _yaml_to_params(yaml_str)
        assert params["governance.transaction_tax_rate"] == 1.0
        assert params["payoff.s_plus"] == 0.0

    def test_int_params_rounded(self):
        """Integer params are rounded."""
        yaml_str = yaml.dump({
            "governance": {"bandwidth_cap": 7.6},
        })
        params = _yaml_to_params(yaml_str)
        assert params["governance.bandwidth_cap"] == 8
        assert isinstance(params["governance.bandwidth_cap"], int)

    def test_unknown_params_ignored(self):
        """Unknown params are silently dropped."""
        yaml_str = yaml.dump({
            "governance": {"nonexistent_param": 42},
            "payoff": {"s_plus": 2.0},
        })
        params = _yaml_to_params(yaml_str)
        assert "governance.nonexistent_param" not in params
        assert params["payoff.s_plus"] == 2.0

    def test_invalid_yaml_raises(self):
        """Non-dict YAML raises ValueError."""
        with pytest.raises(ValueError, match="Expected YAML dict"):
            _yaml_to_params("just a string")

    def test_empty_sections_ok(self):
        """Sections with no valid params produce empty dict."""
        yaml_str = yaml.dump({"governance": "not_a_dict"})
        params = _yaml_to_params(yaml_str)
        assert params == {}


class TestMakeEvaluator:
    """Test evaluator construction (without running full simulations)."""

    def test_evaluator_returns_callable(self):
        scenario = load_scenario(Path("scenarios/baseline.yaml"))
        evaluator = make_evaluator(scenario, eval_epochs=1, eval_steps=1)
        assert callable(evaluator)

    def test_evaluator_handles_bad_yaml(self):
        scenario = load_scenario(Path("scenarios/baseline.yaml"))
        evaluator = make_evaluator(scenario, eval_epochs=1, eval_steps=1)

        score, info = evaluator("not: valid: yaml: [[[")
        assert score == 0.0
        assert "error" in info

    def test_evaluator_handles_empty_params(self):
        scenario = load_scenario(Path("scenarios/baseline.yaml"))
        evaluator = make_evaluator(scenario, eval_epochs=1, eval_steps=1)

        score, info = evaluator("unknown_section:\n  unknown_key: 42\n")
        assert score == 0.0
        assert "error" in info

    def test_evaluator_runs_simulation(self):
        """Integration test: evaluator runs a real simulation."""
        scenario = load_scenario(Path("scenarios/baseline.yaml"))
        evaluator = make_evaluator(scenario, eval_epochs=1, eval_steps=2, seed=42)

        candidate = _params_to_yaml({
            "governance.transaction_tax_rate": 0.0,
            "payoff.s_plus": 2.0,
            "payoff.s_minus": 1.0,
            "payoff.theta": 0.5,
        })
        score, info = evaluator(candidate)

        assert 0.0 <= score <= 1.0
        assert "toxicity" in info
        assert "welfare" in info
        assert "quality_gap" in info
        assert "payoff_gap" in info
