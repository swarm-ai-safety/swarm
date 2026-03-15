"""Tests for cross-scenario family optimization."""

from swarm.analysis.autoresearch import EvalSummary
from swarm.analysis.family_optimization import (
    FamilyEvalSummary,
    _aggregate_family_eval,
)


def test_aggregate_family_eval_minimize() -> None:
    per_scenario = {
        "baseline": EvalSummary(
            primary_metric="toxicity_rate",
            metrics={"toxicity_rate": 0.10, "total_welfare": 5.0},
        ),
        "collusion": EvalSummary(
            primary_metric="toxicity_rate",
            metrics={"toxicity_rate": 0.20, "total_welfare": 3.0},
        ),
    }

    result = _aggregate_family_eval(per_scenario, "toxicity_rate", "minimize")

    assert abs(result.mean_metrics["toxicity_rate"] - 0.15) < 1e-9
    assert abs(result.mean_metrics["total_welfare"] - 4.0) < 1e-9
    assert result.worst_scenario == "collusion"
    assert result.worst_value == 0.20


def test_aggregate_family_eval_maximize() -> None:
    per_scenario = {
        "baseline": EvalSummary(
            primary_metric="total_welfare",
            metrics={"total_welfare": 5.0, "toxicity_rate": 0.10},
        ),
        "governance": EvalSummary(
            primary_metric="total_welfare",
            metrics={"total_welfare": 3.0, "toxicity_rate": 0.05},
        ),
    }

    result = _aggregate_family_eval(per_scenario, "total_welfare", "maximize")

    assert result.mean_metrics["total_welfare"] == 4.0
    assert result.worst_scenario == "governance"
    assert result.worst_value == 3.0


def test_aggregate_family_eval_single_member() -> None:
    per_scenario = {
        "only": EvalSummary(
            primary_metric="toxicity_rate",
            metrics={"toxicity_rate": 0.08},
        ),
    }

    result = _aggregate_family_eval(per_scenario, "toxicity_rate", "minimize")
    assert result.mean_metrics["toxicity_rate"] == 0.08
    assert result.worst_scenario == "only"


def test_family_eval_summary_fields() -> None:
    per = {
        "a": EvalSummary(primary_metric="m", metrics={"m": 1.0}),
        "b": EvalSummary(primary_metric="m", metrics={"m": 2.0}),
    }
    result = _aggregate_family_eval(per, "m", "maximize")

    assert isinstance(result, FamilyEvalSummary)
    assert "a" in result.per_scenario
    assert "b" in result.per_scenario
    assert result.worst_scenario == "a"  # worst for maximize = lowest
    assert result.worst_value == 1.0
