from pathlib import Path

import pytest

from swarm.analysis.autoresearch import (
    EvalSummary,
    Guardrail,
    ObjectiveParseError,
    _guardrails_ok,
    _is_better,
    _validate_scenario_name,
    parse_objective,
)


def test_parse_objective_markdown_yaml_block(tmp_path: Path) -> None:
    f = tmp_path / "program.md"
    f.write_text(
        """
# Objective

```yaml
primary_metric: quality_gap
primary_direction: minimize
min_improvement: 0.02
guardrails:
  - metric: toxicity_rate
    max_regression: 0.01
```
""",
        encoding="utf-8",
    )

    spec = parse_objective(f)
    assert spec.primary_metric == "quality_gap"
    assert spec.primary_direction == "minimize"
    assert spec.min_improvement == 0.02
    assert len(spec.guardrails) == 1
    assert spec.guardrails[0].metric == "toxicity_rate"


def test_is_better_for_minimize_and_maximize() -> None:
    assert _is_better(candidate=0.2, baseline=0.3, direction="minimize", min_improvement=0.05)
    assert not _is_better(candidate=0.27, baseline=0.3, direction="minimize", min_improvement=0.05)
    assert _is_better(candidate=0.8, baseline=0.7, direction="maximize", min_improvement=0.05)


def test_guardrail_regression_detection() -> None:
    baseline = EvalSummary(primary_metric="quality_gap", metrics={"toxicity_rate": 0.10})
    candidate = EvalSummary(primary_metric="quality_gap", metrics={"toxicity_rate": 0.13})
    ok, errors = _guardrails_ok(
        baseline,
        candidate,
        [Guardrail(metric="toxicity_rate", max_regression=0.01)],
    )
    assert not ok
    assert errors


def test_parse_objective_rejects_unsafe_metric_name(tmp_path: Path) -> None:
    """primary_metric must be alphanumeric+underscore to prevent injection."""
    f = tmp_path / "bad.yaml"
    f.write_text("primary_metric: 'metric\\n--allow-empty'\n", encoding="utf-8")
    with pytest.raises(ObjectiveParseError, match="must match"):
        parse_objective(f)


def test_parse_objective_rejects_path_traversal_metric(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("primary_metric: '../../../etc/passwd'\n", encoding="utf-8")
    with pytest.raises(ObjectiveParseError, match="must match"):
        parse_objective(f)


def test_validate_scenario_name_accepts_valid() -> None:
    assert _validate_scenario_name("baseline") == "baseline"
    assert _validate_scenario_name("my-scenario_v2.1") == "my-scenario_v2.1"


def test_validate_scenario_name_rejects_traversal() -> None:
    with pytest.raises(ValueError, match="must match"):
        _validate_scenario_name("../../etc/passwd")


def test_validate_scenario_name_rejects_slashes() -> None:
    with pytest.raises(ValueError, match="must match"):
        _validate_scenario_name("foo/bar")
