"""Tests for the knowledge persistence module."""

from pathlib import Path

import yaml

from swarm.knowledge.lesson_store import Lesson, LessonStore
from swarm.knowledge.run_envelope import RunEnvelope, write_run_yaml

# ---------------------------------------------------------------------------
# LessonStore
# ---------------------------------------------------------------------------


def test_lesson_store_roundtrip(tmp_path: Path) -> None:
    store = LessonStore(root=tmp_path, scenario="baseline")
    assert len(store) == 0

    lesson = Lesson(
        param="transaction_tax_rate",
        old_value=0.0,
        new_value=0.02,
        accepted=True,
        primary_metric="toxicity_rate",
        primary_value=0.08,
        baseline_value=0.10,
        iteration=1,
        scenario="baseline",
    )
    store.add(lesson)
    assert len(store) == 1

    # Reload from disk
    store2 = LessonStore(root=tmp_path, scenario="baseline")
    assert len(store2) == 1
    assert store2.all()[0].param == "transaction_tax_rate"
    assert store2.all()[0].accepted is True


def test_lesson_store_was_tried(tmp_path: Path) -> None:
    store = LessonStore(root=tmp_path, scenario="test")
    store.add(Lesson(
        param="audit_probability",
        old_value=0.1,
        new_value=0.15,
        accepted=False,
        primary_metric="toxicity_rate",
        primary_value=0.12,
        baseline_value=0.10,
        iteration=1,
    ))

    assert store.was_tried("audit_probability", 0.15)
    assert not store.was_tried("audit_probability", 0.20)
    assert not store.was_tried("transaction_tax_rate", 0.15)


def test_lesson_store_was_tried_boolean(tmp_path: Path) -> None:
    store = LessonStore(root=tmp_path, scenario="test")
    store.add(Lesson(
        param="audit_enabled",
        old_value=False,
        new_value=True,
        accepted=True,
        primary_metric="toxicity_rate",
        primary_value=0.05,
        baseline_value=0.10,
        iteration=1,
    ))

    assert store.was_tried("audit_enabled", True)
    assert not store.was_tried("audit_enabled", False)


def test_lesson_store_best_known_value(tmp_path: Path) -> None:
    store = LessonStore(root=tmp_path, scenario="test")
    store.add(Lesson(
        param="transaction_tax_rate",
        old_value=0.0, new_value=0.02,
        accepted=True,
        primary_metric="toxicity_rate",
        primary_value=0.08, baseline_value=0.10,
        iteration=1,
    ))
    store.add(Lesson(
        param="transaction_tax_rate",
        old_value=0.02, new_value=0.04,
        accepted=True,
        primary_metric="toxicity_rate",
        primary_value=0.05, baseline_value=0.08,
        iteration=2,
    ))

    # improvement = primary_value - baseline_value (direction-agnostic)
    # lesson 1: 0.08 - 0.10 = -0.02, lesson 2: 0.05 - 0.08 = -0.03
    # max improvement is -0.02, so best_known is 0.02
    assert store.best_known_value("transaction_tax_rate") == 0.02
    assert store.best_known_value("nonexistent") is None


def test_lesson_store_filtering(tmp_path: Path) -> None:
    store = LessonStore(root=tmp_path, scenario="test")
    store.add(Lesson(
        param="audit_probability", old_value=0.1, new_value=0.15,
        accepted=True, primary_metric="m", primary_value=1, baseline_value=0, iteration=1,
    ))
    store.add(Lesson(
        param="audit_probability", old_value=0.15, new_value=0.20,
        accepted=False, primary_metric="m", primary_value=0, baseline_value=1, iteration=2,
    ))
    store.add(Lesson(
        param="transaction_tax_rate", old_value=0.0, new_value=0.02,
        accepted=True, primary_metric="m", primary_value=1, baseline_value=0, iteration=3,
    ))

    assert len(store.for_param("audit_probability")) == 2
    assert len(store.accepted()) == 2
    assert len(store.rejected()) == 1


def test_lesson_store_summary(tmp_path: Path) -> None:
    store = LessonStore(root=tmp_path, scenario="baseline")
    store.add(Lesson(
        param="audit_probability", old_value=0.1, new_value=0.15,
        accepted=True, primary_metric="m", primary_value=1, baseline_value=0, iteration=1,
    ))
    s = store.summary()
    assert s["scenario"] == "baseline"
    assert s["total_lessons"] == 1
    assert s["accepted"] == 1
    assert "audit_probability" in s["params_tried"]


def test_lesson_store_empty_load(tmp_path: Path) -> None:
    store = LessonStore(root=tmp_path, scenario="never_created")
    assert len(store) == 0
    assert store.all() == []


# ---------------------------------------------------------------------------
# RunEnvelope
# ---------------------------------------------------------------------------


def test_run_envelope_write(tmp_path: Path) -> None:
    envelope = RunEnvelope(
        run_id="20260314-120000_autoresearch_baseline",
        scenario_ref="scenarios/baseline.yaml",
        hypothesis="Optimize toxicity via governance tuning",
        seeds=[7, 11, 19],
        total_iterations=10,
        accepted_iterations=3,
        primary_metric="toxicity_rate",
        primary_result="0.042000",
        baseline_value=0.10,
        best_value=0.042,
        tags=["autoresearch", "governance-tuning", "baseline"],
        significant_findings=["toxicity improved by 0.058"],
    )

    path = write_run_yaml(envelope, tmp_path)
    assert path.exists()
    assert path.name == "run.yaml"

    data = yaml.safe_load(path.read_text())
    assert data["run_id"] == "20260314-120000_autoresearch_baseline"
    assert data["experiment"]["type"] == "autoresearch"
    assert data["experiment"]["seeds"] == [7, 11, 19]
    assert data["results"]["primary_metric"] == "toxicity_rate"
    assert data["results"]["baseline_value"] == 0.10
    assert data["tags"] == ["autoresearch", "governance-tuning", "baseline"]
    assert "provenance" in data
    assert "python_version" in data["provenance"]


def test_run_envelope_artifacts(tmp_path: Path) -> None:
    envelope = RunEnvelope(
        run_id="test",
        scenario_ref="test.yaml",
        hypothesis="test",
        artifacts={"summary": "runs/summary.json"},
    )
    path = write_run_yaml(envelope, tmp_path)
    data = yaml.safe_load(path.read_text())
    assert data["artifacts"]["summary"] == "runs/summary.json"


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------


def test_backpressure_no_halt_before_window() -> None:
    from swarm.analysis.autoresearch import Backpressure

    bp = Backpressure(window=5, min_accepts=1)
    for _ in range(4):
        bp.record(False)
    assert not bp.should_halt()  # window not full yet


def test_backpressure_halt_on_plateau() -> None:
    from swarm.analysis.autoresearch import Backpressure

    bp = Backpressure(window=3, min_accepts=1)
    bp.record(False)
    bp.record(False)
    bp.record(False)
    assert bp.should_halt()


def test_backpressure_no_halt_with_accepts() -> None:
    from swarm.analysis.autoresearch import Backpressure

    bp = Backpressure(window=3, min_accepts=1)
    bp.record(False)
    bp.record(True)
    bp.record(False)
    assert not bp.should_halt()


def test_backpressure_status() -> None:
    from swarm.analysis.autoresearch import Backpressure

    bp = Backpressure(window=3, min_accepts=1)
    bp.record(True)
    bp.record(False)
    assert "1/2" in bp.status()


# ---------------------------------------------------------------------------
# Staged validation helpers
# ---------------------------------------------------------------------------


def test_preflight_rejects_out_of_bounds() -> None:
    from unittest.mock import MagicMock

    from swarm.analysis.autoresearch import _preflight_check

    scenario = MagicMock()
    gov = MagicMock()
    gov.transaction_tax_rate = 999.0  # way out of bounds
    gov.audit_probability = 0.5
    gov.audit_threshold_p = 0.5
    gov.freeze_threshold_toxicity = 0.5
    gov.incoherence_friction_rate = 0.1
    scenario.orchestrator_config.governance_config = gov

    ok, reason = _preflight_check(scenario)
    assert not ok
    assert "transaction_tax_rate" in reason


def test_preflight_accepts_valid() -> None:
    from unittest.mock import MagicMock

    from swarm.analysis.autoresearch import _preflight_check

    scenario = MagicMock()
    gov = MagicMock()
    gov.transaction_tax_rate = 0.05
    gov.audit_probability = 0.5
    gov.audit_threshold_p = 0.5
    gov.freeze_threshold_toxicity = 0.5
    gov.incoherence_friction_rate = 0.1
    scenario.orchestrator_config.governance_config = gov

    ok, reason = _preflight_check(scenario)
    assert ok
    assert reason == ""
