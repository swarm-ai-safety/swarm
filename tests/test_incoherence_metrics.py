"""Tests for incoherence metric contract and semantics."""

import random
from typing import Any, Dict, Hashable, Mapping, Optional

import pytest

from swarm.metrics.incoherence import (
    BenchmarkPolicy,
    DecisionRecord,
    IncoherenceMetrics,
    classify_dual_failure_modes,
    disagreement_rate,
    distributed_coherence,
    error_rate,
    illusion_delta,
    incoherence_index,
    perceived_coherence,
    summarize_incoherence_by_agent_type,
)


class DictBenchmark(BenchmarkPolicy):
    """Simple benchmark policy keyed by (task_family, decision_id)."""

    def __init__(self, mapping: Dict[tuple[str, str], Hashable]):
        self.mapping = mapping

    def action_for(
        self,
        decision_id: str,
        task_family: str,
        metadata: Mapping[str, Any],
    ) -> Optional[Hashable]:
        return self.mapping.get((task_family, decision_id))


def test_disagreement_deterministic_actions_is_zero():
    actions = ["approve", "approve", "approve", "approve"]
    assert disagreement_rate(actions) == 0.0


def test_disagreement_random_actions_is_high():
    rng = random.Random(42)
    actions = [rng.choice(["a", "b", "c", "d"]) for _ in range(1000)]
    assert disagreement_rate(actions) > 0.70


def test_error_rate_with_missing_benchmark_is_zero():
    assert error_rate(["a", "b"], benchmark_action=None) == 0.0


def test_incoherence_index_caps_to_one():
    # High disagreement, very low error should still be clipped.
    assert incoherence_index(disagreement=0.9, error=0.01) == 1.0


def test_incoherence_index_zero_when_error_and_disagreement_zero():
    assert incoherence_index(disagreement=0.0, error=0.0) == 0.0


def test_compute_for_decision_excludes_abstentions():
    benchmark = DictBenchmark({("task", "d1"): "approve"})
    metrics = IncoherenceMetrics(benchmark)
    records = [
        DecisionRecord("d1", "task", 0, "approve"),
        DecisionRecord("d1", "task", 1, "reject"),
        DecisionRecord("d1", "task", 2, None, abstained=True),
    ]

    result = metrics.compute_for_decision(records)

    assert result.n_considered == 2
    assert result.error == pytest.approx(0.5)
    assert 0.0 <= result.incoherence <= 1.0


def test_compute_for_decision_requires_non_empty_records():
    benchmark = DictBenchmark({})
    metrics = IncoherenceMetrics(benchmark)
    with pytest.raises(ValueError, match="at least one"):
        metrics.compute_for_decision([])


def test_summarize_incoherence_by_agent_type():
    rows = [
        {
            "agent_type": "honest",
            "incoherence_index": 0.7,
            "error_rate": 0.4,
            "disagreement_rate": 0.3,
        },
        {
            "agent_type": "honest",
            "incoherence_index": 0.5,
            "error_rate": 0.6,
            "disagreement_rate": 0.4,
        },
        {
            "agent_type": "adversarial",
            "incoherence_index": 0.2,
            "error_rate": 0.3,
            "disagreement_rate": 0.1,
        },
    ]

    summary = summarize_incoherence_by_agent_type(rows)
    assert set(summary.keys()) == {"honest", "adversarial"}
    assert summary["honest"]["n"] == 2.0
    assert summary["honest"]["mean_incoherence_index"] == pytest.approx(0.6)
    assert summary["adversarial"]["mean_disagreement_rate"] == pytest.approx(0.1)


def test_classify_dual_failure_modes_counts_categories():
    incidents = [
        {"harmful": True, "agent_type": "adversarial", "incoherence_index": 0.2},
        {"harmful": True, "agent_type": "deceptive", "incoherence_index": 0.3},
        {"harmful": True, "agent_type": "honest", "incoherence_index": 0.8},
        {"harmful": True, "agent_type": "opportunistic", "incoherence_index": 0.7},
        {"harmful": False, "agent_type": "honest", "incoherence_index": 0.9},
    ]

    result = classify_dual_failure_modes(incidents, incoherence_threshold=0.5)
    assert result.total_harmful_incidents == 4
    assert result.coherent_adversarial_incidents == 2
    assert result.incoherent_benign_incidents == 2
    assert result.coherent_to_incoherent_ratio == pytest.approx(1.0)


def test_classify_dual_failure_modes_handles_zero_denominator():
    incidents = [
        {"harmful": True, "agent_type": "adversarial", "incoherence_index": 0.1},
    ]
    result = classify_dual_failure_modes(incidents, incoherence_threshold=0.5)
    assert result.incoherent_benign_incidents == 0
    assert result.coherent_adversarial_incidents == 1
    assert result.coherent_to_incoherent_ratio == float("inf")


# ---------------------------------------------------------------------------
# Illusion gap (Δ_illusion) tests
# ---------------------------------------------------------------------------


def test_perceived_coherence_high_p():
    """High-p accepted interactions -> high perceived coherence."""
    assert perceived_coherence([0.9, 0.95, 0.88, 0.92]) == pytest.approx(0.9125)


def test_perceived_coherence_empty():
    assert perceived_coherence([]) == 0.0


def test_distributed_coherence_no_disagreement():
    """Zero disagreement across replays -> perfect distributed coherence."""
    assert distributed_coherence([0.0, 0.0, 0.0]) == 1.0


def test_distributed_coherence_full_disagreement():
    """Complete disagreement -> zero distributed coherence."""
    assert distributed_coherence([1.0, 1.0, 1.0]) == 0.0


def test_distributed_coherence_empty():
    """No replay data defaults to perfect coherence (no evidence of instability)."""
    assert distributed_coherence([]) == 1.0


def test_illusion_delta_electric_mind_regime():
    """High p + high disagreement = large positive Δ (electric-mind regime)."""
    result = illusion_delta(
        p_values=[0.9, 0.95, 0.88],
        disagreement_rates=[0.7, 0.8, 0.6],
    )
    assert result.perceived_coherence == pytest.approx(0.91, abs=0.01)
    assert result.distributed_coherence == pytest.approx(0.3, abs=0.01)
    assert result.illusion_delta > 0.5


def test_illusion_delta_aligned_system():
    """High p + low disagreement = Δ near zero (genuinely stable)."""
    result = illusion_delta(
        p_values=[0.9, 0.85, 0.88],
        disagreement_rates=[0.1, 0.05, 0.12],
    )
    assert result.illusion_delta == pytest.approx(0.0, abs=0.15)


def test_illusion_delta_negative():
    """Low p + low disagreement = negative Δ (more consistent than it looks)."""
    result = illusion_delta(
        p_values=[0.3, 0.25, 0.35],
        disagreement_rates=[0.05, 0.1, 0.05],
    )
    assert result.illusion_delta < 0.0


def test_illusion_delta_empty_inputs():
    """Empty inputs: perceived=0.0, distributed=1.0, Δ=-1.0."""
    result = illusion_delta(p_values=[], disagreement_rates=[])
    assert result.perceived_coherence == 0.0
    assert result.distributed_coherence == 1.0
    assert result.illusion_delta == pytest.approx(-1.0)
