"""Tests for multi-agent horizon evaluation metrics.

Verifies that SystemHorizonEvaluator correctly detects:
    - Emergent long-horizon coherence from autocorrelation
    - Adverse selection drift in quality-gap time series
    - Variance-dominated ("hot mess") dynamics
    - Interaction chaining across agents
    - Temporal risk accumulation (harm acceleration)
"""

import numpy as np
import pytest

from swarm.metrics.horizon_eval import (
    HorizonEvalConfig,
    HorizonEvalResult,
    SystemHorizonEvaluator,
    group_by_epoch,
)
from swarm.models.interaction import SoftInteraction
from tests.fixtures.horizon_fixtures import (
    generate_accelerating_harm_epochs,
    generate_chained_handoff_epochs,
    generate_drifting_epochs,
    generate_stable_epochs,
    generate_variance_dominated_epochs,
)

# =====================================================================
# Helpers
# =====================================================================


def _default_evaluator(**overrides) -> SystemHorizonEvaluator:
    """Create evaluator with sensible test defaults."""
    defaults = {
        "agent_horizon_steps": 1,
        "discount_factor": 0.95,
        "coherence_lag_max": 10,
        "drift_window": 5,
        "variance_dominance_threshold": 1.0,
    }
    defaults.update(overrides)
    cfg = HorizonEvalConfig(**defaults)
    return SystemHorizonEvaluator(config=cfg)


# =====================================================================
# Tests: Stable baseline (null hypothesis)
# =====================================================================


class TestStableBaseline:
    """Under stable conditions, horizon metrics should be near-neutral."""

    def test_low_coherence_for_iid_quality(self):
        """Stable random quality should produce low emergent coherence."""
        epochs = generate_stable_epochs(n_epochs=20, seed=42)
        evaluator = _default_evaluator()
        result = evaluator.evaluate(epochs)

        # Coherence should be modest — no systematic trend
        assert result.emergent_coherence < 0.5, (
            f"Expected low coherence for stable data, got {result.emergent_coherence}"
        )

    def test_stable_drift_direction(self):
        """Quality gap should not systematically drift under stable conditions."""
        epochs = generate_stable_epochs(n_epochs=30, seed=123)
        evaluator = _default_evaluator()
        result = evaluator.evaluate(epochs)

        assert result.drift_direction in ("stable", "improving"), (
            f"Expected stable or improving drift, got {result.drift_direction}"
        )
        # Slope magnitude should be small
        assert abs(result.adverse_selection_drift) < 0.05

    def test_low_harm_acceleration(self):
        """Harm should not accelerate under stable conditions."""
        epochs = generate_stable_epochs(n_epochs=20, seed=99)
        evaluator = _default_evaluator()
        result = evaluator.evaluate(epochs)

        assert abs(result.harm_acceleration) < 0.5, (
            f"Expected near-zero harm acceleration, got {result.harm_acceleration}"
        )


# =====================================================================
# Tests: Drifting quality (adverse selection)
# =====================================================================


class TestDriftingQuality:
    """Systematically declining quality should trigger drift detection."""

    def test_positive_coherence_for_drift(self):
        """Drifting quality should show high autocorrelation / coherence."""
        epochs = generate_drifting_epochs(n_epochs=20, seed=42)
        evaluator = _default_evaluator()
        result = evaluator.evaluate(epochs)

        assert result.emergent_coherence > 0.3, (
            f"Expected positive coherence for drifting data, "
            f"got {result.emergent_coherence}"
        )

    def test_effective_horizon_exceeds_agent(self):
        """System effective horizon should exceed the agent horizon of 1."""
        epochs = generate_drifting_epochs(n_epochs=25, seed=42)
        evaluator = _default_evaluator(agent_horizon_steps=1)
        result = evaluator.evaluate(epochs)

        assert result.effective_system_horizon > 1.0, (
            f"Expected system horizon > 1.0, got {result.effective_system_horizon}"
        )

    def test_amplification_ratio_greater_than_one(self):
        """Horizon amplification should be > 1 for drifting system."""
        epochs = generate_drifting_epochs(n_epochs=25, seed=42)
        evaluator = _default_evaluator(agent_horizon_steps=1)
        result = evaluator.evaluate(epochs)

        assert result.horizon_amplification_ratio > 1.0, (
            f"Expected amplification > 1, got {result.horizon_amplification_ratio}"
        )

    def test_worsening_drift_detected(self):
        """Declining quality gap should be detected as worsening drift."""
        epochs = generate_drifting_epochs(
            n_epochs=40,
            interactions_per_epoch=50,
            start_quality=0.85,
            end_quality=0.25,
            seed=42,
        )
        evaluator = _default_evaluator()
        result = evaluator.evaluate(epochs)

        # Drift direction should be "worsening" (negative slope)
        # because adverse-selection dynamics make low-p interactions
        # increasingly accepted relative to high-p ones.
        assert result.adverse_selection_drift < 0, (
            f"Expected negative drift slope, got {result.adverse_selection_drift:.4f}"
        )

    def test_cumulative_harm_increases(self):
        """Cumulative harm should grow over time when quality declines."""
        epochs = generate_drifting_epochs(n_epochs=15, seed=42)
        evaluator = _default_evaluator()
        result = evaluator.evaluate(epochs)

        traj = result.cumulative_harm_trajectory
        assert len(traj) == 15
        # Cumulative harm should be monotonically increasing
        for i in range(1, len(traj)):
            assert traj[i] >= traj[i - 1], (
                f"Cumulative harm should be non-decreasing at epoch {i}"
            )
        # Final harm should exceed initial harm significantly
        assert traj[-1] > traj[0]


# =====================================================================
# Tests: Variance-dominated dynamics
# =====================================================================


class TestVarianceDominance:
    """High-variance scenarios should trigger hot-mess detection."""

    def test_high_variance_dominance_index(self):
        """Bimodal quality distribution should produce high VDI."""
        epochs = generate_variance_dominated_epochs(
            n_epochs=15,
            quality_spread=0.35,
            seed=42,
        )
        evaluator = _default_evaluator(variance_dominance_threshold=1.0)
        result = evaluator.evaluate(epochs)

        assert result.variance_dominance_index > 0.5, (
            f"Expected high VDI for bimodal data, got {result.variance_dominance_index}"
        )

    def test_hot_mess_epochs_detected(self):
        """Some epochs should be classified as hot-mess."""
        epochs = generate_variance_dominated_epochs(
            n_epochs=15,
            quality_spread=0.35,
            seed=42,
        )
        evaluator = _default_evaluator(variance_dominance_threshold=0.8)
        result = evaluator.evaluate(epochs)

        assert result.hot_mess_epochs > 0, "Expected at least some hot-mess epochs"

    def test_stable_has_few_hot_mess_epochs(self):
        """Stable scenario should have few/no hot-mess epochs."""
        epochs = generate_stable_epochs(
            n_epochs=15,
            noise=0.05,
            seed=42,
        )
        evaluator = _default_evaluator(variance_dominance_threshold=1.0)
        result = evaluator.evaluate(epochs)

        # At most a couple from random noise
        assert result.hot_mess_epochs <= 3, (
            f"Expected few hot-mess epochs for stable data, "
            f"got {result.hot_mess_epochs}"
        )


# =====================================================================
# Tests: Chain depth
# =====================================================================


class TestChainDepth:
    """Chained hand-offs should produce measurable chain depth."""

    def test_chained_scenario_has_depth(self):
        """Explicit chain fixtures should yield chain depth > 1."""
        epochs = generate_chained_handoff_epochs(
            n_epochs=10,
            chains_per_epoch=5,
            chain_length=4,
            seed=42,
        )
        evaluator = _default_evaluator()
        result = evaluator.evaluate(epochs)

        assert result.chain_depth_mean > 1.0, (
            f"Expected mean chain depth > 1 for chained data, "
            f"got {result.chain_depth_mean}"
        )
        assert result.chain_depth_max >= 2, (
            f"Expected max chain depth >= 2, got {result.chain_depth_max}"
        )

    def test_unchained_less_than_chained(self):
        """Stable scenario should have lower chain depth than chained scenario."""
        stable_epochs = generate_stable_epochs(n_epochs=10, seed=42)
        chained_epochs = generate_chained_handoff_epochs(
            n_epochs=10,
            chains_per_epoch=5,
            chain_length=4,
            seed=42,
        )
        evaluator = _default_evaluator()
        stable_result = evaluator.evaluate(stable_epochs)
        chained_result = evaluator.evaluate(chained_epochs)

        assert stable_result.chain_depth_mean < chained_result.chain_depth_mean, (
            f"Expected stable chain depth ({stable_result.chain_depth_mean:.2f}) "
            f"< chained ({chained_result.chain_depth_mean:.2f})"
        )


# =====================================================================
# Tests: Harm acceleration
# =====================================================================


class TestHarmAcceleration:
    """Accelerating harm scenarios should produce positive second-differences."""

    def test_quadratic_harm_acceleration(self):
        """Quadratic quality decay should produce positive harm acceleration."""
        epochs = generate_accelerating_harm_epochs(n_epochs=20, seed=42)
        evaluator = _default_evaluator()
        result = evaluator.evaluate(epochs)

        assert result.harm_acceleration > 0.0, (
            f"Expected positive harm acceleration for quadratic decay, "
            f"got {result.harm_acceleration}"
        )

    def test_stable_no_acceleration(self):
        """Stable scenario should have near-zero harm acceleration."""
        epochs = generate_stable_epochs(n_epochs=20, seed=42)
        evaluator = _default_evaluator()
        result = evaluator.evaluate(epochs)

        assert abs(result.harm_acceleration) < 0.3, (
            f"Expected near-zero acceleration for stable data, "
            f"got {result.harm_acceleration}"
        )


# =====================================================================
# Tests: Autocorrelation internals
# =====================================================================


class TestAutocorrelation:
    """Unit tests for the autocorrelation computation."""

    def test_constant_series(self):
        """Constant series should have perfect autocorrelation."""
        acf = SystemHorizonEvaluator._autocorrelation([5.0] * 20, max_lag=5)
        assert len(acf) == 5
        for r in acf:
            assert abs(r - 1.0) < 1e-6

    def test_random_series_low_acf(self):
        """Random series should have low autocorrelation."""
        rng = np.random.default_rng(42)
        series = list(rng.normal(0, 1, size=200))
        acf = SystemHorizonEvaluator._autocorrelation(series, max_lag=10)

        # Individual lags should be small (< 0.2 for iid normal)
        for r in acf:
            assert abs(r) < 0.25, f"ACF lag too high for random series: {r}"

    def test_trending_series_high_acf(self):
        """Linearly trending series should have high lag-1 autocorrelation."""
        series = list(np.linspace(0, 10, 50))
        acf = SystemHorizonEvaluator._autocorrelation(series, max_lag=5)

        assert len(acf) >= 1
        assert acf[0] > 0.9, f"Expected high lag-1 ACF for trend, got {acf[0]}"

    def test_short_series(self):
        """Series shorter than 3 should return empty ACF."""
        assert SystemHorizonEvaluator._autocorrelation([1.0, 2.0], max_lag=5) == []
        assert SystemHorizonEvaluator._autocorrelation([], max_lag=5) == []


# =====================================================================
# Tests: group_by_epoch helper
# =====================================================================


class TestGroupByEpoch:
    """Tests for the group_by_epoch utility."""

    def test_metadata_grouping(self):
        """Interactions with epoch metadata should be grouped correctly."""
        from tests.fixtures.horizon_fixtures import _make_interaction

        interactions = []
        for e in range(3):
            for _ in range(5):
                interactions.append(_make_interaction(0.7, True, "a", "b", epoch=e))

        grouped = group_by_epoch(interactions, n_epochs=3, steps_per_epoch=5)
        assert len(grouped) == 3
        for epoch_list in grouped:
            assert len(epoch_list) == 5

    def test_even_distribution_fallback(self):
        """Without epoch metadata, interactions are distributed evenly."""
        interactions = [SoftInteraction(p=0.5) for _ in range(30)]
        grouped = group_by_epoch(interactions, n_epochs=3, steps_per_epoch=10)
        assert len(grouped) == 3
        assert len(grouped[0]) == 10
        assert len(grouped[1]) == 10
        assert len(grouped[2]) == 10

    def test_empty_input(self):
        """Empty input should return empty epoch groups."""
        grouped = group_by_epoch([], n_epochs=5, steps_per_epoch=10)
        assert len(grouped) == 5
        for g in grouped:
            assert g == []


# =====================================================================
# Tests: Result serialisation
# =====================================================================


class TestResultSerialization:
    """HorizonEvalResult.to_dict() should be JSON-safe."""

    def test_to_dict_contains_all_fields(self):
        result = HorizonEvalResult(
            effective_system_horizon=3.5,
            horizon_amplification_ratio=3.5,
            emergent_coherence=0.6,
            total_epochs=20,
        )
        d = result.to_dict()
        assert d["effective_system_horizon"] == 3.5
        assert d["horizon_amplification_ratio"] == 3.5
        assert d["emergent_coherence"] == 0.6
        assert d["total_epochs"] == 20
        assert isinstance(d["quality_autocorrelation"], list)
        assert isinstance(d["cumulative_harm_trajectory"], list)

    def test_empty_result_serialises(self):
        result = HorizonEvalResult()
        d = result.to_dict()
        assert d["effective_system_horizon"] == 0.0
        assert d["total_epochs"] == 0


# =====================================================================
# Tests: Edge cases
# =====================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_epoch(self):
        """Evaluator should handle a single epoch without crashing."""
        epochs = generate_stable_epochs(n_epochs=1, seed=42)
        evaluator = _default_evaluator()
        result = evaluator.evaluate(epochs)

        assert result.total_epochs == 1
        assert result.effective_system_horizon == 0.0
        assert result.emergent_coherence == 0.0

    def test_empty_epochs(self):
        """Evaluator should handle empty input gracefully."""
        evaluator = _default_evaluator()
        result = evaluator.evaluate([])
        assert result.total_epochs == 0
        assert result.effective_system_horizon == 0.0

    def test_epochs_with_no_accepted(self):
        """Epochs with all-rejected interactions should not crash."""
        from tests.fixtures.horizon_fixtures import _make_interaction

        epochs = []
        for e in range(5):
            epoch_data = [
                _make_interaction(0.3, False, "a", "b", epoch=e) for _ in range(10)
            ]
            epochs.append(epoch_data)

        evaluator = _default_evaluator()
        result = evaluator.evaluate(epochs)
        assert result.total_epochs == 5

    def test_config_defaults(self):
        """Default config should have sensible values."""
        cfg = HorizonEvalConfig()
        assert cfg.agent_horizon_steps == 1
        assert cfg.discount_factor == 0.95
        assert cfg.coherence_lag_max == 10
        assert cfg.drift_window == 5
        assert cfg.variance_dominance_threshold == 1.0

    def test_rejects_oversized_epoch_count(self):
        """Evaluator should reject input exceeding max_epochs."""
        evaluator = _default_evaluator(max_epochs=5)
        epochs = generate_stable_epochs(n_epochs=10, seed=42)
        with pytest.raises(ValueError, match="exceeding max_epochs"):
            evaluator.evaluate(epochs)

    def test_rejects_oversized_epoch_interactions(self):
        """Evaluator should reject an epoch exceeding max_interactions_per_epoch."""
        evaluator = _default_evaluator(max_interactions_per_epoch=10)
        epochs = generate_stable_epochs(
            n_epochs=2,
            interactions_per_epoch=20,
            seed=42,
        )
        with pytest.raises(ValueError, match="exceeding max_interactions_per_epoch"):
            evaluator.evaluate(epochs)
