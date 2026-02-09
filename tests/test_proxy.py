"""Tests for proxy computation."""

import pytest

from swarm.core.proxy import ProxyComputer, ProxyObservables, ProxyWeights
from swarm.core.sigmoid import (
    calibrated_sigmoid,
    effective_uncertainty_band,
    inverse_sigmoid,
    sigmoid_bounds,
    sigmoid_derivative,
)


class TestCalibratedSigmoid:
    """Tests for sigmoid utilities."""

    def test_sigmoid_at_zero(self):
        """sigmoid(0) should be 0.5."""
        assert calibrated_sigmoid(0.0, k=2.0) == pytest.approx(0.5)

    def test_sigmoid_bounds(self):
        """Sigmoid should be in [0, 1]."""
        for v_hat in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            p = calibrated_sigmoid(v_hat, k=2.0)
            assert 0.0 <= p <= 1.0

    def test_sigmoid_monotonic(self):
        """Sigmoid should be monotonically increasing."""
        v_hats = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ps = [calibrated_sigmoid(v, k=2.0) for v in v_hats]

        for i in range(len(ps) - 1):
            assert ps[i] < ps[i + 1]

    def test_sigmoid_symmetry(self):
        """sigmoid(-x) = 1 - sigmoid(x)."""
        for v_hat in [0.1, 0.5, 1.0]:
            p_pos = calibrated_sigmoid(v_hat, k=2.0)
            p_neg = calibrated_sigmoid(-v_hat, k=2.0)
            assert p_pos + p_neg == pytest.approx(1.0)

    def test_k_effect(self):
        """Higher k should make sigmoid sharper."""
        v_hat = 0.5

        p_low_k = calibrated_sigmoid(v_hat, k=1.0)
        p_high_k = calibrated_sigmoid(v_hat, k=4.0)

        # Higher k pushes p further from 0.5
        assert abs(p_high_k - 0.5) > abs(p_low_k - 0.5)

    def test_inverse_sigmoid_roundtrip(self):
        """inverse_sigmoid(sigmoid(v)) should equal v."""
        for v_hat in [-0.8, -0.3, 0.0, 0.4, 0.9]:
            k = 2.0
            p = calibrated_sigmoid(v_hat, k)
            v_back = inverse_sigmoid(p, k)
            assert v_back == pytest.approx(v_hat, rel=1e-6)

    def test_inverse_sigmoid_bounds(self):
        """inverse_sigmoid should raise for p outside (0, 1)."""
        with pytest.raises(ValueError):
            inverse_sigmoid(0.0, k=2.0)

        with pytest.raises(ValueError):
            inverse_sigmoid(1.0, k=2.0)

    def test_derivative_at_midpoint(self):
        """Derivative should be maximal at v_hat=0."""
        k = 2.0
        deriv_0 = sigmoid_derivative(0.0, k)
        deriv_pos = sigmoid_derivative(0.5, k)
        deriv_neg = sigmoid_derivative(-0.5, k)

        assert deriv_0 > deriv_pos
        assert deriv_0 > deriv_neg

    def test_sigmoid_bounds_function(self):
        """sigmoid_bounds should return correct min/max p."""
        k = 2.0
        p_min, p_max = sigmoid_bounds(k)

        assert p_min == pytest.approx(calibrated_sigmoid(-1.0, k))
        assert p_max == pytest.approx(calibrated_sigmoid(1.0, k))

    def test_uncertainty_band(self):
        """effective_uncertainty_band should be symmetric."""
        k = 2.0
        threshold = 0.1

        v_min, v_max = effective_uncertainty_band(k, threshold)

        assert v_min == pytest.approx(-v_max, rel=1e-6)


class TestProxyWeights:
    """Tests for ProxyWeights."""

    def test_default_weights_sum(self):
        """Default weights should sum to 1."""
        weights = ProxyWeights()
        total = (
            weights.task_progress
            + weights.rework_penalty
            + weights.verifier_penalty
            + weights.engagement_signal
        )
        assert total == pytest.approx(1.0)

    def test_normalize(self):
        """Normalized weights should sum to 1."""
        weights = ProxyWeights(
            task_progress=2.0,
            rework_penalty=1.0,
            verifier_penalty=1.0,
            engagement_signal=0.0,
        )

        normalized = weights.normalize()
        total = (
            normalized.task_progress
            + normalized.rework_penalty
            + normalized.verifier_penalty
            + normalized.engagement_signal
        )
        assert total == pytest.approx(1.0)

    def test_normalize_preserves_ratios(self):
        """Normalization should preserve weight ratios."""
        weights = ProxyWeights(
            task_progress=4.0,
            rework_penalty=2.0,
            verifier_penalty=2.0,
            engagement_signal=2.0,
        )

        normalized = weights.normalize()

        assert normalized.task_progress / normalized.rework_penalty == pytest.approx(
            2.0
        )


class TestProxyObservables:
    """Tests for ProxyObservables."""

    def test_from_interaction(self):
        """Should extract observables from interaction."""
        from swarm.models.interaction import SoftInteraction

        interaction = SoftInteraction(
            task_progress_delta=0.7,
            rework_count=2,
            verifier_rejections=1,
            tool_misuse_flags=0,
            counterparty_engagement_delta=0.3,
        )

        obs = ProxyObservables.from_interaction(interaction)

        assert obs.task_progress_delta == 0.7
        assert obs.rework_count == 2
        assert obs.verifier_rejections == 1
        assert obs.tool_misuse_flags == 0
        assert obs.counterparty_engagement_delta == 0.3


class TestProxyComputer:
    """Tests for ProxyComputer."""

    def test_v_hat_bounds(self):
        """v_hat should always be in [-1, +1]."""
        computer = ProxyComputer()

        # Extreme positive
        obs_good = ProxyObservables(
            task_progress_delta=1.0,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=1.0,
        )
        v_hat = computer.compute_v_hat(obs_good)
        assert -1.0 <= v_hat <= 1.0

        # Extreme negative
        obs_bad = ProxyObservables(
            task_progress_delta=-1.0,
            rework_count=10,
            verifier_rejections=10,
            tool_misuse_flags=10,
            counterparty_engagement_delta=-1.0,
        )
        v_hat = computer.compute_v_hat(obs_bad)
        assert -1.0 <= v_hat <= 1.0

    def test_good_observables_positive_v_hat(self):
        """Good observables should produce positive v_hat."""
        computer = ProxyComputer()

        obs = ProxyObservables(
            task_progress_delta=0.8,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=0.5,
        )

        v_hat = computer.compute_v_hat(obs)
        assert v_hat > 0

    def test_bad_observables_negative_v_hat(self):
        """Bad observables should produce negative v_hat."""
        computer = ProxyComputer()

        obs = ProxyObservables(
            task_progress_delta=-0.5,
            rework_count=5,
            verifier_rejections=3,
            tool_misuse_flags=2,
            counterparty_engagement_delta=-0.6,
        )

        v_hat = computer.compute_v_hat(obs)
        assert v_hat < 0

    def test_rework_decay(self):
        """More rework should decrease v_hat."""
        computer = ProxyComputer()

        base_obs = ProxyObservables(
            task_progress_delta=0.5,
            counterparty_engagement_delta=0.3,
        )

        v_hat_0 = computer.compute_v_hat(
            ProxyObservables(**{**base_obs.__dict__, "rework_count": 0})
        )
        v_hat_1 = computer.compute_v_hat(
            ProxyObservables(**{**base_obs.__dict__, "rework_count": 1})
        )
        v_hat_3 = computer.compute_v_hat(
            ProxyObservables(**{**base_obs.__dict__, "rework_count": 3})
        )

        assert v_hat_0 > v_hat_1 > v_hat_3

    def test_compute_p_from_v_hat(self):
        """compute_p should apply sigmoid correctly."""
        computer = ProxyComputer(sigmoid_k=2.0)

        v_hat = 0.5
        p = computer.compute_p(v_hat)

        expected = calibrated_sigmoid(v_hat, k=2.0)
        assert p == pytest.approx(expected)

    def test_compute_labels(self):
        """compute_labels should return both v_hat and p."""
        computer = ProxyComputer()

        obs = ProxyObservables(task_progress_delta=0.6)
        v_hat, p = computer.compute_labels(obs)

        assert isinstance(v_hat, float)
        assert isinstance(p, float)
        assert -1.0 <= v_hat <= 1.0
        assert 0.0 <= p <= 1.0

    def test_custom_weights(self):
        """Custom weights should affect v_hat."""
        # All weight on task_progress
        weights_progress = ProxyWeights(
            task_progress=1.0,
            rework_penalty=0.0,
            verifier_penalty=0.0,
            engagement_signal=0.0,
        )

        # All weight on engagement
        weights_engagement = ProxyWeights(
            task_progress=0.0,
            rework_penalty=0.0,
            verifier_penalty=0.0,
            engagement_signal=1.0,
        )

        computer_progress = ProxyComputer(weights=weights_progress)
        computer_engagement = ProxyComputer(weights=weights_engagement)

        obs = ProxyObservables(
            task_progress_delta=0.8,  # High
            counterparty_engagement_delta=-0.5,  # Low
        )

        v_hat_progress = computer_progress.compute_v_hat(obs)
        v_hat_engagement = computer_engagement.compute_v_hat(obs)

        # Progress-weighted should be positive, engagement-weighted negative
        assert v_hat_progress > 0
        assert v_hat_engagement < 0

    def test_sigmoid_k_effect(self):
        """Higher k should produce more extreme p values."""
        obs = ProxyObservables(task_progress_delta=0.5)

        computer_low_k = ProxyComputer(sigmoid_k=1.0)
        computer_high_k = ProxyComputer(sigmoid_k=5.0)

        _, p_low = computer_low_k.compute_labels(obs)
        _, p_high = computer_high_k.compute_labels(obs)

        # Both should be > 0.5 since obs is positive
        assert p_low > 0.5
        assert p_high > 0.5

        # Higher k should push further from 0.5
        assert abs(p_high - 0.5) > abs(p_low - 0.5)
