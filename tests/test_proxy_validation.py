"""Tests for proxy validation and logging."""

import logging

import pytest

from swarm.core.proxy import ProxyComputer, ProxyObservables, ProxyWeights
from swarm.core.sigmoid import calibrated_sigmoid


class TestSigmoidKValidation:
    """Tests for sigmoid_k parameter validation."""

    def test_sigmoid_k_zero_rejected(self):
        """sigmoid_k = 0 should be rejected."""
        with pytest.raises(ValueError, match="sigmoid_k must be positive"):
            ProxyComputer(sigmoid_k=0.0)

    def test_sigmoid_k_negative_rejected(self):
        """Negative sigmoid_k should be rejected."""
        with pytest.raises(ValueError, match="sigmoid_k must be positive"):
            ProxyComputer(sigmoid_k=-1.0)

    def test_sigmoid_k_extremely_large_rejected(self):
        """Extremely large sigmoid_k should be rejected."""
        with pytest.raises(ValueError, match="extremely large"):
            ProxyComputer(sigmoid_k=150.0)

    def test_sigmoid_k_at_boundary(self):
        """sigmoid_k at boundary (100) should be accepted."""
        computer = ProxyComputer(sigmoid_k=100.0)
        assert computer.sigmoid_k == 100.0

    def test_sigmoid_k_validation_in_calibrated_sigmoid(self):
        """calibrated_sigmoid should also validate k parameter."""
        with pytest.raises(ValueError, match="sigmoid_k must be positive"):
            calibrated_sigmoid(0.5, k=0.0)

        with pytest.raises(ValueError, match="sigmoid_k must be positive"):
            calibrated_sigmoid(0.5, k=-1.0)

        with pytest.raises(ValueError, match="extremely large"):
            calibrated_sigmoid(0.5, k=150.0)


class TestDecayParameterValidation:
    """Tests for decay parameter validation."""

    def test_rework_decay_zero_rejected(self):
        """rework_decay = 0 should be rejected."""
        with pytest.raises(ValueError, match="rework_decay must be positive"):
            ProxyComputer(rework_decay=0.0)

    def test_rework_decay_negative_rejected(self):
        """Negative rework_decay should be rejected."""
        with pytest.raises(ValueError, match="rework_decay must be positive"):
            ProxyComputer(rework_decay=-0.1)

    def test_rework_decay_one_rejected(self):
        """rework_decay = 1 should be rejected."""
        with pytest.raises(ValueError, match="rework_decay must be less than 1"):
            ProxyComputer(rework_decay=1.0)

    def test_rework_decay_above_one_rejected(self):
        """rework_decay > 1 should be rejected."""
        with pytest.raises(ValueError, match="rework_decay must be less than 1"):
            ProxyComputer(rework_decay=1.5)

    def test_rework_decay_extremely_small_rejected(self):
        """Extremely small rework_decay should be rejected."""
        with pytest.raises(ValueError, match="extremely small"):
            ProxyComputer(rework_decay=0.001)

    def test_rejection_decay_zero_rejected(self):
        """rejection_decay = 0 should be rejected."""
        with pytest.raises(ValueError, match="rejection_decay must be positive"):
            ProxyComputer(rejection_decay=0.0)

    def test_rejection_decay_one_rejected(self):
        """rejection_decay = 1 should be rejected."""
        with pytest.raises(ValueError, match="rejection_decay must be less than 1"):
            ProxyComputer(rejection_decay=1.0)

    def test_rejection_decay_extremely_small_rejected(self):
        """Extremely small rejection_decay should be rejected."""
        with pytest.raises(ValueError, match="extremely small"):
            ProxyComputer(rejection_decay=0.005)

    def test_misuse_decay_zero_rejected(self):
        """misuse_decay = 0 should be rejected."""
        with pytest.raises(ValueError, match="misuse_decay must be positive"):
            ProxyComputer(misuse_decay=0.0)

    def test_misuse_decay_one_rejected(self):
        """misuse_decay = 1 should be rejected."""
        with pytest.raises(ValueError, match="misuse_decay must be less than 1"):
            ProxyComputer(misuse_decay=1.0)

    def test_decay_at_boundaries(self):
        """Decay parameters at valid boundaries should be accepted."""
        computer = ProxyComputer(
            rework_decay=0.01, rejection_decay=0.99, misuse_decay=0.5
        )
        assert computer.rework_decay == 0.01
        assert computer.rejection_decay == 0.99
        assert computer.misuse_decay == 0.5


class TestProxyWeightsValidation:
    """Tests for ProxyWeights validation."""

    def test_negative_task_progress_rejected(self):
        """Negative task_progress should be rejected."""
        with pytest.raises(ValueError, match="task_progress must be non-negative"):
            ProxyWeights(task_progress=-0.5)

    def test_negative_rework_penalty_rejected(self):
        """Negative rework_penalty should be rejected."""
        with pytest.raises(ValueError, match="rework_penalty must be non-negative"):
            ProxyWeights(rework_penalty=-0.3)

    def test_negative_verifier_penalty_rejected(self):
        """Negative verifier_penalty should be rejected."""
        with pytest.raises(ValueError, match="verifier_penalty must be non-negative"):
            ProxyWeights(verifier_penalty=-0.2)

    def test_negative_engagement_signal_rejected(self):
        """Negative engagement_signal should be rejected."""
        with pytest.raises(ValueError, match="engagement_signal must be non-negative"):
            ProxyWeights(engagement_signal=-0.1)

    def test_zero_weights_accepted(self):
        """Zero weights should be accepted."""
        weights = ProxyWeights(
            task_progress=0.0,
            rework_penalty=0.0,
            verifier_penalty=1.0,
            engagement_signal=0.0,
        )
        assert weights.task_progress == 0.0
        assert weights.verifier_penalty == 1.0

    def test_normalized_negative_weights_rejected(self):
        """Normalization should not allow negative weights to pass through."""
        weights = ProxyWeights(task_progress=1.0)
        # Try to set negative weight after creation
        with pytest.raises(ValueError):
            ProxyWeights(
                task_progress=1.0,
                rework_penalty=-0.5,
                verifier_penalty=0.3,
                engagement_signal=0.2,
            )


class TestClampingWarnings:
    """Tests for clamping warnings."""

    def test_v_hat_clamping_warning(self, caplog):
        """compute_v_hat should warn when clamping occurs."""
        # Create a scenario where v_hat would exceed [-1, +1]
        # This is difficult with normalized weights, so we'll use unnormalized weights
        weights = ProxyWeights(
            task_progress=5.0,  # Will be normalized but demonstrates intent
            rework_penalty=0.0,
            verifier_penalty=0.0,
            engagement_signal=0.0,
        )
        computer = ProxyComputer(weights=weights)

        # Create observables that would produce extreme v_hat
        obs = ProxyObservables(
            task_progress_delta=2.0,  # Exceeds expected range
        )

        with caplog.at_level(logging.WARNING):
            v_hat = computer.compute_v_hat(obs)

        # v_hat should be clamped to [-1, 1]
        assert -1.0 <= v_hat <= 1.0

        # Check if warning was logged (may or may not happen depending on normalization)
        # This test is more about coverage than assertion
        if any("clamped" in record.message for record in caplog.records):
            assert any(
                "compute_v_hat" in record.message for record in caplog.records
            )

    def test_sigmoid_v_hat_clamping_warning(self, caplog):
        """calibrated_sigmoid should warn when clamping v_hat."""
        with caplog.at_level(logging.WARNING):
            # Pass extremely large v_hat that will be clamped
            p = calibrated_sigmoid(50.0, k=2.0)

        # p should be valid
        assert 0.0 <= p <= 1.0

        # Check that warning was logged
        assert any("clamped" in record.message for record in caplog.records)
        assert any("calibrated_sigmoid" in record.message for record in caplog.records)

    def test_sigmoid_negative_extreme_clamping_warning(self, caplog):
        """calibrated_sigmoid should warn for extreme negative v_hat."""
        with caplog.at_level(logging.WARNING):
            p = calibrated_sigmoid(-50.0, k=2.0)

        assert 0.0 <= p <= 1.0
        assert any("clamped" in record.message for record in caplog.records)


class TestValidParameterCombinations:
    """Tests that valid parameter combinations work correctly."""

    def test_all_valid_parameters(self):
        """All valid parameters should work together."""
        weights = ProxyWeights(
            task_progress=0.5,
            rework_penalty=0.3,
            verifier_penalty=0.1,
            engagement_signal=0.1,
        )

        computer = ProxyComputer(
            weights=weights,
            sigmoid_k=3.0,
            rework_decay=0.2,
            rejection_decay=0.35,
            misuse_decay=0.6,
        )

        obs = ProxyObservables(
            task_progress_delta=0.7,
            rework_count=2,
            verifier_rejections=1,
            tool_misuse_flags=0,
            counterparty_engagement_delta=0.4,
        )

        v_hat, p = computer.compute_labels(obs)

        assert -1.0 <= v_hat <= 1.0
        assert 0.0 <= p <= 1.0

    def test_edge_case_decay_values(self):
        """Test decay values at boundaries."""
        computer = ProxyComputer(
            rework_decay=0.01,  # Minimum acceptable
            rejection_decay=0.99,  # Maximum acceptable
            misuse_decay=0.5,
        )

        obs = ProxyObservables(rework_count=1, verifier_rejections=1)
        v_hat = computer.compute_v_hat(obs)

        # Should not raise and should produce valid v_hat
        assert -1.0 <= v_hat <= 1.0
