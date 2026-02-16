"""Regression tests for evaluation metrics suite."""

import pytest

from swarm.evaluation.eval_metrics import (
    aggregate_success_metrics,
    audit_effectiveness,
    calls_per_success,
    deception_detection_rate,
    loopiness_score,
    success_rate,
)


class TestSuccessRate:
    """Tests for success rate metric."""

    def test_empty_attempts(self):
        """Empty list should return 0.0."""
        assert success_rate([]) == 0.0

    def test_all_successes(self):
        """All successful attempts should return 1.0."""
        attempts = [
            {"success": True},
            {"success": True},
            {"success": True},
        ]
        assert success_rate(attempts) == 1.0

    def test_no_successes(self):
        """No successful attempts should return 0.0."""
        attempts = [
            {"success": False},
            {"success": False},
            {"success": False},
        ]
        assert success_rate(attempts) == 0.0

    def test_mixed_successes(self):
        """Mixed success/failure should return correct fraction."""
        attempts = [
            {"success": True},
            {"success": False},
            {"success": True},
            {"success": False},
        ]
        assert success_rate(attempts) == 0.5

    def test_custom_success_key(self):
        """Should work with custom success key."""
        attempts = [
            {"passed": True},
            {"passed": False},
            {"passed": True},
        ]
        assert success_rate(attempts, success_key="passed") == pytest.approx(
            2.0 / 3.0
        )

    def test_missing_success_key(self):
        """Missing key should be treated as False."""
        attempts = [
            {"success": True},
            {"other": "value"},
            {"success": True},
        ]
        assert success_rate(attempts) == pytest.approx(2.0 / 3.0)


class TestCallsPerSuccess:
    """Tests for calls per success metric."""

    def test_empty_attempts(self):
        """Empty list should return infinity."""
        assert calls_per_success([]) == float("inf")

    def test_no_successes(self):
        """No successes should return infinity."""
        attempts = [
            {"success": False, "calls": 5},
            {"success": False, "calls": 3},
        ]
        assert calls_per_success(attempts) == float("inf")

    def test_single_success(self):
        """Single success should return its call count."""
        attempts = [
            {"success": True, "calls": 7},
        ]
        assert calls_per_success(attempts) == 7.0

    def test_multiple_successes(self):
        """Multiple successes should return average calls."""
        attempts = [
            {"success": True, "calls": 3},
            {"success": False, "calls": 5},
            {"success": True, "calls": 2},
            {"success": True, "calls": 4},
        ]
        # Average of 3, 2, 4 = 3.0
        assert calls_per_success(attempts) == 3.0

    def test_custom_keys(self):
        """Should work with custom keys."""
        attempts = [
            {"passed": True, "api_calls": 5},
            {"passed": True, "api_calls": 3},
        ]
        result = calls_per_success(
            attempts, success_key="passed", calls_key="api_calls"
        )
        assert result == 4.0

    def test_zero_calls(self):
        """Should handle zero calls correctly."""
        attempts = [
            {"success": True, "calls": 0},
            {"success": True, "calls": 0},
        ]
        assert calls_per_success(attempts) == 0.0


class TestLoopinessScore:
    """Tests for loopiness score metric."""

    def test_empty_sequence(self):
        """Empty sequence should return 0.0."""
        assert loopiness_score([]) == 0.0

    def test_too_short_sequence(self):
        """Sequence shorter than window should return 0.0."""
        assert loopiness_score(["a", "b", "c"], window_size=5) == 0.0

    def test_no_loops(self):
        """No repetition should return 0.0."""
        sequence = ["a", "b", "c", "d", "e", "f"]
        assert loopiness_score(sequence, window_size=3) == 0.0

    def test_perfect_loop(self):
        """Perfect repetition should return 1.0."""
        # Window size 4: [a,b,a,b] repeats perfectly
        sequence = ["a", "b", "a", "b"]
        assert loopiness_score(sequence, window_size=4) == 1.0

    def test_partial_loop(self):
        """Partial repetition should return intermediate value."""
        # "a,b,a,b,c,d,a,b" has some loops in windows
        sequence = ["a", "b", "a", "b", "c", "d", "a", "b"]
        score = loopiness_score(sequence, window_size=4)
        assert 0.0 < score < 1.0

    def test_different_window_sizes(self):
        """Should detect loops at different granularities."""
        sequence = ["a", "b", "a", "b", "a", "b"]
        # Window size 4 should detect [a,b,a,b] which is a perfect repeat of [a,b]
        score_4 = loopiness_score(sequence, window_size=4)
        assert score_4 > 0.0
        
        # For window size 2, test with a different sequence that has repeating pairs
        sequence2 = ["x", "x", "y", "y", "z", "z"]
        score_2 = loopiness_score(sequence2, window_size=2)
        # Windows [x,x], [y,y], [z,z] are all perfect loops of single element
        assert score_2 > 0.0

    def test_longer_sequence_with_loops(self):
        """Longer sequence with embedded loops."""
        # [a,a,a,a] is a perfect loop of "a"
        sequence = ["x", "y", "z", "a", "a", "a", "a", "m", "n"]
        score = loopiness_score(sequence, window_size=4)
        # Should detect the "aaaa" loop
        assert score > 0.0


class TestAuditEffectiveness:
    """Tests for audit effectiveness metric."""

    def test_empty_inputs(self):
        """Empty inputs should return sensible defaults."""
        result = audit_effectiveness([], [])
        assert result["precision"] == 1.0  # No false positives
        assert result["recall"] == 0.0  # No ground truth
        assert result["f1_score"] == 0.0
        assert result["detection_rate"] == 0.0

    def test_no_malicious_actions(self):
        """No ground truth malicious actions."""
        audited = [{"action_id": "a1"}]
        result = audit_effectiveness(audited, [])
        assert result["precision"] == 0.0  # All are false positives
        assert result["recall"] == 0.0
        assert result["f1_score"] == 0.0

    def test_perfect_detection(self):
        """All and only malicious actions audited."""
        audited = [
            {"action_id": "a1"},
            {"action_id": "a2"},
        ]
        malicious = [
            {"action_id": "a1"},
            {"action_id": "a2"},
        ]
        result = audit_effectiveness(audited, malicious)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1_score"] == 1.0
        assert result["detection_rate"] == 1.0

    def test_partial_detection(self):
        """Some malicious detected, some missed, some false positives."""
        audited = [
            {"action_id": "a1"},
            {"action_id": "a2"},
            {"action_id": "a3"},
        ]
        malicious = [
            {"action_id": "a1"},  # Detected (TP)
            {"action_id": "a2"},  # Detected (TP)
            {"action_id": "a4"},  # Missed (FN)
        ]
        # a3 is FP, a4 is FN
        # TP=2, FP=1, FN=1
        result = audit_effectiveness(audited, malicious)
        assert result["precision"] == pytest.approx(2.0 / 3.0)  # 2/(2+1)
        assert result["recall"] == pytest.approx(2.0 / 3.0)  # 2/(2+1)
        assert result["detection_rate"] == pytest.approx(2.0 / 3.0)
        # F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3
        assert result["f1_score"] == pytest.approx(2.0 / 3.0)

    def test_no_detection(self):
        """No malicious actions detected."""
        audited = [
            {"action_id": "a5"},
            {"action_id": "a6"},
        ]
        malicious = [
            {"action_id": "a1"},
            {"action_id": "a2"},
        ]
        result = audit_effectiveness(audited, malicious)
        assert result["precision"] == 0.0  # All FP
        assert result["recall"] == 0.0  # All FN
        assert result["f1_score"] == 0.0

    def test_custom_action_key(self):
        """Should work with custom action ID key."""
        audited = [{"id": "a1"}]
        malicious = [{"id": "a1"}]
        result = audit_effectiveness(audited, malicious, action_id_key="id")
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0


class TestDeceptionDetectionRate:
    """Tests for deception detection rate metric."""

    def test_empty_behaviors(self):
        """Empty list should return zeros."""
        result = deception_detection_rate([])
        assert result["true_positive_rate"] == 0.0
        assert result["false_positive_rate"] == 0.0
        assert result["accuracy"] == 0.0
        assert result["detection_rate"] == 0.0

    def test_perfect_detection(self):
        """All deceptive detected, all honest correctly classified."""
        behaviors = [
            {"is_deceptive": True, "detected_as_deceptive": True},
            {"is_deceptive": True, "detected_as_deceptive": True},
            {"is_deceptive": False, "detected_as_deceptive": False},
            {"is_deceptive": False, "detected_as_deceptive": False},
        ]
        result = deception_detection_rate(behaviors)
        assert result["true_positive_rate"] == 1.0
        assert result["false_positive_rate"] == 0.0
        assert result["accuracy"] == 1.0
        assert result["detection_rate"] == 1.0

    def test_no_detection(self):
        """No deceptive behaviors detected."""
        behaviors = [
            {"is_deceptive": True, "detected_as_deceptive": False},
            {"is_deceptive": True, "detected_as_deceptive": False},
            {"is_deceptive": False, "detected_as_deceptive": False},
        ]
        result = deception_detection_rate(behaviors)
        assert result["true_positive_rate"] == 0.0
        assert result["false_positive_rate"] == 0.0
        # Accuracy is 1/3 (only honest correctly classified)
        assert result["accuracy"] == pytest.approx(1.0 / 3.0)

    def test_mixed_detection(self):
        """Mix of correct and incorrect detections."""
        behaviors = [
            {"is_deceptive": True, "detected_as_deceptive": True},  # TP
            {"is_deceptive": True, "detected_as_deceptive": False},  # FN
            {"is_deceptive": False, "detected_as_deceptive": False},  # TN
            {"is_deceptive": False, "detected_as_deceptive": True},  # FP
        ]
        # TP=1, FN=1, TN=1, FP=1
        result = deception_detection_rate(behaviors)
        assert result["true_positive_rate"] == 0.5  # 1/(1+1)
        assert result["false_positive_rate"] == 0.5  # 1/(1+1)
        assert result["accuracy"] == 0.5  # (1+1)/4
        assert result["detection_rate"] == 0.5

    def test_all_deceptive(self):
        """All behaviors are deceptive."""
        behaviors = [
            {"is_deceptive": True, "detected_as_deceptive": True},
            {"is_deceptive": True, "detected_as_deceptive": False},
            {"is_deceptive": True, "detected_as_deceptive": True},
        ]
        result = deception_detection_rate(behaviors)
        assert result["true_positive_rate"] == pytest.approx(2.0 / 3.0)
        assert result["false_positive_rate"] == 0.0  # No honest behaviors
        assert result["accuracy"] == pytest.approx(2.0 / 3.0)

    def test_all_honest(self):
        """All behaviors are honest."""
        behaviors = [
            {"is_deceptive": False, "detected_as_deceptive": False},
            {"is_deceptive": False, "detected_as_deceptive": True},
            {"is_deceptive": False, "detected_as_deceptive": False},
        ]
        result = deception_detection_rate(behaviors)
        assert result["true_positive_rate"] == 0.0  # No deceptive behaviors
        assert result["false_positive_rate"] == pytest.approx(1.0 / 3.0)
        assert result["accuracy"] == pytest.approx(2.0 / 3.0)

    def test_custom_keys(self):
        """Should work with custom keys."""
        behaviors = [
            {"malicious": True, "flagged": True},
            {"malicious": False, "flagged": False},
        ]
        result = deception_detection_rate(
            behaviors, deceptive_key="malicious", detected_key="flagged"
        )
        assert result["true_positive_rate"] == 1.0
        assert result["accuracy"] == 1.0


class TestAggregateSuccessMetrics:
    """Tests for aggregate success metrics."""

    def test_empty_experiments(self):
        """Empty list should return zeros."""
        result = aggregate_success_metrics([])
        assert result["mean_success_rate"] == 0.0
        assert result["std_success_rate"] == 0.0
        assert result["min_success_rate"] == 0.0
        assert result["max_success_rate"] == 0.0
        assert result["total_attempts"] == 0
        assert result["total_successes"] == 0

    def test_single_experiment(self):
        """Single experiment should return its metrics."""
        experiments = [
            {
                "attempts": [
                    {"success": True},
                    {"success": False},
                    {"success": True},
                ]
            }
        ]
        result = aggregate_success_metrics(experiments)
        assert result["mean_success_rate"] == pytest.approx(2.0 / 3.0)
        assert result["std_success_rate"] == 0.0  # Only one experiment
        assert result["min_success_rate"] == pytest.approx(2.0 / 3.0)
        assert result["max_success_rate"] == pytest.approx(2.0 / 3.0)
        assert result["total_attempts"] == 3
        assert result["total_successes"] == 2

    def test_multiple_experiments(self):
        """Multiple experiments should aggregate correctly."""
        experiments = [
            {
                "attempts": [
                    {"success": True},
                    {"success": False},
                ]
            },  # 50%
            {
                "attempts": [
                    {"success": True},
                    {"success": True},
                ]
            },  # 100%
            {
                "attempts": [
                    {"success": False},
                    {"success": False},
                ]
            },  # 0%
        ]
        result = aggregate_success_metrics(experiments)
        # Mean: (0.5 + 1.0 + 0.0) / 3 = 0.5
        assert result["mean_success_rate"] == 0.5
        # Min/Max
        assert result["min_success_rate"] == 0.0
        assert result["max_success_rate"] == 1.0
        # Std
        assert result["std_success_rate"] > 0.0
        # Totals
        assert result["total_attempts"] == 6
        assert result["total_successes"] == 3

    def test_experiments_without_attempts(self):
        """Should handle experiments with empty attempts."""
        experiments = [
            {"attempts": []},
            {"attempts": [{"success": True}]},
        ]
        result = aggregate_success_metrics(experiments)
        # Only second experiment counts
        assert result["mean_success_rate"] == 1.0
        assert result["total_attempts"] == 1
        assert result["total_successes"] == 1

    def test_custom_success_key(self):
        """Should work with custom success key."""
        experiments = [
            {
                "attempts": [
                    {"passed": True},
                    {"passed": False},
                ]
            }
        ]
        result = aggregate_success_metrics(experiments, success_key="passed")
        assert result["mean_success_rate"] == 0.5
        assert result["total_successes"] == 1


class TestEdgeCases:
    """Edge case tests for all metrics."""

    def test_success_rate_with_none_values(self):
        """Should handle None values gracefully."""
        attempts = [
            {"success": True},
            {"success": None},
            {"success": False},
        ]
        # None should be treated as False
        rate = success_rate(attempts)
        assert rate == pytest.approx(1.0 / 3.0)

    def test_calls_per_success_with_negative_calls(self):
        """Should handle negative call counts (shouldn't happen but defensive)."""
        attempts = [
            {"success": True, "calls": -5},
            {"success": True, "calls": 10},
        ]
        # Average of -5 and 10 = 2.5
        assert calls_per_success(attempts) == 2.5

    def test_loopiness_with_single_element_sequence(self):
        """Single element sequence."""
        assert loopiness_score(["a"], window_size=2) == 0.0

    def test_audit_effectiveness_with_duplicate_ids(self):
        """Should handle duplicate action IDs."""
        audited = [
            {"action_id": "a1"},
            {"action_id": "a1"},  # Duplicate
        ]
        malicious = [
            {"action_id": "a1"},
        ]
        result = audit_effectiveness(audited, malicious)
        # Sets will deduplicate, so should be perfect match
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_deception_with_missing_keys(self):
        """Should treat missing keys as False."""
        behaviors = [
            {"is_deceptive": True, "detected_as_deceptive": True},
            {},  # Missing both keys
            {"is_deceptive": False},  # Missing detected key
        ]
        result = deception_detection_rate(behaviors)
        # First: TP, Second: TN, Third: TN
        # TP=1, FN=0, TN=2, FP=0
        assert result["accuracy"] == 1.0
