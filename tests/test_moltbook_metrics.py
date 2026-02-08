"""Tests for swarm.metrics.moltbook_metrics."""

import pytest

from swarm.metrics.moltbook_metrics import (
    captcha_effectiveness,
    challenge_pass_rate,
    content_throughput,
    karma_concentration,
    rate_limit_governance_impact,
    rate_limit_hit_rate,
    verification_latency_distribution,
    wasted_action_rate,
)


class TestChallengePassRate:
    def test_empty(self):
        assert challenge_pass_rate([]) == 0.0

    def test_all_pass(self):
        assert challenge_pass_rate([True, True, True]) == 1.0

    def test_all_fail(self):
        assert challenge_pass_rate([False, False]) == 0.0

    def test_mixed(self):
        assert challenge_pass_rate([True, False, True, False]) == pytest.approx(0.5)


class TestRateLimitHitRate:
    def test_zero_attempts(self):
        assert rate_limit_hit_rate(5, 0) == 0.0

    def test_negative_attempts(self):
        assert rate_limit_hit_rate(5, -1) == 0.0

    def test_normal(self):
        assert rate_limit_hit_rate(3, 10) == pytest.approx(0.3)


class TestContentThroughput:
    def test_zero_epochs(self):
        assert content_throughput(10, 0) == 0.0

    def test_negative_epochs(self):
        assert content_throughput(10, -1) == 0.0

    def test_normal(self):
        assert content_throughput(20, 4) == pytest.approx(5.0)


class TestVerificationLatencyDistribution:
    def test_empty(self):
        result = verification_latency_distribution([])
        assert result == {"mean": 0.0, "p50": 0.0, "p90": 0.0}

    def test_single_value(self):
        result = verification_latency_distribution([5])
        assert result["mean"] == pytest.approx(5.0)
        assert result["p50"] == pytest.approx(5.0)
        assert result["p90"] == pytest.approx(5.0)

    def test_multiple_values(self):
        result = verification_latency_distribution([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert result["mean"] == pytest.approx(5.5)
        assert result["p50"] == 6
        assert result["p90"] == 9


class TestKarmaConcentration:
    def test_empty(self):
        assert karma_concentration({}) == 0.0

    def test_equal_distribution(self):
        result = karma_concentration({"a": 10.0, "b": 10.0, "c": 10.0})
        assert result == pytest.approx(0.0, abs=0.01)

    def test_all_zero(self):
        assert karma_concentration({"a": 0.0, "b": 0.0}) == 0.0

    def test_unequal_distribution(self):
        result = karma_concentration({"a": 100.0, "b": 1.0, "c": 1.0})
        assert result > 0.0

    def test_negative_values_clamped(self):
        # Negative karma should be treated as 0
        result = karma_concentration({"a": -5.0, "b": -3.0})
        assert result == 0.0


class TestWastedActionRate:
    def test_zero_total(self):
        assert wasted_action_rate(5, 0) == 0.0

    def test_negative_total(self):
        assert wasted_action_rate(5, -1) == 0.0

    def test_normal(self):
        assert wasted_action_rate(2, 10) == pytest.approx(0.2)


class TestCaptchaEffectiveness:
    def test_zero_bot_success(self):
        assert captcha_effectiveness(0.1, 0.0) == 0.0

    def test_negative_bot_success(self):
        assert captcha_effectiveness(0.1, -0.5) == 0.0

    def test_normal(self):
        assert captcha_effectiveness(0.2, 0.5) == pytest.approx(0.4)


class TestRateLimitGovernanceImpact:
    def test_zero_without_limits(self):
        assert rate_limit_governance_impact(5.0, 0.0) == 0.0

    def test_negative_without_limits(self):
        assert rate_limit_governance_impact(5.0, -1.0) == 0.0

    def test_no_reduction(self):
        assert rate_limit_governance_impact(10.0, 10.0) == pytest.approx(0.0)

    def test_partial_reduction(self):
        result = rate_limit_governance_impact(5.0, 10.0)
        assert result == pytest.approx(0.5)

    def test_full_reduction(self):
        result = rate_limit_governance_impact(0.0, 10.0)
        assert result == pytest.approx(1.0)

    def test_clamped_at_one(self):
        # with_limits > without_limits could give negative, clamped to 0
        result = rate_limit_governance_impact(15.0, 10.0)
        assert result == pytest.approx(0.0)
