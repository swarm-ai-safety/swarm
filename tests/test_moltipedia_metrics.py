"""Tests for swarm.metrics.moltipedia_metrics."""

import pytest

from swarm.metrics.moltipedia_metrics import (
    content_quality_trend,
    governance_effectiveness,
    pair_farming_rate,
    point_concentration,
    policy_fix_exploitation_rate,
)
from swarm.models.interaction import SoftInteraction


class TestPointConcentration:
    def test_empty(self):
        assert point_concentration({}) == 0.0

    def test_equal(self):
        result = point_concentration({"a": 5.0, "b": 5.0, "c": 5.0})
        assert result == pytest.approx(0.0, abs=0.01)

    def test_all_zero(self):
        assert point_concentration({"a": 0.0, "b": 0.0}) == 0.0

    def test_concentrated(self):
        result = point_concentration({"a": 100.0, "b": 0.0, "c": 0.0})
        assert result > 0.5

    def test_negative_values_clamped(self):
        result = point_concentration({"a": -1.0, "b": -2.0})
        assert result == 0.0


class TestPairFarmingRate:
    def test_empty(self):
        assert pair_farming_rate([]) == 0.0

    def test_no_moltipedia_interactions(self):
        interactions = [SoftInteraction(metadata={})]
        assert pair_farming_rate(interactions) == 0.0

    def test_no_repeated_pairs(self):
        interactions = [
            SoftInteraction(
                initiator="a", counterparty="b",
                metadata={"moltipedia": True, "points": 1},
            ),
            SoftInteraction(
                initiator="c", counterparty="d",
                metadata={"moltipedia": True, "points": 2},
            ),
        ]
        assert pair_farming_rate(interactions) == pytest.approx(0.0)

    def test_repeated_pairs(self):
        interactions = [
            SoftInteraction(
                initiator="a", counterparty="b",
                metadata={"moltipedia": True, "points": 1},
            ),
            SoftInteraction(
                initiator="a", counterparty="b",
                metadata={"moltipedia": True, "points": 2},
            ),
            SoftInteraction(
                initiator="c", counterparty="d",
                metadata={"moltipedia": True, "points": 3},
            ),
        ]
        # 2 repeated out of 3 total scored
        result = pair_farming_rate(interactions)
        assert result == pytest.approx(2 / 3)

    def test_reversed_pair_order(self):
        interactions = [
            SoftInteraction(
                initiator="b", counterparty="a",
                metadata={"moltipedia": True, "points": 1},
            ),
            SoftInteraction(
                initiator="a", counterparty="b",
                metadata={"moltipedia": True, "points": 2},
            ),
        ]
        result = pair_farming_rate(interactions)
        assert result == pytest.approx(2 / 2)


class TestPolicyFixExploitationRate:
    def test_empty(self):
        assert policy_fix_exploitation_rate([]) == 0.0

    def test_no_policy_fixes(self):
        interactions = [SoftInteraction(metadata={"moltipedia": True})]
        assert policy_fix_exploitation_rate(interactions) == 0.0

    def test_no_exploitation(self):
        interactions = [
            SoftInteraction(
                task_progress_delta=0.5,
                metadata={"moltipedia": True, "edit_type": "policy_fix"},
            ),
        ]
        assert policy_fix_exploitation_rate(interactions) == pytest.approx(0.0)

    def test_all_exploited(self):
        interactions = [
            SoftInteraction(
                task_progress_delta=0.01,
                metadata={"moltipedia": True, "edit_type": "policy_fix"},
            ),
            SoftInteraction(
                task_progress_delta=0.0,
                metadata={"moltipedia": True, "edit_type": "policy_fix"},
            ),
        ]
        assert policy_fix_exploitation_rate(interactions) == pytest.approx(1.0)

    def test_mixed(self):
        interactions = [
            SoftInteraction(
                task_progress_delta=0.01,
                metadata={"moltipedia": True, "edit_type": "policy_fix"},
            ),
            SoftInteraction(
                task_progress_delta=0.5,
                metadata={"moltipedia": True, "edit_type": "policy_fix"},
            ),
        ]
        assert policy_fix_exploitation_rate(interactions) == pytest.approx(0.5)


class TestContentQualityTrend:
    def test_empty(self):
        assert content_quality_trend([]) == 0.0

    def test_single(self):
        assert content_quality_trend([0.8]) == pytest.approx(0.8)

    def test_average(self):
        assert content_quality_trend([0.6, 0.8, 1.0]) == pytest.approx(0.8)


class TestGovernanceEffectiveness:
    def test_zero_total(self):
        assert governance_effectiveness(0.0, 5.0) == 0.0

    def test_negative_total(self):
        assert governance_effectiveness(-1.0, 5.0) == 0.0

    def test_normal(self):
        assert governance_effectiveness(100.0, 30.0) == pytest.approx(0.3)

    def test_clamped_at_one(self):
        result = governance_effectiveness(10.0, 50.0)
        assert result == pytest.approx(1.0)

    def test_clamped_at_zero(self):
        result = governance_effectiveness(10.0, -5.0)
        assert result == pytest.approx(0.0)
