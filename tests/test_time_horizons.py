"""Tests for swarm.metrics.time_horizons."""

import pytest

from swarm.metrics.time_horizons import (
    CAPABILITY_PROFILES,
    AgentCapabilityProfile,
    ComputeConstraints,
    TimeHorizonBucket,
    TimeHorizonMetrics,
)


class TestTimeHorizonBucket:
    def test_defaults(self):
        b = TimeHorizonBucket(horizon_minutes=10)
        assert b.total_tasks == 0
        assert b.successful_tasks == 0
        assert b.reliability == 0.0
        assert b.mean_quality == 0.0
        assert b.mean_duration == 0.0

    def test_record_success(self):
        b = TimeHorizonBucket(horizon_minutes=10)
        b.record(success=True, duration_minutes=8.0, quality=0.9)
        assert b.total_tasks == 1
        assert b.successful_tasks == 1
        assert b.reliability == pytest.approx(1.0)
        assert b.mean_quality == pytest.approx(0.9)
        assert b.mean_duration == pytest.approx(8.0)

    def test_record_failure(self):
        b = TimeHorizonBucket(horizon_minutes=10)
        b.record(success=False, duration_minutes=5.0)
        assert b.total_tasks == 1
        assert b.successful_tasks == 0
        assert b.reliability == pytest.approx(0.0)
        assert b.mean_quality == 0.0  # No quality scores added for failures

    def test_mixed_records(self):
        b = TimeHorizonBucket(horizon_minutes=10)
        b.record(success=True, duration_minutes=8.0, quality=0.8)
        b.record(success=False, duration_minutes=9.0)
        b.record(success=True, duration_minutes=7.0, quality=1.0)
        assert b.total_tasks == 3
        assert b.successful_tasks == 2
        assert b.reliability == pytest.approx(2 / 3)
        assert b.mean_quality == pytest.approx(0.9)
        assert b.mean_duration == pytest.approx(24.0 / 3)


class TestTimeHorizonMetrics:
    def test_initialization(self):
        m = TimeHorizonMetrics()
        assert len(m.buckets) == len(TimeHorizonMetrics.HORIZON_BUCKETS)
        for horizon in TimeHorizonMetrics.HORIZON_BUCKETS:
            assert horizon in m.buckets

    def test_record_task_short(self):
        m = TimeHorizonMetrics()
        m.record_task(duration_minutes=3.0, success=True, quality=0.9)
        assert m.buckets[5].total_tasks == 1

    def test_record_task_long(self):
        m = TimeHorizonMetrics()
        m.record_task(duration_minutes=2000.0, success=True)
        # Should go to the largest bucket (1440)
        assert m.buckets[1440].total_tasks == 1

    def test_reliability_curve_empty(self):
        m = TimeHorizonMetrics()
        assert m.reliability_curve() == {}

    def test_reliability_curve(self):
        m = TimeHorizonMetrics()
        m.record_task(duration_minutes=3.0, success=True)
        m.record_task(duration_minutes=3.0, success=True)
        m.record_task(duration_minutes=50.0, success=True)
        m.record_task(duration_minutes=50.0, success=False)
        curve = m.reliability_curve()
        assert curve[5] == pytest.approx(1.0)
        assert curve[60] == pytest.approx(0.5)

    def test_effective_horizon_none(self):
        m = TimeHorizonMetrics()
        assert m.effective_horizon() is None

    def test_effective_horizon(self):
        m = TimeHorizonMetrics()
        # Record 100% success for short tasks
        for _ in range(5):
            m.record_task(duration_minutes=3.0, success=True)
        # Record 100% success for 10min tasks
        for _ in range(5):
            m.record_task(duration_minutes=8.0, success=True)
        # Record 50% success for 30min tasks
        for i in range(4):
            m.record_task(duration_minutes=25.0, success=(i < 2))
        result = m.effective_horizon(threshold=0.8)
        assert result == 10

    def test_horizon_gap_no_effective(self):
        m = TimeHorizonMetrics()
        assert m.horizon_gap() == 0.0

    def test_horizon_gap(self):
        m = TimeHorizonMetrics()
        for _ in range(10):
            m.record_task(duration_minutes=3.0, success=True)
        for _ in range(10):
            m.record_task(duration_minutes=8.0, success=True)
        # effective_horizon should be 10 min, target is 480 min
        gap = m.horizon_gap(target_horizon=480)
        assert gap == pytest.approx(10 / 480)

    def test_to_dict(self):
        m = TimeHorizonMetrics()
        m.record_task(duration_minutes=3.0, success=True, quality=0.95)
        d = m.to_dict()
        assert "reliability_curve" in d
        assert "effective_horizon_80" in d
        assert "effective_horizon_90" in d
        assert "horizon_gap" in d
        assert "buckets" in d
        assert 5 in d["buckets"]


class TestAgentCapabilityProfile:
    def test_defaults(self):
        p = AgentCapabilityProfile()
        assert p.capability_level == 1.0
        assert p.base_reliability_10min == 0.8

    def test_reliability_at_zero(self):
        p = AgentCapabilityProfile()
        assert p.reliability_at_horizon(0) == p.base_reliability_10min

    def test_reliability_at_10min(self):
        p = AgentCapabilityProfile()
        result = p.reliability_at_horizon(10)
        # At 10 min, log10(10/10) = 0, so no decay
        assert result == pytest.approx(p.base_reliability_10min * p.capability_level)

    def test_reliability_decreases_with_duration(self):
        p = AgentCapabilityProfile()
        r10 = p.reliability_at_horizon(10)
        r60 = p.reliability_at_horizon(60)
        r480 = p.reliability_at_horizon(480)
        assert r10 > r60
        assert r60 > r480

    def test_reliability_clamped(self):
        p = AgentCapabilityProfile(
            base_reliability_10min=0.1, horizon_decay_rate=0.5
        )
        result = p.reliability_at_horizon(100000)
        assert result >= 0.0

    def test_expected_quality(self):
        p = AgentCapabilityProfile()
        q = p.expected_quality(10)
        r = p.reliability_at_horizon(10)
        assert q == pytest.approx(r ** 1.2)

    def test_compute_cost(self):
        p = AgentCapabilityProfile(compute_cost_per_minute=2.0)
        assert p.compute_cost(30) == pytest.approx(60.0)


class TestCapabilityProfiles:
    def test_profiles_exist(self):
        assert "frontier" in CAPABILITY_PROFILES
        assert "standard" in CAPABILITY_PROFILES
        assert "distilled" in CAPABILITY_PROFILES
        assert "edge" in CAPABILITY_PROFILES

    def test_frontier_best(self):
        f = CAPABILITY_PROFILES["frontier"]
        e = CAPABILITY_PROFILES["edge"]
        assert f.reliability_at_horizon(60) > e.reliability_at_horizon(60)


class TestComputeConstraints:
    def test_defaults(self):
        c = ComputeConstraints()
        assert c.total_capacity == 125_000.0
        assert c.allocated == 0.0

    def test_available(self):
        c = ComputeConstraints(total_capacity=100.0, allocated=30.0)
        assert c.available() == pytest.approx(70.0)

    def test_available_clamped(self):
        c = ComputeConstraints(total_capacity=10.0, allocated=20.0)
        assert c.available() == 0.0

    def test_utilization(self):
        c = ComputeConstraints(total_capacity=100.0, allocated=25.0)
        assert c.utilization() == pytest.approx(0.25)

    def test_can_allocate(self):
        c = ComputeConstraints(total_capacity=100.0, allocated=90.0)
        assert c.can_allocate(10.0)
        assert not c.can_allocate(11.0)

    def test_allocate_success(self):
        c = ComputeConstraints(total_capacity=100.0, allocated=0.0)
        assert c.allocate(50.0)
        assert c.allocated == pytest.approx(50.0)

    def test_allocate_fail(self):
        c = ComputeConstraints(total_capacity=100.0, allocated=95.0)
        assert not c.allocate(10.0)
        assert c.allocated == pytest.approx(95.0)

    def test_release(self):
        c = ComputeConstraints(total_capacity=100.0, allocated=50.0)
        c.release(20.0)
        assert c.allocated == pytest.approx(30.0)

    def test_release_clamped(self):
        c = ComputeConstraints(total_capacity=100.0, allocated=10.0)
        c.release(50.0)
        assert c.allocated == 0.0

    def test_max_concurrent_agents(self):
        c = ComputeConstraints(total_capacity=1000.0)
        p = AgentCapabilityProfile(compute_cost_per_minute=1.0)
        # 1000 / (1.0 * 60) = 16
        assert c.max_concurrent_agents(p, task_minutes=60) == 16

    def test_max_concurrent_agents_zero_cost(self):
        c = ComputeConstraints(total_capacity=1000.0)
        p = AgentCapabilityProfile(compute_cost_per_minute=0.0)
        assert c.max_concurrent_agents(p, task_minutes=60) == 0
