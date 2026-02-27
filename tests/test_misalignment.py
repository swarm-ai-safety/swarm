"""Tests for the misalignment module (Kierans et al. framework)."""

import math
import random

import pytest

from swarm.metrics.misalignment import (
    DistanceMetric,
    IssueSpace,
    MisalignmentModule,
    MisalignmentProfile,
    WeightAggregation,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def two_issue_space():
    """Simple 2-issue space for testing."""
    return IssueSpace(issues=["safety", "efficiency"])


@pytest.fixture
def three_issue_space():
    """3-issue space for clustering tests."""
    return IssueSpace(issues=["safety", "efficiency", "fairness"])


@pytest.fixture
def module(two_issue_space):
    """Module with 2-issue space."""
    return MisalignmentModule(issue_space=two_issue_space)


# ======================================================================
# MisalignmentProfile validation
# ======================================================================


class TestMisalignmentProfile:
    def test_valid_profile(self):
        p = MisalignmentProfile("a1", [0.5, -0.3], [0.6, 0.4])
        assert p.agent_id == "a1"
        assert p.prefs == [0.5, -0.3]
        assert abs(sum(p.salience) - 1.0) < 1e-9

    def test_auto_normalizes_salience(self):
        p = MisalignmentProfile("a1", [0.0, 0.0], [3.0, 7.0])
        assert abs(p.salience[0] - 0.3) < 1e-9
        assert abs(p.salience[1] - 0.7) < 1e-9

    def test_prefs_out_of_range(self):
        with pytest.raises(ValueError, match="not in"):
            MisalignmentProfile("a1", [1.5, 0.0], [0.5, 0.5])

    def test_negative_salience(self):
        with pytest.raises(ValueError, match="negative"):
            MisalignmentProfile("a1", [0.0, 0.0], [-0.5, 0.5])

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            MisalignmentProfile("a1", [0.0, 0.0, 0.0], [0.5, 0.5])


# ======================================================================
# Pairwise misalignment
# ======================================================================


class TestPairwiseMisalignment:
    def test_zero_when_identical(self, module):
        module.register_agent("a", [0.5, 0.5], [0.5, 0.5])
        module.register_agent("b", [0.5, 0.5], [0.5, 0.5])
        assert module.pairwise_misalignment("a", "b") == 0.0

    def test_symmetric(self, module):
        module.register_agent("a", [1.0, -1.0], [0.5, 0.5])
        module.register_agent("b", [-1.0, 1.0], [0.5, 0.5])
        m_ab = module.pairwise_misalignment("a", "b")
        m_ba = module.pairwise_misalignment("b", "a")
        assert abs(m_ab - m_ba) < 1e-12

    def test_maximal_disagreement_l1(self, module):
        """Max L1 distance with uniform salience on 2 issues: 2 * 0.5 = 1.0."""
        module.register_agent("a", [1.0, 1.0], [0.5, 0.5])
        module.register_agent("b", [-1.0, -1.0], [0.5, 0.5])
        m = module.pairwise_misalignment("a", "b")
        # Each issue contributes 0.5 * 2.0 = 1.0, total = 2.0
        assert abs(m - 2.0) < 1e-9

    def test_salience_weighting(self, module):
        """High salience on disagreed issue amplifies misalignment."""
        module.register_agent("a", [1.0, 0.0], [0.9, 0.1])
        module.register_agent("b", [-1.0, 0.0], [0.9, 0.1])
        m_weighted = module.pairwise_misalignment("a", "b")

        module.register_agent("c", [1.0, 0.0], [0.1, 0.9])
        module.register_agent("d", [-1.0, 0.0], [0.1, 0.9])
        m_low_sal = module.pairwise_misalignment("c", "d")

        # Same position difference but different salience -> different M
        assert m_weighted > m_low_sal

    def test_monotonic_in_distance(self, module):
        """Increasing preference distance increases misalignment."""
        module.register_agent("a", [0.0, 0.0], [0.5, 0.5])
        vals = []
        for delta in [0.2, 0.5, 0.8, 1.0]:
            module.register_agent(f"b_{delta}", [delta, delta], [0.5, 0.5])
            vals.append(module.pairwise_misalignment("a", f"b_{delta}"))
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]

    def test_l2_distance(self, two_issue_space):
        two_issue_space.distance = DistanceMetric.L2
        mod = MisalignmentModule(issue_space=two_issue_space)
        mod.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        mod.register_agent("b", [0.0, 1.0], [0.5, 0.5])
        m = mod.pairwise_misalignment("a", "b")
        # sqrt(0.5 * 1^2 + 0.5 * 1^2) = sqrt(1.0) = 1.0
        assert abs(m - 1.0) < 1e-9

    def test_issue_contributions(self, module):
        module.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [-1.0, 0.0], [0.5, 0.5])
        contribs = module.pairwise_issue_contributions("a", "b")
        # Issue 0: 0.5 * 2.0 = 1.0, Issue 1: 0.5 * 0.0 = 0.0
        assert abs(contribs[0] - 1.0) < 1e-9
        assert abs(contribs[1] - 0.0) < 1e-9


# ======================================================================
# Weight aggregation modes
# ======================================================================


class TestWeightAggregation:
    def test_min_aggregation(self, two_issue_space):
        two_issue_space.weight_agg = WeightAggregation.MIN
        mod = MisalignmentModule(issue_space=two_issue_space)
        mod.register_agent("a", [1.0, 0.0], [0.9, 0.1])
        mod.register_agent("b", [-1.0, 0.0], [0.1, 0.9])
        m = mod.pairwise_misalignment("a", "b")
        # Issue 0: min(0.9, 0.1) * 2 = 0.2, Issue 1: min(0.1, 0.9) * 0 = 0
        assert abs(m - 0.2) < 1e-9

    def test_max_aggregation(self, two_issue_space):
        two_issue_space.weight_agg = WeightAggregation.MAX
        mod = MisalignmentModule(issue_space=two_issue_space)
        mod.register_agent("a", [1.0, 0.0], [0.9, 0.1])
        mod.register_agent("b", [-1.0, 0.0], [0.1, 0.9])
        m = mod.pairwise_misalignment("a", "b")
        # Issue 0: max(0.9, 0.1) * 2 = 1.8
        assert abs(m - 1.8) < 1e-9

    def test_geom_mean_aggregation(self, two_issue_space):
        two_issue_space.weight_agg = WeightAggregation.GEOM_MEAN
        mod = MisalignmentModule(issue_space=two_issue_space)
        mod.register_agent("a", [1.0, 0.0], [0.9, 0.1])
        mod.register_agent("b", [-1.0, 0.0], [0.1, 0.9])
        m = mod.pairwise_misalignment("a", "b")
        expected = math.sqrt(0.9 * 0.1) * 2.0  # ~0.6
        assert abs(m - expected) < 1e-9


# ======================================================================
# Global misalignment
# ======================================================================


class TestGlobalMisalignment:
    def test_homogeneous_population(self, module):
        for i in range(5):
            module.register_agent(f"a{i}", [0.5, 0.5], [0.5, 0.5])
        assert module.global_misalignment() == 0.0

    def test_two_agent_equals_pairwise(self, module):
        module.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [0.0, 1.0], [0.5, 0.5])
        assert abs(
            module.global_misalignment()
            - module.pairwise_misalignment("a", "b")
        ) < 1e-12

    def test_fewer_than_two(self, module):
        module.register_agent("a", [0.0, 0.0], [0.5, 0.5])
        assert module.global_misalignment() == 0.0

    def test_subset(self, module):
        module.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [-1.0, 0.0], [0.5, 0.5])
        module.register_agent("c", [1.0, 0.0], [0.5, 0.5])
        # Subset of a, c should be 0 (identical)
        assert module.global_misalignment(["a", "c"]) == 0.0

    def test_global_issue_contributions_sum(self, module):
        module.register_agent("a", [1.0, -0.5], [0.5, 0.5])
        module.register_agent("b", [-1.0, 0.5], [0.5, 0.5])
        contribs = module.global_issue_contributions()
        total = sum(contribs.values())
        assert abs(total - module.global_misalignment()) < 1e-9


# ======================================================================
# Local misalignment
# ======================================================================


class TestLocalMisalignment:
    def test_no_neighbors(self, module):
        module.register_agent("a", [0.0, 0.0], [0.5, 0.5])
        assert module.local_misalignment("a", []) == 0.0

    def test_single_neighbor(self, module):
        module.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [0.0, 0.0], [0.5, 0.5])
        m_local = module.local_misalignment("a", ["b"])
        m_pair = module.pairwise_misalignment("a", "b")
        assert abs(m_local - m_pair) < 1e-12

    def test_weighted_edges(self, module):
        module.register_agent("a", [0.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("c", [-1.0, 0.0], [0.5, 0.5])
        # Heavily weight b -> local misalignment biased toward M(a,b)
        m = module.local_misalignment("a", ["b", "c"], {"b": 10.0, "c": 1.0})
        m_ab = module.pairwise_misalignment("a", "b")
        m_ac = module.pairwise_misalignment("a", "c")
        expected = (10.0 * m_ab + 1.0 * m_ac) / 11.0
        assert abs(m - expected) < 1e-9

    def test_all_local(self, module):
        module.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [0.0, 0.0], [0.5, 0.5])
        module.register_agent("c", [-1.0, 0.0], [0.5, 0.5])
        graph = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
        result = module.all_local_misalignment(graph)
        assert set(result.keys()) == {"a", "b", "c"}
        # b has neighbors a and c with equal weight
        m_ba = module.pairwise_misalignment("b", "a")
        m_bc = module.pairwise_misalignment("b", "c")
        assert abs(result["b"] - (m_ba + m_bc) / 2) < 1e-9


# ======================================================================
# Governance-adjusted misalignment
# ======================================================================


class TestEffectiveMisalignment:
    def test_zero_pressure(self, module):
        module.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [-1.0, 0.0], [0.5, 0.5])
        raw = module.pairwise_misalignment("a", "b")
        eff = module.effective_misalignment("a", "b", 0.0)
        assert abs(eff - raw) < 1e-12

    def test_full_suppression(self, module):
        module.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [-1.0, 0.0], [0.5, 0.5])
        raw = module.pairwise_misalignment("a", "b")
        eff = module.effective_misalignment("a", "b", raw + 1.0)
        assert eff == 0.0

    def test_partial_reduction(self, module):
        module.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [-1.0, 0.0], [0.5, 0.5])
        raw = module.pairwise_misalignment("a", "b")
        eff = module.effective_misalignment("a", "b", raw / 2)
        assert abs(eff - raw / 2) < 1e-9

    def test_global_effective_uniform(self, module):
        module.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [-1.0, 0.0], [0.5, 0.5])
        raw = module.global_misalignment()
        eff = module.global_effective_misalignment(uniform_pressure=0.0)
        assert abs(eff - raw) < 1e-12

    def test_gov_lambda_scaling(self, two_issue_space):
        mod = MisalignmentModule(issue_space=two_issue_space, gov_lambda=2.0)
        mod.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        mod.register_agent("b", [-1.0, 0.0], [0.5, 0.5])
        raw = mod.pairwise_misalignment("a", "b")
        eff = mod.effective_misalignment("a", "b", 0.1)
        assert abs(eff - (raw - 2.0 * 0.1)) < 1e-9


# ======================================================================
# Sampled misalignment
# ======================================================================


class TestSampledMisalignment:
    def test_convergence(self, two_issue_space):
        mod = MisalignmentModule(issue_space=two_issue_space)
        rng = random.Random(42)
        for i in range(50):
            prefs = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
            mod.register_agent(f"a{i}", prefs, [0.5, 0.5])

        exact = mod.global_misalignment()
        sampled = mod.sampled_global_misalignment(k=5000, rng=random.Random(99))
        # Within 10% of exact
        assert abs(sampled - exact) / exact < 0.1


# ======================================================================
# Polarization & fragmentation
# ======================================================================


class TestDiagnostics:
    def test_polarization_high_for_separated_clusters(self, three_issue_space):
        mod = MisalignmentModule(issue_space=three_issue_space)
        # Cluster 1: all at [1, 1, 1]
        for i in range(5):
            mod.register_agent(f"pos{i}", [1.0, 1.0, 1.0], [1 / 3] * 3)
        # Cluster 2: all at [-1, -1, -1]
        for i in range(5):
            mod.register_agent(f"neg{i}", [-1.0, -1.0, -1.0], [1 / 3] * 3)
        pol = mod.polarization_index(n_clusters=2)
        assert pol > 1.0  # Between-cluster >> within-cluster

    def test_polarization_low_for_homogeneous(self, three_issue_space):
        mod = MisalignmentModule(issue_space=three_issue_space)
        for i in range(10):
            mod.register_agent(f"a{i}", [0.5, 0.5, 0.5], [1 / 3] * 3)
        pol = mod.polarization_index(n_clusters=2)
        assert pol == 0.0  # All identical

    def test_fragmentation_max_for_equal_clusters(self, three_issue_space):
        mod = MisalignmentModule(issue_space=three_issue_space)
        # 3 equal-size, well-separated clusters
        for i in range(3):
            mod.register_agent(f"a{i}", [1.0, 0.0, 0.0], [1 / 3] * 3)
        for i in range(3):
            mod.register_agent(f"b{i}", [0.0, 1.0, 0.0], [1 / 3] * 3)
        for i in range(3):
            mod.register_agent(f"c{i}", [0.0, 0.0, 1.0], [1 / 3] * 3)
        frag = mod.fragmentation_index(n_clusters=3)
        assert frag > 0.9  # Near-perfect entropy


# ======================================================================
# Snapshot
# ======================================================================


class TestSnapshot:
    def test_snapshot_basic(self, module):
        module.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [-1.0, 0.0], [0.5, 0.5])
        snap = module.compute_snapshot(step=0)
        assert snap.step == 0
        assert snap.m_pref_global > 0
        assert snap.m_eff_global == snap.m_pref_global  # No governance
        assert "safety" in snap.issue_contributions
        assert "efficiency" in snap.issue_contributions

    def test_snapshot_with_graph(self, module):
        module.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [-1.0, 0.0], [0.5, 0.5])
        module.register_agent("c", [0.0, 0.0], [0.5, 0.5])
        graph = {"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"]}
        snap = module.compute_snapshot(step=5, graph=graph)
        assert len(snap.local) == 3
        assert "a" in snap.local

    def test_snapshot_governance_reduces_eff(self, module):
        module.register_agent("a", [1.0, 0.0], [0.5, 0.5])
        module.register_agent("b", [-1.0, 0.0], [0.5, 0.5])
        snap = module.compute_snapshot(step=0, uniform_pressure=0.3)
        assert snap.m_eff_global < snap.m_pref_global

    def test_snapshot_alert_on_spike(self, module):
        module.register_agent("a", [1.0, 1.0], [0.5, 0.5])
        module.register_agent("b", [-1.0, -1.0], [0.5, 0.5])
        snap = module.compute_snapshot(step=0)
        # M_pref should be 2.0, triggering spike alert
        assert any("spike" in a for a in snap.alerts)

    def test_snapshot_to_dict(self, module):
        module.register_agent("a", [0.5, 0.0], [0.5, 0.5])
        module.register_agent("b", [-0.5, 0.0], [0.5, 0.5])
        snap = module.compute_snapshot(step=1)
        d = snap.to_dict()
        assert d["step"] == 1
        assert "M_pref" in d["global"]
        assert "M_eff" in d["global"]


# ======================================================================
# Registration & update
# ======================================================================


class TestRegistration:
    def test_register_and_retrieve(self, module):
        module.register_agent("a", [0.5, -0.5], [0.5, 0.5])
        assert "a" in module.profiles
        assert module.profiles["a"].prefs == [0.5, -0.5]

    def test_update_prefs(self, module):
        module.register_agent("a", [0.5, -0.5], [0.5, 0.5])
        module.update_agent("a", prefs=[0.0, 0.0])
        assert module.profiles["a"].prefs == [0.0, 0.0]

    def test_update_nonexistent_raises(self, module):
        with pytest.raises(KeyError):
            module.update_agent("nonexistent", prefs=[0.0, 0.0])

    def test_wrong_dimension_raises(self, module):
        with pytest.raises(ValueError):
            module.register_agent("a", [0.0], [1.0])

    def test_overwrite_via_register(self, module):
        module.register_agent("a", [0.5, 0.5], [0.5, 0.5])
        module.register_agent("a", [1.0, 1.0], [0.5, 0.5])
        assert module.profiles["a"].prefs == [1.0, 1.0]
