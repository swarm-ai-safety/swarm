"""Smoke tests for the graph-structural ROC benchmark harness."""

from __future__ import annotations

import pytest

from experiments.graph_structural_roc import (
    DETECTORS,
    apply_decision_rule,
    generate_benign,
    generate_collusion_ring,
    generate_sybil_cluster,
    generate_threshold_dancing,
    roc_auc,
    run_family,
    run_sweep,
)


class TestGenerators:
    def test_benign_no_plant(self):
        s = generate_benign(n_agents=20, edge_density=0.1, seed=0)
        assert s.planted == set()
        assert len(s.nodes) == 20

    def test_collusion_ring_plants_ring(self):
        s = generate_collusion_ring(n_agents=30, ring_size=4, seed=1)
        assert s.planted == {"r0", "r1", "r2", "r3"}

    def test_sybil_cluster_plants_sybils(self):
        s = generate_sybil_cluster(n_agents=30, cluster_size=4, seed=2)
        assert s.planted == {"s0", "s1", "s2", "s3"}

    def test_threshold_dancing_plants_cluster(self):
        s = generate_threshold_dancing(n_agents=30, cluster_size=4, seed=3)
        assert s.planted == {"c0", "c1", "c2", "c3"}


class TestROCAUC:
    def test_perfect_separation(self):
        # planted nodes get score 1.0, rest 0.0
        scores = {"p1": 1.0, "p2": 1.0, "n1": 0.0, "n2": 0.0}
        auc = roc_auc(scores, {"p1", "p2"}, list(scores))
        assert auc == pytest.approx(1.0)

    def test_random_scores(self):
        scores = {"p1": 0.5, "p2": 0.5, "n1": 0.5, "n2": 0.5}
        auc = roc_auc(scores, {"p1", "p2"}, list(scores))
        assert auc == pytest.approx(0.5)

    def test_no_positives_returns_half(self):
        scores = {"a": 0.9, "b": 0.1}
        assert roc_auc(scores, set(), list(scores)) == 0.5


class TestDetectorAdapters:
    @pytest.mark.parametrize("name", list(DETECTORS))
    def test_returns_score_per_node(self, name):
        s = generate_collusion_ring(n_agents=20, ring_size=3, seed=4)
        scores = DETECTORS[name](s)
        # every node should appear in the returned scores dict
        assert set(scores) == set(s.nodes)
        # scores should be in [0, 1] (loose check)
        for v in scores.values():
            assert 0.0 <= v <= 1.1  # allow slight overshoot from raw similarity


class TestRunFamily:
    def test_small_run_completes(self):
        results = run_family(
            generate_collusion_ring,
            "collusion_ring_small",
            replicates=2,
            base_params={"n_agents": 20, "ring_size": 3},
            seed_base=10,
        )
        assert set(results) == set(DETECTORS)
        for r in results.values():
            assert r.n_replicates == 2


class TestDecisionRule:
    def test_dominance_recommends_wiring(self):
        from experiments.graph_structural_roc import RunResult
        rs = [
            RunResult("f1", "graph_structural", 0.9, 0.85, 0.95, 10, []),
            RunResult("f1", "identity_jaccard", 0.6, 0.55, 0.65, 10, []),
            RunResult("f1", "reputation_mutual", 0.5, 0.45, 0.55, 10, []),
            RunResult("f1", "collusion_score", 0.7, 0.65, 0.75, 10, []),
        ]
        verdict = apply_decision_rule(rs)
        assert "strictly dominates" in verdict

    def test_tie_recommends_secondary_metric(self):
        from experiments.graph_structural_roc import RunResult
        rs = [
            RunResult("f1", "graph_structural", 0.7, 0.65, 0.75, 10, []),
            RunResult("f1", "identity_jaccard", 0.7, 0.65, 0.75, 10, []),
            RunResult("f1", "reputation_mutual", 0.7, 0.65, 0.75, 10, []),
            RunResult("f1", "collusion_score", 0.7, 0.65, 0.75, 10, []),
        ]
        verdict = apply_decision_rule(rs)
        assert "no governance wiring" in verdict.lower()


@pytest.mark.slow
class TestSweep:
    def test_full_sweep_smoke(self):
        results = run_sweep(replicates=2, seed_base=0)
        # 8 families * 4 detectors = 32 results
        assert len(results) == 8 * 4
