"""Tests for work_regime_metrics — especially compute_coalition_strength."""

from datetime import datetime
from typing import List, Tuple

import pytest

from swarm.metrics.work_regime_metrics import (
    WorkRegimeMetrics,
    compute_coalition_strength,
    compute_work_regime_metrics,
)
from swarm.models.interaction import InteractionType, SoftInteraction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pairs(*edges: Tuple[str, str]) -> List[Tuple[str, str]]:
    return list(edges)


def make_interaction(initiator: str, counterparty: str) -> SoftInteraction:
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        interaction_type=InteractionType.COLLABORATION,
        accepted=True,
        p=0.8,
        timestamp=datetime(2024, 1, 1),
    )


# ---------------------------------------------------------------------------
# compute_coalition_strength — edge cases
# ---------------------------------------------------------------------------


class TestComputeCoalitionStrengthEdgeCases:
    def test_empty_pairs_returns_zero(self):
        assert compute_coalition_strength([], ["a", "b", "c"]) == 0.0

    def test_fewer_than_three_agents_returns_zero(self):
        assert compute_coalition_strength([("a", "b")], ["a", "b"]) == 0.0

    def test_no_node_has_degree_two_returns_zero(self):
        # Star topology: centre has degree 3, leaves degree 1
        pairs = _pairs(("centre", "a"), ("centre", "b"), ("centre", "c"))
        agents = ["centre", "a", "b", "c"]
        # Leaves can't form triangles; centre can but a-b, a-c, b-c edges missing
        result = compute_coalition_strength(pairs, agents)
        assert result == 0.0

    def test_self_loops_ignored(self):
        # Self-loop pairs should not be added to the adjacency list
        pairs = _pairs(("a", "a"), ("b", "b"), ("a", "b"), ("b", "c"), ("a", "c"))
        agents = ["a", "b", "c"]
        result = compute_coalition_strength(pairs, agents)
        assert 0.0 <= result <= 1.0

    def test_pairs_outside_agent_ids_ignored(self):
        pairs = _pairs(("x", "y"), ("a", "b"), ("b", "c"), ("a", "c"))
        agents = ["a", "b", "c"]  # x, y not in agent_ids
        result = compute_coalition_strength(pairs, agents)
        # x-y edge ignored; a-b-c triangle should yield perfect clustering
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_coalition_strength — correctness
# ---------------------------------------------------------------------------


class TestComputeCoalitionStrengthCorrectness:
    def test_complete_triangle_returns_one(self):
        """Three agents all connected → clustering = 1.0."""
        pairs = _pairs(("a", "b"), ("b", "c"), ("a", "c"))
        result = compute_coalition_strength(pairs, ["a", "b", "c"])
        assert result == pytest.approx(1.0)

    def test_complete_graph_k4_returns_one(self):
        """K4 complete graph: every node's neighbours are all connected."""
        agents = ["a", "b", "c", "d"]
        pairs = _pairs(
            ("a", "b"), ("a", "c"), ("a", "d"),
            ("b", "c"), ("b", "d"),
            ("c", "d"),
        )
        result = compute_coalition_strength(pairs, agents)
        assert result == pytest.approx(1.0)

    def test_line_graph_returns_zero(self):
        """a-b-c-d line: no node has two neighbours that are also connected."""
        pairs = _pairs(("a", "b"), ("b", "c"), ("c", "d"))
        result = compute_coalition_strength(pairs, ["a", "b", "c", "d"])
        # b and c have degree 2 but their neighbours are not connected to each other
        assert result == pytest.approx(0.0)

    def test_result_in_unit_interval(self):
        """Coalition strength is always in [0, 1]."""
        pairs = _pairs(
            ("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"),
            ("a", "c"),  # one diagonal only
        )
        result = compute_coalition_strength(pairs, ["a", "b", "c", "d"])
        assert 0.0 <= result <= 1.0

    def test_duplicate_pairs_handled(self):
        """Repeated edges don't inflate the clustering coefficient."""
        pairs = _pairs(
            ("a", "b"), ("a", "b"), ("b", "c"), ("a", "c"),
        )
        result_dup = compute_coalition_strength(pairs, ["a", "b", "c"])
        pairs_dedup = _pairs(("a", "b"), ("b", "c"), ("a", "c"))
        result_dedup = compute_coalition_strength(pairs_dedup, ["a", "b", "c"])
        assert result_dup == pytest.approx(result_dedup)

    def test_directed_pairs_treated_as_undirected(self):
        """(a,b) and (b,a) should produce the same graph as just (a,b)."""
        pairs_bi = _pairs(("a", "b"), ("b", "a"), ("b", "c"), ("a", "c"))
        pairs_uni = _pairs(("a", "b"), ("b", "c"), ("a", "c"))
        result_bi = compute_coalition_strength(pairs_bi, ["a", "b", "c"])
        result_uni = compute_coalition_strength(pairs_uni, ["a", "b", "c"])
        assert result_bi == pytest.approx(result_uni)


# ---------------------------------------------------------------------------
# compute_coalition_strength — sampling
# ---------------------------------------------------------------------------


class TestComputeCoalitionStrengthSampling:
    def _build_clique(self, n: int) -> Tuple[List[str], List[Tuple[str, str]]]:
        agents = [f"a{i}" for i in range(n)]
        pairs = [
            (agents[i], agents[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        return agents, pairs

    def test_sampling_triggered_for_large_network(self):
        """When n_eligible > max_sample_nodes the function still returns a value."""
        agents, pairs = self._build_clique(60)
        result = compute_coalition_strength(pairs, agents, max_sample_nodes=10)
        # K60 is complete, so clustering == 1.0 regardless of which nodes sampled
        assert result == pytest.approx(1.0)

    def test_no_sampling_when_within_limit(self):
        agents, pairs = self._build_clique(10)
        result = compute_coalition_strength(pairs, agents, max_sample_nodes=500)
        assert result == pytest.approx(1.0)

    def test_sampling_is_deterministic(self):
        """Same inputs → same result (no hidden randomness)."""
        agents, pairs = self._build_clique(40)
        r1 = compute_coalition_strength(pairs, agents, max_sample_nodes=5)
        r2 = compute_coalition_strength(pairs, agents, max_sample_nodes=5)
        assert r1 == r2

    def test_max_sample_nodes_one_does_not_crash(self):
        agents, pairs = self._build_clique(20)
        result = compute_coalition_strength(pairs, agents, max_sample_nodes=1)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_work_regime_metrics
# ---------------------------------------------------------------------------


class TestComputeWorkRegimeMetrics:
    def test_empty_interactions_returns_defaults(self):
        result = compute_work_regime_metrics([])
        assert isinstance(result, WorkRegimeMetrics)
        assert result.coalition_strength == 0.0
        assert result.regime_label == "neutral"

    def test_cooperative_label_for_dense_network(self):
        agents = ["a", "b", "c", "d"]
        interactions = [
            make_interaction("a", "b"),
            make_interaction("b", "c"),
            make_interaction("a", "c"),
            make_interaction("b", "d"),
            make_interaction("a", "d"),
            make_interaction("c", "d"),
        ]
        result = compute_work_regime_metrics(interactions, agent_ids=agents)
        assert result.regime_label == "cooperative"
        assert result.coalition_strength >= 0.4

    def test_competitive_label_for_sparse_network(self):
        agents = ["a", "b", "c", "d", "e"]
        # Line graph: a-b-c-d-e, no triangles
        interactions = [
            make_interaction("a", "b"),
            make_interaction("b", "c"),
            make_interaction("c", "d"),
            make_interaction("d", "e"),
        ]
        result = compute_work_regime_metrics(
            interactions,
            agent_ids=agents,
            competitive_threshold=0.05,
        )
        assert result.coalition_strength == pytest.approx(0.0)
        assert result.regime_label == "competitive"

    def test_inferred_agent_ids(self):
        interactions = [
            make_interaction("x", "y"),
            make_interaction("y", "z"),
            make_interaction("x", "z"),
        ]
        result = compute_work_regime_metrics(interactions)  # no agent_ids
        assert result.coalition_strength == pytest.approx(1.0)

    def test_per_agent_clustering_populated(self):
        agents = ["a", "b", "c"]
        interactions = [
            make_interaction("a", "b"),
            make_interaction("b", "c"),
            make_interaction("a", "c"),
        ]
        result = compute_work_regime_metrics(interactions, agent_ids=agents)
        # All three agents form a triangle — each should have cc == 1.0
        for agent, cc in result.per_agent_clustering.items():
            assert cc == pytest.approx(1.0), f"agent {agent} has cc={cc}"

    def test_n_nodes_sampled_at_most_max_sample_nodes(self):
        n = 30
        agents = [f"a{i}" for i in range(n)]
        interactions = [
            make_interaction(agents[i], agents[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        result = compute_work_regime_metrics(
            interactions, agent_ids=agents, max_sample_nodes=5
        )
        assert result.n_nodes_sampled <= 5
        assert result.n_eligible_nodes == n
