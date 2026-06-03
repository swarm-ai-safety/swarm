"""Graph-structural sybil/coordination detectors.

Existing detectors in swarm/ (identity.detect_sybil_clusters,
reputation_governor.detect_collusion_clusters, collusion.CollusionDetector)
score pairs/groups against a threshold. This module flags coalitions by
*topological* structure relative to a degree-preserving null model:

    * Densest subgraph extraction (Charikar 1/2-approx peeling).
    * k-core decomposition.
    * Reciprocity z-score vs configuration model.
    * Label-propagation community detection.
    * Configuration-model null sampler -> p-values, not magic thresholds.

Zero external dependency (matches swarm/analysis/network.py convention).
Inputs are a list of directed edges (weights accepted but currently used
only as a *positivity filter* — see ``DiGraph.from_edges``; densities and
reciprocity use edge presence, not weight). Edges are easily produced from
``SoftInteraction`` via ``edges_from_interactions``.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from swarm.models.interaction import SoftInteraction

Edge = Tuple[str, str, float]


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------


@dataclass
class DiGraph:
    """Minimal directed multigraph keyed by string node ids.

    Stores aggregated edge weight w(u, v) = sum of weights of all u->v edges.
    """

    nodes: Set[str] = field(default_factory=set)
    out: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    in_: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )

    @classmethod
    def from_edges(cls, edges: Iterable[Edge]) -> "DiGraph":
        g = cls()
        for u, v, w in edges:
            if u == v or w <= 0.0:
                # ignore self-loops and non-positive-weight edges (a p=0.0
                # SoftInteraction in p-weighted mode must not contribute
                # topology, since downstream metrics use edge presence)
                continue
            g.nodes.add(u)
            g.nodes.add(v)
            g.out[u][v] += w
            g.in_[v][u] += w
        return g

    def undirected_neighbors(self, u: str) -> Set[str]:
        return set(self.out.get(u, {})) | set(self.in_.get(u, {}))

    def undirected_degree(self, u: str) -> int:
        return len(self.undirected_neighbors(u))

    def induced_edge_count(self, subset: Set[str]) -> int:
        """Count directed edges with both endpoints in ``subset``."""
        n = 0
        for u in subset:
            for v in self.out.get(u, {}):
                if v in subset:
                    n += 1
        return n

    def reciprocity(self, subset: Optional[Set[str]] = None) -> float:
        """Fraction of directed edges (u, v) for which (v, u) also exists."""
        nodes = self.nodes if subset is None else subset
        total = 0
        reciprocated = 0
        for u in nodes:
            for v in self.out.get(u, {}):
                if subset is not None and v not in subset:
                    continue
                total += 1
                if u in self.out.get(v, {}):
                    reciprocated += 1
        return reciprocated / total if total else 0.0


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------


def edges_from_interactions(
    interactions: Sequence[SoftInteraction],
    *,
    weight: str = "count",
) -> List[Edge]:
    """Aggregate ``SoftInteraction`` records into weighted directed edges.

    Note: the structural detector currently uses edge *presence* (weights
    only act as a positivity filter in ``DiGraph.from_edges``), so the
    choice of weighting mode affects only which interactions register as
    an edge at all — e.g. a ``p=0.0`` interaction in ``"p"`` mode drops out
    entirely. The mode hook is in place for a future weighted variant.

    Args:
        interactions: list of SoftInteraction records.
        weight: one of ``"count"`` (1 per interaction), ``"p"`` (sum of p), or
            ``"mutual_benefit"`` (sum of accepted=True interactions only).
    """
    agg: Dict[Tuple[str, str], float] = defaultdict(float)
    for x in interactions:
        if not x.initiator or not x.counterparty:
            continue
        key = (x.initiator, x.counterparty)
        if weight == "count":
            agg[key] += 1.0
        elif weight == "p":
            agg[key] += x.p
        elif weight == "mutual_benefit":
            if x.accepted:
                agg[key] += 1.0
        else:
            raise ValueError(f"unknown weight mode: {weight!r}")
    return [(u, v, w) for (u, v), w in agg.items()]


# ---------------------------------------------------------------------------
# k-core
# ---------------------------------------------------------------------------


def k_core_decomposition(g: DiGraph) -> Dict[str, int]:
    """Coreness on the undirected projection (Batagelj-Zaversnik style peel).

    Not the strict O(m + n) bucket-array variant — degree updates here use a
    bounded backward scan rather than constant-time bucket pointers, so
    worst-case is super-linear. Fine for the graph sizes this detector runs on.
    """
    deg = {u: g.undirected_degree(u) for u in g.nodes}
    neigh = {u: g.undirected_neighbors(u) for u in g.nodes}
    # Tie-break by node id for determinism.
    order: List[str] = sorted(g.nodes, key=lambda u: (deg[u], u))
    pos = {u: i for i, u in enumerate(order)}
    core: Dict[str, int] = {}
    for u in order:
        core[u] = deg[u]
        for v in neigh[u]:
            if v in core:
                continue  # already peeled
            if deg[v] > deg[u]:
                dv = deg[v]
                # swap v with the first node of degree dv to keep order sorted
                first_idx = pos[v]
                # find the leftmost index with degree dv
                while first_idx > 0 and deg[order[first_idx - 1]] == dv:
                    first_idx -= 1
                w = order[first_idx]
                if w != v:
                    order[first_idx], order[pos[v]] = v, w
                    pos[v], pos[w] = first_idx, pos[v]
                deg[v] -= 1
    return core


# ---------------------------------------------------------------------------
# Densest subgraph (Charikar 1/2-approx)
# ---------------------------------------------------------------------------


def densest_subgraph(g: DiGraph) -> Tuple[Set[str], float]:
    """Charikar peeling: repeatedly remove the min-degree node, return the
    snapshot maximizing |E(S)| / |S| (directed edges, undirected support).
    """
    deg = {u: g.undirected_degree(u) for u in g.nodes}
    neigh = {u: set(g.undirected_neighbors(u)) for u in g.nodes}
    remaining = set(g.nodes)
    edge_count = g.induced_edge_count(remaining)

    best_set = set(remaining)
    best_density = edge_count / len(remaining) if remaining else 0.0

    while len(remaining) > 1:
        # pick min-degree node (tie-break by id for determinism)
        u = min(remaining, key=lambda x: (deg[x], x))
        # count directed edges incident to u within `remaining`
        removed = 0
        for v in g.out.get(u, {}):
            if v in remaining and v != u:
                removed += 1
        for v in g.in_.get(u, {}):
            if v in remaining and v != u:
                removed += 1
        edge_count -= removed
        remaining.discard(u)
        for v in neigh[u]:
            if v in remaining:
                deg[v] -= 1
        if remaining:
            density = edge_count / len(remaining)
            if density > best_density:
                best_density = density
                best_set = set(remaining)
    return best_set, best_density


# ---------------------------------------------------------------------------
# Label propagation
# ---------------------------------------------------------------------------


def label_propagation(
    g: DiGraph, *, max_iter: int = 30, seed: int = 0
) -> Dict[str, str]:
    """Asynchronous label propagation on the undirected projection.

    Returns ``node -> community_label``. Deterministic given ``seed``.
    """
    rng = random.Random(seed)
    labels: Dict[str, str] = {u: u for u in g.nodes}
    nodes = sorted(g.nodes)  # deterministic order; rng.shuffle reseeds each iter
    for _ in range(max_iter):
        rng.shuffle(nodes)
        changed = False
        for u in nodes:
            neigh = g.undirected_neighbors(u)
            if not neigh:
                continue
            counts: Dict[str, int] = defaultdict(int)
            for v in neigh:
                counts[labels[v]] += 1
            top = max(counts.values())
            candidates = sorted(lab for lab, c in counts.items() if c == top)
            new = candidates[0]  # deterministic tie-break
            if labels[u] != new:
                labels[u] = new
                changed = True
        if not changed:
            break
    return labels


# ---------------------------------------------------------------------------
# Configuration-model null
# ---------------------------------------------------------------------------


def configuration_model_null(g: DiGraph, *, seed: int = 0) -> DiGraph:
    """Generate a directed graph with the same in/out degree sequence.

    Stubs are randomly matched; self-loops and multi-edges are merged into
    weighted edges (matching DiGraph aggregation semantics).
    """
    rng = random.Random(seed)
    out_stubs: List[str] = []
    in_stubs: List[str] = []
    # Iterate in sorted order so the (seed, graph) pair is reproducible
    # across PYTHONHASHSEED values; raw set iteration is not.
    for u in sorted(g.nodes):
        out_stubs.extend([u] * sum(1 for _ in g.out.get(u, {})))
        in_stubs.extend([u] * sum(1 for _ in g.in_.get(u, {})))
    rng.shuffle(out_stubs)
    rng.shuffle(in_stubs)
    edges: List[Edge] = []
    for u, v in zip(out_stubs, in_stubs, strict=False):
        if u != v:
            edges.append((u, v, 1.0))
    return DiGraph.from_edges(edges)


def reciprocity_zscore(
    g: DiGraph,
    subset: Optional[Set[str]] = None,
    *,
    n_samples: int = 50,
    seed: int = 0,
) -> Tuple[float, float]:
    """Return (observed_reciprocity, z_score) for ``subset`` vs null model."""
    observed = g.reciprocity(subset)
    samples = []
    for i in range(n_samples):
        null = configuration_model_null(g, seed=seed + i)
        samples.append(null.reciprocity(subset if subset is None else subset & null.nodes))
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / len(samples)
    std = var ** 0.5
    # Floor std so a tight null (all zeros) against a positive observed
    # still scores as anomalous rather than collapsing to 0.
    z = (observed - mean) / max(std, 0.05)
    return observed, z


def density_pvalue(
    g: DiGraph,
    subset: Set[str],
    *,
    n_samples: int = 50,
    seed: int = 0,
) -> float:
    """Empirical p-value that *any* subset of a degree-matched null graph
    has internal density >= observed density of ``subset``.

    This is an upper bound on the size-matched p-value: we run Charikar
    peeling on each null sample and compare against its globally densest
    subgraph (any size), which is at least as dense as the best subgraph
    of size ``|subset|``. Conservative for ranking — it never under-reports
    significance — and avoids the cost of a size-conditioned search per
    sample. Replace with a size-matched scan if that conservatism matters.
    """
    if not subset:
        return 1.0
    observed_edges = g.induced_edge_count(subset)
    observed_density = observed_edges / len(subset)
    hits = 0
    for i in range(n_samples):
        null = configuration_model_null(g, seed=seed + i)
        # take densest subgraph of any size as upper bound
        _, d = densest_subgraph(null)
        if d >= observed_density:
            hits += 1
    return (hits + 1) / (n_samples + 1)


# ---------------------------------------------------------------------------
# Top-level detector
# ---------------------------------------------------------------------------


@dataclass
class StructuralAnomaly:
    """Suspicious subgraph flagged by structural signals."""

    members: Set[str]
    n_internal_edges: int
    density: float  # edges per node (directed)
    k_core: int  # min coreness within members
    reciprocity: float
    reciprocity_z: float
    modularity_label: Optional[str] = None
    pvalue: float = 1.0

    @property
    def is_suspicious(self) -> bool:
        # Pre-registered combination: dense AND reciprocal AND rare under null.
        return (
            len(self.members) >= 3
            and self.density >= 1.0
            and self.reciprocity_z >= 2.0
            and self.pvalue <= 0.05
        )


def detect_structural_anomalies(
    edges: Sequence[Edge],
    *,
    min_size: int = 3,
    n_null_samples: int = 50,
    seed: int = 0,
) -> List[StructuralAnomaly]:
    """End-to-end detector: build graph, run all four signals, return
    one anomaly record per candidate cluster.

    Candidates come from (a) Charikar densest subgraph and (b) each label-
    propagation community of size >= ``min_size``. Duplicates are merged.
    """
    g = DiGraph.from_edges(edges)
    if len(g.nodes) < min_size:
        return []

    coreness = k_core_decomposition(g)
    communities: Dict[str, Set[str]] = defaultdict(set)
    for node, node_label in label_propagation(g, seed=seed).items():
        communities[node_label].add(node)

    candidates: List[Set[str]] = []
    dense_set, _ = densest_subgraph(g)
    if len(dense_set) >= min_size:
        candidates.append(dense_set)
    for _lab, members in communities.items():
        if len(members) >= min_size and not any(members == c for c in candidates):
            candidates.append(members)

    results: List[StructuralAnomaly] = []
    for cand in candidates:
        n_edges = g.induced_edge_count(cand)
        density = n_edges / len(cand)
        rec, z = reciprocity_zscore(g, cand, n_samples=n_null_samples, seed=seed)
        pval = density_pvalue(g, cand, n_samples=n_null_samples, seed=seed)
        # find community label, if any
        lab: Optional[str] = next(
            (cl for cl, m in communities.items() if m == cand), None
        )
        results.append(
            StructuralAnomaly(
                members=cand,
                n_internal_edges=n_edges,
                density=density,
                k_core=min(coreness[u] for u in cand),
                reciprocity=rec,
                reciprocity_z=z,
                modularity_label=lab,
                pvalue=pval,
            )
        )
    return results
