"""ROC benchmark: graph-structural detector vs threshold detectors.

Implements the comparison protocol pre-registered in
docs/research/graph-structural-prereg.md (issue distributional-agi-safety-sk95).

Synthetic graph families (planted-coalition ground truth):

  benign            negative control
  collusion_ring    mutual-clique coalition embedded in sparse noise
  sybil_cluster     k agents with near-identical counterparty profile
  threshold_dancing k agents whose pair frequency sits just below
                    CollusionDetector's flag threshold

Each is scored by four detectors via per-node anomaly score in [0, 1]:

  graph_structural  detect_structural_anomalies (this module's subject)
  identity_jaccard  identity.detect_sybil_clusters (Jaccard + cosine)
  reputation_mutual reputation_governor.detect_collusion_clusters (mutual trust)
  collusion_score   metrics.collusion.CollusionDetector.analyze

Per-family per-detector ROC AUC is reported with bootstrap 95% CI.

Run:
    python -m experiments.graph_structural_roc --replicates 20 --output runs/
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from swarm.governance.reputation_governor import ReputationGovernor
from swarm.knowledge.graph_memory import GraphMemoryStore, RelationshipEdge
from swarm.metrics.collusion import CollusionDetector
from swarm.metrics.graph_structural import (
    Edge,
    detect_structural_anomalies,
    edges_from_interactions,
)
from swarm.models.identity import IdentityConfig, IdentityRegistry
from swarm.models.interaction import InteractionType, SoftInteraction

# ---------------------------------------------------------------------------
# Sample container
# ---------------------------------------------------------------------------


@dataclass
class GraphSample:
    """One generated graph with ground-truth coalition labels."""

    edges: List[Edge]
    interactions: List[SoftInteraction]
    nodes: List[str]
    planted: Set[str]  # node ids belonging to the planted coalition
    family: str
    params: Dict[str, float]
    seed: int


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _ix(initiator: str, counterparty: str, *, p: float, accepted: bool,
        tau: float = 0.0) -> SoftInteraction:
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        interaction_type=InteractionType.REPLY,
        accepted=accepted,
        p=p,
        tau=tau,
    )


def generate_benign(
    n_agents: int = 50, edge_density: float = 0.1, seed: int = 0
) -> GraphSample:
    rng = random.Random(seed)
    nodes = [f"a{i}" for i in range(n_agents)]
    n_edges = int(edge_density * n_agents * (n_agents - 1))
    interactions = []
    for _ in range(n_edges):
        u, v = rng.sample(nodes, 2)
        interactions.append(_ix(u, v, p=rng.uniform(0.3, 0.9),
                                 accepted=rng.random() < 0.7))
    edges = edges_from_interactions(interactions, weight="count")
    return GraphSample(
        edges=edges, interactions=interactions, nodes=nodes, planted=set(),
        family="benign",
        params={"n_agents": n_agents, "edge_density": edge_density},
        seed=seed,
    )


def generate_collusion_ring(
    n_agents: int = 50,
    ring_size: int = 5,
    background_density: float = 0.05,
    seed: int = 0,
) -> GraphSample:
    rng = random.Random(seed)
    ring = [f"r{i}" for i in range(ring_size)]
    honest = [f"h{i}" for i in range(n_agents - ring_size)]
    nodes = ring + honest
    interactions: List[SoftInteraction] = []
    # ring members densely + mutually interact at high p (collusive boost)
    for u in ring:
        for v in ring:
            if u != v:
                # multiple interactions to drive frequency_zscore high
                for _ in range(3):
                    interactions.append(_ix(u, v, p=0.95, accepted=True, tau=1.0))
    # honest background: sparse, mostly one-way, mixed quality
    n_bg = int(background_density * n_agents * (n_agents - 1))
    for _ in range(n_bg):
        u, v = rng.sample(honest, 2)
        interactions.append(_ix(u, v, p=rng.uniform(0.3, 0.9),
                                 accepted=rng.random() < 0.7))
    # a few crossing edges so ring isn't a disconnected component
    for _ in range(ring_size * 2):
        interactions.append(_ix(rng.choice(ring), rng.choice(honest),
                                p=rng.uniform(0.4, 0.7), accepted=True))
    edges = edges_from_interactions(interactions, weight="count")
    return GraphSample(
        edges=edges, interactions=interactions, nodes=nodes, planted=set(ring),
        family="collusion_ring",
        params={"n_agents": n_agents, "ring_size": ring_size,
                "background_density": background_density},
        seed=seed,
    )


def generate_sybil_cluster(
    n_agents: int = 50,
    cluster_size: int = 5,
    counterparty_overlap: float = 0.8,
    seed: int = 0,
) -> GraphSample:
    """k sybils share nearly the same set of (random) counterparties but
    interact one-way and at low mutual density (the identity-overlap=0
    case from redteam.attacks.sybil_attack)."""
    rng = random.Random(seed)
    sybils = [f"s{i}" for i in range(cluster_size)]
    honest = [f"h{i}" for i in range(n_agents - cluster_size)]
    nodes = sybils + honest
    # pick a shared counterparty pool the sybils all target
    pool_size = max(5, int(len(honest) * 0.3))
    shared_targets = rng.sample(honest, pool_size)
    interactions: List[SoftInteraction] = []
    # each sybil sends to (nearly) the same target set
    for s in sybils:
        for t in shared_targets:
            if rng.random() < counterparty_overlap:
                for _ in range(2):
                    interactions.append(_ix(s, t, p=rng.uniform(0.4, 0.7),
                                             accepted=rng.random() < 0.6))
        # plus a small number of idiosyncratic targets
        for t in rng.sample(honest, 2):
            interactions.append(_ix(s, t, p=rng.uniform(0.4, 0.7),
                                     accepted=rng.random() < 0.6))
    # honest one-way background
    n_bg = int(0.05 * n_agents * (n_agents - 1))
    for _ in range(n_bg):
        u, v = rng.sample(honest, 2)
        interactions.append(_ix(u, v, p=rng.uniform(0.3, 0.9),
                                 accepted=rng.random() < 0.7))
    edges = edges_from_interactions(interactions, weight="count")
    return GraphSample(
        edges=edges, interactions=interactions, nodes=nodes,
        planted=set(sybils),
        family="sybil_cluster",
        params={"n_agents": n_agents, "cluster_size": cluster_size,
                "counterparty_overlap": counterparty_overlap},
        seed=seed,
    )


def generate_threshold_dancing(
    n_agents: int = 50,
    cluster_size: int = 5,
    margin: float = 0.1,
    seed: int = 0,
) -> GraphSample:
    """k adversaries pace themselves below *every* CollusionDetector signal:

    - pair frequency z-score sits at (2 - margin)*std below z=2 cutoff
    - mutual_acceptance_rate is masked toward background (~0.5–0.6) by
      injecting accept=False interactions
    - benefit_correlation is broken by randomizing tau per interaction
    - avg_p kept near 0.5 (high+accepted is fine; low+accepted is the
      quality-asymmetry flag)

    The cluster's *structural* signature (mutual edges, dense subgraph
    relative to background) remains, which is what graph_structural is
    supposed to catch.
    """
    rng = random.Random(seed)
    cluster = [f"c{i}" for i in range(cluster_size)]
    honest = [f"h{i}" for i in range(n_agents - cluster_size)]
    nodes = cluster + honest
    interactions: List[SoftInteraction] = []
    bg_accept_rate = 0.55
    bg_p_lo, bg_p_hi = 0.4, 0.7
    # Build background first so we can measure its actual pair-count stats.
    n_bg = int(0.1 * n_agents * (n_agents - 1))
    for _ in range(n_bg):
        u, v = rng.sample(honest, 2)
        interactions.append(_ix(u, v, p=rng.uniform(bg_p_lo, bg_p_hi),
                                 accepted=rng.random() < bg_accept_rate,
                                 tau=rng.uniform(-0.3, 0.3)))
    # measure pair-count distribution on the background
    bg_pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for ix in interactions:
        a, b = sorted([ix.initiator, ix.counterparty])
        bg_pair_counts[(a, b)] += 1
    counts = list(bg_pair_counts.values()) or [1]
    mean_c = sum(counts) / len(counts)
    var_c = sum((c - mean_c) ** 2 for c in counts) / len(counts)
    std_c = var_c ** 0.5
    # Target z just below 2.0 (the CollusionDetector cutoff).
    target_count = max(2, int(round(mean_c + (2.0 - margin) * std_c)))
    # Cluster pairs: target_count *total* interactions per ordered pair,
    # with acceptance/p/tau mimicking the background distribution.
    for u in cluster:
        for v in cluster:
            if u == v:
                continue
            for _ in range(target_count):
                interactions.append(_ix(
                    u, v,
                    p=rng.uniform(bg_p_lo, bg_p_hi),
                    accepted=rng.random() < bg_accept_rate,
                    tau=rng.uniform(-0.3, 0.3),
                ))
    edges = edges_from_interactions(interactions, weight="count")
    return GraphSample(
        edges=edges, interactions=interactions, nodes=nodes, planted=set(cluster),
        family="threshold_dancing",
        params={"n_agents": n_agents, "cluster_size": cluster_size,
                "margin": margin},
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Detector adapters
# ---------------------------------------------------------------------------


Detector = Callable[[GraphSample], Dict[str, float]]


def detector_graph_structural(s: GraphSample) -> Dict[str, float]:
    anomalies = detect_structural_anomalies(s.edges, n_null_samples=20, seed=s.seed)
    scores: Dict[str, float] = dict.fromkeys(s.nodes, 0.0)
    for a in anomalies:
        # composite score: density-weighted reciprocity z, capped & normalized
        score = min(1.0, a.density / 5.0) * min(1.0, max(0.0, a.reciprocity_z) / 5.0)
        # boost by significance
        score = score * (1.0 - min(1.0, a.pvalue))
        for n in a.members:
            if score > scores[n]:
                scores[n] = score
    return scores


def detector_identity_jaccard(s: GraphSample) -> Dict[str, float]:
    # Build interaction patterns: agent -> {counterparty: count}
    patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ix in s.interactions:
        patterns[ix.initiator][ix.counterparty] += 1
        patterns[ix.counterparty][ix.initiator] += 1
    # Run at a permissive threshold so we get continuous similarity signal,
    # then score each node by its max similarity to any partner in its cluster.
    reg = IdentityRegistry(
        IdentityConfig(behavioral_similarity_threshold=0.3,
                       sybil_detection_enabled=True))
    clusters = reg.detect_sybil_clusters({k: dict(v) for k, v in patterns.items()})
    scores: Dict[str, float] = dict.fromkeys(s.nodes, 0.0)
    for cluster in clusters:
        members = list(cluster)
        for i, a in enumerate(members):
            best = 0.0
            for j, b in enumerate(members):
                if i == j:
                    continue
                sim = reg._behavioral_similarity(
                    patterns.get(a, {}), patterns.get(b, {}))
                if sim > best:
                    best = sim
            if best > scores[a]:
                scores[a] = best
    return scores


def detector_reputation_mutual(s: GraphSample) -> Dict[str, float]:
    """Mutual-trust connected component detector. Trust seeded from p."""
    store = GraphMemoryStore(store_path=None)
    # Aggregate per-pair p and counts.
    pair_p: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
    for ix in s.interactions:
        a, b = sorted([ix.initiator, ix.counterparty])
        pair_p[(a, b)].append(ix.p)
        pair_count[(a, b)] += 1
    # Convert to directed RelationshipEdges with trust = avg p.
    for (a, b), ps in pair_p.items():
        avg_p = sum(ps) / len(ps)
        edge = RelationshipEdge(
            agent_a=a, agent_b=b,
            trust_a_to_b=avg_p, trust_b_to_a=avg_p,
            interaction_count=pair_count[(a, b)], avg_p=avg_p,
        )
        store._relationships.append(edge.to_dict())
        # add reverse edge for the b->a direction
        rev_edge = RelationshipEdge(
            agent_a=b, agent_b=a,
            trust_a_to_b=avg_p, trust_b_to_a=avg_p,
            interaction_count=pair_count[(a, b)], avg_p=avg_p,
        )
        store._relationships.append(rev_edge.to_dict())
    gov = ReputationGovernor(store)
    # Sweep at a permissive threshold; per-node score = number of high-mutual
    # trust partners normalized by max possible. Continuous score so the
    # ROC isn't a single point.
    edges_obj = store.get_all_relationships()
    mutual_neighbors: Dict[str, Set[str]] = defaultdict(set)
    rev_lookup = {(e.agent_a, e.agent_b): e for e in edges_obj}
    for e in edges_obj:
        rev: Optional[RelationshipEdge] = rev_lookup.get((e.agent_b, e.agent_a))
        if rev is None:
            continue
        if e.trust_a_to_b >= 0.6 and rev.trust_a_to_b >= 0.6:
            mutual_neighbors[e.agent_a].add(e.agent_b)
    n_max = max(1, max((len(v) for v in mutual_neighbors.values()), default=1))
    scores: Dict[str, float] = dict.fromkeys(s.nodes, 0.0)
    for n, ms in mutual_neighbors.items():
        scores[n] = len(ms) / n_max
    # tie to original gov detector: also boost if in a returned cluster
    clusters = gov.detect_collusion_clusters(min_mutual_trust=0.8, min_size=2)
    in_cluster = {n for c in clusters for n in c}
    for n in in_cluster:
        scores[n] = max(scores[n], 0.5 + scores.get(n, 0.0) / 2.0)
    return scores


def detector_collusion_score(s: GraphSample) -> Dict[str, float]:
    det = CollusionDetector(seed=s.seed)
    report = det.analyze(s.interactions, agent_ids=s.nodes)
    return {n: report.agent_collusion_risk.get(n, 0.0) for n in s.nodes}


DETECTORS: Dict[str, Detector] = {
    "graph_structural": detector_graph_structural,
    "identity_jaccard": detector_identity_jaccard,
    "reputation_mutual": detector_reputation_mutual,
    "collusion_score": detector_collusion_score,
}


# ---------------------------------------------------------------------------
# ROC AUC (no sklearn dependency)
# ---------------------------------------------------------------------------


def roc_auc(scores: Dict[str, float], planted: Set[str], nodes: List[str]) -> float:
    """Per-node ROC AUC. Returns 0.5 for degenerate (all-same-label) cases."""
    pos = [scores[n] for n in nodes if n in planted]
    neg = [scores[n] for n in nodes if n not in planted]
    if not pos or not neg:
        return 0.5
    # Mann-Whitney U / |pos| / |neg| equals ROC AUC.
    total = 0.0
    for p_s in pos:
        for n_s in neg:
            if p_s > n_s:
                total += 1.0
            elif p_s == n_s:
                total += 0.5
    return total / (len(pos) * len(neg))


def bootstrap_ci(values: List[float], *, n: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    samples = []
    k = len(values)
    for _ in range(n):
        boot = [values[rng.randrange(k)] for _ in range(k)]
        samples.append(sum(boot) / k)
    samples.sort()
    lo = samples[int(alpha / 2 * n)]
    hi = samples[int((1 - alpha / 2) * n)]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    family: str
    detector: str
    auc_mean: float
    auc_lo: float
    auc_hi: float
    n_replicates: int
    aucs: List[float]


def run_family(
    generator: Callable[..., GraphSample],
    family_name: str,
    *,
    replicates: int,
    base_params: Dict[str, float],
    seed_base: int = 0,
) -> Dict[str, RunResult]:
    aucs_by_det: Dict[str, List[float]] = defaultdict(list)
    for r in range(replicates):
        sample = generator(seed=seed_base + r, **base_params)
        for det_name, det_fn in DETECTORS.items():
            try:
                scores = det_fn(sample)
                auc = roc_auc(scores, sample.planted, sample.nodes)
            except Exception:
                auc = float("nan")
            aucs_by_det[det_name].append(auc)
    results = {}
    for det_name, aucs in aucs_by_det.items():
        clean = [a for a in aucs if not math.isnan(a)]
        if not clean:
            results[det_name] = RunResult(family_name, det_name, float("nan"),
                                          float("nan"), float("nan"),
                                          len(aucs), aucs)
            continue
        mean = sum(clean) / len(clean)
        lo, hi = bootstrap_ci(clean, seed=seed_base + 1234)
        results[det_name] = RunResult(family_name, det_name, mean, lo, hi,
                                      len(aucs), aucs)
    return results


def run_sweep(replicates: int = 20, seed_base: int = 0) -> List[RunResult]:
    out: List[RunResult] = []
    # benign included for sanity (all detectors should give ~0.5 with no plant)
    families: List[Tuple[Callable[..., GraphSample], str, Dict[str, float]]] = [
        (generate_benign, "benign",
         {"n_agents": 50, "edge_density": 0.1}),
        (generate_collusion_ring, "collusion_ring_size3",
         {"n_agents": 50, "ring_size": 3}),
        (generate_collusion_ring, "collusion_ring_size5",
         {"n_agents": 50, "ring_size": 5}),
        (generate_collusion_ring, "collusion_ring_size8",
         {"n_agents": 50, "ring_size": 8}),
        (generate_sybil_cluster, "sybil_cluster_overlap80",
         {"n_agents": 50, "cluster_size": 5, "counterparty_overlap": 0.8}),
        (generate_sybil_cluster, "sybil_cluster_overlap95",
         {"n_agents": 50, "cluster_size": 5, "counterparty_overlap": 0.95}),
        (generate_threshold_dancing, "threshold_dancing_margin10",
         {"n_agents": 50, "cluster_size": 5, "margin": 0.1}),
        (generate_threshold_dancing, "threshold_dancing_margin05",
         {"n_agents": 50, "cluster_size": 5, "margin": 0.05}),
    ]
    for gen, name, params in families:
        results = run_family(gen, name, replicates=replicates,
                             base_params=params, seed_base=seed_base)
        out.extend(results.values())
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(results: List[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "detector", "auc_mean", "auc_lo95", "auc_hi95",
                    "n_replicates"])
        for r in results:
            w.writerow([r.family, r.detector,
                        f"{r.auc_mean:.4f}",
                        f"{r.auc_lo:.4f}",
                        f"{r.auc_hi:.4f}",
                        r.n_replicates])


def print_table(results: List[RunResult]) -> None:
    by_family: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        by_family[r.family].append(r)
    for fam, rs in by_family.items():
        print(f"\n{fam}")
        print(f"  {'detector':<22} {'AUC':>7}  {'95% CI':>20}")
        for r in sorted(rs, key=lambda x: -x.auc_mean):
            ci = f"[{r.auc_lo:.3f}, {r.auc_hi:.3f}]"
            print(f"  {r.detector:<22} {r.auc_mean:>7.3f}  {ci:>20}")


def apply_decision_rule(results: List[RunResult]) -> str:
    """Return a one-line verdict per the pre-reg decision rule."""
    by_family: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        if r.family == "benign":
            continue  # sanity, not part of the rule
        by_family[r.family].append(r)
    families_dominated = []
    for fam, rs in by_family.items():
        gs = next((r for r in rs if r.detector == "graph_structural"), None)
        if gs is None:
            continue
        others = [r for r in rs if r.detector != "graph_structural"]
        # strict dominance: graph_structural lower CI bound > each other's upper CI bound
        if all(gs.auc_lo > o.auc_hi for o in others):
            families_dominated.append(fam)
    if families_dominated:
        return (f"VERDICT: graph_structural strictly dominates on "
                f"{len(families_dominated)} family/families "
                f"({', '.join(families_dominated)}). "
                f"Per pre-reg, file governance-wiring follow-up.")
    return ("VERDICT: graph_structural does not strictly dominate on any "
            "family. Per pre-reg, ship as secondary metric only — no "
            "governance wiring.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=20,
                        help="replicates per (family, params) cell "
                             "(pre-reg target: 100; default 20 for CI speed)")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("runs"),
                        help="output directory (a timestamped subdir is created)")
    args = parser.parse_args(argv)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output / f"{ts}_graph_structural_roc"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.replicates} replicates per family...")
    results = run_sweep(replicates=args.replicates, seed_base=args.seed_base)
    print_table(results)
    csv_path = run_dir / "auc_table.csv"
    write_csv(results, csv_path)
    verdict = apply_decision_rule(results)
    print(f"\n{verdict}\n")
    (run_dir / "verdict.txt").write_text(verdict + "\n")
    print(f"Results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
