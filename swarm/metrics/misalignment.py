"""Misalignment metrics for multi-agent preference divergence.

Implements the sociotechnical misalignment framework from Kierans et al.
(arXiv:2406.04231), adapted for SWARM's governance topology. Measures
weighted preference disagreement across agent populations, with support
for graph-local computation and governance-adjusted effective misalignment.

Key concepts:
- Each agent has a preference vector p_k in [-1, 1]^m over m issues
- Each agent has salience weights w_k (normalized, non-negative)
- Pairwise misalignment = salience-weighted L1 distance in preference space
- Governance reduces effective misalignment via taxes, audits, reputation
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class DistanceMetric(Enum):
    """Distance metric for preference disagreement."""

    L1 = "l1"
    L2 = "l2"
    COSINE = "cosine"


class WeightAggregation(Enum):
    """How to aggregate salience weights between two agents."""

    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    GEOM_MEAN = "geom_mean"


@dataclass
class MisalignmentProfile:
    """Per-agent preference and salience representation.

    Attributes:
        agent_id: Unique agent identifier.
        prefs: Preference vector p_k in [-1, 1]^m.
        salience: Importance weights w_k, non-negative, sums to 1.
        gov_sensitivity: Optional governance channel sensitivities.
    """

    agent_id: str
    prefs: List[float]
    salience: List[float]
    gov_sensitivity: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if len(self.prefs) != len(self.salience):
            raise ValueError(
                f"prefs length {len(self.prefs)} != salience length {len(self.salience)}"
            )
        for i, p in enumerate(self.prefs):
            if not (-1.0 <= p <= 1.0):
                raise ValueError(f"prefs[{i}]={p} not in [-1, 1]")
        for i, w in enumerate(self.salience):
            if w < 0:
                raise ValueError(f"salience[{i}]={w} is negative")
        total = sum(self.salience)
        if total > 0 and abs(total - 1.0) > 1e-6:
            # Auto-normalize
            self.salience = [w / total for w in self.salience]


@dataclass
class IssueSpace:
    """Definition of the issue/dimension space.

    Attributes:
        issues: Named dimensions of disagreement.
        distance: Distance metric for preference vectors.
        weight_agg: How to combine salience weights between agent pairs.
    """

    issues: List[str]
    distance: DistanceMetric = DistanceMetric.L1
    weight_agg: WeightAggregation = WeightAggregation.MEAN

    @property
    def m(self) -> int:
        return len(self.issues)


@dataclass
class MisalignmentSnapshot:
    """Misalignment state at a single simulation step.

    Attributes:
        step: Simulation step number.
        m_pref_global: Population-level raw misalignment.
        m_eff_global: Population-level governance-adjusted misalignment.
        polarization: Between-cluster / within-cluster distance ratio.
        fragmentation: Entropy over cluster sizes.
        local: Per-agent local misalignment scores.
        issue_contributions: Per-issue contribution to global misalignment.
        alerts: Early warning signals.
    """

    step: int
    m_pref_global: float
    m_eff_global: float
    polarization: float = 0.0
    fragmentation: float = 0.0
    local: Dict[str, float] = field(default_factory=dict)
    local_eff: Dict[str, float] = field(default_factory=dict)
    issue_contributions: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "global": {
                "M_pref": self.m_pref_global,
                "M_eff": self.m_eff_global,
                "polarization": self.polarization,
                "fragmentation": self.fragmentation,
            },
            "local": self.local,
            "local_eff": self.local_eff,
            "issue_contributions": self.issue_contributions,
            "alerts": self.alerts,
        }


class MisalignmentModule:
    """Computes preference-space misalignment metrics for agent populations.

    Implements pairwise, local (graph-conditioned), and global misalignment
    with optional governance adjustment. Supports exact, graph-local, and
    sampled computation modes for scalability.
    """

    def __init__(
        self,
        issue_space: IssueSpace,
        gov_lambda: float = 1.0,
    ) -> None:
        """
        Args:
            issue_space: Issue dimension definitions.
            gov_lambda: Scaling factor for governance pressure reduction.
        """
        self.issue_space = issue_space
        self.gov_lambda = gov_lambda
        self._profiles: Dict[str, MisalignmentProfile] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        prefs: List[float],
        salience: List[float],
        gov_sensitivity: Optional[List[float]] = None,
    ) -> None:
        """Register or update an agent's misalignment profile."""
        if len(prefs) != self.issue_space.m:
            raise ValueError(
                f"prefs length {len(prefs)} != issue_space.m={self.issue_space.m}"
            )
        if len(salience) != self.issue_space.m:
            raise ValueError(
                f"salience length {len(salience)} != issue_space.m={self.issue_space.m}"
            )
        self._profiles[agent_id] = MisalignmentProfile(
            agent_id=agent_id,
            prefs=list(prefs),
            salience=list(salience),
            gov_sensitivity=list(gov_sensitivity) if gov_sensitivity else None,
        )

    def update_agent(
        self,
        agent_id: str,
        prefs: Optional[List[float]] = None,
        salience: Optional[List[float]] = None,
    ) -> None:
        """Update an existing agent's preferences or salience."""
        if agent_id not in self._profiles:
            raise KeyError(f"Agent {agent_id} not registered")
        profile = self._profiles[agent_id]
        if prefs is not None:
            if len(prefs) != self.issue_space.m:
                raise ValueError(
                    f"prefs length {len(prefs)} != issue_space.m={self.issue_space.m}"
                )
            profile.prefs = list(prefs)
        if salience is not None:
            if len(salience) != self.issue_space.m:
                raise ValueError(
                    f"salience length {len(salience)} != issue_space.m={self.issue_space.m}"
                )
            profile.salience = list(salience)
            # Re-normalize
            total = sum(profile.salience)
            if total > 0 and abs(total - 1.0) > 1e-6:
                profile.salience = [w / total for w in profile.salience]

    @property
    def profiles(self) -> Dict[str, MisalignmentProfile]:
        return self._profiles

    # ------------------------------------------------------------------
    # Pairwise misalignment
    # ------------------------------------------------------------------

    def _aggregate_weights(self, w_k: float, w_l: float) -> float:
        """Aggregate salience weights for a pair of agents on one issue."""
        agg = self.issue_space.weight_agg
        if agg == WeightAggregation.MEAN:
            return (w_k + w_l) / 2.0
        elif agg == WeightAggregation.MIN:
            return min(w_k, w_l)
        elif agg == WeightAggregation.MAX:
            return max(w_k, w_l)
        elif agg == WeightAggregation.GEOM_MEAN:
            return math.sqrt(w_k * w_l)
        raise ValueError(f"Unknown aggregation: {agg}")

    def pairwise_misalignment(
        self,
        agent_k: str,
        agent_l: str,
    ) -> float:
        """Compute salience-weighted pairwise misalignment M_pref(k, l).

        Returns value in [0, 1] for L1 distance with normalized salience.
        """
        pk = self._profiles[agent_k]
        pl = self._profiles[agent_l]
        m = self.issue_space.m
        dist = self.issue_space.distance

        if dist == DistanceMetric.L1:
            return sum(
                self._aggregate_weights(pk.salience[j], pl.salience[j])
                * abs(pk.prefs[j] - pl.prefs[j])
                for j in range(m)
            )
        elif dist == DistanceMetric.L2:
            return math.sqrt(
                sum(
                    self._aggregate_weights(pk.salience[j], pl.salience[j])
                    * (pk.prefs[j] - pl.prefs[j]) ** 2
                    for j in range(m)
                )
            )
        elif dist == DistanceMetric.COSINE:
            # Salience-weighted cosine distance
            dot = sum(
                self._aggregate_weights(pk.salience[j], pl.salience[j])
                * pk.prefs[j] * pl.prefs[j]
                for j in range(m)
            )
            norm_k = math.sqrt(
                sum(
                    self._aggregate_weights(pk.salience[j], pl.salience[j])
                    * pk.prefs[j] ** 2
                    for j in range(m)
                )
            )
            norm_l = math.sqrt(
                sum(
                    self._aggregate_weights(pk.salience[j], pl.salience[j])
                    * pl.prefs[j] ** 2
                    for j in range(m)
                )
            )
            if norm_k == 0 or norm_l == 0:
                return 0.0
            cos_sim = dot / (norm_k * norm_l)
            return 1.0 - cos_sim
        raise ValueError(f"Unknown distance: {dist}")

    def pairwise_issue_contributions(
        self,
        agent_k: str,
        agent_l: str,
    ) -> List[float]:
        """Per-issue contribution to pairwise misalignment (L1 only)."""
        pk = self._profiles[agent_k]
        pl = self._profiles[agent_l]
        return [
            self._aggregate_weights(pk.salience[j], pl.salience[j])
            * abs(pk.prefs[j] - pl.prefs[j])
            for j in range(self.issue_space.m)
        ]

    # ------------------------------------------------------------------
    # Population-level misalignment
    # ------------------------------------------------------------------

    def global_misalignment(
        self,
        agent_ids: Optional[Sequence[str]] = None,
    ) -> float:
        """Average pairwise misalignment across all agent pairs.

        M_global = (2 / n(n-1)) * sum_{k<l} M_pref(k, l)

        Args:
            agent_ids: Subset of agents to consider (default: all registered).

        Returns:
            Average pairwise misalignment. 0.0 if fewer than 2 agents.
        """
        ids = list(agent_ids) if agent_ids else list(self._profiles.keys())
        n = len(ids)
        if n < 2:
            return 0.0

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += self.pairwise_misalignment(ids[i], ids[j])
                count += 1

        return total / count

    def global_issue_contributions(
        self,
        agent_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Per-issue average contribution to global misalignment."""
        ids = list(agent_ids) if agent_ids else list(self._profiles.keys())
        n = len(ids)
        if n < 2:
            return dict.fromkeys(self.issue_space.issues, 0.0)

        m = self.issue_space.m
        totals = [0.0] * m
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                contribs = self.pairwise_issue_contributions(ids[i], ids[j])
                for d in range(m):
                    totals[d] += contribs[d]
                count += 1

        return {
            self.issue_space.issues[d]: totals[d] / count for d in range(m)
        }

    # ------------------------------------------------------------------
    # Graph-local misalignment
    # ------------------------------------------------------------------

    def local_misalignment(
        self,
        agent_id: str,
        neighbors: Sequence[str],
        edge_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Misalignment of an agent relative to its graph neighbors.

        M_local(k) = (1/Z_k) * sum_{l in N(k)} omega_kl * M_pref(k, l)

        Args:
            agent_id: The focal agent.
            neighbors: Neighbor agent IDs.
            edge_weights: Optional weights per neighbor ID (default: uniform).

        Returns:
            Weighted-average misalignment with neighbors. 0.0 if no neighbors.
        """
        if not neighbors:
            return 0.0

        total = 0.0
        z = 0.0
        for nb in neighbors:
            w = edge_weights.get(nb, 1.0) if edge_weights else 1.0
            total += w * self.pairwise_misalignment(agent_id, nb)
            z += w

        return total / z if z > 0 else 0.0

    def all_local_misalignment(
        self,
        graph: Dict[str, List[str]],
        edge_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """Compute local misalignment for every agent in a graph.

        Args:
            graph: Adjacency list {agent_id: [neighbor_ids]}.
            edge_weights: Optional nested weights {agent_id: {neighbor: weight}}.

        Returns:
            {agent_id: M_local} for each agent in the graph.
        """
        result = {}
        for agent_id, neighbors in graph.items():
            if agent_id not in self._profiles:
                continue
            valid_neighbors = [n for n in neighbors if n in self._profiles]
            ew = edge_weights.get(agent_id) if edge_weights else None
            result[agent_id] = self.local_misalignment(agent_id, valid_neighbors, ew)
        return result

    # ------------------------------------------------------------------
    # Sampled misalignment (for large populations)
    # ------------------------------------------------------------------

    def sampled_global_misalignment(
        self,
        k: int = 1000,
        agent_ids: Optional[Sequence[str]] = None,
        rng: Optional[random.Random] = None,
    ) -> float:
        """Estimate global misalignment from k random pairs.

        Args:
            k: Number of random pairs to sample.
            agent_ids: Subset of agents (default: all).
            rng: Random number generator for reproducibility.

        Returns:
            Estimated average pairwise misalignment.
        """
        ids = list(agent_ids) if agent_ids else list(self._profiles.keys())
        n = len(ids)
        if n < 2:
            return 0.0

        _rng = rng or random.Random()
        total = 0.0
        for _ in range(k):
            i, j = _rng.sample(range(n), 2)
            total += self.pairwise_misalignment(ids[i], ids[j])

        return total / k

    # ------------------------------------------------------------------
    # Governance-adjusted misalignment
    # ------------------------------------------------------------------

    def effective_misalignment(
        self,
        agent_k: str,
        agent_l: str,
        governance_pressure: float = 0.0,
    ) -> float:
        """Governance-adjusted pairwise misalignment.

        M_eff(k, l) = max(0, M_pref(k, l) - lambda * G_kl)

        Args:
            agent_k: First agent.
            agent_l: Second agent.
            governance_pressure: Governance pressure G_kl for this pair.

        Returns:
            Effective misalignment, floored at 0.
        """
        raw = self.pairwise_misalignment(agent_k, agent_l)
        return max(0.0, raw - self.gov_lambda * governance_pressure)

    def global_effective_misalignment(
        self,
        governance_pressures: Optional[Dict[Tuple[str, str], float]] = None,
        uniform_pressure: float = 0.0,
        agent_ids: Optional[Sequence[str]] = None,
    ) -> float:
        """Population-level governance-adjusted misalignment.

        Args:
            governance_pressures: Per-pair pressures {(k,l): G_kl}.
            uniform_pressure: Applied to all pairs if per-pair not given.
            agent_ids: Subset of agents.

        Returns:
            Average effective misalignment.
        """
        ids = list(agent_ids) if agent_ids else list(self._profiles.keys())
        n = len(ids)
        if n < 2:
            return 0.0

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair = (ids[i], ids[j])
                g = (
                    governance_pressures.get(pair, uniform_pressure)
                    if governance_pressures
                    else uniform_pressure
                )
                total += self.effective_misalignment(ids[i], ids[j], g)
                count += 1

        return total / count

    # ------------------------------------------------------------------
    # Diagnostics: polarization & fragmentation
    # ------------------------------------------------------------------

    def polarization_index(
        self,
        n_clusters: int = 2,
        agent_ids: Optional[Sequence[str]] = None,
    ) -> float:
        """Polarization: between-cluster / within-cluster distance ratio.

        Uses k-means-style clustering on salience-weighted preference vectors.
        Higher values indicate stronger polarization.

        Args:
            n_clusters: Number of clusters to fit (default: 2).
            agent_ids: Subset of agents.

        Returns:
            Polarization ratio. 0.0 if insufficient agents.
        """
        ids = list(agent_ids) if agent_ids else list(self._profiles.keys())
        if len(ids) < n_clusters:
            return 0.0

        # Weighted preference vectors for clustering
        vectors = []
        for aid in ids:
            p = self._profiles[aid]
            vectors.append(
                [p.salience[j] * p.prefs[j] for j in range(self.issue_space.m)]
            )

        # Simple k-means (few iterations, deterministic seed)
        labels = self._kmeans(vectors, n_clusters, max_iter=20)

        # Compute between/within cluster distances
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)

        # Within-cluster average distance
        within = 0.0
        within_count = 0
        for members in clusters.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    within += self._vec_l1(vectors[members[i]], vectors[members[j]])
                    within_count += 1

        # Between-cluster: distance between cluster centroids
        centroids = {}
        for label, members in clusters.items():
            m = self.issue_space.m
            centroid = [0.0] * m
            for idx in members:
                for d in range(m):
                    centroid[d] += vectors[idx][d]
            centroids[label] = [c / len(members) for c in centroid]

        between = 0.0
        between_count = 0
        cluster_labels = list(centroids.keys())
        for i in range(len(cluster_labels)):
            for j in range(i + 1, len(cluster_labels)):
                between += self._vec_l1(
                    centroids[cluster_labels[i]], centroids[cluster_labels[j]]
                )
                between_count += 1

        if within_count == 0 or between_count == 0:
            return 0.0

        avg_within = within / within_count
        avg_between = between / between_count

        if avg_within == 0:
            return float("inf") if avg_between > 0 else 0.0

        return avg_between / avg_within

    def fragmentation_index(
        self,
        n_clusters: int = 3,
        agent_ids: Optional[Sequence[str]] = None,
    ) -> float:
        """Fragmentation: entropy over cluster sizes (normalized to [0, 1]).

        Higher values indicate more evenly distributed clusters
        (fragmented population). Lower values indicate one dominant cluster.

        Args:
            n_clusters: Number of clusters to fit.
            agent_ids: Subset of agents.

        Returns:
            Normalized entropy in [0, 1]. 0.0 if insufficient agents.
        """
        ids = list(agent_ids) if agent_ids else list(self._profiles.keys())
        if len(ids) < n_clusters:
            return 0.0

        vectors = []
        for aid in ids:
            p = self._profiles[aid]
            vectors.append(
                [p.salience[j] * p.prefs[j] for j in range(self.issue_space.m)]
            )

        labels = self._kmeans(vectors, n_clusters, max_iter=20)

        # Cluster sizes
        sizes: Dict[int, int] = {}
        for label in labels:
            sizes[label] = sizes.get(label, 0) + 1

        n = len(ids)
        entropy = 0.0
        for count in sizes.values():
            if count > 0:
                frac = count / n
                entropy -= frac * math.log(frac)

        max_entropy = math.log(n_clusters) if n_clusters > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    # ------------------------------------------------------------------
    # Snapshot: full diagnostic computation
    # ------------------------------------------------------------------

    def compute_snapshot(
        self,
        step: int,
        graph: Optional[Dict[str, List[str]]] = None,
        edge_weights: Optional[Dict[str, Dict[str, float]]] = None,
        governance_pressures: Optional[Dict[Tuple[str, str], float]] = None,
        uniform_pressure: float = 0.0,
        agent_ids: Optional[Sequence[str]] = None,
    ) -> MisalignmentSnapshot:
        """Compute a full misalignment snapshot for the current step.

        Args:
            step: Current simulation step.
            graph: Adjacency list for local misalignment.
            edge_weights: Nested edge weights for graph.
            governance_pressures: Per-pair governance pressures.
            uniform_pressure: Uniform governance pressure fallback.
            agent_ids: Subset of agents (default: all).

        Returns:
            MisalignmentSnapshot with all computed metrics.
        """
        ids = list(agent_ids) if agent_ids else list(self._profiles.keys())

        m_pref = self.global_misalignment(ids)
        m_eff = self.global_effective_misalignment(
            governance_pressures, uniform_pressure, ids
        )

        # Local misalignment
        local_scores: Dict[str, float] = {}
        local_eff_scores: Dict[str, float] = {}
        if graph:
            local_scores = self.all_local_misalignment(graph, edge_weights)
            # Effective local: use uniform pressure as proxy
            for aid, m_local in local_scores.items():
                local_eff_scores[aid] = max(
                    0.0, m_local - self.gov_lambda * uniform_pressure
                )

        # Issue contributions
        issue_contribs = self.global_issue_contributions(ids)

        # Diagnostics
        pol = self.polarization_index(agent_ids=ids) if len(ids) >= 2 else 0.0
        frag = self.fragmentation_index(agent_ids=ids) if len(ids) >= 3 else 0.0

        # Alerts
        alerts = []
        if m_pref > 0.8:
            alerts.append("ALERT: Misalignment spike (M_pref > 0.8)")
        if pol > 3.0:
            alerts.append("ALERT: High polarization (ratio > 3.0)")
        if m_eff > 0 and m_pref > 0 and m_eff / m_pref > 0.9:
            alerts.append(
                "ALERT: Governance not reducing effective misalignment"
            )
        if local_scores:
            max_local = max(local_scores.values())
            if max_local > 1.5:
                alerts.append(
                    f"ALERT: Local percolation risk (max M_local={max_local:.3f})"
                )

        return MisalignmentSnapshot(
            step=step,
            m_pref_global=m_pref,
            m_eff_global=m_eff,
            polarization=pol,
            fragmentation=frag,
            local=local_scores,
            local_eff=local_eff_scores,
            issue_contributions=issue_contribs,
            alerts=alerts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vec_l1(a: List[float], b: List[float]) -> float:
        return sum(abs(ai - bi) for ai, bi in zip(a, b, strict=True))

    @staticmethod
    def _kmeans(
        vectors: List[List[float]],
        k: int,
        max_iter: int = 20,
    ) -> List[int]:
        """Simple deterministic k-means. Returns cluster labels."""
        n = len(vectors)
        if n == 0:
            return []
        m = len(vectors[0])

        # Initialize centroids: spread across data (maximin)
        centroids = [list(vectors[0])]
        for _ in range(1, k):
            # Pick vector farthest from all existing centroids
            best_idx = 0
            best_dist = -1.0
            for idx in range(n):
                min_dist = min(
                    sum(abs(vectors[idx][d] - c[d]) for d in range(m))
                    for c in centroids
                )
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = idx
            centroids.append(list(vectors[best_idx]))
        labels = [0] * n

        for _ in range(max_iter):
            # Assign
            new_labels = []
            for vec in vectors:
                best = 0
                best_dist = float("inf")
                for c_idx, centroid in enumerate(centroids):
                    d = sum(abs(vec[d] - centroid[d]) for d in range(m))
                    if d < best_dist:
                        best_dist = d
                        best = c_idx
                new_labels.append(best)

            if new_labels == labels:
                break
            labels = new_labels

            # Update centroids
            for c_idx in range(k):
                members = [i for i, lbl in enumerate(labels) if lbl == c_idx]
                if members:
                    centroids[c_idx] = [
                        sum(vectors[i][d] for i in members) / len(members)
                        for d in range(m)
                    ]

        return labels
