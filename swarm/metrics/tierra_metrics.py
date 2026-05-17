"""Metrics for Tierra-inspired artificial life scenarios.

Functions for genome diversity, resource inequality, parasitism/cooperation
fractions, and distance-based speciation.
"""

from __future__ import annotations

import math
from typing import Dict, List


def genome_diversity(genomes: List[Dict[str, float]]) -> float:
    """Average pairwise Euclidean distance between genome vectors.

    Returns 0.0 for fewer than 2 genomes.
    """
    n = len(genomes)
    if n < 2:
        return 0.0

    keys = sorted(genomes[0].keys())
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = sum(
                (genomes[i][k] - genomes[j][k]) ** 2 for k in keys
            )
            total += math.sqrt(dist_sq)
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


def resource_gini(resources: List[float]) -> float:
    """Gini coefficient for resource distribution.  0 = perfect equality."""
    n = len(resources)
    if n == 0:
        return 0.0
    sorted_r = sorted(resources)
    total = sum(sorted_r)
    if total == 0:
        return 0.0
    cumulative = 0.0
    gini_sum = 0.0
    for _i, r in enumerate(sorted_r):
        cumulative += r
        gini_sum += cumulative
    # Gini = 1 - 2*B  where B = sum of cumulative / (n * total)
    return 1.0 - 2.0 * gini_sum / (n * total) + 1.0 / n


def parasitism_fraction(genomes: List[Dict[str, float]], threshold: float = 0.5) -> float:
    """Fraction of agents with exploitation_tendency above *threshold*."""
    if not genomes:
        return 0.0
    count = sum(1 for g in genomes if g.get("exploitation_tendency", 0) > threshold)
    return count / len(genomes)


def cooperation_fraction(genomes: List[Dict[str, float]], threshold: float = 0.5) -> float:
    """Fraction of agents with cooperation_bias above *threshold*."""
    if not genomes:
        return 0.0
    count = sum(1 for g in genomes if g.get("cooperation_bias", 0) > threshold)
    return count / len(genomes)


def speciation_count(
    genomes: List[Dict[str, float]],
    distance_threshold: float = 0.5,
) -> int:
    """Count species via single-linkage clustering at *distance_threshold*.

    Two genomes belong to the same species if their Euclidean distance
    is below the threshold.  Returns the number of disjoint clusters.
    """
    n = len(genomes)
    if n == 0:
        return 0

    keys = sorted(genomes[0].keys())

    def dist(i: int, j: int) -> float:
        return math.sqrt(sum((genomes[i][k] - genomes[j][k]) ** 2 for k in keys))

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if dist(i, j) < distance_threshold:
                union(i, j)

    return len({find(i) for i in range(n)})


def species_clusters(
    genomes: List[Dict[str, float]],
    distance_threshold: float = 0.5,
) -> Dict[int, List[int]]:
    """Return species clusters as {cluster_id: [genome_indices]}.

    Uses the same single-linkage clustering as :func:`speciation_count`.
    Cluster IDs are the canonical root index from union-find.
    """
    n = len(genomes)
    if n == 0:
        return {}

    keys = sorted(genomes[0].keys())

    def dist(i: int, j: int) -> float:
        return math.sqrt(sum((genomes[i][k] - genomes[j][k]) ** 2 for k in keys))

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if dist(i, j) < distance_threshold:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)
    return clusters
