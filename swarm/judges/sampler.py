"""Stratified sampler for the judge anchor.

Per arm B of the calibration pre-reg, the judge rates a held-out set of
**accepted** interactions stratified across `p` bins. This module does the
sampling. Stratification uses the simulation's `p` (not v_hat) because it
is the authoritative latent under the static-baseline data we draw from;
the judge never sees `p` — that's the JudgeView's job to enforce.
"""

from __future__ import annotations

import random
from typing import Any, Iterable


def stratified_sample(
    interactions: Iterable[Any],
    per_bin: int,
    bin_edges: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = 0,
    require_accepted: bool = True,
) -> list[Any]:
    """Sample `per_bin` accepted interactions per p-bin, without replacement.

    Bins with fewer than `per_bin` accepted interactions return all of
    them and the caller is responsible for noticing the underflow (we log
    it via the return value's length, not via a hard failure — a bin
    being short is often the point).
    """
    rng = random.Random(seed)
    pool = list(interactions)
    if require_accepted:
        pool = [i for i in pool if getattr(i, "accepted", False)]

    by_bin: list[list[Any]] = [[] for _ in range(len(bin_edges) - 1)]
    for interaction in pool:
        p = float(getattr(interaction, "p", 0.5))
        for b in range(len(bin_edges) - 1):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            in_range = lo <= p < hi if b < len(bin_edges) - 2 else lo <= p <= hi
            if in_range:
                by_bin[b].append(interaction)
                break

    sampled: list[Any] = []
    for bucket in by_bin:
        if len(bucket) <= per_bin:
            sampled.extend(bucket)
        else:
            sampled.extend(rng.sample(bucket, per_bin))
    return sampled


def bin_counts(
    interactions: Iterable[Any],
    bin_edges: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> list[int]:
    """Count interactions per p-bin (using same bin convention as the sampler)."""
    counts = [0] * (len(bin_edges) - 1)
    for interaction in interactions:
        p = float(getattr(interaction, "p", 0.5))
        for b in range(len(bin_edges) - 1):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            in_range = lo <= p < hi if b < len(bin_edges) - 2 else lo <= p <= hi
            if in_range:
                counts[b] += 1
                break
    return counts
