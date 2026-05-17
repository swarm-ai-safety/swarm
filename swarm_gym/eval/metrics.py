"""System metrics aggregation across episodes."""

from __future__ import annotations

from typing import Any, Dict, List


def aggregate_outcomes(episode_outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate outcome metrics across multiple episodes.

    For each numeric key, computes mean. For boolean keys (like 'catastrophe'),
    computes rate.

    Returns:
        Dictionary with aggregated metrics (prefixed with 'mean_' or '_rate').
    """
    if not episode_outcomes:
        return {}

    agg: Dict[str, Any] = {}
    keys = set()
    for ep in episode_outcomes:
        keys.update(ep.keys())

    for key in sorted(keys):
        values = [ep.get(key) for ep in episode_outcomes if key in ep]
        if not values:
            continue

        if all(isinstance(v, bool) for v in values):
            rate = sum(1 for v in values if v) / len(values)
            agg[f"{key}_rate"] = round(rate, 4)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            mean = sum(values) / len(values)
            agg[f"mean_{key}"] = round(mean, 4)
        elif all(v is None for v in values):
            agg[key] = None
        # Skip non-numeric, non-bool keys

    return agg


def compute_variance(values: List[float]) -> float:
    """Compute population variance."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
