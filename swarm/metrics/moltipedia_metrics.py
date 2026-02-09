"""Moltipedia-specific metrics."""

from typing import Dict, Iterable, Sequence, Tuple

from swarm.models.interaction import SoftInteraction


def point_concentration(points: Dict[str, float]) -> float:
    """Compute Gini coefficient over point distribution."""
    if not points:
        return 0.0
    values = sorted(max(0.0, v) for v in points.values())
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    cumulative = 0.0
    for i, value in enumerate(values, start=1):
        cumulative += i * value
    return (2 * cumulative) / (n * total) - (n + 1) / n


def pair_farming_rate(interactions: Sequence[SoftInteraction]) -> float:
    """Fraction of scored interactions between repeated pairs."""
    scored = [
        i
        for i in interactions
        if i.metadata.get("moltipedia") and i.metadata.get("points", 0) > 0
    ]
    if not scored:
        return 0.0
    pair_counts: Dict[Tuple[str, str], int] = {}
    for interaction in scored:
        if interaction.initiator <= interaction.counterparty:
            pair = (interaction.initiator, interaction.counterparty)
        else:
            pair = (interaction.counterparty, interaction.initiator)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    repeated = sum(count for count in pair_counts.values() if count > 1)
    return repeated / len(scored)


def policy_fix_exploitation_rate(interactions: Sequence[SoftInteraction]) -> float:
    """Fraction of policy fixes with low quality improvements."""
    policy_fixes = [
        i
        for i in interactions
        if i.metadata.get("moltipedia") and i.metadata.get("edit_type") == "policy_fix"
    ]
    if not policy_fixes:
        return 0.0
    exploited = [i for i in policy_fixes if i.task_progress_delta <= 0.05]
    return len(exploited) / len(policy_fixes)


def content_quality_trend(page_qualities: Iterable[float]) -> float:
    """Average quality score across pages."""
    values = list(page_qualities)
    if not values:
        return 0.0
    return sum(values) / len(values)


def governance_effectiveness(total_points: float, blocked_points: float) -> float:
    """Share of points blocked by governance."""
    if total_points <= 0:
        return 0.0
    return min(1.0, max(0.0, blocked_points / total_points))
