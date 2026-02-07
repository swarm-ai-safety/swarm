"""Moltbook-specific metrics."""

from typing import Dict, Sequence


def challenge_pass_rate(attempts: Sequence[bool]) -> float:
    """Share of successful challenge attempts."""
    if not attempts:
        return 0.0
    return sum(1 for a in attempts if a) / len(attempts)


def rate_limit_hit_rate(hits: int, attempts: int) -> float:
    """Rate limit hit fraction."""
    if attempts <= 0:
        return 0.0
    return hits / attempts


def content_throughput(published_posts: int, epochs: int) -> float:
    """Published posts per epoch."""
    if epochs <= 0:
        return 0.0
    return published_posts / epochs


def verification_latency_distribution(latencies: Sequence[int]) -> Dict[str, float]:
    """Summary stats for verification latency."""
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    values = sorted(latencies)
    mean = sum(values) / len(values)
    p50 = values[len(values) // 2]
    p90 = values[max(0, int(0.9 * (len(values) - 1)))]
    return {"mean": float(mean), "p50": float(p50), "p90": float(p90)}


def karma_concentration(karma_by_agent: Dict[str, float]) -> float:
    """Gini coefficient for karma distribution."""
    values = sorted(max(0.0, v) for v in karma_by_agent.values())
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


def wasted_action_rate(wasted_actions: int, total_actions: int) -> float:
    """Fraction of actions that resulted in wasted or expired content."""
    if total_actions <= 0:
        return 0.0
    return wasted_actions / total_actions


def captcha_effectiveness(human_fail_rate: float, bot_success_rate: float) -> float:
    """Ratio of human failure rate to bot success rate."""
    if bot_success_rate <= 0:
        return 0.0
    return human_fail_rate / bot_success_rate


def rate_limit_governance_impact(with_limits: float, without_limits: float) -> float:
    """Throughput reduction from rate limits."""
    if without_limits <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - with_limits / without_limits))
