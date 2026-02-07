"""Memory-tier-specific metrics for shared-memory simulations."""

from typing import Dict, Sequence

from swarm.env.memory_tiers import MemoryEntry, MemoryEntryStatus, MemoryStore, MemoryTier


def poisoning_rate(store: MemoryStore, tier: MemoryTier = MemoryTier.GRAPH) -> float:
    """Fraction of active entries at the given tier that are poisoned."""
    entries = [
        e for e in store.get_entries_by_tier(tier)
        if e.status == MemoryEntryStatus.ACTIVE
    ]
    if not entries:
        return 0.0
    poisoned = sum(1 for e in entries if e.is_poisoned)
    return poisoned / len(entries)


def promotion_accuracy(store: MemoryStore) -> float:
    """Fraction of promoted entries (Tier 2+) that are not poisoned."""
    promoted = [
        e for e in store.all_entries()
        if e.tier in (MemoryTier.STRUCTURED, MemoryTier.GRAPH)
        and e.status == MemoryEntryStatus.ACTIVE
    ]
    if not promoted:
        return 1.0
    accurate = sum(1 for e in promoted if not e.is_poisoned)
    return accurate / len(promoted)


def cache_corruption(store: MemoryStore) -> float:
    """Fraction of hot cache entries that are poisoned."""
    cache = store.hot_cache
    if not cache:
        return 0.0
    poisoned = sum(1 for e in cache if e.is_poisoned)
    return poisoned / len(cache)


def information_asymmetry(store: MemoryStore) -> float:
    """Gini coefficient of search hit distribution across agents.

    High values indicate some agents have much better recall than others.
    """
    hits = store.search_hits
    if not hits:
        return 0.0
    values = sorted(max(0, v) for v in hits.values())
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


def write_concentration(store: MemoryStore) -> float:
    """Gini coefficient of write distribution across agents.

    High values indicate write flooding by a few agents.
    """
    writes = store.writes_this_epoch
    if not writes:
        return 0.0
    values = sorted(max(0, v) for v in writes.values())
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


def governance_filter_rate(store: MemoryStore) -> float:
    """Fraction of poisoned entries that were reverted (caught by governance).

    Higher is better — means governance is catching bad entries.
    """
    all_poisoned = [e for e in store.all_entries() if e.is_poisoned]
    if not all_poisoned:
        return 1.0  # No poisoned entries to catch
    reverted = sum(1 for e in all_poisoned if e.status == MemoryEntryStatus.REVERTED)
    return reverted / len(all_poisoned)


def tier_distribution(store: MemoryStore) -> Dict[str, int]:
    """Count of active entries per tier."""
    return store.entry_count()


def content_quality_by_tier(store: MemoryStore) -> Dict[str, float]:
    """Average quality score of active entries per tier."""
    result: Dict[str, float] = {}
    for tier in MemoryTier:
        entries = [
            e for e in store.get_entries_by_tier(tier)
            if e.status == MemoryEntryStatus.ACTIVE
        ]
        if entries:
            result[tier.value] = sum(e.quality_score for e in entries) / len(entries)
        else:
            result[tier.value] = 0.0
    return result
