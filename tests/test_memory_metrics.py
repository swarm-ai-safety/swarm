from __future__ import annotations

from swarm.env.memory_tiers import MemoryEntryStatus, MemoryStore, MemoryTier
from swarm.metrics import memory_metrics


def test_poisoning_and_promotion_accuracy() -> None:
    store = MemoryStore(seed=1)

    clean = store.write(
        agent_id="a",
        content="clean",
        quality_score=0.9,
        is_poisoned=False,
        epoch=0,
        step=0,
    )
    poisoned = store.write(
        agent_id="b",
        content="poisoned",
        quality_score=0.2,
        is_poisoned=True,
        epoch=0,
        step=1,
    )
    store.write(
        agent_id="c",
        content="ephemeral",
        quality_score=0.5,
        is_poisoned=True,
        epoch=0,
        step=2,
    )

    assert store.promote(clean.entry_id, promoter_id="mod")
    assert store.promote(poisoned.entry_id, promoter_id="mod")

    structured = store.get_entries_by_tier(MemoryTier.STRUCTURED)
    clean_structured = next(e for e in structured if not e.is_poisoned)
    assert store.promote(clean_structured.entry_id, promoter_id="mod")

    assert memory_metrics.poisoning_rate(store, MemoryTier.EPHEMERAL) == 1.0
    assert memory_metrics.promotion_accuracy(store) == 0.5


def test_cache_asymmetry_and_write_concentration() -> None:
    store = MemoryStore(seed=2)
    entry_a = store.write(
        agent_id="a",
        content="cache-a",
        quality_score=0.6,
        is_poisoned=False,
        epoch=0,
        step=0,
    )
    entry_b = store.write(
        agent_id="b",
        content="cache-b",
        quality_score=0.4,
        is_poisoned=True,
        epoch=0,
        step=1,
    )

    store._hot_cache = [entry_a, entry_b]
    store._search_hits = {"a": 3, "b": 1}
    store._writes_this_epoch = {"a": 2, "b": 2}

    assert memory_metrics.cache_corruption(store) == 0.5
    assert memory_metrics.information_asymmetry(store) == 0.25
    assert memory_metrics.write_concentration(store) == 0.0


def test_governance_filter_rate_and_tier_metrics() -> None:
    store = MemoryStore(seed=3)

    poison_a = store.write(
        agent_id="a",
        content="poison-a",
        quality_score=0.1,
        is_poisoned=True,
        epoch=0,
        step=0,
    )
    store.write(
        agent_id="b",
        content="poison-b",
        quality_score=0.2,
        is_poisoned=True,
        epoch=0,
        step=1,
    )
    store.revert(poison_a.entry_id)

    assert memory_metrics.governance_filter_rate(store) == 0.5

    store = MemoryStore(seed=4)
    ephem = store.write(
        agent_id="a",
        content="ephem",
        quality_score=0.2,
        is_poisoned=False,
        epoch=0,
        step=0,
    )
    structured = store.write(
        agent_id="b",
        content="structured",
        quality_score=0.8,
        is_poisoned=False,
        epoch=0,
        step=1,
    )
    structured.tier = MemoryTier.STRUCTURED
    graph = store.write(
        agent_id="c",
        content="graph",
        quality_score=0.6,
        is_poisoned=False,
        epoch=0,
        step=2,
    )
    graph.tier = MemoryTier.GRAPH
    reverted = store.write(
        agent_id="d",
        content="reverted",
        quality_score=0.9,
        is_poisoned=False,
        epoch=0,
        step=3,
    )
    reverted.status = MemoryEntryStatus.REVERTED

    tier_counts = memory_metrics.tier_distribution(store)
    assert tier_counts[MemoryTier.EPHEMERAL.value] == 1
    assert tier_counts[MemoryTier.STRUCTURED.value] == 1
    assert tier_counts[MemoryTier.GRAPH.value] == 1

    quality = memory_metrics.content_quality_by_tier(store)
    assert quality[MemoryTier.EPHEMERAL.value] == ephem.quality_score
    assert quality[MemoryTier.STRUCTURED.value] == structured.quality_score
    assert quality[MemoryTier.GRAPH.value] == graph.quality_score
