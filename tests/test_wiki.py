"""Tests for Moltipedia wiki environment."""

from swarm.env.wiki import (
    EditType,
    PageStatus,
    PolicyViolationType,
    WikiPage,
    WikiTaskPool,
)


def test_wiki_task_pool_seed_and_retrieve():
    pool = WikiTaskPool(seed=1)
    pool.seed_pages(5)
    assert len(pool.all_pages()) == 5


def test_wiki_page_apply_edit_updates_state():
    page = WikiPage(title="Test", content="Base", status=PageStatus.DRAFT, quality_score=0.3)
    page.apply_edit(
        editor_id="agent_1",
        new_content="Updated",
        edit_type=EditType.EDIT,
        delta_quality=0.2,
        policy_violations=[PolicyViolationType.SOURCING],
        epoch=0,
        step=1,
    )
    assert page.content == "Updated"
    assert page.quality_score == 0.5
    assert page.policy_violations == [PolicyViolationType.SOURCING]
    assert page.edit_history


def test_contested_queue_respects_cooldown():
    pool = WikiTaskPool(seed=2)
    page = WikiPage(title="Contested", status=PageStatus.CONTESTED, cooldown_until=5)
    pool.add_page(page)
    assert pool.get_contested_pages(limit=5, current_step=3) == []
    assert pool.get_contested_pages(limit=5, current_step=6)
