"""Tests for moltbook_catalog seeding."""

import random

from swarm.env.moltbook import ContentStatus, MoltbookFeed
from swarm.env.moltbook_catalog import (
    EXPECTED_SUBMOLTS,
    MOLTBOOK_CATALOG,
    seed_from_catalog,
)


class TestMoltbookCatalogStructure:
    """Validate catalog entries have the expected shape."""

    def test_catalog_has_entries(self):
        assert len(MOLTBOOK_CATALOG) >= 20

    def test_entries_have_required_keys(self):
        for entry in MOLTBOOK_CATALOG:
            assert "content" in entry
            assert "submolt" in entry
            assert "author_tag" in entry

    def test_submolts_from_expected_set(self):
        for entry in MOLTBOOK_CATALOG:
            assert entry["submolt"] in EXPECTED_SUBMOLTS, (
                f"Unexpected submolt: {entry['submolt']}"
            )

    def test_content_is_nonempty(self):
        for entry in MOLTBOOK_CATALOG:
            assert len(entry["content"]) > 0

    def test_all_expected_submolts_represented(self):
        submolts_present = {e["submolt"] for e in MOLTBOOK_CATALOG}
        assert submolts_present == EXPECTED_SUBMOLTS


class TestMoltbookSeedFromCatalog:
    """Test the seed_from_catalog function."""

    def test_seeds_correct_number_of_posts(self):
        feed = MoltbookFeed()
        rng = random.Random(42)
        seed_from_catalog(feed, 10, rng)
        published = feed.get_published_posts(limit=100)
        assert len(published) == 10

    def test_caps_at_catalog_size(self):
        feed = MoltbookFeed()
        rng = random.Random(42)
        seed_from_catalog(feed, 9999, rng)
        published = feed.get_published_posts(limit=100)
        assert len(published) == len(MOLTBOOK_CATALOG)

    def test_posts_are_published(self):
        feed = MoltbookFeed()
        rng = random.Random(42)
        seed_from_catalog(feed, 5, rng)
        for post in feed.get_published_posts(limit=100):
            assert post.status == ContentStatus.PUBLISHED

    def test_posts_have_seed_bot_author(self):
        feed = MoltbookFeed()
        rng = random.Random(42)
        seed_from_catalog(feed, 5, rng)
        for post in feed.get_published_posts(limit=100):
            assert post.author_id == "seed_bot"

    def test_deterministic_with_same_seed(self):
        feed1 = MoltbookFeed()
        seed_from_catalog(feed1, 10, random.Random(99))
        content1 = sorted(p.content for p in feed1.get_published_posts(limit=100))

        feed2 = MoltbookFeed()
        seed_from_catalog(feed2, 10, random.Random(99))
        content2 = sorted(p.content for p in feed2.get_published_posts(limit=100))

        assert content1 == content2

    def test_zero_posts(self):
        feed = MoltbookFeed()
        rng = random.Random(42)
        seed_from_catalog(feed, 0, rng)
        assert len(feed.get_published_posts(limit=100)) == 0
