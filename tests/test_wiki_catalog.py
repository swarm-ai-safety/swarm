"""Tests for wiki_catalog seeding."""

import random

from swarm.env.wiki import PageStatus, WikiTaskPool
from swarm.env.wiki_catalog import WIKI_CATALOG, seed_from_catalog


class TestWikiCatalogStructure:
    """Validate catalog entries have the expected shape."""

    def test_catalog_has_entries(self):
        assert len(WIKI_CATALOG) >= 25

    def test_entries_have_required_keys(self):
        for entry in WIKI_CATALOG:
            assert "title" in entry
            assert "content" in entry
            assert "quality_score" in entry

    def test_quality_scores_in_range(self):
        for entry in WIKI_CATALOG:
            assert 0.0 < entry["quality_score"] <= 1.0

    def test_titles_are_unique(self):
        titles = [e["title"] for e in WIKI_CATALOG]
        assert len(titles) == len(set(titles))

    def test_content_is_nonempty(self):
        for entry in WIKI_CATALOG:
            assert len(entry["content"]) > 0

    def test_research_relevant_titles(self):
        titles_lower = " ".join(e["title"].lower() for e in WIKI_CATALOG)
        assert "distributional" in titles_lower
        assert "governance" in titles_lower
        assert "collusion" in titles_lower
        assert "rain" in titles_lower or "river" in titles_lower
        assert "adverse selection" in titles_lower


class TestSeedFromCatalog:
    """Test the seed_from_catalog function."""

    def test_seeds_correct_number_of_pages(self):
        pool = WikiTaskPool(seed=42)
        rng = random.Random(42)
        seed_from_catalog(pool, 10, rng)
        assert len(pool.all_pages()) == 10

    def test_caps_at_catalog_size(self):
        pool = WikiTaskPool(seed=42)
        rng = random.Random(42)
        seed_from_catalog(pool, 9999, rng)
        assert len(pool.all_pages()) == len(WIKI_CATALOG)

    def test_pages_have_valid_fields(self):
        pool = WikiTaskPool(seed=42)
        rng = random.Random(42)
        seed_from_catalog(pool, 5, rng)
        for page in pool.all_pages():
            assert page.title
            assert page.content
            assert 0.0 < page.quality_score <= 1.0
            assert page.created_by == "catalog_seed"
            assert page.status in (PageStatus.STUB, PageStatus.DRAFT)

    def test_deterministic_with_same_seed(self):
        pool1 = WikiTaskPool(seed=1)
        seed_from_catalog(pool1, 10, random.Random(99))
        titles1 = sorted(p.title for p in pool1.all_pages())

        pool2 = WikiTaskPool(seed=2)
        seed_from_catalog(pool2, 10, random.Random(99))
        titles2 = sorted(p.title for p in pool2.all_pages())

        assert titles1 == titles2

    def test_zero_pages(self):
        pool = WikiTaskPool(seed=42)
        rng = random.Random(42)
        seed_from_catalog(pool, 0, rng)
        assert len(pool.all_pages()) == 0
