"""Tests for the peer review system.

Covers:
- PeerReview dataclass (serialization, properties, compatibility)
- ReviewStore (append, load, filtering, average, corruption handling)
- AgentRxivBridge review methods
- QualityGate integration with peer reviews
"""

import json
import tempfile
from pathlib import Path

from swarm.research.agents import (
    _RECOMMENDATION_RATINGS,
    Critique,
    PeerReview,
    Review,
    review_to_peer_review,
)
from swarm.research.quality import QualityGates
from swarm.research.swarm_papers.agentrxiv_bridge import AgentRxivBridge
from swarm.research.swarm_papers.review_store import ReviewStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_review(
    paper_id: str = "paper_1",
    reviewer_id: str = "reviewer_a",
    recommendation: str = "accept",
    rating: int = 5,
    critiques: list[Critique] | None = None,
) -> PeerReview:
    return PeerReview(
        review_id="rev_001",
        paper_id=paper_id,
        reviewer_id=reviewer_id,
        critiques=critiques or [],
        recommendation=recommendation,
        summary=f"Test review: {recommendation}",
        rating=rating,
    )


# ===========================================================================
# TestPeerReview
# ===========================================================================


class TestPeerReview:
    """Test PeerReview dataclass."""

    def test_to_dict_from_dict_roundtrip(self):
        """Roundtrip serialization preserves all fields."""
        original = _make_review(
            critiques=[
                Critique(
                    severity="medium",
                    category="methodology",
                    issue="Missing seeds",
                    suggestion="Add seeds",
                    addressed=True,
                ),
            ]
        )
        data = original.to_dict()
        restored = PeerReview.from_dict(data)

        assert restored.review_id == original.review_id
        assert restored.paper_id == original.paper_id
        assert restored.reviewer_id == original.reviewer_id
        assert restored.recommendation == original.recommendation
        assert restored.summary == original.summary
        assert restored.rating == original.rating
        assert len(restored.critiques) == 1
        assert restored.critiques[0].severity == "medium"
        assert restored.critiques[0].addressed is True

    def test_high_severity_count(self):
        """high_severity_count counts high and critical critiques."""
        review = _make_review(
            critiques=[
                Critique(severity="low", category="x", issue="a"),
                Critique(severity="high", category="x", issue="b"),
                Critique(severity="critical", category="x", issue="c"),
                Critique(severity="medium", category="x", issue="d"),
            ]
        )
        assert review.high_severity_count == 2

    def test_all_critiques_addressed(self):
        """all_critiques_addressed returns True when all are addressed."""
        review = _make_review(
            critiques=[
                Critique(severity="low", category="x", issue="a", addressed=True),
                Critique(severity="high", category="x", issue="b", addressed=True),
            ]
        )
        assert review.all_critiques_addressed is True

        review2 = _make_review(
            critiques=[
                Critique(severity="low", category="x", issue="a", addressed=True),
                Critique(severity="high", category="x", issue="b", addressed=False),
            ]
        )
        assert review2.all_critiques_addressed is False

    def test_all_critiques_addressed_empty(self):
        """Empty critique list means all addressed."""
        review = _make_review(critiques=[])
        assert review.all_critiques_addressed is True

    def test_to_review_compatibility(self):
        """to_review() produces a Review usable with QualityGate."""
        peer = _make_review(
            recommendation="accept",
            critiques=[
                Critique(severity="low", category="x", issue="a", addressed=True),
            ],
        )
        review = peer.to_review()
        assert isinstance(review, Review)
        assert review.recommendation == "accept"
        assert len(review.critiques) == 1
        assert review.high_severity_count == 0
        assert review.all_critiques_addressed is True

    def test_rating_mapping(self):
        """review_to_peer_review maps recommendations to correct ratings."""
        for rec, expected_rating in _RECOMMENDATION_RATINGS.items():
            review = Review(recommendation=rec, summary="test")
            peer = review_to_peer_review(review, "paper_1", "reviewer_1")
            assert peer.rating == expected_rating, f"{rec} should map to {expected_rating}"

    def test_review_to_peer_review_unknown_recommendation(self):
        """Unknown recommendation defaults to rating 3."""
        review = Review(recommendation="unknown_status", summary="test")
        peer = review_to_peer_review(review, "paper_1", "reviewer_1")
        assert peer.rating == 3

    def test_review_to_peer_review_preserves_critiques(self):
        """Critiques from Review are carried into PeerReview."""
        crit = Critique(severity="high", category="stats", issue="p-hacking")
        review = Review(
            critiques=[crit],
            recommendation="major_revision",
            summary="Needs work",
        )
        peer = review_to_peer_review(review, "paper_x", "rev_y")
        assert len(peer.critiques) == 1
        assert peer.critiques[0].issue == "p-hacking"
        assert peer.paper_id == "paper_x"
        assert peer.reviewer_id == "rev_y"


# ===========================================================================
# TestReviewStore
# ===========================================================================


class TestReviewStore:
    """Test ReviewStore JSONL backend."""

    def test_append_and_load(self):
        """Append + load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReviewStore(Path(tmpdir) / "reviews.jsonl")
            r1 = _make_review(paper_id="p1", rating=5)
            r2 = _make_review(paper_id="p2", rating=3)
            store.append(r1)
            store.append(r2)

            loaded = store.load()
            assert len(loaded) == 2
            assert loaded[0].paper_id == "p1"
            assert loaded[1].paper_id == "p2"

    def test_get_for_paper_filtering(self):
        """get_for_paper returns only matching reviews."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReviewStore(Path(tmpdir) / "reviews.jsonl")
            store.append(_make_review(paper_id="p1", rating=5))
            store.append(_make_review(paper_id="p2", rating=3))
            store.append(_make_review(paper_id="p1", rating=4))

            p1_reviews = store.get_for_paper("p1")
            assert len(p1_reviews) == 2
            assert all(r.paper_id == "p1" for r in p1_reviews)

    def test_average_rating(self):
        """average_rating computes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReviewStore(Path(tmpdir) / "reviews.jsonl")
            store.append(_make_review(paper_id="p1", rating=5))
            store.append(_make_review(paper_id="p1", rating=3))

            avg = store.average_rating("p1")
            assert avg == 4.0

    def test_average_rating_no_reviews(self):
        """average_rating returns None for unknown paper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReviewStore(Path(tmpdir) / "reviews.jsonl")
            assert store.average_rating("nonexistent") is None

    def test_empty_store(self):
        """load() on nonexistent file returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReviewStore(Path(tmpdir) / "reviews.jsonl")
            assert store.load() == []

    def test_corrupted_line_skipped(self):
        """Corrupted JSON lines are skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reviews.jsonl"
            valid = _make_review(paper_id="good")
            with path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(valid.to_dict()) + "\n")
                fh.write("NOT VALID JSON\n")
                fh.write(json.dumps(_make_review(paper_id="also_good").to_dict()) + "\n")

            store = ReviewStore(path)
            loaded = store.load()
            assert len(loaded) == 2
            assert loaded[0].paper_id == "good"
            assert loaded[1].paper_id == "also_good"

    def test_creates_parent_dirs(self):
        """ReviewStore creates parent directories on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "reviews.jsonl"
            store = ReviewStore(nested)
            store.append(_make_review())
            assert nested.exists()


# ===========================================================================
# TestBridgeReviews
# ===========================================================================


class TestBridgeReviews:
    """Test AgentRxivBridge review methods."""

    def test_submit_review_stores(self):
        """submit_review persists via ReviewStore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = AgentRxivBridge()
            bridge.set_review_store(Path(tmpdir) / "reviews.jsonl")

            review = _make_review(paper_id="paper_x")
            assert bridge.submit_review(review) is True

            loaded = bridge.get_reviews("paper_x")
            assert len(loaded) == 1
            assert loaded[0].paper_id == "paper_x"

    def test_get_reviews_filters(self):
        """get_reviews returns only matching paper reviews."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = AgentRxivBridge()
            bridge.set_review_store(Path(tmpdir) / "reviews.jsonl")

            bridge.submit_review(_make_review(paper_id="a"))
            bridge.submit_review(_make_review(paper_id="b"))
            bridge.submit_review(_make_review(paper_id="a"))

            assert len(bridge.get_reviews("a")) == 2
            assert len(bridge.get_reviews("b")) == 1
            assert len(bridge.get_reviews("c")) == 0

    def test_review_summary_aggregation(self):
        """review_summary returns count, avg_rating, recommendation_counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = AgentRxivBridge()
            bridge.set_review_store(Path(tmpdir) / "reviews.jsonl")

            bridge.submit_review(
                _make_review(paper_id="p", rating=5, recommendation="accept")
            )
            bridge.submit_review(
                _make_review(paper_id="p", rating=3, recommendation="minor_revision")
            )

            summary = bridge.review_summary("p")
            assert summary["count"] == 2
            assert summary["avg_rating"] == 4.0
            assert summary["recommendation_counts"]["accept"] == 1
            assert summary["recommendation_counts"]["minor_revision"] == 1

    def test_review_summary_no_reviews(self):
        """review_summary returns empty data for unknown paper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = AgentRxivBridge()
            bridge.set_review_store(Path(tmpdir) / "reviews.jsonl")

            summary = bridge.review_summary("nonexistent")
            assert summary["count"] == 0
            assert summary["avg_rating"] is None

    def test_no_store_graceful_fallback(self):
        """Bridge methods work gracefully without a ReviewStore."""
        bridge = AgentRxivBridge()
        # No set_review_store called
        assert bridge.submit_review(_make_review()) is False
        assert bridge.get_reviews("anything") == []
        summary = bridge.review_summary("anything")
        assert summary["count"] == 0

    def test_constructor_review_path(self):
        """Bridge can receive review_path at construction time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reviews.jsonl"
            bridge = AgentRxivBridge(review_path=path)

            bridge.submit_review(_make_review(paper_id="init_test"))
            assert len(bridge.get_reviews("init_test")) == 1


# ===========================================================================
# TestReviewGateIntegration
# ===========================================================================


class TestReviewGateIntegration:
    """Test QualityGate integration with peer reviews."""

    def test_clean_review_passes_gate(self):
        """A review with no high-severity issues and positive recommendation passes."""
        review = _make_review(
            recommendation="accept",
            critiques=[
                Critique(severity="low", category="x", issue="minor", addressed=True),
            ],
        )
        gate = QualityGates.review_gate()
        result = gate.run(review.to_review())
        assert result.passed is True

    def test_high_severity_fails_gate(self):
        """A review with high-severity unaddressed critiques fails the gate."""
        review = _make_review(
            recommendation="major_revision",
            critiques=[
                Critique(
                    severity="high",
                    category="stats",
                    issue="Missing CIs",
                    addressed=False,
                ),
            ],
        )
        gate = QualityGates.review_gate()
        result = gate.run(review.to_review())
        assert result.passed is False

    def test_review_to_peer_review_gate_flow(self):
        """Full flow: Review -> PeerReview -> to_review -> gate check."""
        review = Review(
            critiques=[
                Critique(severity="low", category="x", issue="y", addressed=True),
            ],
            recommendation="accept",
            summary="Looks good",
        )
        peer = review_to_peer_review(review, "paper_1", "reviewer_1")
        gate = QualityGates.review_gate()
        result = gate.run(peer.to_review())
        assert result.passed is True
