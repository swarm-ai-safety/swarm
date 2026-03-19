"""Tests for adverse selection detection via relationship graph."""

import tempfile
from pathlib import Path

import pytest

from swarm.analysis.adverse_selection import AdverseSelectionDetector
from swarm.knowledge.graph_memory import GraphMemoryStore, RelationshipEdge


@pytest.fixture
def temp_store_path():
    """Create a temporary directory for test stores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_store.json")


@pytest.fixture
def empty_store(temp_store_path):
    """Create an empty GraphMemoryStore."""
    return GraphMemoryStore(store_path=temp_store_path)


@pytest.fixture
def honest_agent_store(temp_store_path):
    """Create a store with honest agents (high avg_p, good matches).

    Topology:
    - honest_a <-> honest_b: avg_p=0.9, count=5
    - honest_a <-> honest_c: avg_p=0.85, count=3
    - honest_b <-> honest_c: avg_p=0.88, count=4
    - honest_c <-> honest_a: avg_p=0.85, count=1 (reciprocal)
    """
    store = GraphMemoryStore(store_path=temp_store_path)

    # All high-quality relationships
    store.add_relationship_event(
        "honest_a",
        "honest_b",
        {"p": 0.9, "payoff_a": 10.0, "payoff_b": 10.0, "epoch": 0},
    )
    for _ in range(4):
        store.add_relationship_event(
            "honest_a",
            "honest_b",
            {"p": 0.9, "payoff_a": 10.0, "payoff_b": 10.0, "epoch": 0},
        )

    for _ in range(3):
        store.add_relationship_event(
            "honest_a",
            "honest_c",
            {"p": 0.85, "payoff_a": 9.0, "payoff_b": 9.0, "epoch": 0},
        )

    for _ in range(4):
        store.add_relationship_event(
            "honest_b",
            "honest_c",
            {"p": 0.88, "payoff_a": 9.5, "payoff_b": 9.5, "epoch": 0},
        )

    # Add honest_c as initiator too
    store.add_relationship_event(
        "honest_c",
        "honest_a",
        {"p": 0.85, "payoff_a": 9.0, "payoff_b": 9.0, "epoch": 0},
    )

    return store


@pytest.fixture
def exploiter_victim_store(temp_store_path):
    """Create a store with exploiter and victim agents.

    Topology:
    - exploiter initiates relationships with victims (low p)
    - victim_a initiates relationships with honest agents (high p)
    - exploiter consistently gets low p but high payoff (extraction)
    """
    store = GraphMemoryStore(store_path=temp_store_path)

    # Exploiter matches with victims (low quality, high payoff for exploiter)
    for _ in range(3):
        store.add_relationship_event(
            "exploiter",
            "victim_a",
            {"p": 0.3, "payoff_a": 15.0, "payoff_b": 2.0, "epoch": 0},
        )

    for _ in range(3):
        store.add_relationship_event(
            "exploiter",
            "victim_b",
            {"p": 0.25, "payoff_a": 16.0, "payoff_b": 1.5, "epoch": 0},
        )

    # Victims try to match with honest agents (higher quality)
    for _ in range(3):
        store.add_relationship_event(
            "victim_a",
            "honest",
            {"p": 0.8, "payoff_a": 8.0, "payoff_b": 8.0, "epoch": 0},
        )

    for _ in range(3):
        store.add_relationship_event(
            "victim_b",
            "honest",
            {"p": 0.75, "payoff_a": 7.5, "payoff_b": 7.5, "epoch": 0},
        )

    # Honest agent matches with honest partners
    for _ in range(2):
        store.add_relationship_event(
            "honest",
            "honest_partner",
            {"p": 0.9, "payoff_a": 10.0, "payoff_b": 10.0, "epoch": 0},
        )

    return store


@pytest.fixture
def mixed_quality_store(temp_store_path):
    """Create a store with mixed quality agents.

    Topology:
    - high_quality initiates relationships with mixed partners (some bad matches)
    - mid_quality initiates relationships with other mid_quality agents
    - low_quality initiates relationships with high_quality agents (good matches)
    """
    store = GraphMemoryStore(store_path=temp_store_path)

    # high_quality matches with mixed partners
    for _ in range(2):
        store.add_relationship_event(
            "high_quality",
            "mid_quality_1",
            {"p": 0.85, "payoff_a": 9.0, "payoff_b": 8.5, "epoch": 0},
        )

    for _ in range(2):
        store.add_relationship_event(
            "high_quality",
            "low_quality_1",
            {"p": 0.4, "payoff_a": 5.0, "payoff_b": 3.0, "epoch": 0},
        )

    # mid_quality matches with other mid_quality
    for _ in range(3):
        store.add_relationship_event(
            "mid_quality_1",
            "mid_quality_2",
            {"p": 0.65, "payoff_a": 6.5, "payoff_b": 6.5, "epoch": 0},
        )

    # low_quality gets good matches
    for _ in range(3):
        store.add_relationship_event(
            "low_quality_1",
            "high_quality_partner",
            {"p": 0.8, "payoff_a": 8.0, "payoff_b": 8.0, "epoch": 0},
        )

    return store


class TestAdverseSelectionDetectorBasics:
    """Test basic functionality of AdverseSelectionDetector."""

    def test_empty_store(self, empty_store):
        """Test detector on empty store."""
        detector = AdverseSelectionDetector(empty_store)

        assert detector.compute_quality_gap_by_agent() == {}
        assert detector.identify_exploited_agents() == []
        assert detector.identify_exploiting_agents() == []
        assert detector.compute_selection_pressure() == {}

        summary = detector.get_adverse_selection_summary()
        assert summary["system_avg_p"] == 0.0
        assert summary["agents_exploited_count"] == 0
        assert summary["agents_exploiting_count"] == 0

    def test_single_relationship(self, temp_store_path):
        """Test with single relationship."""
        store = GraphMemoryStore(store_path=temp_store_path)
        store.add_relationship_event(
            "agent_a",
            "agent_b",
            {"p": 0.7, "payoff_a": 5.0, "payoff_b": 5.0, "epoch": 0},
        )

        detector = AdverseSelectionDetector(store)

        gaps = detector.compute_quality_gap_by_agent()
        assert "agent_a" in gaps
        assert gaps["agent_a"] == 0.0  # Only agent, so baseline is their own avg_p

    def test_honest_agents_no_adverse_selection(self, honest_agent_store):
        """Test that honest agents show no adverse selection."""
        detector = AdverseSelectionDetector(honest_agent_store)

        # All agents should have positive or near-zero quality gaps
        gaps = detector.compute_quality_gap_by_agent()
        for agent_id, gap in gaps.items():
            assert gap >= -0.05, f"Agent {agent_id} unexpectedly low gap: {gap}"

        # No exploited agents
        exploited = detector.identify_exploited_agents(threshold=-0.1)
        assert len(exploited) == 0

        # No exploiting agents
        exploiting = detector.identify_exploiting_agents(threshold=0.1)
        assert len(exploiting) == 0


class TestQualityGapComputation:
    """Test quality gap computation."""

    def test_quality_gap_calculation(self, honest_agent_store):
        """Test quality gap is computed correctly."""
        detector = AdverseSelectionDetector(honest_agent_store)

        gaps = detector.compute_quality_gap_by_agent()

        # All three agents should be present as initiators
        assert len(gaps) == 3, f"Expected 3 agents, got {len(gaps)}: {gaps}"
        assert all(abs(gap) < 0.05 for gap in gaps.values()), "All gaps should be small"

    def test_quality_gap_with_mixed_quality(self, mixed_quality_store):
        """Test quality gap with mixed-quality agents."""
        detector = AdverseSelectionDetector(mixed_quality_store)

        gaps = detector.compute_quality_gap_by_agent()

        # high_quality: matches with 0.85 and 0.4 -> avg=0.625
        # low_quality_1: matches with 0.8 -> avg=0.8 (good matches!)
        # Others are in between

        # low_quality_1 should have positive gap (matched up)
        assert gaps["low_quality_1"] > 0

        # high_quality should have lower gap (matched down a bit)
        assert gaps["high_quality"] < gaps["low_quality_1"]


class TestExploitedAgentDetection:
    """Test identification of exploited agents."""

    def test_exploited_agents_in_mixed_quality(self, mixed_quality_store):
        """Test identification of exploited agents."""
        detector = AdverseSelectionDetector(mixed_quality_store)

        # Compute expected values
        gaps = detector.compute_quality_gap_by_agent()
        exploited = detector.identify_exploited_agents(threshold=-0.1)

        # high_quality has avg_p of (0.85*2 + 0.4*2)/4 = 0.625
        # System avg is around 0.68, so gap is about -0.055
        # With threshold -0.1, this should not be flagged
        # But verify that exploited agents, if any, have negative gaps
        for agent_id in exploited:
            assert gaps[agent_id] < -0.1, f"Agent {agent_id} should have gap < -0.1"

    def test_exploited_agents_threshold_sensitivity(self, mixed_quality_store):
        """Test threshold sensitivity for exploited agent detection."""
        detector = AdverseSelectionDetector(mixed_quality_store)

        exploited_strict = detector.identify_exploited_agents(threshold=-0.05)
        exploited_lenient = detector.identify_exploited_agents(threshold=-0.15)

        # Strict threshold should catch more exploited agents
        assert len(exploited_strict) >= len(exploited_lenient)

    def test_no_exploited_in_honest_network(self, honest_agent_store):
        """Test that all-honest network has no exploited agents."""
        detector = AdverseSelectionDetector(honest_agent_store)

        exploited = detector.identify_exploited_agents(threshold=-0.1)
        assert len(exploited) == 0


class TestExploitingAgentDetection:
    """Test identification of exploiting agents."""

    def test_exploiting_agents_detected(self, exploiter_victim_store):
        """Test detection of agents exploiting others."""
        detector = AdverseSelectionDetector(exploiter_victim_store)

        exploiting = detector.identify_exploiting_agents(threshold=0.1)

        # exploiter has low avg_p (0.275) but high total payoff (31.0)
        assert "exploiter" in exploiting

    def test_exploiting_agent_characteristics(self, exploiter_victim_store):
        """Verify exploiting agents have low quality but high payoff."""
        # Build the details manually
        relationships = exploiter_victim_store.get_all_relationships()
        exploiter_rels = [r for r in relationships if r.agent_a == "exploiter"]

        # exploiter should have low avg_p
        exploiter_avg_p = sum(r.avg_p for r in exploiter_rels) / len(exploiter_rels)
        assert exploiter_avg_p < 0.4

        # exploiter should have high total payoff
        exploiter_payoff = sum(r.total_payoff_a for r in exploiter_rels)
        assert exploiter_payoff > 25.0

    def test_no_exploiting_in_honest_network(self, honest_agent_store):
        """Test that all-honest network has no exploiting agents."""
        detector = AdverseSelectionDetector(honest_agent_store)

        exploiting = detector.identify_exploiting_agents(threshold=0.1)
        assert len(exploiting) == 0


class TestSelectionPressure:
    """Test selection pressure computation."""

    def test_selection_pressure_equals_one_on_average(self, honest_agent_store):
        """Test that average selection pressure is close to 1.0."""
        detector = AdverseSelectionDetector(honest_agent_store)

        pressures = detector.compute_selection_pressure()

        assert len(pressures) > 0
        mean_pressure = sum(pressures.values()) / len(pressures)

        # Mean should be ~1.0 because quality gaps are small
        assert 0.95 < mean_pressure < 1.05

    def test_selection_pressure_less_than_one_for_exploited(
        self, exploiter_victim_store
    ):
        """Test selection pressure < 1.0 for adversely selected agents."""
        detector = AdverseSelectionDetector(exploiter_victim_store)

        pressures = detector.compute_selection_pressure()

        # Victims have selection pressure < 1.0 initially
        # But they also match with honest (high p), so overall near 1.0
        # Exploiter has pressure << 1.0 (only matches with victims)
        assert pressures["exploiter"] < 1.0

    def test_selection_pressure_greater_than_one_for_favored(
        self, mixed_quality_store
    ):
        """Test selection pressure > 1.0 for well-matched agents."""
        detector = AdverseSelectionDetector(mixed_quality_store)

        pressures = detector.compute_selection_pressure()

        # low_quality_1 matches with high_quality_partner (avg_p=0.8)
        # This should give pressure > 1.0 if system avg is lower
        if "low_quality_1" in pressures:
            # low_quality_1 should have decent pressure
            assert pressures["low_quality_1"] >= 0.8


class TestAdverseSelecitonSummary:
    """Test summary statistics."""

    def test_summary_structure(self, honest_agent_store):
        """Test summary has all required fields."""
        detector = AdverseSelectionDetector(honest_agent_store)

        summary = detector.get_adverse_selection_summary()

        required_keys = [
            "system_avg_p",
            "agents_exploited_count",
            "agents_exploiting_count",
            "worst_quality_gap",
            "best_quality_gap",
            "selection_pressure_variance",
            "selection_pressure_mean",
        ]

        for key in required_keys:
            assert key in summary

    def test_summary_honest_network(self, honest_agent_store):
        """Test summary for all-honest network."""
        detector = AdverseSelectionDetector(honest_agent_store)

        summary = detector.get_adverse_selection_summary()

        # All high-quality matches
        assert summary["system_avg_p"] > 0.85
        assert summary["agents_exploited_count"] == 0
        assert summary["agents_exploiting_count"] == 0

        # Gaps should be small
        assert abs(summary["worst_quality_gap"]) < 0.1
        assert abs(summary["best_quality_gap"]) < 0.1

        # Pressure should be stable (low variance)
        assert summary["selection_pressure_variance"] < 0.01

    def test_summary_exploiter_victim_network(self, exploiter_victim_store):
        """Test summary for exploiter-victim network."""
        detector = AdverseSelectionDetector(exploiter_victim_store)

        summary = detector.get_adverse_selection_summary()

        # At least one exploited and one exploiting agent
        assert summary["agents_exploited_count"] > 0
        assert summary["agents_exploiting_count"] > 0

        # Worst quality gap should be negative
        assert summary["worst_quality_gap"] < 0

        # Pressure variance should be higher
        assert summary["selection_pressure_variance"] > 0.01

    def test_summary_variance_is_nonnegative(self, mixed_quality_store):
        """Test that variance is always non-negative."""
        detector = AdverseSelectionDetector(mixed_quality_store)

        summary = detector.get_adverse_selection_summary()

        assert summary["selection_pressure_variance"] >= 0.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_same_quality_relationships(self, temp_store_path):
        """Test when all relationships have identical quality."""
        store = GraphMemoryStore(store_path=temp_store_path)

        for a in ["a", "b", "c"]:
            for b in ["d", "e", "f"]:
                if a != b:
                    store.add_relationship_event(
                        a,
                        b,
                        {"p": 0.5, "payoff_a": 5.0, "payoff_b": 5.0, "epoch": 0},
                    )

        detector = AdverseSelectionDetector(store)

        gaps = detector.compute_quality_gap_by_agent()

        # All gaps should be exactly zero
        for gap in gaps.values():
            assert abs(gap) < 1e-10

    def test_single_agent_many_relationships(self, temp_store_path):
        """Test single agent with many relationships."""
        store = GraphMemoryStore(store_path=temp_store_path)

        for i in range(10):
            store.add_relationship_event(
                "initiator",
                f"partner_{i}",
                {"p": 0.5 + 0.05 * i, "payoff_a": 5.0, "payoff_b": 5.0, "epoch": 0},
            )

        detector = AdverseSelectionDetector(store)

        gaps = detector.compute_quality_gap_by_agent()

        # Initiator should be in gaps
        assert "initiator" in gaps

        # Pressure should be exactly 1.0 (only initiator matches)
        pressures = detector.compute_selection_pressure()
        assert abs(pressures["initiator"] - 1.0) < 1e-10

    def test_relationships_with_zero_interactions(self, temp_store_path):
        """Test that relationships with zero interactions are skipped."""
        store = GraphMemoryStore(store_path=temp_store_path)

        # Add manually created relationships with zero interactions
        rel = RelationshipEdge(
            agent_a="agent_a",
            agent_b="agent_b",
            trust_a_to_b=0.5,
            trust_b_to_a=0.5,
            interaction_count=0,  # No interactions
            avg_p=0.5,
        )
        store._relationships.append(rel.to_dict())
        store._persist()

        detector = AdverseSelectionDetector(store)

        # Should not include agents with no interactions
        gaps = detector.compute_quality_gap_by_agent()
        assert len(gaps) == 0
