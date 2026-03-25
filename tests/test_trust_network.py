"""Tests for trust network analysis and visualization."""

import tempfile
from pathlib import Path

from swarm.analysis.trust_network import TrustNetworkAnalyzer, plot_trust_network
from swarm.knowledge.graph_memory import (
    AgentMemorySnapshot,
    GraphMemoryStore,
)


class TestTrustNetworkAnalyzer:
    """Tests for TrustNetworkAnalyzer class."""

    def test_build_trust_matrix_empty(self):
        """Empty store should produce empty matrix."""
        store = GraphMemoryStore()
        analyzer = TrustNetworkAnalyzer(store)

        matrix = analyzer.build_trust_matrix()

        assert matrix == {}

    def test_build_trust_matrix_single_relationship(self):
        """Single relationship should be reflected in matrix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            # Add snapshots
            snap_a = AgentMemorySnapshot(agent_id="a0", agent_type="honest")
            snap_b = AgentMemorySnapshot(agent_id="a1", agent_type="opportunistic")
            store.save("a0", snap_a)
            store.save("a1", snap_b)

            # Add relationship
            store.add_relationship_event(
                "a0",
                "a1",
                {
                    "p": 0.8,
                    "payoff_a": 10.0,
                    "payoff_b": 5.0,
                    "epoch": 0,
                    "trust_a_to_b": 0.9,
                    "trust_b_to_a": 0.7,
                },
            )

            analyzer = TrustNetworkAnalyzer(store)
            matrix = analyzer.build_trust_matrix()

            assert "a0" in matrix
            assert "a1" in matrix
            # Average trust should be (0.9 + 0.7) / 2 = 0.8
            assert matrix["a0"]["a1"] == 0.8
            assert matrix["a1"]["a0"] == 0.8

    def test_build_trust_matrix_multiple_relationships(self):
        """Multiple relationships should all be in matrix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            # Add snapshots for 3 agents
            for agent_id in ["a0", "a1", "a2"]:
                snap = AgentMemorySnapshot(agent_id=agent_id, agent_type="honest")
                store.save(agent_id, snap)

            # Add relationships
            store.add_relationship_event(
                "a0",
                "a1",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.9, "trust_b_to_a": 0.7},
            )
            store.add_relationship_event(
                "a1",
                "a2",
                {"p": 0.6, "payoff_a": 5.0, "payoff_b": 8.0, "epoch": 0, "trust_a_to_b": 0.5, "trust_b_to_a": 0.6},
            )

            analyzer = TrustNetworkAnalyzer(store)
            matrix = analyzer.build_trust_matrix()

            assert len(matrix) == 3
            assert matrix["a0"]["a1"] == 0.8  # (0.9 + 0.7) / 2
            assert matrix["a1"]["a2"] == 0.55  # (0.5 + 0.6) / 2

    def test_compute_clusters_no_high_trust_edges(self):
        """No high-trust edges should result in singleton clusters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            # Add snapshots
            for agent_id in ["a0", "a1", "a2"]:
                snap = AgentMemorySnapshot(agent_id=agent_id, agent_type="honest")
                store.save(agent_id, snap)

            # Add low-trust relationships (below 0.7 threshold)
            store.add_relationship_event(
                "a0",
                "a1",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.4, "trust_b_to_a": 0.3},
            )

            analyzer = TrustNetworkAnalyzer(store)
            clusters = analyzer.compute_clusters(threshold=0.7)

            # Each agent should be in its own cluster
            assert len(clusters) == 3
            assert all(len(c) == 1 for c in clusters)

    def test_compute_clusters_high_trust_pair(self):
        """High-trust pair should form a single cluster."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            # Add snapshots
            for agent_id in ["a0", "a1", "a2"]:
                snap = AgentMemorySnapshot(agent_id=agent_id, agent_type="honest")
                store.save(agent_id, snap)

            # Add high-trust relationship (both > 0.7)
            store.add_relationship_event(
                "a0",
                "a1",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.8, "trust_b_to_a": 0.9},
            )

            analyzer = TrustNetworkAnalyzer(store)
            clusters = analyzer.compute_clusters(threshold=0.7)

            # a0 and a1 should be in one cluster, a2 in another
            assert len(clusters) == 2
            cluster_sizes = sorted([len(c) for c in clusters])
            assert cluster_sizes == [1, 2]

            # Find the 2-agent cluster
            two_agent_cluster = [c for c in clusters if len(c) == 2][0]
            assert set(two_agent_cluster) == {"a0", "a1"}

    def test_compute_clusters_triangle(self):
        """Three agents with mutual high trust should form one cluster."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            # Add snapshots
            for agent_id in ["a0", "a1", "a2"]:
                snap = AgentMemorySnapshot(agent_id=agent_id, agent_type="honest")
                store.save(agent_id, snap)

            # Add three high-trust relationships forming a triangle
            store.add_relationship_event(
                "a0",
                "a1",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.8, "trust_b_to_a": 0.8},
            )
            store.add_relationship_event(
                "a1",
                "a2",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.8, "trust_b_to_a": 0.8},
            )
            store.add_relationship_event(
                "a0",
                "a2",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.8, "trust_b_to_a": 0.8},
            )

            analyzer = TrustNetworkAnalyzer(store)
            clusters = analyzer.compute_clusters(threshold=0.7)

            # All three should be in one cluster
            assert len(clusters) == 1
            assert set(clusters[0]) == {"a0", "a1", "a2"}

    def test_compute_isolation_score_isolated_agent(self):
        """Agent with no relationships should have isolation score 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            snap = AgentMemorySnapshot(agent_id="a0", agent_type="honest")
            store.save("a0", snap)

            analyzer = TrustNetworkAnalyzer(store)
            score = analyzer.compute_isolation_score("a0")

            assert score == 1.0

    def test_compute_isolation_score_well_trusted(self):
        """Agent with high incoming trust should have low isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            # Add snapshots
            snap_a = AgentMemorySnapshot(agent_id="a0", agent_type="honest")
            snap_b = AgentMemorySnapshot(agent_id="a1", agent_type="honest")
            snap_c = AgentMemorySnapshot(agent_id="a2", agent_type="honest")
            store.save("a0", snap_a)
            store.save("a1", snap_b)
            store.save("a2", snap_c)

            # a0 receives high trust from a1 and a2
            store.add_relationship_event(
                "a1",
                "a0",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.9, "trust_b_to_a": 0.5},
            )
            store.add_relationship_event(
                "a2",
                "a0",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.8, "trust_b_to_a": 0.5},
            )

            analyzer = TrustNetworkAnalyzer(store)
            score = analyzer.compute_isolation_score("a0")

            # Incoming trust: 0.9 and 0.8, mean = 0.85
            # Isolation = 1 - 0.85 = 0.15
            assert abs(score - 0.15) < 0.01

    def test_compute_isolation_score_low_trusted(self):
        """Agent with low incoming trust should have high isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            # Add snapshots
            snap_a = AgentMemorySnapshot(agent_id="a0", agent_type="honest")
            snap_b = AgentMemorySnapshot(agent_id="a1", agent_type="honest")
            store.save("a0", snap_a)
            store.save("a1", snap_b)

            # a0 receives low trust from a1
            store.add_relationship_event(
                "a1",
                "a0",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.2, "trust_b_to_a": 0.5},
            )

            analyzer = TrustNetworkAnalyzer(store)
            score = analyzer.compute_isolation_score("a0")

            # Incoming trust: 0.2, mean = 0.2
            # Isolation = 1 - 0.2 = 0.8
            assert abs(score - 0.8) < 0.01

    def test_get_network_summary_empty(self):
        """Empty store should produce summary with defaults."""
        store = GraphMemoryStore()
        analyzer = TrustNetworkAnalyzer(store)

        summary = analyzer.get_network_summary()

        assert summary["n_agents"] == 0
        assert summary["n_edges"] == 0
        assert summary["cluster_count"] == 0
        assert summary["isolated_agents"] == []

    def test_get_network_summary_with_relationships(self):
        """Summary should include all expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            # Add snapshots
            for agent_id in ["a0", "a1", "a2"]:
                snap = AgentMemorySnapshot(agent_id=agent_id, agent_type="honest")
                store.save(agent_id, snap)

            # Add relationships
            store.add_relationship_event(
                "a0",
                "a1",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.8, "trust_b_to_a": 0.8},
            )
            store.add_relationship_event(
                "a1",
                "a2",
                {"p": 0.6, "payoff_a": 5.0, "payoff_b": 8.0, "epoch": 0, "trust_a_to_b": 0.3, "trust_b_to_a": 0.2},
            )

            analyzer = TrustNetworkAnalyzer(store)
            summary = analyzer.get_network_summary()

            assert summary["n_agents"] == 3
            assert summary["n_edges"] == 2
            assert "mean_trust" in summary
            assert "trust_reciprocity" in summary
            assert "cluster_count" in summary
            assert "isolated_agents" in summary

            # With high trust in first edge and low in second, mean should be around 0.575
            assert 0.5 < summary["mean_trust"] < 0.7

    def test_get_network_summary_isolated_agents(self):
        """Summary should identify isolated agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            # Add snapshots
            for agent_id in ["a0", "a1", "a2"]:
                snap = AgentMemorySnapshot(agent_id=agent_id, agent_type="honest")
                store.save(agent_id, snap)

            # Add low-trust relationship
            store.add_relationship_event(
                "a0",
                "a1",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.1, "trust_b_to_a": 0.15},
            )
            # a2 has no relationships (fully isolated)

            analyzer = TrustNetworkAnalyzer(store)
            summary = analyzer.get_network_summary()

            # a2 should be isolated (isolation=1.0 > 0.8)
            # a0 and a1 should be isolated (isolation ~0.85 > 0.8) due to low trust
            assert "a2" in summary["isolated_agents"]
            assert "a0" in summary["isolated_agents"]
            assert "a1" in summary["isolated_agents"]


class TestPlotTrustNetwork:
    """Tests for plot_trust_network function."""

    def test_plot_empty_network(self):
        """Plotting empty network should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore()
            analyzer = TrustNetworkAnalyzer(store)

            output_path = Path(tmpdir) / "empty_network.png"
            plot_trust_network(analyzer, str(output_path))

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_single_edge(self):
        """Plotting single edge should create valid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            # Add snapshots
            snap_a = AgentMemorySnapshot(agent_id="a0", agent_type="honest")
            snap_b = AgentMemorySnapshot(agent_id="a1", agent_type="deceptive")
            store.save("a0", snap_a)
            store.save("a1", snap_b)

            # Add relationship
            store.add_relationship_event(
                "a0",
                "a1",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.7, "trust_b_to_a": 0.8},
            )

            analyzer = TrustNetworkAnalyzer(store)
            output_path = Path(tmpdir) / "single_edge.png"
            plot_trust_network(analyzer, str(output_path))

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_complex_network(self):
        """Plotting complex network should create valid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            # Add 5 agents of different types
            agent_types = ["honest", "honest", "deceptive", "opportunistic", "adversarial"]
            for i, agent_type in enumerate(agent_types):
                snap = AgentMemorySnapshot(agent_id=f"a{i}", agent_type=agent_type)
                store.save(f"a{i}", snap)

            # Add multiple relationships
            store.add_relationship_event(
                "a0",
                "a1",
                {"p": 0.8, "payoff_a": 10.0, "payoff_b": 5.0, "epoch": 0, "trust_a_to_b": 0.9, "trust_b_to_a": 0.85},
            )
            store.add_relationship_event(
                "a1",
                "a2",
                {"p": 0.6, "payoff_a": 5.0, "payoff_b": 8.0, "epoch": 0, "trust_a_to_b": 0.4, "trust_b_to_a": 0.3},
            )
            store.add_relationship_event(
                "a2",
                "a3",
                {"p": 0.5, "payoff_a": 3.0, "payoff_b": 4.0, "epoch": 0, "trust_a_to_b": 0.2, "trust_b_to_a": 0.1},
            )
            store.add_relationship_event(
                "a0",
                "a4",
                {"p": 0.9, "payoff_a": 15.0, "payoff_b": 20.0, "epoch": 0, "trust_a_to_b": 0.95, "trust_b_to_a": 0.92},
            )

            analyzer = TrustNetworkAnalyzer(store)
            output_path = Path(tmpdir) / "complex_network.png"
            plot_trust_network(analyzer, str(output_path), title="Test Network")

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_plot_with_custom_figsize(self):
        """Plot should respect custom figure size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            snap = AgentMemorySnapshot(agent_id="a0", agent_type="honest")
            store.save("a0", snap)

            analyzer = TrustNetworkAnalyzer(store)
            output_path = Path(tmpdir) / "custom_size.png"
            plot_trust_network(analyzer, str(output_path), figsize=(16, 10))

            assert output_path.exists()
            # File size should be larger with larger figure
            assert output_path.stat().st_size > 0

    def test_plot_varying_trust_levels(self):
        """Plot should handle edges with varying trust levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = GraphMemoryStore(str(store_path))

            # Add 3 agents
            for i in range(3):
                snap = AgentMemorySnapshot(agent_id=f"a{i}", agent_type="honest")
                store.save(f"a{i}", snap)

            # Add edges with different trust levels
            trust_levels = [(0.1, 0.15), (0.5, 0.5), (0.9, 0.95)]
            for i, (t_a_to_b, t_b_to_a) in enumerate(trust_levels):
                store.add_relationship_event(
                    f"a{i}",
                    f"a{(i+1)%3}",
                    {
                        "p": 0.7,
                        "payoff_a": 10.0,
                        "payoff_b": 5.0,
                        "epoch": 0,
                        "trust_a_to_b": t_a_to_b,
                        "trust_b_to_a": t_b_to_a,
                    },
                )

            analyzer = TrustNetworkAnalyzer(store)
            output_path = Path(tmpdir) / "varying_trust.png"
            plot_trust_network(analyzer, str(output_path))

            assert output_path.exists()
            assert output_path.stat().st_size > 0
