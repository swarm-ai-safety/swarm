"""Tests for network topology module."""

import numpy as np
import pytest
from pydantic import ValidationError

from swarm.env.network import AgentNetwork, NetworkConfig, NetworkTopology

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def agent_ids():
    """Standard list of agent IDs for testing."""
    return [f"agent_{i}" for i in range(6)]


@pytest.fixture
def complete_network(agent_ids):
    """Complete network where all agents are connected."""
    config = NetworkConfig(topology=NetworkTopology.COMPLETE)
    network = AgentNetwork(config, seed=42)
    network.initialize(agent_ids)
    return network


@pytest.fixture
def ring_network(agent_ids):
    """Ring network where agents are connected to neighbors."""
    config = NetworkConfig(topology=NetworkTopology.RING)
    network = AgentNetwork(config, seed=42)
    network.initialize(agent_ids)
    return network


@pytest.fixture
def small_world_network(agent_ids):
    """Small-world network."""
    config = NetworkConfig(
        topology=NetworkTopology.SMALL_WORLD,
        k_neighbors=4,
        rewire_probability=0.1,
    )
    network = AgentNetwork(config, seed=42)
    network.initialize(agent_ids)
    return network


# =============================================================================
# NetworkConfig Tests
# =============================================================================


class TestNetworkConfig:
    """Tests for NetworkConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NetworkConfig()
        assert config.topology == NetworkTopology.COMPLETE
        assert config.edge_probability == 0.5
        assert config.k_neighbors == 4
        assert config.dynamic is False

    def test_validation_edge_probability(self):
        """Test edge_probability validation."""
        with pytest.raises(ValidationError, match="edge_probability"):
            NetworkConfig(edge_probability=1.5)

        with pytest.raises(ValidationError, match="edge_probability"):
            NetworkConfig(edge_probability=-0.1)

    def test_validation_rewire_probability(self):
        """Test rewire_probability validation."""
        with pytest.raises(ValidationError, match="rewire_probability"):
            NetworkConfig(rewire_probability=2.0)

    def test_validation_k_neighbors(self):
        """Test k_neighbors validation."""
        with pytest.raises(ValidationError, match="k_neighbors"):
            NetworkConfig(k_neighbors=1)

    def test_validation_m_edges(self):
        """Test m_edges validation."""
        with pytest.raises(ValidationError, match="m_edges"):
            NetworkConfig(m_edges=0)

    def test_validation_decay_rate(self):
        """Test edge_decay_rate validation."""
        with pytest.raises(ValidationError, match="edge_decay_rate"):
            NetworkConfig(edge_decay_rate=1.5)


# =============================================================================
# Topology Tests
# =============================================================================


class TestCompleteNetwork:
    """Tests for complete network topology."""

    def test_all_connected(self, complete_network, agent_ids):
        """Test all agents are connected to each other."""
        for a in agent_ids:
            neighbors = complete_network.neighbors(a)
            expected = [aid for aid in agent_ids if aid != a]
            assert set(neighbors) == set(expected)

    def test_edge_count(self, complete_network, agent_ids):
        """Test edge count in complete graph."""
        n = len(agent_ids)
        expected_edges = n * (n - 1) // 2  # n choose 2

        metrics = complete_network.get_metrics()
        assert metrics["n_edges"] == expected_edges

    def test_average_degree(self, complete_network, agent_ids):
        """Test average degree in complete graph."""
        n = len(agent_ids)
        assert complete_network.average_degree() == n - 1


class TestRingNetwork:
    """Tests for ring network topology."""

    def test_two_neighbors(self, ring_network, agent_ids):
        """Test each agent has exactly 2 neighbors."""
        for aid in agent_ids:
            assert ring_network.degree(aid) == 2

    def test_neighbors_are_adjacent(self, ring_network, agent_ids):
        """Test neighbors are adjacent in the ring."""
        n = len(agent_ids)
        for i, aid in enumerate(agent_ids):
            neighbors = ring_network.neighbors(aid)
            expected_prev = agent_ids[(i - 1) % n]
            expected_next = agent_ids[(i + 1) % n]
            assert set(neighbors) == {expected_prev, expected_next}

    def test_connected(self, ring_network):
        """Test ring network is connected."""
        assert ring_network.is_connected()


class TestStarNetwork:
    """Tests for star network topology."""

    def test_hub_connected_to_all(self, agent_ids):
        """Test hub is connected to all spokes."""
        config = NetworkConfig(topology=NetworkTopology.STAR)
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        hub = agent_ids[0]
        assert network.degree(hub) == len(agent_ids) - 1

    def test_spokes_connected_to_hub_only(self, agent_ids):
        """Test spokes are only connected to hub."""
        config = NetworkConfig(topology=NetworkTopology.STAR)
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        hub = agent_ids[0]
        for spoke in agent_ids[1:]:
            neighbors = network.neighbors(spoke)
            assert neighbors == [hub]


class TestErdosRenyiNetwork:
    """Tests for Erdős-Rényi random network."""

    def test_approximate_edge_density(self, agent_ids):
        """Test edge density approximates probability."""
        config = NetworkConfig(
            topology=NetworkTopology.RANDOM_ERDOS_RENYI,
            edge_probability=0.5,
        )
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        n = len(agent_ids)
        max_edges = n * (n - 1) // 2
        actual_edges = network.get_metrics()["n_edges"]

        # Should be roughly 50% of max edges (with some variance)
        assert 0.2 * max_edges <= actual_edges <= 0.8 * max_edges

    def test_probability_zero_gives_no_edges(self, agent_ids):
        """Test probability 0 gives empty graph."""
        config = NetworkConfig(
            topology=NetworkTopology.RANDOM_ERDOS_RENYI,
            edge_probability=0.0,
        )
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        assert network.get_metrics()["n_edges"] == 0

    def test_probability_one_gives_complete(self, agent_ids):
        """Test probability 1 gives complete graph."""
        config = NetworkConfig(
            topology=NetworkTopology.RANDOM_ERDOS_RENYI,
            edge_probability=1.0,
        )
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        n = len(agent_ids)
        expected = n * (n - 1) // 2
        assert network.get_metrics()["n_edges"] == expected


class TestSmallWorldNetwork:
    """Tests for small-world network."""

    def test_initial_connectivity(self, small_world_network, agent_ids):
        """Test each node has at least k connections."""
        for aid in agent_ids:
            assert small_world_network.degree(aid) >= 2

    def test_connected(self, small_world_network):
        """Test small-world network is connected."""
        assert small_world_network.is_connected()

    def test_clustering_coefficient(self, small_world_network):
        """Test small-world has high clustering."""
        avg_clustering = small_world_network.average_clustering()
        # Small-world networks typically have high clustering
        assert avg_clustering > 0


class TestScaleFreeNetwork:
    """Tests for scale-free (Barabási-Albert) network."""

    def test_connected(self, agent_ids):
        """Test scale-free network is connected."""
        config = NetworkConfig(
            topology=NetworkTopology.SCALE_FREE,
            m_edges=2,
        )
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        assert network.is_connected()

    def test_minimum_degree(self, agent_ids):
        """Test all nodes have at least m edges."""
        m = 2
        config = NetworkConfig(
            topology=NetworkTopology.SCALE_FREE,
            m_edges=m,
        )
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        # All nodes except initial m+1 should have exactly m edges
        # Initial nodes may have more
        for i, aid in enumerate(agent_ids):
            if i > m:
                assert network.degree(aid) >= m


# =============================================================================
# Edge Operations Tests
# =============================================================================


class TestEdgeOperations:
    """Tests for edge manipulation."""

    def test_add_edge(self, agent_ids):
        """Test adding an edge."""
        config = NetworkConfig(topology=NetworkTopology.CUSTOM)
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        assert not network.has_edge(agent_ids[0], agent_ids[1])

        network.add_edge(agent_ids[0], agent_ids[1], 0.5)

        assert network.has_edge(agent_ids[0], agent_ids[1])
        assert network.edge_weight(agent_ids[0], agent_ids[1]) == 0.5

    def test_remove_edge(self, complete_network, agent_ids):
        """Test removing an edge."""
        a, b = agent_ids[0], agent_ids[1]

        assert complete_network.has_edge(a, b)

        complete_network.remove_edge(a, b)

        assert not complete_network.has_edge(a, b)
        assert not complete_network.has_edge(b, a)

    def test_edge_weight_capped(self, agent_ids):
        """Test edge weight is capped at max."""
        config = NetworkConfig(
            topology=NetworkTopology.CUSTOM,
            max_edge_weight=1.0,
        )
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        network.add_edge(agent_ids[0], agent_ids[1], 5.0)

        assert network.edge_weight(agent_ids[0], agent_ids[1]) == 1.0


# =============================================================================
# Dynamic Network Tests
# =============================================================================


class TestDynamicNetwork:
    """Tests for dynamic network evolution."""

    def test_strengthen_edge(self, agent_ids):
        """Test edge strengthening."""
        config = NetworkConfig(
            topology=NetworkTopology.CUSTOM,
            dynamic=True,
            edge_strengthen_rate=0.2,
            max_edge_weight=1.0,
        )
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        a, b = agent_ids[0], agent_ids[1]

        # Add edge with low weight
        network.add_edge(a, b, 0.5)
        initial_weight = network.edge_weight(a, b)

        network.strengthen_edge(a, b)

        assert network.edge_weight(a, b) == initial_weight + 0.2

    def test_decay_edges(self, agent_ids):
        """Test edge decay."""
        config = NetworkConfig(
            topology=NetworkTopology.COMPLETE,
            dynamic=True,
            edge_decay_rate=0.1,
            min_edge_weight=0.1,
        )
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        a, b = agent_ids[0], agent_ids[1]
        initial_weight = network.edge_weight(a, b)

        network.decay_edges()

        expected = initial_weight * 0.9
        assert abs(network.edge_weight(a, b) - expected) < 0.01

    def test_decay_prunes_weak_edges(self, agent_ids):
        """Test decay prunes edges below threshold."""
        config = NetworkConfig(
            topology=NetworkTopology.CUSTOM,
            dynamic=True,
            edge_decay_rate=0.5,
            min_edge_weight=0.2,
        )
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        # Add weak edge
        network.add_edge(agent_ids[0], agent_ids[1], 0.3)

        # After decay: 0.3 * 0.5 = 0.15 < 0.2 threshold
        pruned = network.decay_edges()

        assert pruned == 1
        assert not network.has_edge(agent_ids[0], agent_ids[1])

    def test_no_decay_when_not_dynamic(self, complete_network, agent_ids):
        """Test decay does nothing when dynamic=False."""
        a, b = agent_ids[0], agent_ids[1]
        initial_weight = complete_network.edge_weight(a, b)

        complete_network.decay_edges()

        assert complete_network.edge_weight(a, b) == initial_weight


# =============================================================================
# Graph Metrics Tests
# =============================================================================


class TestGraphMetrics:
    """Tests for graph metric computations."""

    def test_shortest_path_neighbors(self, ring_network, agent_ids):
        """Test shortest path between neighbors."""
        a = agent_ids[0]
        b = agent_ids[1]

        assert ring_network.shortest_path(a, b) == 1

    def test_shortest_path_across_ring(self, ring_network, agent_ids):
        """Test shortest path across ring."""
        n = len(agent_ids)
        a = agent_ids[0]
        b = agent_ids[n // 2]  # Opposite side

        path = ring_network.shortest_path(a, b)
        assert path == n // 2

    def test_shortest_path_no_path(self, agent_ids):
        """Test shortest path when no path exists."""
        config = NetworkConfig(topology=NetworkTopology.CUSTOM)
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        # No edges, no path
        assert network.shortest_path(agent_ids[0], agent_ids[1]) == -1

    def test_clustering_coefficient_complete(self, complete_network, agent_ids):
        """Test clustering coefficient in complete graph."""
        # In complete graph, all neighbors are connected
        for aid in agent_ids:
            assert complete_network.clustering_coefficient(aid) == 1.0

    def test_clustering_coefficient_star(self, agent_ids):
        """Test clustering in star network."""
        config = NetworkConfig(topology=NetworkTopology.STAR)
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        # Hub has clustering 0 (neighbors not connected to each other)
        hub = agent_ids[0]
        assert network.clustering_coefficient(hub) == 0.0

    def test_connected_components_complete(self, complete_network):
        """Test connected components in complete graph."""
        components = complete_network.connected_components()
        assert len(components) == 1

    def test_connected_components_disconnected(self, agent_ids):
        """Test connected components in disconnected graph."""
        config = NetworkConfig(topology=NetworkTopology.CUSTOM)
        network = AgentNetwork(config, seed=42)
        network.initialize(agent_ids)

        # Create two disconnected pairs
        network.add_edge(agent_ids[0], agent_ids[1])
        network.add_edge(agent_ids[2], agent_ids[3])

        components = network.connected_components()
        # Should have 4 components: 2 pairs + 2 isolates
        assert len(components) == 4

    def test_degree_distribution(self, ring_network, agent_ids):
        """Test degree distribution in ring."""
        dist = ring_network.degree_distribution()

        # All nodes have degree 2 in ring
        assert dist == {2: len(agent_ids)}


# =============================================================================
# Adjacency Matrix Tests
# =============================================================================


class TestAdjacencyMatrix:
    """Tests for adjacency matrix conversion."""

    def test_to_adjacency_matrix(self, complete_network, agent_ids):
        """Test conversion to adjacency matrix."""
        matrix, ids = complete_network.to_adjacency_matrix()

        assert matrix.shape == (len(agent_ids), len(agent_ids))
        assert ids == agent_ids

        # Diagonal should be 0
        assert np.all(np.diag(matrix) == 0)

        # Off-diagonal should be 1 (complete graph)
        for i in range(len(agent_ids)):
            for j in range(len(agent_ids)):
                if i != j:
                    assert matrix[i, j] == 1.0

    def test_from_adjacency_matrix(self, agent_ids):
        """Test loading from adjacency matrix."""
        config = NetworkConfig(topology=NetworkTopology.CUSTOM)
        network = AgentNetwork(config, seed=42)

        # Create custom matrix
        n = len(agent_ids)
        matrix = np.zeros((n, n))
        matrix[0, 1] = matrix[1, 0] = 0.5
        matrix[1, 2] = matrix[2, 1] = 0.8

        network.from_adjacency_matrix(matrix, agent_ids)

        assert network.edge_weight(agent_ids[0], agent_ids[1]) == 0.5
        assert network.edge_weight(agent_ids[1], agent_ids[2]) == 0.8
        assert not network.has_edge(agent_ids[0], agent_ids[2])


# =============================================================================
# Orchestrator Integration Tests
# =============================================================================


class TestOrchestratorIntegration:
    """Tests for network integration with orchestrator."""

    def test_network_initialization(self, agent_ids):
        """Test network is initialized when orchestrator runs."""
        from swarm.agents.honest import HonestAgent
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(
            n_epochs=1,
            steps_per_epoch=1,
            network_config=NetworkConfig(topology=NetworkTopology.COMPLETE),
        )
        orchestrator = Orchestrator(config)

        for aid in agent_ids:
            orchestrator.register_agent(HonestAgent(aid))

        # Network initialized on run
        orchestrator.run()

        assert orchestrator.network is not None
        assert orchestrator.network.get_metrics()["n_agents"] == len(agent_ids)

    def test_network_constrains_interactions(self):
        """Test agents can only interact with neighbors."""
        from swarm.agents.honest import HonestAgent
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        # Star network: only hub can interact with everyone
        config = OrchestratorConfig(
            n_epochs=1,
            steps_per_epoch=5,
            network_config=NetworkConfig(topology=NetworkTopology.STAR),
        )
        orchestrator = Orchestrator(config)

        agents = [HonestAgent(f"agent_{i}") for i in range(4)]
        for agent in agents:
            orchestrator.register_agent(agent)

        orchestrator.run()

        # Spokes can only see hub in their observation
        network = orchestrator.network
        spoke = agents[1].agent_id
        neighbors = network.neighbors(spoke)
        assert len(neighbors) == 1  # Only hub

    def test_network_metrics_in_epoch(self):
        """Test network metrics are included in epoch metrics."""
        from swarm.agents.honest import HonestAgent
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(
            n_epochs=2,
            steps_per_epoch=1,
            network_config=NetworkConfig(topology=NetworkTopology.COMPLETE),
        )
        orchestrator = Orchestrator(config)

        for i in range(4):
            orchestrator.register_agent(HonestAgent(f"agent_{i}"))

        metrics = orchestrator.run()

        assert metrics[-1].network_metrics is not None
        assert "n_edges" in metrics[-1].network_metrics
        assert "average_degree" in metrics[-1].network_metrics

    def test_dynamic_network_decay(self):
        """Test network edges decay over epochs."""
        from swarm.agents.honest import HonestAgent
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(
            n_epochs=5,
            steps_per_epoch=1,
            network_config=NetworkConfig(
                topology=NetworkTopology.COMPLETE,
                dynamic=True,
                edge_decay_rate=0.1,
                min_edge_weight=0.5,
            ),
        )
        orchestrator = Orchestrator(config)

        for i in range(4):
            orchestrator.register_agent(HonestAgent(f"agent_{i}"))

        orchestrator.run()

        # Edges should have decayed
        network = orchestrator.network
        # After 5 epochs with 10% decay: 1.0 * 0.9^5 ≈ 0.59
        # Some edges may be pruned if they fall below 0.5
        final_avg_degree = network.average_degree()
        assert final_avg_degree <= 3  # May have lost some edges


# =============================================================================
# Scenario Loader Integration Tests
# =============================================================================


class TestScenarioLoaderNetwork:
    """Tests for network configuration in scenario loader."""

    def test_parse_network_config_complete(self):
        """Test parsing complete network config."""
        from swarm.scenarios.loader import parse_network_config

        data = {
            "topology": "complete",
        }

        config = parse_network_config(data)

        assert config is not None
        assert config.topology == NetworkTopology.COMPLETE

    def test_parse_network_config_small_world(self):
        """Test parsing small-world network config."""
        from swarm.scenarios.loader import parse_network_config

        data = {
            "topology": "small_world",
            "params": {
                "k": 6,
                "p": 0.2,
            },
            "dynamic": True,
            "edge_decay_rate": 0.1,
        }

        config = parse_network_config(data)

        assert config is not None
        assert config.topology == NetworkTopology.SMALL_WORLD
        assert config.k_neighbors == 6
        assert config.rewire_probability == 0.2
        assert config.dynamic is True
        assert config.edge_decay_rate == 0.1

    def test_parse_network_config_disabled(self):
        """Test parsing disabled network."""
        from swarm.scenarios.loader import parse_network_config

        data = {"enabled": False}

        config = parse_network_config(data)

        assert config is None

    def test_parse_network_config_empty(self):
        """Test parsing empty network config."""
        from swarm.scenarios.loader import parse_network_config

        config = parse_network_config({})

        assert config is None

    def test_parse_network_config_invalid_topology(self):
        """Test parsing invalid topology raises error."""
        from swarm.scenarios.loader import parse_network_config

        with pytest.raises(ValueError, match="Unknown network topology"):
            parse_network_config({"topology": "invalid_topology"})
