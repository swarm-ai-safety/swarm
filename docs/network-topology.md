# Network Topology

The network module controls which agents can interact, enabling study of information cascades and coalition formation.

## Quick Start

```python
from src.core.orchestrator import Orchestrator, OrchestratorConfig
from src.env.network import NetworkConfig, NetworkTopology

# Configure small-world network
network_config = NetworkConfig(
    topology=NetworkTopology.SMALL_WORLD,
    k_neighbors=4,           # Initial connections to nearest neighbors
    rewire_probability=0.1,  # 10% chance to rewire each edge
    dynamic=True,            # Enable edge evolution
    edge_strengthen_rate=0.1,  # Strengthen on interaction
    edge_decay_rate=0.05,      # 5% decay per epoch
    min_edge_weight=0.2,       # Prune weak edges
)

config = OrchestratorConfig(
    n_epochs=20,
    network_config=network_config,
)
orchestrator = Orchestrator(config=config)
```

## Topology Types

| Topology | Description | Use Case |
|----------|-------------|----------|
| **Complete** | All agents connected | Baseline, small populations |
| **Ring** | Circular chain | Local information flow |
| **Star** | Hub-and-spoke | Centralized coordination |
| **Random (Erdos-Renyi)** | Edges with probability p | Random networks |
| **Small-World** | High clustering, short paths | Social networks |
| **Scale-Free** | Power-law degree distribution | Hubs and influence |
| **Custom** | Manual edge specification | Specific topologies |

## Dynamic Network Evolution

When `dynamic=True`, the network evolves based on agent behavior:

1. **Edge Strengthening**: Successful interactions increase edge weight
2. **Edge Decay**: All edges decay each epoch (models relationship maintenance costs)
3. **Edge Pruning**: Edges below `min_edge_weight` are removed
4. **Weight Capping**: Edges capped at `max_edge_weight`

```python
# After simulation, inspect network state
network = orchestrator.network
print(f"Connected components: {network.connected_components()}")
print(f"Average clustering: {network.get_metrics()['avg_clustering']}")
```

## Network Metrics

Each epoch includes network metrics:

| Metric | Description |
|--------|-------------|
| `n_edges` | Current edge count |
| `avg_degree` | Average connections per agent |
| `avg_clustering` | Local clustering coefficient |
| `n_components` | Number of connected components |
| `avg_path_length` | Average shortest path (if connected) |
| `density` | Edge density (edges / possible edges) |

## YAML Configuration

```yaml
network:
  topology: small_world
  params:
    k: 4     # k_neighbors
    p: 0.1   # rewire_probability

  dynamic: true
  edge_strengthen_rate: 0.1
  edge_decay_rate: 0.05
  min_edge_weight: 0.2
  max_edge_weight: 1.0

  # Optional: disconnect from agents with bad reputation
  # reputation_disconnect_threshold: -2.0

success_criteria:
  min_connected_components: 1
  max_avg_path_length: 4.0
```

## Integration with Orchestrator

The network constrains interactions:
- **Observation filtering**: Agents only see neighbors in their observations
- **Interaction validation**: Actions targeting non-neighbors are rejected
- **Edge strengthening**: Accepted interactions strengthen the edge between agents

Run the network effects scenario:
```bash
python examples/run_scenario.py scenarios/network_effects.yaml
```
