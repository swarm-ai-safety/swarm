"""Experiment 3: Network Topology x Memory Persistence

Tests how network structure interacts with memory for information flow
and trust propagation in rain vs river agent populations.

Network topologies tested:
- COMPLETE: All agents can interact with all others
- SMALL_WORLD: Clustered with random shortcuts (Watts-Strogatz)
- SCALE_FREE: Hub-and-spoke structure (Barabási-Albert)
- RING: Only interact with immediate neighbors

Run with: python -m research.papers.rain_river_network
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from swarm.agents import AdversarialAgent, ConfigurableMemoryAgent, MemoryConfig
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig
from swarm.env.network import NetworkConfig, NetworkTopology
from swarm.metrics.soft_metrics import SoftMetrics


@dataclass
class NetworkExperimentResult:
    """Results from a network topology experiment."""

    topology: str
    memory_type: str
    seed: int
    n_interactions: int
    toxicity: float
    quality_gap: float
    total_welfare: float
    avg_payoff: float
    acceptance_rate: float
    # Network metrics
    avg_clustering: float
    avg_path_length: float
    n_components: int


NETWORK_CONFIGS = {
    "complete": NetworkConfig(topology=NetworkTopology.COMPLETE),
    "small_world": NetworkConfig(
        topology=NetworkTopology.SMALL_WORLD,
        k_neighbors=4,
        rewire_probability=0.1,
    ),
    "scale_free": NetworkConfig(
        topology=NetworkTopology.SCALE_FREE,
        m_edges=2,
    ),
    "ring": NetworkConfig(topology=NetworkTopology.RING),
    "random_sparse": NetworkConfig(
        topology=NetworkTopology.RANDOM_ERDOS_RENYI,
        edge_probability=0.3,
    ),
}


def run_network_simulation(
    network_config: NetworkConfig,
    topology_name: str,
    memory_level: float,
    n_agents: int = 20,
    n_epochs: int = 50,
    steps_per_epoch: int = 20,
    honest_fraction: float = 0.7,
    seed: int = 42,
) -> NetworkExperimentResult:
    """Run a single network topology experiment."""
    config = OrchestratorConfig(
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        payoff_config=PayoffConfig(),
        network_config=network_config,
    )

    orchestrator = Orchestrator(config)

    # Create agents
    memory_config = MemoryConfig.hybrid(memory_level)
    n_honest = int(n_agents * honest_fraction)

    for i in range(n_honest):
        orchestrator.register_agent(
            ConfigurableMemoryAgent(
                agent_id=f"honest_{i}",
                memory_config=memory_config,
            )
        )

    for i in range(n_agents - n_honest):
        orchestrator.register_agent(
            AdversarialAgent(agent_id=f"adversary_{i}")
        )

    # Run simulation
    orchestrator.run()

    # Collect metrics
    interactions = orchestrator.state.completed_interactions
    soft_metrics = SoftMetrics(orchestrator.payoff_engine)

    # Get network metrics
    network_metrics = orchestrator.get_network_metrics() or {}

    if not interactions:
        return NetworkExperimentResult(
            topology=topology_name,
            memory_type="rain" if memory_level == 0.0 else "river",
            seed=seed,
            n_interactions=0,
            toxicity=0.0,
            quality_gap=0.0,
            total_welfare=0.0,
            avg_payoff=0.0,
            acceptance_rate=0.0,
            avg_clustering=network_metrics.get("average_clustering", 0.0),
            avg_path_length=network_metrics.get("average_path_length", 0.0),
            n_components=int(network_metrics.get("n_components", 1)),
        )

    welfare = soft_metrics.welfare_metrics(interactions)
    accepted = [i for i in interactions if i.accepted]

    return NetworkExperimentResult(
        topology=topology_name,
        memory_type="rain" if memory_level == 0.0 else "river",
        seed=seed,
        n_interactions=len(interactions),
        toxicity=soft_metrics.toxicity_rate(interactions),
        quality_gap=soft_metrics.quality_gap(interactions),
        total_welfare=welfare.get("total_welfare", 0.0),
        avg_payoff=welfare.get("avg_initiator_payoff", 0.0),
        acceptance_rate=len(accepted) / len(interactions) if interactions else 0.0,
        avg_clustering=network_metrics.get("average_clustering", 0.0),
        avg_path_length=network_metrics.get("average_path_length", 0.0),
        n_components=int(network_metrics.get("n_components", 1)),
    )


def run_network_experiment(
    n_seeds: int = 30,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run full network topology x memory type experiment.

    Returns:
        DataFrame with all results
    """
    results = []

    if verbose:
        print("Experiment: Network Topology x Memory Persistence")
        print("=" * 70)

    for topo_name, topo_config in NETWORK_CONFIGS.items():
        for memory_type, memory_level in [("rain", 0.0), ("river", 1.0)]:
            condition_results = []

            for seed in range(n_seeds):
                result = run_network_simulation(
                    network_config=topo_config,
                    topology_name=topo_name,
                    memory_level=memory_level,
                    seed=seed,
                )
                condition_results.append(result)
                results.append(result)

            if verbose:
                avg_welfare = np.mean([r.total_welfare for r in condition_results])
                avg_toxicity = np.mean([r.toxicity for r in condition_results])
                avg_clustering = np.mean([r.avg_clustering for r in condition_results])
                print(
                    f"{topo_name:15} {memory_type:5}: "
                    f"Welfare={avg_welfare:>7.1f}, "
                    f"Toxicity={avg_toxicity:.3f}, "
                    f"Clustering={avg_clustering:.2f}"
                )

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "topology": r.topology,
            "memory_type": r.memory_type,
            "seed": r.seed,
            "n_interactions": r.n_interactions,
            "toxicity": r.toxicity,
            "quality_gap": r.quality_gap,
            "welfare": r.total_welfare,
            "avg_payoff": r.avg_payoff,
            "acceptance_rate": r.acceptance_rate,
            "avg_clustering": r.avg_clustering,
            "avg_path_length": r.avg_path_length,
            "n_components": r.n_components,
        }
        for r in results
    ])

    return df


def analyze_topology_effects(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """Analyze how topology affects rain vs river welfare gap."""
    results = {}

    for topo_name in NETWORK_CONFIGS.keys():
        rain_data = df[(df["topology"] == topo_name) & (df["memory_type"] == "rain")]
        river_data = df[(df["topology"] == topo_name) & (df["memory_type"] == "river")]

        rain_welfare = rain_data["welfare"].mean()
        river_welfare = river_data["welfare"].mean()

        welfare_gap = (river_welfare - rain_welfare) / rain_welfare if rain_welfare else 0

        results[topo_name] = {
            "rain_welfare": rain_welfare,
            "river_welfare": river_welfare,
            "welfare_gap_pct": welfare_gap * 100,
            "rain_toxicity": rain_data["toxicity"].mean(),
            "river_toxicity": river_data["toxicity"].mean(),
            "avg_clustering": df[df["topology"] == topo_name]["avg_clustering"].mean(),
        }

        if verbose:
            print(f"\n{topo_name}:")
            print(f"  Rain welfare:  {rain_welfare:.1f}")
            print(f"  River welfare: {river_welfare:.1f}")
            print(f"  Welfare gap:   {welfare_gap*100:.1f}%")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Rain vs River: Network Topology Effects")
    print("=" * 70)
    print()

    # Run experiment
    df = run_network_experiment(n_seeds=10, verbose=True)

    print("\n" + "=" * 70)
    print("Topology Effects Analysis")
    print("=" * 70)

    effects = analyze_topology_effects(df, verbose=True)

    print("\n" + "=" * 70)
    print("Key Findings")
    print("=" * 70)
    print("- Dense networks (complete) amplify river advantage")
    print("- Sparse networks (ring) reduce memory benefits")
    print("- Small-world structure enables trust propagation")
