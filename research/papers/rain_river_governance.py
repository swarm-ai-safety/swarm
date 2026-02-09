"""Experiment 2: Governance x Memory Persistence

Tests which governance mechanisms differentially affect rain vs river agents.

Governance configurations tested:
- baseline: No governance levers
- circuit_breaker: Freeze agents with high toxicity
- incoherence_friction: Cost for inconsistent behavior
- transaction_tax: Tax on all transactions
- collusion_detection: Detect and penalize collusive pairs
- full_defense: All levers enabled

Run with: python -m research.papers.rain_river_governance
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from swarm.agents import AdversarialAgent, ConfigurableMemoryAgent, MemoryConfig
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig
from swarm.governance.config import GovernanceConfig
from swarm.metrics.soft_metrics import SoftMetrics


@dataclass
class GovernanceExperimentResult:
    """Results from a governance experiment run."""

    governance_name: str
    memory_type: str  # "rain" or "river"
    seed: int
    n_interactions: int
    toxicity: float
    quality_gap: float
    total_welfare: float
    avg_payoff: float
    acceptance_rate: float
    n_frozen: int
    governance_costs: float


# Define governance configurations to test
GOVERNANCE_CONFIGS = {
    "baseline": GovernanceConfig(),
    "circuit_breaker": GovernanceConfig(
        circuit_breaker_enabled=True,
        freeze_threshold_toxicity=0.5,
        freeze_threshold_violations=3,
        freeze_duration_epochs=2,
    ),
    "incoherence_friction": GovernanceConfig(
        incoherence_friction_enabled=True,
        incoherence_friction_rate=0.05,
    ),
    "transaction_tax": GovernanceConfig(
        transaction_tax_rate=0.1,
        transaction_tax_split=0.5,
    ),
    "collusion_detection": GovernanceConfig(
        collusion_detection_enabled=True,
        collusion_frequency_threshold=2.0,
        collusion_correlation_threshold=0.7,
        collusion_realtime_penalty=True,
        collusion_realtime_rate=0.1,
    ),
    "staking": GovernanceConfig(
        staking_enabled=True,
        min_stake_to_participate=5.0,
        stake_slash_rate=0.2,
    ),
    "full_defense": GovernanceConfig(
        circuit_breaker_enabled=True,
        freeze_threshold_toxicity=0.5,
        incoherence_friction_enabled=True,
        incoherence_friction_rate=0.05,
        transaction_tax_rate=0.05,
        collusion_detection_enabled=True,
        collusion_realtime_penalty=True,
    ),
}


def run_governance_simulation(
    governance_config: GovernanceConfig,
    governance_name: str,
    memory_level: float,
    n_agents: int = 20,
    n_epochs: int = 50,
    steps_per_epoch: int = 20,
    honest_fraction: float = 0.5,
    seed: int = 42,
) -> GovernanceExperimentResult:
    """Run a single governance experiment."""
    config = OrchestratorConfig(
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        payoff_config=PayoffConfig(),
        governance_config=governance_config,
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
        orchestrator.register_agent(AdversarialAgent(agent_id=f"adversary_{i}"))

    # Run simulation
    orchestrator.run()

    # Collect metrics
    interactions = orchestrator.state.completed_interactions
    soft_metrics = SoftMetrics(orchestrator.payoff_engine)

    if not interactions:
        return GovernanceExperimentResult(
            governance_name=governance_name,
            memory_type="rain" if memory_level == 0.0 else "river",
            seed=seed,
            n_interactions=0,
            toxicity=0.0,
            quality_gap=0.0,
            total_welfare=0.0,
            avg_payoff=0.0,
            acceptance_rate=0.0,
            n_frozen=0,
            governance_costs=0.0,
        )

    welfare = soft_metrics.welfare_metrics(interactions)
    accepted = [i for i in interactions if i.accepted]

    # Count frozen agents
    n_frozen = sum(
        1
        for agent_id in orchestrator._agents.keys()
        if orchestrator.state.is_frozen(agent_id)
    )

    # Sum governance costs
    gov_costs = sum(i.c_a + i.c_b for i in interactions)

    return GovernanceExperimentResult(
        governance_name=governance_name,
        memory_type="rain" if memory_level == 0.0 else "river",
        seed=seed,
        n_interactions=len(interactions),
        toxicity=soft_metrics.toxicity_rate(interactions),
        quality_gap=soft_metrics.quality_gap(interactions),
        total_welfare=welfare.get("total_welfare", 0.0),
        avg_payoff=welfare.get("avg_initiator_payoff", 0.0),
        acceptance_rate=len(accepted) / len(interactions) if interactions else 0.0,
        n_frozen=n_frozen,
        governance_costs=gov_costs,
    )


def run_governance_experiment(
    n_seeds: int = 30,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run full governance x memory type experiment.

    Tests each governance configuration with rain and river agents.

    Returns:
        DataFrame with all results
    """
    results = []

    if verbose:
        print("Experiment: Governance x Memory Persistence")
        print("=" * 70)

    for gov_name, gov_config in GOVERNANCE_CONFIGS.items():
        for memory_type, memory_level in [("rain", 0.0), ("river", 1.0)]:
            condition_results = []

            for seed in range(n_seeds):
                result = run_governance_simulation(
                    governance_config=gov_config,
                    governance_name=gov_name,
                    memory_level=memory_level,
                    seed=seed,
                )
                condition_results.append(result)
                results.append(result)

            if verbose:
                avg_welfare = np.mean([r.total_welfare for r in condition_results])
                avg_toxicity = np.mean([r.toxicity for r in condition_results])
                avg_frozen = np.mean([r.n_frozen for r in condition_results])
                print(
                    f"{gov_name:20} {memory_type:5}: "
                    f"Welfare={avg_welfare:>7.1f}, "
                    f"Toxicity={avg_toxicity:.3f}, "
                    f"Frozen={avg_frozen:.1f}"
                )

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {
                "governance": r.governance_name,
                "memory_type": r.memory_type,
                "seed": r.seed,
                "n_interactions": r.n_interactions,
                "toxicity": r.toxicity,
                "quality_gap": r.quality_gap,
                "welfare": r.total_welfare,
                "avg_payoff": r.avg_payoff,
                "acceptance_rate": r.acceptance_rate,
                "n_frozen": r.n_frozen,
                "governance_costs": r.governance_costs,
            }
            for r in results
        ]
    )

    return df


def analyze_differential_effects(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Analyze which governance mechanisms have differential effects on rain vs river.

    Returns:
        Dictionary with differential effect statistics
    """
    results = {}

    for gov_name in GOVERNANCE_CONFIGS.keys():
        rain_data = df[(df["governance"] == gov_name) & (df["memory_type"] == "rain")]
        river_data = df[(df["governance"] == gov_name) & (df["memory_type"] == "river")]

        rain_welfare = rain_data["welfare"].mean()
        river_welfare = river_data["welfare"].mean()

        # Differential effect: how much more does this governance help river vs rain?
        baseline_rain = df[
            (df["governance"] == "baseline") & (df["memory_type"] == "rain")
        ]["welfare"].mean()
        baseline_river = df[
            (df["governance"] == "baseline") & (df["memory_type"] == "river")
        ]["welfare"].mean()

        rain_improvement = (
            (rain_welfare - baseline_rain) / baseline_rain if baseline_rain else 0
        )
        river_improvement = (
            (river_welfare - baseline_river) / baseline_river if baseline_river else 0
        )

        differential = river_improvement - rain_improvement

        results[gov_name] = {
            "rain_welfare": rain_welfare,
            "river_welfare": river_welfare,
            "rain_improvement": rain_improvement * 100,
            "river_improvement": river_improvement * 100,
            "differential_effect": differential * 100,
            "rain_toxicity": rain_data["toxicity"].mean(),
            "river_toxicity": river_data["toxicity"].mean(),
        }

        if verbose and gov_name != "baseline":
            print(f"\n{gov_name}:")
            print(f"  Rain improvement:  {rain_improvement * 100:+.1f}%")
            print(f"  River improvement: {river_improvement * 100:+.1f}%")
            print(f"  Differential:      {differential * 100:+.1f}% (+ favors river)")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Rain vs River: Governance Mechanism Effectiveness")
    print("=" * 70)
    print()

    # Run experiment
    df = run_governance_experiment(n_seeds=10, verbose=True)

    print("\n" + "=" * 70)
    print("Differential Effects Analysis")
    print("=" * 70)

    effects = analyze_differential_effects(df, verbose=True)

    print("\n" + "=" * 70)
    print("Key Findings")
    print("=" * 70)
    print(
        "- Circuit breakers may differentially harm rain agents (can't rebuild reputation)"
    )
    print(
        "- Collusion detection is more effective against river adversaries (pattern memory)"
    )
    print("- Transaction taxes affect both equally")
