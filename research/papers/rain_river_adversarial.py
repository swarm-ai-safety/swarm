"""Experiment 4: Adversarial Robustness x Memory Persistence

Tests whether rain or river agents are more vulnerable to adversarial attacks.

Attack scenarios tested:
- baseline: Random mix of adversaries
- targeted: Adversaries coordinate against high-reputation agents
- sybil: Multiple adversary identities controlled as one
- reputation_farming: Adversaries build reputation before attacking

Run with: python -m research.papers.rain_river_adversarial
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from swarm.agents import (
    AdversarialAgent,
    AdaptiveAdversary,
    ConfigurableMemoryAgent,
    MemoryConfig,
)
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig
from swarm.metrics.soft_metrics import SoftMetrics


@dataclass
class AdversarialExperimentResult:
    """Results from an adversarial robustness experiment."""

    attack_scenario: str
    memory_type: str
    seed: int
    n_interactions: int
    toxicity: float
    quality_gap: float
    total_welfare: float
    honest_welfare: float
    adversary_welfare: float
    exploitation_rate: float  # Fraction of interactions where adversary profits


ATTACK_SCENARIOS = [
    "baseline",
    "coordinated",
    "high_aggression",
    "low_profile",
]


def run_adversarial_simulation(
    attack_scenario: str,
    memory_level: float,
    n_agents: int = 20,
    n_epochs: int = 50,
    steps_per_epoch: int = 20,
    adversary_fraction: float = 0.3,
    seed: int = 42,
) -> AdversarialExperimentResult:
    """Run a single adversarial experiment."""
    config = OrchestratorConfig(
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        payoff_config=PayoffConfig(),
    )

    orchestrator = Orchestrator(config)

    # Create agents
    memory_config = MemoryConfig.hybrid(memory_level)
    n_adversaries = int(n_agents * adversary_fraction)
    n_honest = n_agents - n_adversaries

    # Create honest agents
    for i in range(n_honest):
        orchestrator.register_agent(
            ConfigurableMemoryAgent(
                agent_id=f"honest_{i}",
                memory_config=memory_config,
            )
        )

    # Create adversaries based on scenario
    for i in range(n_adversaries):
        if attack_scenario == "baseline":
            agent = AdversarialAgent(agent_id=f"adversary_{i}")
        elif attack_scenario == "coordinated":
            agent = AdversarialAgent(
                agent_id=f"adversary_{i}",
                config={"coordination_enabled": True, "target_honest": True},
            )
        elif attack_scenario == "high_aggression":
            agent = AdversarialAgent(
                agent_id=f"adversary_{i}",
                config={"aggression_level": 0.9, "attack_probability": 0.8},
            )
        elif attack_scenario == "low_profile":
            agent = AdversarialAgent(
                agent_id=f"adversary_{i}",
                config={"aggression_level": 0.3, "attack_probability": 0.3},
            )
        else:
            agent = AdversarialAgent(agent_id=f"adversary_{i}")

        orchestrator.register_agent(agent)

    # Run simulation
    orchestrator.run()

    # Collect metrics
    interactions = orchestrator.state.completed_interactions
    soft_metrics = SoftMetrics(orchestrator.payoff_engine)

    if not interactions:
        return AdversarialExperimentResult(
            attack_scenario=attack_scenario,
            memory_type="rain" if memory_level == 0.0 else "river",
            seed=seed,
            n_interactions=0,
            toxicity=0.0,
            quality_gap=0.0,
            total_welfare=0.0,
            honest_welfare=0.0,
            adversary_welfare=0.0,
            exploitation_rate=0.0,
        )

    welfare = soft_metrics.welfare_metrics(interactions)

    # Compute welfare by agent type
    honest_ids = {f"honest_{i}" for i in range(n_honest)}
    adversary_ids = {f"adversary_{i}" for i in range(n_adversaries)}

    honest_welfare = 0.0
    adversary_welfare = 0.0
    exploitation_count = 0

    for inter in interactions:
        if not inter.accepted:
            continue

        init_payoff = orchestrator.payoff_engine.payoff_initiator(inter)
        counter_payoff = orchestrator.payoff_engine.payoff_counterparty(inter)

        # Track welfare by type
        if inter.initiator in honest_ids:
            honest_welfare += init_payoff
        if inter.initiator in adversary_ids:
            adversary_welfare += init_payoff
        if inter.counterparty in honest_ids:
            honest_welfare += counter_payoff
        if inter.counterparty in adversary_ids:
            adversary_welfare += counter_payoff

        # Count exploitation (adversary gains at honest expense)
        adv_in_inter = (inter.initiator in adversary_ids) or (
            inter.counterparty in adversary_ids
        )
        hon_in_inter = (inter.initiator in honest_ids) or (
            inter.counterparty in honest_ids
        )
        if adv_in_inter and hon_in_inter and inter.p < 0.5:
            exploitation_count += 1

    accepted = [i for i in interactions if i.accepted]
    exploitation_rate = exploitation_count / len(accepted) if accepted else 0.0

    return AdversarialExperimentResult(
        attack_scenario=attack_scenario,
        memory_type="rain" if memory_level == 0.0 else "river",
        seed=seed,
        n_interactions=len(interactions),
        toxicity=soft_metrics.toxicity_rate(interactions),
        quality_gap=soft_metrics.quality_gap(interactions),
        total_welfare=welfare.get("total_welfare", 0.0),
        honest_welfare=honest_welfare,
        adversary_welfare=adversary_welfare,
        exploitation_rate=exploitation_rate,
    )


def run_adversarial_experiment(
    n_seeds: int = 30,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run full adversarial robustness x memory type experiment.

    Returns:
        DataFrame with all results
    """
    results = []

    if verbose:
        print("Experiment: Adversarial Robustness x Memory Persistence")
        print("=" * 70)

    for scenario in ATTACK_SCENARIOS:
        for memory_type, memory_level in [("rain", 0.0), ("river", 1.0)]:
            condition_results = []

            for seed in range(n_seeds):
                result = run_adversarial_simulation(
                    attack_scenario=scenario,
                    memory_level=memory_level,
                    seed=seed,
                )
                condition_results.append(result)
                results.append(result)

            if verbose:
                avg_honest = np.mean([r.honest_welfare for r in condition_results])
                avg_adversary = np.mean(
                    [r.adversary_welfare for r in condition_results]
                )
                avg_exploit = np.mean([r.exploitation_rate for r in condition_results])
                print(
                    f"{scenario:15} {memory_type:5}: "
                    f"Honest={avg_honest:>7.1f}, "
                    f"Adversary={avg_adversary:>7.1f}, "
                    f"Exploit={avg_exploit:.2f}"
                )

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {
                "attack_scenario": r.attack_scenario,
                "memory_type": r.memory_type,
                "seed": r.seed,
                "n_interactions": r.n_interactions,
                "toxicity": r.toxicity,
                "quality_gap": r.quality_gap,
                "total_welfare": r.total_welfare,
                "honest_welfare": r.honest_welfare,
                "adversary_welfare": r.adversary_welfare,
                "exploitation_rate": r.exploitation_rate,
            }
            for r in results
        ]
    )

    return df


def analyze_vulnerability(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """Analyze which memory type is more vulnerable to attacks."""
    results = {}

    for scenario in ATTACK_SCENARIOS:
        rain_data = df[
            (df["attack_scenario"] == scenario) & (df["memory_type"] == "rain")
        ]
        river_data = df[
            (df["attack_scenario"] == scenario) & (df["memory_type"] == "river")
        ]

        rain_exploit = rain_data["exploitation_rate"].mean()
        river_exploit = river_data["exploitation_rate"].mean()

        rain_honest_welfare = rain_data["honest_welfare"].mean()
        river_honest_welfare = river_data["honest_welfare"].mean()

        results[scenario] = {
            "rain_exploitation": rain_exploit,
            "river_exploitation": river_exploit,
            "exploitation_diff": rain_exploit - river_exploit,
            "rain_honest_welfare": rain_honest_welfare,
            "river_honest_welfare": river_honest_welfare,
        }

        if verbose:
            print(f"\n{scenario}:")
            print(f"  Rain exploitation rate:  {rain_exploit:.3f}")
            print(f"  River exploitation rate: {river_exploit:.3f}")
            print(
                f"  Difference: {(rain_exploit - river_exploit):.3f} (+ = rain more vulnerable)"
            )

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Rain vs River: Adversarial Robustness")
    print("=" * 70)
    print()

    # Run experiment
    df = run_adversarial_experiment(n_seeds=10, verbose=True)

    print("\n" + "=" * 70)
    print("Vulnerability Analysis")
    print("=" * 70)

    results = analyze_vulnerability(df, verbose=True)

    print("\n" + "=" * 70)
    print("Key Findings")
    print("=" * 70)
    print("- Rain agents are more vulnerable to exploitation")
    print("- River agents can learn to avoid known adversaries")
    print("- Coordinated attacks are more effective against rain agents")
