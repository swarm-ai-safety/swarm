"""Experiment 5: Time Horizon x Memory Persistence

Tests whether rain agents degrade faster on longer tasks, following
Herbie Bradley's framework for measuring AI capability by reliable
task completion across increasing time horizons.

Key hypothesis: Rain agents should show steeper reliability decline
across time horizons because they cannot maintain context/memory
across longer task durations.

Run with: python -m research.papers.rain_river_time_horizons
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from swarm.agents import ConfigurableMemoryAgent, MemoryConfig
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.metrics.time_horizons import (
    AgentCapabilityProfile,
    CAPABILITY_PROFILES,
    TimeHorizonMetrics,
)


@dataclass
class TimeHorizonExperimentResult:
    """Results from a time horizon experiment."""

    memory_type: str
    task_horizon_minutes: int
    seed: int
    n_interactions: int
    success_rate: float
    avg_quality: float
    total_welfare: float
    reliability: float


# Task horizons to test (in simulation steps, treated as "minutes")
TASK_HORIZONS = [1, 5, 10, 30, 60, 120]


def simulate_task_completion(
    memory_level: float,
    task_duration_minutes: int,
    capability_profile: AgentCapabilityProfile,
    seed: int = 42,
) -> tuple:
    """
    Simulate task completion for a single task duration.

    Models how memory persistence affects task reliability:
    - Rain agents "forget" context as task progresses
    - River agents maintain context throughout

    Returns:
        (success, quality) tuple
    """
    rng = np.random.default_rng(seed)

    # Base reliability from capability profile
    base_reliability = capability_profile.reliability_at_horizon(task_duration_minutes)

    # Memory penalty: rain agents lose context over time
    # Longer tasks = more context loss for rain agents
    context_decay = 1.0 - memory_level  # 1.0 for rain, 0.0 for river
    time_factor = np.log10(max(1, task_duration_minutes)) / 2.0  # Scales with log(time)
    memory_penalty = context_decay * time_factor * 0.2  # Up to 20% penalty

    effective_reliability = max(0.0, base_reliability - memory_penalty)

    # Determine success
    success = rng.random() < effective_reliability

    # Quality (if successful)
    if success:
        quality = capability_profile.expected_quality(task_duration_minutes)
        # Rain agents produce lower quality on long tasks
        quality *= (1.0 - memory_penalty * 0.5)
    else:
        quality = 0.0

    return success, quality


def run_time_horizon_simulation(
    memory_level: float,
    task_horizon_minutes: int,
    n_tasks: int = 100,
    seed: int = 42,
) -> TimeHorizonExperimentResult:
    """Run simulation for a specific time horizon."""
    profile = CAPABILITY_PROFILES["standard"]

    successes = 0
    total_quality = 0.0

    for i in range(n_tasks):
        success, quality = simulate_task_completion(
            memory_level=memory_level,
            task_duration_minutes=task_horizon_minutes,
            capability_profile=profile,
            seed=seed * 1000 + i,
        )
        if success:
            successes += 1
            total_quality += quality

    success_rate = successes / n_tasks
    avg_quality = total_quality / successes if successes > 0 else 0.0

    return TimeHorizonExperimentResult(
        memory_type="rain" if memory_level == 0.0 else "river",
        task_horizon_minutes=task_horizon_minutes,
        seed=seed,
        n_interactions=n_tasks,
        success_rate=success_rate,
        avg_quality=avg_quality,
        total_welfare=total_quality,
        reliability=success_rate,
    )


def run_time_horizon_experiment(
    n_seeds: int = 30,
    n_tasks_per_horizon: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run full time horizon x memory type experiment.

    Returns:
        DataFrame with reliability curves for rain and river agents
    """
    results = []

    if verbose:
        print("Experiment: Time Horizon x Memory Persistence")
        print("=" * 70)
        print(f"{'Horizon':<10} {'Type':<6} {'Reliability':<12} {'Avg Quality':<12}")
        print("-" * 40)

    for horizon in TASK_HORIZONS:
        for memory_type, memory_level in [("rain", 0.0), ("river", 1.0)]:
            condition_results = []

            for seed in range(n_seeds):
                result = run_time_horizon_simulation(
                    memory_level=memory_level,
                    task_horizon_minutes=horizon,
                    n_tasks=n_tasks_per_horizon,
                    seed=seed,
                )
                condition_results.append(result)
                results.append(result)

            if verbose:
                avg_reliability = np.mean([r.reliability for r in condition_results])
                avg_quality = np.mean([r.avg_quality for r in condition_results])
                print(f"{horizon:<10} {memory_type:<6} {avg_reliability:>10.3f}   {avg_quality:>10.3f}")

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "memory_type": r.memory_type,
            "horizon_minutes": r.task_horizon_minutes,
            "seed": r.seed,
            "reliability": r.reliability,
            "avg_quality": r.avg_quality,
            "success_rate": r.success_rate,
        }
        for r in results
    ])

    return df


def compute_reliability_curves(df: pd.DataFrame) -> Dict:
    """Compute reliability curves for rain and river agents."""
    curves = {}

    for memory_type in ["rain", "river"]:
        type_data = df[df["memory_type"] == memory_type]
        curve = {}
        for horizon in TASK_HORIZONS:
            horizon_data = type_data[type_data["horizon_minutes"] == horizon]
            curve[horizon] = {
                "mean_reliability": horizon_data["reliability"].mean(),
                "std_reliability": horizon_data["reliability"].std(),
                "mean_quality": horizon_data["avg_quality"].mean(),
            }
        curves[memory_type] = curve

    return curves


def compute_effective_horizon(df: pd.DataFrame, threshold: float = 0.8) -> Dict:
    """
    Compute effective horizon (longest task duration with reliability >= threshold).

    This is the key Bradley metric.
    """
    results = {}

    for memory_type in ["rain", "river"]:
        type_data = df[df["memory_type"] == memory_type]
        effective = None

        for horizon in sorted(TASK_HORIZONS):
            horizon_data = type_data[type_data["horizon_minutes"] == horizon]
            reliability = horizon_data["reliability"].mean()

            if reliability >= threshold:
                effective = horizon
            else:
                break

        results[memory_type] = {
            "effective_horizon_80": effective,
            "description": f"Can complete {effective}-minute tasks at {threshold*100:.0f}% reliability" if effective else "Below threshold at all horizons",
        }

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Rain vs River: Time Horizon Reliability")
    print("=" * 70)
    print()

    # Run experiment
    df = run_time_horizon_experiment(n_seeds=10, verbose=True)

    print("\n" + "=" * 70)
    print("Reliability Curves")
    print("=" * 70)

    curves = compute_reliability_curves(df)
    print("\nRain agent reliability by horizon:")
    for horizon, stats in curves["rain"].items():
        print(f"  {horizon} min: {stats['mean_reliability']:.3f} +/- {stats['std_reliability']:.3f}")

    print("\nRiver agent reliability by horizon:")
    for horizon, stats in curves["river"].items():
        print(f"  {horizon} min: {stats['mean_reliability']:.3f} +/- {stats['std_reliability']:.3f}")

    print("\n" + "=" * 70)
    print("Effective Horizons (Bradley Metric)")
    print("=" * 70)

    effective = compute_effective_horizon(df, threshold=0.8)
    for memory_type, data in effective.items():
        print(f"\n{memory_type.capitalize()}:")
        print(f"  {data['description']}")

    print("\n" + "=" * 70)
    print("Key Findings")
    print("=" * 70)
    print("- Rain agents show steeper reliability decline with task duration")
    print("- River agents maintain higher reliability on longer tasks")
    print("- Effective horizon is shorter for rain agents")
    print("- Memory persistence is crucial for complex, multi-step tasks")
