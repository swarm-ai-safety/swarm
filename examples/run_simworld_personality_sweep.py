"""SimWorld Delivery Personality Distribution Sweep

Runs the delivery scenario across different personality compositions
(Big Five → delivery persona mappings) and traces how the Pareto frontier
shifts with population personality distribution.

The Pareto frontier is traced over two axes:
  - X: Delivery rate (throughput/efficiency)
  - Y: 1 - adverse_selection_signal (safety)

Each personality distribution produces a cloud of (efficiency, safety) points
across seeds, and we compute the Pareto front for each.

Usage:
    python examples/run_simworld_personality_sweep.py [--seeds 5] [--epochs 10]
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm.domains.simworld_delivery.config import DeliveryConfig
from swarm.domains.simworld_delivery.metrics import DeliveryMetrics
from swarm.domains.simworld_delivery.runner import DeliveryScenarioRunner

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Scenario files ordered by personality dominance
PERSONALITY_SCENARIOS = [
    ("conscientious", "scenarios/simworld_delivery_personality_conscientious.yaml"),
    ("baseline", "scenarios/simworld_delivery_baseline.yaml"),
    ("cautious", "scenarios/simworld_delivery_personality_cautious.yaml"),
    ("aggressive", "scenarios/simworld_delivery_personality_aggressive.yaml"),
    ("opportunistic", "scenarios/simworld_delivery_personality_opportunistic.yaml"),
]


@dataclass
class SweepPoint:
    """A single point in the efficiency-safety Pareto space."""

    personality: str
    seed: int
    delivery_rate: float
    adverse_selection: float
    earnings_gini: float
    exploit_frequency: float
    overbid_rate: float
    variance_amplification: float
    governance_cost_ratio: float
    total_earnings: float
    idle_fraction: float
    mean_reputation: float

    @property
    def safety(self) -> float:
        """Safety score: 1 - adverse_selection."""
        return 1.0 - self.adverse_selection

    @property
    def efficiency(self) -> float:
        """Efficiency score: delivery_rate."""
        return self.delivery_rate


def load_scenario(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_scenario_seed(
    scenario: dict, seed: int,
) -> List[DeliveryMetrics]:
    """Run a single scenario seed."""
    delivery_cfg = scenario.get("delivery", {})
    config = DeliveryConfig.from_dict({**delivery_cfg, "seed": seed})

    sim = scenario.get("simulation", {})
    runner = DeliveryScenarioRunner(
        config=config,
        agent_specs=scenario.get("agents", []),
        n_epochs=sim.get("n_epochs", 10),
        steps_per_epoch=sim.get("steps_per_epoch", 20),
        seed=seed,
    )
    return runner.run()


def compute_pareto_front(
    points: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Compute the Pareto front (maximizing both dimensions).

    Returns points sorted by x-coordinate.
    """
    sorted_pts = sorted(points, key=lambda p: (-p[0], -p[1]))
    front = []
    max_y = -float("inf")
    for x, y in sorted_pts:
        if y > max_y:
            front.append((x, y))
            max_y = y
    return sorted(front, key=lambda p: p[0])


def run_sweep(
    n_seeds: int, n_epochs: int,
) -> Dict[str, List[SweepPoint]]:
    """Run the full personality sweep."""
    results: Dict[str, List[SweepPoint]] = {}

    for personality, scenario_path in PERSONALITY_SCENARIOS:
        print(f"\n--- {personality.upper()} ---")
        scenario = load_scenario(scenario_path)
        scenario.setdefault("simulation", {})["n_epochs"] = n_epochs

        points: List[SweepPoint] = []
        for seed in range(n_seeds):
            print(f"  seed {seed}...", end=" ", flush=True)
            metrics_list = run_scenario_seed(scenario, seed)
            final = metrics_list[-1]

            point = SweepPoint(
                personality=personality,
                seed=seed,
                delivery_rate=final.delivery_rate,
                adverse_selection=final.adverse_selection_signal,
                earnings_gini=final.earnings_gini,
                exploit_frequency=final.exploit_frequency,
                overbid_rate=final.overbid_rate,
                variance_amplification=final.variance_amplification,
                governance_cost_ratio=final.governance_cost_ratio,
                total_earnings=final.total_earnings,
                idle_fraction=final.idle_fraction,
                mean_reputation=final.mean_reputation,
            )
            points.append(point)
            print(
                f"rate={final.delivery_rate:.3f} "
                f"adv_sel={final.adverse_selection_signal:.3f} "
                f"gini={final.earnings_gini:.3f}",
            )

        results[personality] = points

    return results


def print_analysis(results: Dict[str, List[SweepPoint]]) -> None:
    """Print comparative analysis and Pareto fronts."""
    print("\n" + "=" * 80)
    print("PERSONALITY DISTRIBUTION SWEEP — PARETO FRONTIER ANALYSIS")
    print("=" * 80)

    # Summary table
    print(f"\n{'Personality':<16} {'Efficiency':>10} {'Safety':>8} "
          f"{'Gini':>6} {'Overbid':>8} {'Exploit':>8} {'VarAmp':>7} "
          f"{'GovCost':>8} {'Idle':>6}")
    print("-" * 90)

    pareto_points: Dict[str, List[Tuple[float, float]]] = {}

    for personality, points in results.items():
        eff = statistics.mean(p.efficiency for p in points)
        saf = statistics.mean(p.safety for p in points)
        gini = statistics.mean(p.earnings_gini for p in points)
        overbid = statistics.mean(p.overbid_rate for p in points)
        exploit = statistics.mean(p.exploit_frequency for p in points)
        var_amp = statistics.mean(p.variance_amplification for p in points)
        gov_cost = statistics.mean(p.governance_cost_ratio for p in points)
        idle = statistics.mean(p.idle_fraction for p in points)

        print(
            f"{personality:<16} {eff:>10.3f} {saf:>8.3f} "
            f"{gini:>6.3f} {overbid:>8.3f} {exploit:>8.3f} {var_amp:>7.3f} "
            f"{gov_cost:>8.3f} {idle:>6.3f}",
        )

        pareto_points[personality] = [
            (p.efficiency, p.safety) for p in points
        ]

    # Pareto fronts
    print("\n── Pareto Fronts (Efficiency × Safety) ──")
    for personality, pts in pareto_points.items():
        front = compute_pareto_front(pts)
        front_str = " → ".join(f"({x:.3f}, {y:.3f})" for x, y in front)
        print(f"  {personality:<16}: {front_str}")

    # Key findings
    print("\n── Key Findings ──")

    # Which personality achieves best efficiency?
    best_eff = max(
        results.items(),
        key=lambda kv: statistics.mean(p.efficiency for p in kv[1]),
    )
    print(f"  Highest efficiency: {best_eff[0]} "
          f"({statistics.mean(p.efficiency for p in best_eff[1]):.3f})")

    # Which personality achieves best safety?
    best_saf = max(
        results.items(),
        key=lambda kv: statistics.mean(p.safety for p in kv[1]),
    )
    print(f"  Highest safety:     {best_saf[0]} "
          f"({statistics.mean(p.safety for p in best_saf[1]):.3f})")

    # Which personality achieves lowest inequality?
    best_gini = min(
        results.items(),
        key=lambda kv: statistics.mean(p.earnings_gini for p in kv[1]),
    )
    print(f"  Lowest inequality:  {best_gini[0]} "
          f"(Gini {statistics.mean(p.earnings_gini for p in best_gini[1]):.3f})")

    # Tradeoff analysis
    print("\n── Tradeoff Analysis ──")
    for personality, points in results.items():
        eff = statistics.mean(p.efficiency for p in points)
        saf = statistics.mean(p.safety for p in points)
        gini = statistics.mean(p.earnings_gini for p in points)

        if eff > 0.5 and saf > 0.9:
            print(f"  {personality}: PARETO-OPTIMAL region (high eff + high safety)")
        elif eff > 0.5 and saf < 0.8:
            print(f"  {personality}: efficiency-focused (safety tradeoff)")
        elif eff < 0.4 and saf > 0.9:
            print(f"  {personality}: safety-focused (efficiency tradeoff)")
        elif gini > 0.4:
            print(f"  {personality}: high inequality (structural concern)")
        else:
            print(f"  {personality}: moderate tradeoff region")

    print("\n" + "=" * 80)


def export_results(
    results: Dict[str, List[SweepPoint]],
) -> Path:
    """Export sweep results to CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_personality_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "sweep_results.csv"
    fieldnames = [
        "personality", "seed", "delivery_rate", "safety",
        "adverse_selection", "earnings_gini", "exploit_frequency",
        "overbid_rate", "variance_amplification", "governance_cost_ratio",
        "total_earnings", "idle_fraction", "mean_reputation",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _personality, points in results.items():
            for p in points:
                writer.writerow({
                    "personality": p.personality,
                    "seed": p.seed,
                    "delivery_rate": f"{p.delivery_rate:.4f}",
                    "safety": f"{p.safety:.4f}",
                    "adverse_selection": f"{p.adverse_selection:.4f}",
                    "earnings_gini": f"{p.earnings_gini:.4f}",
                    "exploit_frequency": f"{p.exploit_frequency:.4f}",
                    "overbid_rate": f"{p.overbid_rate:.4f}",
                    "variance_amplification": f"{p.variance_amplification:.4f}",
                    "governance_cost_ratio": f"{p.governance_cost_ratio:.4f}",
                    "total_earnings": f"{p.total_earnings:.2f}",
                    "idle_fraction": f"{p.idle_fraction:.4f}",
                    "mean_reputation": f"{p.mean_reputation:.4f}",
                })

    # Also export Pareto fronts
    front_path = out_dir / "pareto_fronts.csv"
    with open(front_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["personality", "efficiency", "safety"])
        for personality, points in results.items():
            pts = [(p.efficiency, p.safety) for p in points]
            front = compute_pareto_front(pts)
            for x, y in front:
                writer.writerow([personality, f"{x:.4f}", f"{y:.4f}"])

    print(f"\nResults exported to {out_dir}/")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimWorld Delivery Personality Distribution Sweep",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    results = run_sweep(args.seeds, args.epochs)
    print_analysis(results)
    export_results(results)


if __name__ == "__main__":
    main()
