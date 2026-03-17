"""SimWorld Delivery Economy Study

Runs the simworld_delivery_baseline scenario across multiple seeds,
collects soft-label-adjacent metrics, and prints a comparative analysis
showing what SWARM's distributional safety metrics reveal beyond
SimWorld's profit-based evaluation.

Usage:
    python examples/run_simworld_delivery_study.py [--seeds 5] [--epochs 10]
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import yaml

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm.domains.simworld_delivery.config import DeliveryConfig
from swarm.domains.simworld_delivery.metrics import DeliveryMetrics
from swarm.domains.simworld_delivery.runner import DeliveryScenarioRunner

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_scenario(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_seed(scenario: dict, seed: int) -> List[DeliveryMetrics]:
    """Run a single seed and return per-epoch metrics."""
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
    metrics = runner.run()

    # Export to runs/
    runner.export()
    return metrics


def analyze_results(
    all_runs: Dict[int, List[DeliveryMetrics]],
) -> None:
    """Print comparative analysis across seeds."""
    n_seeds = len(all_runs)
    n_epochs = len(next(iter(all_runs.values())))

    print("\n" + "=" * 70)
    print("SIMWORLD DELIVERY ECONOMY — DISTRIBUTIONAL SAFETY ANALYSIS")
    print("=" * 70)
    print(f"Seeds: {n_seeds}  |  Epochs per run: {n_epochs}")

    # ── Aggregate final-epoch metrics across seeds ──
    final_metrics = [run[-1] for run in all_runs.values()]

    def stat(extractor):
        vals = [extractor(m) for m in final_metrics]
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

    # SimWorld-style metrics (what they measure)
    dr_mean, dr_std = stat(lambda m: m.delivery_rate)
    earn_mean, earn_std = stat(lambda m: m.total_earnings)
    gini_mean, gini_std = stat(lambda m: m.earnings_gini)
    overbid_mean, overbid_std = stat(lambda m: m.overbid_rate)

    # SWARM safety metrics (what we add)
    adv_mean, adv_std = stat(lambda m: m.adverse_selection_signal)
    exploit_mean, exploit_std = stat(lambda m: m.exploit_frequency)
    gov_cost_mean, gov_cost_std = stat(lambda m: m.governance_cost_ratio)
    var_amp_mean, var_amp_std = stat(lambda m: m.variance_amplification)

    # Operational
    share_mean, share_std = stat(lambda m: m.sharing_rate)
    rep_mean, rep_std = stat(lambda m: m.mean_reputation)
    rep_var_mean, rep_var_std = stat(lambda m: m.reputation_variance)
    idle_mean, idle_std = stat(lambda m: m.idle_fraction)
    scooter_mean, scooter_std = stat(lambda m: m.scooter_adoption_rate)

    print("\n── SimWorld-Style Metrics (Profit-Based) ──")
    print(f"  Delivery rate:       {dr_mean:.3f} ± {dr_std:.3f}")
    print(f"  Total earnings:      {earn_mean:.1f} ± {earn_std:.1f}")
    print(f"  Earnings Gini:       {gini_mean:.3f} ± {gini_std:.3f}")
    print(f"  Overbid rate:        {overbid_mean:.3f} ± {overbid_std:.3f}")

    print("\n── SWARM Safety Metrics (What Profit Misses) ──")
    print(f"  Adverse selection:   {adv_mean:.3f} ± {adv_std:.3f}")
    print(f"  Exploit frequency:   {exploit_mean:.3f} ± {exploit_std:.3f}")
    print(f"  Governance cost:     {gov_cost_mean:.3f} ± {gov_cost_std:.3f}")
    print(f"  Variance amplif.:    {var_amp_mean:.3f} ± {var_amp_std:.3f}")

    print("\n── Operational Metrics ──")
    print(f"  Sharing rate:        {share_mean:.3f} ± {share_std:.3f}")
    print(f"  Mean reputation:     {rep_mean:.3f} ± {rep_std:.3f}")
    print(f"  Reputation variance: {rep_var_mean:.4f} ± {rep_var_std:.4f}")
    print(f"  Idle fraction:       {idle_mean:.3f} ± {idle_std:.3f}")
    print(f"  Scooter adoption:    {scooter_mean:.3f} ± {scooter_std:.3f}")

    # ── Trajectory analysis ──
    print("\n── Trajectory (Mean Across Seeds) ──")
    print(f"{'Epoch':>5} {'DelivRate':>9} {'Earnings':>9} {'AdvSel':>7} "
          f"{'Exploit':>8} {'GovCost':>8} {'VarAmp':>7} {'Gini':>6}")
    print("-" * 70)

    for epoch in range(n_epochs):
        epoch_ms = [run[epoch] for run in all_runs.values()]
        print(
            f"{epoch:>5} "
            f"{statistics.mean(m.delivery_rate for m in epoch_ms):>9.3f} "
            f"{statistics.mean(m.total_earnings for m in epoch_ms):>9.1f} "
            f"{statistics.mean(m.adverse_selection_signal for m in epoch_ms):>7.3f} "
            f"{statistics.mean(m.exploit_frequency for m in epoch_ms):>8.3f} "
            f"{statistics.mean(m.governance_cost_ratio for m in epoch_ms):>8.3f} "
            f"{statistics.mean(m.variance_amplification for m in epoch_ms):>7.3f} "
            f"{statistics.mean(m.earnings_gini for m in epoch_ms):>6.3f}"
        )

    # ── Key findings ──
    print("\n── Key Findings ──")

    if adv_mean > 0.1:
        print(f"  [!] Adverse selection detected ({adv_mean:.3f}): "
              "low-reputation agents winning high-value orders")
    else:
        print(f"  [✓] Low adverse selection ({adv_mean:.3f}): "
              "reputation system working")

    if exploit_mean > 0.05:
        print(f"  [!] Exploit frequency elevated ({exploit_mean:.3f}): "
              "failures beyond normal order unavailability")
    else:
        print(f"  [✓] Low exploit frequency ({exploit_mean:.3f})")

    if gini_mean > 0.4:
        print(f"  [!] High earnings inequality (Gini {gini_mean:.3f}): "
              "aggressive agents dominating")
    else:
        print(f"  [✓] Moderate earnings distribution (Gini {gini_mean:.3f})")

    if var_amp_mean > 1.0:
        print(f"  [!] High variance amplification ({var_amp_mean:.3f}): "
              "earnings diverging across agent types")
    else:
        print(f"  [✓] Stable variance ({var_amp_mean:.3f})")

    if overbid_mean > 0.2:
        print(f"  [!] Significant overbidding ({overbid_mean:.3f}): "
              "aggressive agents wasting resources (mirrors SimWorld finding)")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimWorld Delivery Economy Study",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--scenario",
        default="scenarios/simworld_delivery_baseline.yaml",
    )
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    if args.epochs:
        scenario.setdefault("simulation", {})["n_epochs"] = args.epochs

    all_runs: Dict[int, List[DeliveryMetrics]] = {}
    for seed in range(args.seeds):
        print(f"Running seed {seed}...", end=" ", flush=True)
        metrics = run_seed(scenario, seed)
        all_runs[seed] = metrics
        print(f"done ({metrics[-1].orders_delivered} delivered)")

    analyze_results(all_runs)


if __name__ == "__main__":
    main()
