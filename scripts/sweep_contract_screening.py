#!/usr/bin/env python3
"""Multi-seed sweep on the contract screening scenario.

Runs N seeds through the contract_screening scenario and exports
a CSV with per-seed contract metrics (separation_quality, infiltration_rate,
welfare_delta, attack_displacement, per-pool quality/welfare).

Usage:
    python scripts/sweep_contract_screening.py [--seeds N] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

from swarm.analysis.sweep import SweepConfig, SweepRunner
from swarm.scenarios import load_scenario


def main():
    parser = argparse.ArgumentParser(description="Contract screening multi-seed sweep")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds (default: 10)")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed (default: 42)")
    parser.add_argument(
        "--out",
        type=str,
        default="runs/contract_screening_sweep",
        help="Output directory (default: runs/contract_screening_sweep)",
    )
    args = parser.parse_args()

    scenario_path = Path("scenarios/contract_screening.yaml")
    if not scenario_path.exists():
        print(f"Error: {scenario_path} not found")
        return 1

    print(f"Loading scenario from {scenario_path}...")
    scenario = load_scenario(scenario_path)

    out_dir = Path(args.out)
    csv_path = out_dir / "sweep_results.csv"

    config = SweepConfig(
        base_scenario=scenario,
        parameters=[],  # No parameter sweep, just multi-seed
        runs_per_config=args.seeds,
        seed_base=args.seed_base,
    )

    print(f"Running {config.total_runs()} seeds (base={args.seed_base})...")

    def progress(current, total, params):
        print(f"  [{current}/{total}] seed={args.seed_base + current}")

    runner = SweepRunner(config, progress_callback=progress)
    runner.run()

    # Export CSV
    runner.to_csv(csv_path)
    print(f"\nCSV saved to {csv_path}")

    # Print summary
    summary = runner.summary()
    print(f"\n{'='*60}")
    print(f"Contract Screening Sweep Summary ({summary['total_runs']} runs)")
    print(f"{'='*60}")

    for s in summary.get("summaries", []):
        n = s["n_runs"]
        print(f"  Runs: {n}")
        print(f"  Mean welfare:          {s['mean_welfare']:.4f}")
        print(f"  Mean toxicity:         {s['mean_toxicity']:.4f}")

        if "mean_separation_quality" in s:
            print(f"  Mean separation:       {s['mean_separation_quality']:.4f}")
            print(f"  Mean infiltration:     {s['mean_infiltration_rate']:.4f}")
            print(f"  Mean welfare delta:    {s['mean_welfare_delta']:.4f}")
            print(f"  Mean attack displ.:    {s['mean_attack_displacement']:.4f}")

    # Per-seed detail table
    print(f"\n{'seed':>6} {'sep_q':>8} {'infil':>8} {'w_delta':>8} {'atk_disp':>8}")
    print("-" * 42)
    for r in runner.results:
        print(
            f"{r.seed:>6} {r.separation_quality:>8.4f} {r.infiltration_rate:>8.4f} "
            f"{r.welfare_delta:>8.4f} {r.attack_displacement:>8.4f}"
        )

    print(f"\nDone. Results in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
