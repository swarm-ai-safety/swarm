#!/usr/bin/env python
"""
Team-of-Rivals ablation sweep.

Sweeps across the four governance modes (single_agent, advisory, council,
team_of_rivals) to replicate the central finding from Vijayaraghavan et al.
(arXiv:2601.14351): staged veto + retry outperforms advisory and council.

Usage:
    python examples/rivals_ablation_sweep.py
    python examples/rivals_ablation_sweep.py --output runs/<dir>/sweep_results.csv
    python examples/rivals_ablation_sweep.py --runs_per_config 5 --epochs 20
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis import SweepConfig, SweepParameter, SweepRunner
from swarm.scenarios import load_scenario


def progress(current: int, total: int, params: dict) -> None:
    """Progress callback."""
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"  [{current}/{total}] {param_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Team-of-Rivals ablation sweep across 4 governance modes"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output CSV path (default: auto-generated under runs/)",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=Path,
        default=Path("scenarios/team_of_rivals.yaml"),
        help="Base scenario file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for sweep",
        dest="seed_base",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs per run",
    )
    parser.add_argument(
        "--runs_per_config",
        type=int,
        default=3,
        help="Number of runs per mode (different seeds)",
    )
    args = parser.parse_args()

    # Auto-generate output path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(f"runs/{timestamp}_rivals_ablation_sweep")
    if args.output is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        args.output = run_dir / "sweep_results.csv"
    else:
        run_dir = args.output.parent
        run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Team-of-Rivals Ablation Sweep")
    print("=" * 60)
    print()
    print("Modes: single_agent, advisory, council, team_of_rivals")
    print(f"Epochs per run: {args.epochs}")
    print(f"Runs per mode:  {args.runs_per_config}")
    print(f"Base seed:      {args.seed_base}")
    print()

    # Load base scenario
    print(f"Loading base scenario: {args.scenario}")
    base_scenario = load_scenario(args.scenario)
    base_scenario.orchestrator_config.n_epochs = args.epochs

    # Configure sweep — single parameter: rivals.mode
    sweep_config = SweepConfig(
        base_scenario=base_scenario,
        parameters=[
            SweepParameter(
                name="rivals.mode",
                values=[
                    "single_agent",
                    "advisory",
                    "council",
                    "team_of_rivals",
                ],
            ),
        ],
        runs_per_config=args.runs_per_config,
        seed_base=args.seed_base,
    )

    total = sweep_config.total_runs()
    print(f"Total runs: {total} (4 modes x {args.runs_per_config} seeds)")
    print()

    # Run sweep
    print("Running sweep...")
    runner = SweepRunner(sweep_config, progress_callback=progress)
    runner.run()

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    # Summary
    summary = runner.summary()
    print(f"\nTotal runs completed: {summary['total_runs']}")
    print()

    # Print per-mode results
    header = (
        f"{'Mode':<20s} {'Welfare':>10s} {'Toxicity':>10s} "
        f"{'Interactions':>14s} {'AvgPayoff':>10s}"
    )
    print(header)
    print("-" * len(header))

    for cfg in summary.get("configs", []):
        mode = cfg.get("rivals.mode", "?")
        print(
            f"{mode:<20s} "
            f"{cfg.get('mean_welfare', 0):>10.2f} "
            f"{cfg.get('mean_toxicity_rate', 0):>10.4f} "
            f"{cfg.get('mean_total_interactions', 0):>14.0f} "
            f"{cfg.get('mean_avg_payoff', 0):>10.4f}"
        )

    print()

    # Export CSV
    runner.to_csv(args.output)
    print(f"CSV exported: {args.output}")

    # Export summary JSON
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary JSON: {summary_path}")

    print()
    print(f"Plot with: /plot {run_dir}")


if __name__ == "__main__":
    main()
