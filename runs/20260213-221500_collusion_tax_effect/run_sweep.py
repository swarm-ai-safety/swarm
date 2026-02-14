#!/usr/bin/env python
"""Collusion tax effect parameter sweep for full_study pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from swarm.analysis import SweepConfig, SweepParameter, SweepRunner
from swarm.scenarios import load_scenario

RUN_DIR = Path(__file__).resolve().parent
SCENARIO = Path(__file__).resolve().parent.parent.parent / "scenarios" / "rlm_recursive_collusion.yaml"
SEEDS = 10


def progress(current: int, total: int, params: dict) -> None:
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"  [{current}/{total}] {param_str}")


def main() -> int:
    print("=" * 60)
    print("Collusion Tax Effect Parameter Sweep")
    print("=" * 60)

    base_scenario = load_scenario(SCENARIO)

    sweep_config = SweepConfig(
        base_scenario=base_scenario,
        parameters=[
            SweepParameter(
                name="governance.transaction_tax_rate",
                values=[0.0, 0.02, 0.05, 0.10],
            ),
            SweepParameter(
                name="governance.collusion_penalty_multiplier",
                values=[0.5, 1.0, 1.5, 2.0],
            ),
        ],
        runs_per_config=SEEDS,
        seed_base=300,
    )

    total = sweep_config.total_runs()
    print("\nParameters:")
    for p in sweep_config.parameters:
        print(f"  {p.name}: {p.values}")
    print(f"Seeds per config: {SEEDS}")
    print(f"Total runs: {total}")
    print()

    print("Running sweep...")
    runner = SweepRunner(sweep_config, progress_callback=progress)
    runner.run()

    csv_path = RUN_DIR / "sweep_results.csv"
    runner.to_csv(csv_path)
    print(f"\nResults written to: {csv_path}")

    summary = runner.summary()
    print(f"Total runs: {summary['total_runs']}")
    print(f"Parameter combinations: {summary['param_combinations']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
