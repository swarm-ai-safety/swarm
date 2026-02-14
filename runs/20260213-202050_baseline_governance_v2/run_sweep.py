#!/usr/bin/env python
"""Baseline governance v2 — improved sweep per council recommendations.

Changes from v1:
  - 50 seeds per config (up from 10) for better statistical power
  - Finer tax granularity: 0%, 2.5%, 5%, 7.5%, 10%, 12.5%, 15%
  - Same circuit breaker toggle for interaction analysis
  - Total runs: 7 tax × 2 CB × 50 seeds = 700
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from swarm.analysis import SweepConfig, SweepParameter, SweepRunner
from swarm.scenarios import load_scenario

RUN_DIR = Path(__file__).resolve().parent
SCENARIO = Path(__file__).resolve().parent.parent.parent / "scenarios" / "baseline.yaml"
SEEDS = 50


def progress(current: int, total: int, params: dict) -> None:
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    pct = current / total * 100
    print(f"  [{current}/{total}] ({pct:.0f}%) {param_str}")


def main() -> int:
    print("=" * 60)
    print("Baseline Governance v2 — Council-Improved Sweep")
    print("=" * 60)

    base_scenario = load_scenario(SCENARIO)

    sweep_config = SweepConfig(
        base_scenario=base_scenario,
        parameters=[
            SweepParameter(
                name="governance.transaction_tax_rate",
                values=[0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15],
            ),
            SweepParameter(
                name="governance.circuit_breaker_enabled",
                values=[False, True],
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
