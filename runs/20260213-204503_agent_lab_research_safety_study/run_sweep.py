#!/usr/bin/env python
"""
Full study sweep: Agent Lab Research Safety

Sweeps governance parameters over the agent_lab_research_safety scenario
to study how transaction taxes, circuit breakers, and collusion detection
affect welfare and safety in autonomous research pipelines.

Sweep axes:
  - governance.transaction_tax_rate: [0.0, 0.03, 0.06, 0.10]
  - governance.circuit_breaker_enabled: [False, True]
  - governance.collusion_detection_enabled: [False, True]

16 configurations × 10 seeds = 160 runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swarm.analysis import SweepConfig, SweepParameter, SweepRunner
from swarm.scenarios import load_scenario

RUN_DIR = Path(__file__).parent
SCENARIO = Path(__file__).parent.parent.parent / "scenarios" / "agent_lab_research_safety.yaml"
OUTPUT_CSV = RUN_DIR / "sweep_results.csv"

SEEDS = 10
SEED_BASE = 42
EPOCHS = 4  # matches AgentLab's 4 phases


def progress(current: int, total: int, params: dict) -> None:
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"  [{current:>3}/{total}] {param_str}")


def main() -> int:
    print("=" * 60)
    print("Full Study Sweep: Agent Lab Research Safety")
    print("=" * 60)

    scenario = load_scenario(SCENARIO)
    scenario.orchestrator_config.n_epochs = EPOCHS

    sweep_config = SweepConfig(
        base_scenario=scenario,
        parameters=[
            SweepParameter(
                name="governance.transaction_tax_rate",
                values=[0.0, 0.03, 0.06, 0.10],
            ),
            SweepParameter(
                name="governance.circuit_breaker_enabled",
                values=[False, True],
            ),
            SweepParameter(
                name="governance.collusion_detection_enabled",
                values=[False, True],
            ),
        ],
        runs_per_config=SEEDS,
        seed_base=SEED_BASE,
    )

    print(f"\n  Scenario: {SCENARIO.name}")
    print(f"  Parameters: {len(sweep_config.parameters)}")
    for p in sweep_config.parameters:
        print(f"    - {p.name}: {p.values}")
    print(f"  Seeds: {SEEDS} (base={SEED_BASE})")
    print(f"  Total runs: {sweep_config.total_runs()}")
    print()

    runner = SweepRunner(sweep_config, progress_callback=progress)
    runner.run()

    print(f"\nExporting results to: {OUTPUT_CSV}")
    runner.to_csv(OUTPUT_CSV)

    # Also export parquet if available
    try:
        runner.to_parquet(OUTPUT_CSV.with_suffix(".parquet"))
        print(f"Exported parquet to: {OUTPUT_CSV.with_suffix('.parquet')}")
    except Exception:
        pass

    summary = runner.summary()
    print(f"\nTotal runs: {summary['total_runs']}")
    print(f"Configurations: {summary['param_combinations']}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
