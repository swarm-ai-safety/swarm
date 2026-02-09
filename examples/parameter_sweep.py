#!/usr/bin/env python
"""
Parameter sweep example.

Demonstrates sweeping over governance parameters to study their effects.

Usage:
    python examples/parameter_sweep.py
    python examples/parameter_sweep.py --output results.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis import SweepConfig, SweepParameter, SweepRunner
from swarm.scenarios import load_scenario


def progress(current: int, total: int, params: dict) -> None:
    """Progress callback."""
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"  [{current}/{total}] {param_str}")


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("logs/sweep_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=Path,
        default=Path("scenarios/baseline.yaml"),
        help="Base scenario file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Parameter Sweep - Governance Effects Study")
    print("=" * 60)
    print()

    # Load base scenario
    print(f"Loading base scenario: {args.scenario}")
    base_scenario = load_scenario(args.scenario)

    # Reduce epochs for faster sweep
    base_scenario.orchestrator_config.n_epochs = 5

    # Configure sweep
    print("\nConfiguring parameter sweep...")
    sweep_config = SweepConfig(
        base_scenario=base_scenario,
        parameters=[
            SweepParameter(
                name="governance.transaction_tax_rate",
                values=[0.0, 0.05, 0.10, 0.15],
            ),
            SweepParameter(
                name="governance.circuit_breaker_enabled",
                values=[False, True],
            ),
        ],
        runs_per_config=2,  # Multiple runs for statistical significance
        seed_base=42,
    )

    print(f"  Parameters: {len(sweep_config.parameters)}")
    for p in sweep_config.parameters:
        print(f"    - {p.name}: {p.values}")
    print(f"  Runs per config: {sweep_config.runs_per_config}")
    print(f"  Total runs: {sweep_config.total_runs()}")
    print()

    # Run sweep
    print("Running sweep...")
    runner = SweepRunner(sweep_config, progress_callback=progress)
    runner.run()

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)

    # Get summary
    summary = runner.summary()

    print(f"\nTotal runs: {summary['total_runs']}")
    print(f"Parameter combinations: {summary['param_combinations']}")
    print()

    # Print table
    print(
        f"{'Tax Rate':<10} {'Circuit':<10} {'Welfare':<12} {'Toxicity':<12} {'Frozen':<8} {'Honest':<12} {'Adversarial':<12}"
    )
    print("-" * 86)

    for s in summary["summaries"]:
        tax = s.get("governance.transaction_tax_rate", 0)
        circuit = s.get("governance.circuit_breaker_enabled", False)
        print(
            f"{tax:<10.2f} "
            f"{'Yes' if circuit else 'No':<10} "
            f"{s['mean_welfare']:<12.2f} "
            f"{s['mean_toxicity']:<12.4f} "
            f"{s['mean_frozen']:<8.1f} "
            f"{s['mean_honest_payoff']:<12.2f} "
            f"{s['mean_adversarial_payoff']:<12.2f}"
        )

    print()

    # Export to CSV
    print(f"Exporting results to: {args.output}")
    runner.to_csv(args.output)
    print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
