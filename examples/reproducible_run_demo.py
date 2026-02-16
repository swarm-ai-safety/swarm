#!/usr/bin/env python
"""
Demonstrate a complete reproducible SWARM run with artifact generation.

This script shows the recommended workflow for running reproducible experiments:
1. Create timestamped run directory
2. Run simulation with fixed seed
3. Export all artifacts (JSON history, CSV metrics)
4. Save run metadata

Usage:
    python examples/reproducible_run_demo.py
    python examples/reproducible_run_demo.py scenarios/strict_governance.yaml
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.scenarios import build_orchestrator, load_scenario


def main():
    # Configuration
    scenario_path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/baseline.yaml"
    seed = 42
    epochs = 10
    steps_per_epoch = 10

    print("=" * 70)
    print("SWARM Reproducible Run Demo")
    print("=" * 70)
    print()

    # Create timestamped run directory
    scenario_name = Path(scenario_path).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}_{scenario_name}_seed{seed}"
    run_dir = Path(f"runs/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Scenario: {scenario_path}")
    print(f"Seed: {seed}")
    print(f"Epochs: {epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print()

    # Load scenario
    print("Loading scenario...")
    scenario = load_scenario(scenario_path)

    # Override simulation parameters
    scenario.orchestrator_config.seed = seed
    scenario.orchestrator_config.n_epochs = epochs
    scenario.orchestrator_config.steps_per_epoch = steps_per_epoch

    # Build orchestrator
    print("Building orchestrator...")
    orchestrator = build_orchestrator(scenario)

    print(f"Registered {len(orchestrator.get_all_agents())} agents")
    print()

    # Run simulation
    print("Running simulation...")
    print("-" * 70)
    metrics_history = orchestrator.run()
    print("-" * 70)
    print()

    # Export artifacts
    print("Exporting artifacts...")

    # Build simulation history from metrics
    from swarm.analysis.aggregation import EpochSnapshot, SimulationHistory

    history = SimulationHistory(
        simulation_id=scenario.scenario_id,
        n_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        n_agents=len(orchestrator.get_all_agents()),
        seed=seed,
    )

    for m in metrics_history:
        snapshot = EpochSnapshot(
            epoch=m.epoch,
            total_interactions=m.total_interactions,
            accepted_interactions=m.accepted_interactions,
            rejected_interactions=m.total_interactions - m.accepted_interactions,
            toxicity_rate=m.toxicity_rate,
            quality_gap=m.quality_gap,
            total_welfare=m.total_welfare,
            avg_payoff=m.avg_payoff,
            n_agents=len(orchestrator.get_all_agents()),
        )
        history.add_epoch_snapshot(snapshot)

    # 1. Export history JSON
    from swarm.analysis.export import export_to_csv, export_to_json

    history_path = run_dir / "history.json"
    export_to_json(history, str(history_path))
    print(f"  ✓ History: {history_path}")

    # 2. Export CSV metrics
    csv_dir = run_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    paths = export_to_csv(history, str(csv_dir), prefix=scenario.scenario_id)
    for kind, path in paths.items():
        print(f"  ✓ CSV ({kind}): {path}")

    # 3. Save run metadata
    metadata = {
        "scenario": scenario_path,
        "scenario_id": scenario.scenario_id,
        "seed": seed,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "n_agents": len(orchestrator.get_all_agents()),
        "timestamp": timestamp,
        "run_id": run_id,
    }
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata: {metadata_path}")
    print()

    # Summary statistics
    total_interactions = sum(m.total_interactions for m in metrics_history)
    total_accepted = sum(m.accepted_interactions for m in metrics_history)
    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0
    )
    final_welfare = metrics_history[-1].total_welfare if metrics_history else 0

    print("Summary:")
    print(f"  Total interactions: {total_interactions}")
    print(f"  Accepted interactions: {total_accepted}")
    print(f"  Average toxicity: {avg_toxicity:.4f}")
    print(f"  Final welfare: {final_welfare:.2f}")
    print()

    print("=" * 70)
    print("Run complete!")
    print(f"All artifacts saved to: {run_dir}")
    print()
    print("To reproduce this run:")
    print(f"  python -m swarm run {scenario_path} \\")
    print(f"    --seed {seed} \\")
    print(f"    --epochs {epochs} \\")
    print(f"    --steps {steps_per_epoch} \\")
    print("    --export-json runs/verify/history.json \\")
    print("    --export-csv runs/verify/csv/")
    print()
    print("To generate plots:")
    print(f"  python examples/plot_run.py {run_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
