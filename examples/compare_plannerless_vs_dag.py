#!/usr/bin/env python
"""Compare plannerless_coordination vs dag_planner_screening on matched seeds.

Both scenarios now share an identical ``payoff:`` block so this study isolates
the coordination mechanism (emergent artifact exchange vs hierarchical DAG
plan screening) from payoff confounds.

Usage:
    python examples/compare_plannerless_vs_dag.py --seeds 42,43,44 --epochs 20

Writes per-run histories to ``runs/compare_plannerless_vs_dag_seed<seed>/``
and a summary CSV at ``runs/compare_plannerless_vs_dag/summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis.aggregation import MetricsAggregator
from swarm.analysis.export import export_to_json
from swarm.scenarios import build_orchestrator, load_scenario

SCENARIOS = {
    "plannerless": "scenarios/plannerless_coordination.yaml",
    "dag_planner": "scenarios/dag_planner_screening.yaml",
}


def run_one(label: str, scenario_path: str, seed: int, n_epochs: int, run_root: Path):
    scenario = load_scenario(scenario_path)
    scenario.orchestrator_config.seed = seed
    if n_epochs:
        scenario.orchestrator_config.n_epochs = n_epochs

    orchestrator = build_orchestrator(scenario)
    aggregator = MetricsAggregator()
    cfg = scenario.orchestrator_config
    aggregator.start_simulation(
        simulation_id=f"{label}_seed{seed}",
        n_epochs=cfg.n_epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        n_agents=len(orchestrator.get_all_agents()),
        seed=seed,
    )

    def _on_interaction(interaction, p_init, p_cp):
        interaction.metadata["epoch"] = orchestrator.state.current_epoch
        interaction.metadata["step"] = orchestrator.state.current_step
        aggregator.record_interaction(interaction)
        aggregator.record_payoff(interaction.initiator, p_init)
        aggregator.record_payoff(interaction.counterparty, p_cp)
        aggregator.get_history().interactions.append(interaction)

    def _on_epoch(epoch_metrics):
        aggregator.finalize_epoch(
            epoch=epoch_metrics.epoch,
            agent_states=orchestrator.state.agents,
            frozen_agents=orchestrator.state.frozen_agents,
            quarantined_agents=getattr(orchestrator.state, "quarantined_agents", set()),
        )

    orchestrator.on_interaction_complete(_on_interaction)
    orchestrator.on_epoch_end(_on_epoch)
    metrics_history = orchestrator.run()

    history = aggregator.end_simulation()
    out_dir = run_root / f"{label}_seed{seed}"
    export_to_json(history, out_dir / "history.json", include_events=True)

    total_ix = sum(m.total_interactions for m in metrics_history)
    total_accepted = sum(m.accepted_interactions for m in metrics_history)
    avg_tox = sum(m.toxicity_rate for m in metrics_history) / max(len(metrics_history), 1)
    total_welfare = sum(m.total_welfare for m in metrics_history)

    return {
        "scenario": label,
        "seed": seed,
        "epochs": len(metrics_history),
        "interactions": total_ix,
        "accepted": total_accepted,
        "acceptance_rate": total_accepted / total_ix if total_ix else 0.0,
        "avg_toxicity": avg_tox,
        "total_welfare": total_welfare,
        "history_path": str((out_dir / "history.json").resolve()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds")
    ap.add_argument("--epochs", type=int, default=0, help="Override n_epochs (0 = scenario default)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run_id = uuid.uuid4().hex[:8]
    run_root = Path("runs") / f"compare_plannerless_vs_dag_{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        for label, path in SCENARIOS.items():
            print(f"[{label} seed={seed}] running {path} ...")
            rows.append(run_one(label, path, seed, args.epochs, run_root))

    summary_path = run_root / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written: {summary_path}")
    print(f"{'scenario':<15} {'seed':<6} {'acc_rate':<10} {'avg_tox':<10} {'welfare':<10}")
    for r in rows:
        print(f"{r['scenario']:<15} {r['seed']:<6} {r['acceptance_rate']:<10.4f} "
              f"{r['avg_toxicity']:<10.4f} {r['total_welfare']:<10.2f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
