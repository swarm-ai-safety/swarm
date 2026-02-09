"""Run incoherence scaling sweeps over horizon/branching tiers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from swarm.analysis.aggregation import aggregate_incoherence_scaling
from swarm.replay import EpisodeSpec, ReplayRunner
from swarm.scenarios import load_scenario

SCENARIO_MAP = {
    ("short", "low"): Path("scenarios/incoherence/short_low_branching.yaml"),
    ("medium", "medium"): Path("scenarios/incoherence/medium_medium_branching.yaml"),
    ("long", "high"): Path("scenarios/incoherence/long_high_branching.yaml"),
}


def _proxy_incoherence_from_toxicity(toxicity_values: List[float]) -> Dict[str, float]:
    """
    Temporary proxy for incoherence signal from replay toxicity variation.

    This is a bridge artifact until step-level decision replay metrics are
    fully integrated into the experiment runner.
    """
    if not toxicity_values:
        return {"incoherence_index": 0.0, "error_rate": 0.0, "disagreement_rate": 0.0}

    mean_toxicity = sum(toxicity_values) / len(toxicity_values)
    disagreement = sum(abs(value - mean_toxicity) for value in toxicity_values) / len(
        toxicity_values
    )
    error = mean_toxicity
    incoherence = min(1.0, disagreement / (error + 1e-8)) if error > 0 else 0.0
    return {
        "incoherence_index": incoherence,
        "error_rate": error,
        "disagreement_rate": disagreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run incoherence scaling sweeps.")
    parser.add_argument("--replay-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("logs/incoherence_scaling")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    replay_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for (horizon_tier, branching_tier), path in SCENARIO_MAP.items():
        scenario = load_scenario(path)
        scenario.orchestrator_config.log_path = None
        scenario.orchestrator_config.log_events = False

        spec = EpisodeSpec(scenario=scenario, seed=args.seed, replay_k=args.replay_k)
        results = ReplayRunner(spec).run()
        toxicity_values = [result.avg_toxicity for result in results]
        proxy = _proxy_incoherence_from_toxicity(toxicity_values)

        for result in results:
            replay_rows.append(
                {
                    "horizon_tier": horizon_tier,
                    "branching_tier": branching_tier,
                    "replay_index": result.replay_index,
                    "seed": result.seed,
                    "total_interactions": result.total_interactions,
                    "avg_toxicity": result.avg_toxicity,
                    "avg_quality_gap": result.avg_quality_gap,
                    "total_welfare": result.total_welfare,
                    "incoherence_index": proxy["incoherence_index"],
                    "error_rate": proxy["error_rate"],
                    "disagreement_rate": proxy["disagreement_rate"],
                }
            )

        summary_rows.extend(
            aggregate_incoherence_scaling(
                [
                    {
                        "horizon_tier": horizon_tier,
                        "branching_tier": branching_tier,
                        "incoherence_index": proxy["incoherence_index"],
                        "error_rate": proxy["error_rate"],
                        "disagreement_rate": proxy["disagreement_rate"],
                    }
                ]
            )
        )

    replay_csv = args.output_dir / "incoherence_scaling_replays.csv"
    summary_csv = args.output_dir / "incoherence_scaling_summary.csv"

    with open(replay_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replay_rows[0].keys()))
        writer.writeheader()
        writer.writerows(replay_rows)

    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote replay rows to {replay_csv}")
    print(f"Wrote summary rows to {summary_csv}")


if __name__ == "__main__":
    main()
