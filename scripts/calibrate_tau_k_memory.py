#!/usr/bin/env python3
"""Run a seeded baseline vs Arm A vs Arm B calibration for tau_min and K_max.

This script uses the memory-tier scenario and maps the two-gate policy to
governance controls that are active in the current SWARM implementation:

- tau_min proxy: ``governance.refinery_p_threshold``
- K_max proxy: ``governance.memory_write_rate_limit_per_epoch``

Arms:
- baseline: no refinery gate, no write cap
- arm_a: refinery gate only (sweep tau)
- arm_b: refinery gate + write cap (sweep tau x K)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from swarm.env.memory_tiers import MemoryTier
from swarm.metrics import memory_metrics
from swarm.models.events import EventType
from swarm.scenarios.loader import build_orchestrator, load_scenario


@dataclass(frozen=True)
class ArmConfig:
    arm: str
    tau_min: Optional[float]
    k_max: Optional[int]
    seed: int


def _parse_csv_list(raw: str, cast_fn):
    return [cast_fn(part.strip()) for part in raw.split(",") if part.strip()]


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _stdev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    return float(statistics.stdev(vals))


def _configure_arm(
    base_path: Path,
    arm_cfg: ArmConfig,
    n_epochs: int,
    steps_per_epoch: int,
):
    scenario = load_scenario(base_path)
    scenario = deepcopy(scenario)

    scenario.orchestrator_config.seed = arm_cfg.seed
    scenario.orchestrator_config.n_epochs = n_epochs
    scenario.orchestrator_config.steps_per_epoch = steps_per_epoch

    # Disable file logging; collect events in-memory for speed and reproducibility.
    scenario.orchestrator_config.log_path = None
    scenario.orchestrator_config.log_events = False

    if scenario.orchestrator_config.memory_tier_config is not None:
        scenario.orchestrator_config.memory_tier_config.seed = arm_cfg.seed

    gov = scenario.orchestrator_config.governance_config
    if gov is None:
        raise RuntimeError("Scenario did not provide a governance config")

    # Keep non-memory governance disabled to isolate the two-gate mechanism.
    gov.collusion_detection_enabled = False
    gov.circuit_breaker_enabled = False
    gov.audit_enabled = False
    gov.transaction_tax_rate = 0.0
    gov.reputation_decay_rate = 1.0

    # Shared governance defaults for calibration.
    gov.refinery_enabled = False
    gov.memory_write_rate_limit_enabled = False
    gov.memory_promotion_gate_enabled = False
    gov.memory_cross_verification_enabled = False
    gov.memory_provenance_enabled = False

    if arm_cfg.arm == "baseline":
        pass
    elif arm_cfg.arm == "arm_a":
        gov.refinery_enabled = True
        gov.refinery_p_threshold = float(arm_cfg.tau_min)
    elif arm_cfg.arm == "arm_b":
        gov.refinery_enabled = True
        gov.refinery_p_threshold = float(arm_cfg.tau_min)
        gov.memory_write_rate_limit_enabled = True
        gov.memory_write_rate_limit_per_epoch = int(arm_cfg.k_max)
    else:
        raise ValueError(f"Unknown arm: {arm_cfg.arm}")

    return scenario


def _run_single(base_path: Path, arm_cfg: ArmConfig, n_epochs: int, steps_per_epoch: int) -> Dict[str, Any]:
    scenario = _configure_arm(base_path, arm_cfg, n_epochs, steps_per_epoch)
    orchestrator = build_orchestrator(scenario)
    events = []
    orchestrator.subscribe_events(events.append)
    metrics_history = orchestrator.run()

    if orchestrator._memory_handler is None:  # type: ignore[attr-defined]
        raise RuntimeError("Memory handler not enabled in selected scenario")
    store = orchestrator._memory_handler.store  # type: ignore[attr-defined]

    total_interactions = sum(m.total_interactions for m in metrics_history)
    accepted_interactions = sum(m.accepted_interactions for m in metrics_history)
    avg_toxicity = _mean(m.toxicity_rate for m in metrics_history)
    total_welfare = float(sum(m.total_welfare for m in metrics_history))
    avg_welfare_per_epoch = _mean(m.total_welfare for m in metrics_history)

    governance_cost_events = [
        e for e in events if e.event_type == EventType.GOVERNANCE_COST_APPLIED
    ]
    write_cap_hits = sum(
        1 for e in governance_cost_events if "memory_write_rate_limit" in e.payload.get("levers", [])
    )

    row = {
        "arm": arm_cfg.arm,
        "seed": arm_cfg.seed,
        "tau_min": arm_cfg.tau_min,
        "k_max": arm_cfg.k_max,
        "epochs": n_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_interactions": total_interactions,
        "accepted_interactions": accepted_interactions,
        "rejected_interactions": total_interactions - accepted_interactions,
        "acceptance_rate": (accepted_interactions / total_interactions) if total_interactions else 0.0,
        "avg_toxicity": avg_toxicity,
        "total_welfare": total_welfare,
        "avg_welfare_per_epoch": avg_welfare_per_epoch,
        "tier3_poisoning_rate": memory_metrics.poisoning_rate(store, MemoryTier.GRAPH),
        "cache_corruption": memory_metrics.cache_corruption(store),
        "information_asymmetry": memory_metrics.information_asymmetry(store),
        "governance_filter_rate": memory_metrics.governance_filter_rate(store),
        "promotion_accuracy": memory_metrics.promotion_accuracy(store),
        "write_concentration": memory_metrics.write_concentration(store),
        "memory_writes": sum(1 for e in events if e.event_type == EventType.MEMORY_WRITTEN),
        "memory_promotions": sum(1 for e in events if e.event_type == EventType.MEMORY_PROMOTED),
        "memory_challenges": sum(1 for e in events if e.event_type == EventType.MEMORY_CHALLENGED),
        "memory_reverted": sum(1 for e in events if e.event_type == EventType.MEMORY_REVERTED),
        "write_cap_hits": write_cap_hits,
        "governance_cost_events": len(governance_cost_events),
    }
    return row


def _group_key(row: Dict[str, Any]) -> tuple:
    return (row["arm"], row["tau_min"], row["k_max"])


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    metrics = [
        "acceptance_rate",
        "rejected_interactions",
        "avg_toxicity",
        "total_welfare",
        "avg_welfare_per_epoch",
        "tier3_poisoning_rate",
        "cache_corruption",
        "information_asymmetry",
        "governance_filter_rate",
        "promotion_accuracy",
        "write_concentration",
        "memory_writes",
        "memory_promotions",
        "memory_challenges",
        "memory_reverted",
        "write_cap_hits",
        "governance_cost_events",
    ]

    out: List[Dict[str, Any]] = []
    for key in sorted(groups.keys(), key=lambda k: (k[0], str(k[1]), str(k[2]))):
        arm, tau_min, k_max = key
        bucket = groups[key]
        record: Dict[str, Any] = {
            "arm": arm,
            "tau_min": tau_min,
            "k_max": k_max,
            "n_runs": len(bucket),
            "seeds": [r["seed"] for r in bucket],
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in bucket]
            record[f"{metric}_mean"] = _mean(vals)
            record[f"{metric}_std"] = _stdev(vals)
        out.append(record)
    return out


def _score(summary: Dict[str, Any], baseline: Dict[str, Any]) -> float:
    # Higher is better. Positive terms represent safety improvements vs baseline.
    poison_gain = baseline["tier3_poisoning_rate_mean"] - summary["tier3_poisoning_rate_mean"]
    cache_gain = baseline["cache_corruption_mean"] - summary["cache_corruption_mean"]
    filter_gain = summary["governance_filter_rate_mean"] - baseline["governance_filter_rate_mean"]
    tox_gain = baseline["avg_toxicity_mean"] - summary["avg_toxicity_mean"]
    welfare_ratio = summary["total_welfare_mean"] / max(abs(baseline["total_welfare_mean"]), 1.0)
    acceptance_delta = summary["acceptance_rate_mean"] - baseline["acceptance_rate_mean"]

    return (
        3.0 * poison_gain
        + 2.0 * cache_gain
        + 2.0 * filter_gain
        + 1.0 * tox_gain
        + 0.5 * welfare_ratio
        + 2.0 * acceptance_delta
    )


def _select_recommendation(agg: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline = next((r for r in agg if r["arm"] == "baseline"), None)
    if baseline is None:
        raise RuntimeError("No baseline aggregate row found")

    arm_a = [r for r in agg if r["arm"] == "arm_a"]
    arm_b = [r for r in agg if r["arm"] == "arm_b"]

    for row in arm_a:
        row["composite_score"] = _score(row, baseline)
    for row in arm_b:
        row["composite_score"] = _score(row, baseline)

    if not arm_a:
        raise RuntimeError("No Arm A rows found")
    best_a = max(
        arm_a,
        key=lambda r: (
            r["composite_score"],
            r["total_welfare_mean"],
            -float(r["tau_min"] if r["tau_min"] is not None else 0.0),
        ),
    )
    chosen_tau = float(best_a["tau_min"])

    b_with_tau = [r for r in arm_b if float(r["tau_min"]) == chosen_tau]
    if not b_with_tau:
        b_with_tau = arm_b
    if not b_with_tau:
        raise RuntimeError("No Arm B rows found")

    best_b = max(
        b_with_tau,
        key=lambda r: (
            r["composite_score"],
            r["total_welfare_mean"],
            r["k_max"] if r["k_max"] is not None else -1,
        ),
    )

    return {
        "recommended_tau_min": chosen_tau,
        "recommended_k_max": int(best_b["k_max"]),
        "baseline": baseline,
        "best_arm_a": best_a,
        "best_arm_b": best_b,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate tau_min and K_max on memory tiers")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("scenarios/memory_tiers.yaml"),
        help="Base scenario file",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="101,202,303,404,505",
        help="Comma-separated seed list",
    )
    parser.add_argument(
        "--tau-values",
        type=str,
        default="0.45,0.55,0.65",
        help="Comma-separated tau_min candidates for Arm A/B",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="2,4,6",
        help="Comma-separated K_max candidates for Arm B",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per run")
    parser.add_argument("--steps", type=int, default=8, help="Steps per epoch")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/<timestamp>_tau_k_calibration)",
    )
    args = parser.parse_args()

    if not args.scenario.exists():
        raise FileNotFoundError(f"Scenario not found: {args.scenario}")

    seeds = _parse_csv_list(args.seeds, int)
    tau_values = _parse_csv_list(args.tau_values, float)
    k_values = _parse_csv_list(args.k_values, int)

    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"{stamp}_tau_k_calibration"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configs: List[ArmConfig] = []
    for seed in seeds:
        configs.append(ArmConfig(arm="baseline", tau_min=None, k_max=None, seed=seed))
        for tau in tau_values:
            configs.append(ArmConfig(arm="arm_a", tau_min=tau, k_max=None, seed=seed))
            for k in k_values:
                configs.append(ArmConfig(arm="arm_b", tau_min=tau, k_max=k, seed=seed))

    total = len(configs)
    rows: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(configs, start=1):
        print(
            f"[{idx:03d}/{total}] arm={cfg.arm} seed={cfg.seed} tau={cfg.tau_min} k={cfg.k_max}",
            flush=True,
        )
        row = _run_single(args.scenario, cfg, args.epochs, args.steps)
        rows.append(row)

    agg = _aggregate(rows)
    recommendation = _select_recommendation(agg)

    runs_csv = output_dir / "runs.csv"
    summary_json = output_dir / "summary.json"
    rec_json = output_dir / "recommendation.json"

    _write_csv(runs_csv, rows)
    with open(summary_json, "w") as f:
        json.dump(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "scenario": str(args.scenario),
                "seeds": seeds,
                "tau_values": tau_values,
                "k_values": k_values,
                "epochs": args.epochs,
                "steps_per_epoch": args.steps,
                "aggregates": agg,
            },
            f,
            indent=2,
        )
    with open(rec_json, "w") as f:
        json.dump(recommendation, f, indent=2)

    print()
    print(f"runs.csv: {runs_csv}")
    print(f"summary.json: {summary_json}")
    print(f"recommendation.json: {rec_json}")
    print(
        "recommended "
        f"tau_min={recommendation['recommended_tau_min']:.2f}, "
        f"K_max={recommendation['recommended_k_max']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
