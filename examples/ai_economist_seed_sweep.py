#!/usr/bin/env python
"""AI Economist GTB Multi-Seed Sweep.

Run the AI Economist Gather-Trade-Build scenario across multiple seeds
to establish statistical rigor for three findings: emergent progressive
taxation, collusion failure, and the cost of evasion.

Usage:
    python examples/ai_economist_seed_sweep.py
    python examples/ai_economist_seed_sweep.py --seeds 10 --epochs 20 --steps 10
    python examples/ai_economist_seed_sweep.py --scenario scenarios/ai_economist_full.yaml

Smoke test:
    python examples/ai_economist_seed_sweep.py --seeds 2 --epochs 3 --steps 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from swarm.domains.gather_trade_build.config import GTBConfig
from swarm.domains.gather_trade_build.runner import GTBScenarioRunner


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SeedResult:
    """Scalar summary from one seed run."""

    seed: int = 0
    n_epochs: int = 0

    # Final-epoch metrics
    final_production: float = 0.0
    final_tax_revenue: float = 0.0
    final_gini: float = 0.0
    final_welfare: float = 0.0
    final_effective_tax_rate: float = 0.0
    final_bunching_intensity: float = 0.0
    final_collusion_events: int = 0
    final_collusion_suspicion: float = 0.0
    final_exploit_frequency: float = 0.0
    final_variance_amplification: float = 0.0

    # Cumulative enforcement
    total_audits: int = 0
    total_catches: int = 0
    total_fines: float = 0.0

    # Tax schedule progressivity
    progressivity_index: float = 0.0  # max_rate - min_rate

    # Per-type mean wealth
    honest_mean_wealth: float = 0.0
    gaming_mean_wealth: float = 0.0
    evasive_mean_wealth: float = 0.0
    collusive_mean_wealth: float = 0.0

    # Tax bracket rates (final)
    bracket_rate_0: float = 0.0
    bracket_rate_1: float = 0.0
    bracket_rate_2: float = 0.0
    bracket_rate_3: float = 0.0


def _agent_type(agent_id: str) -> str:
    """Extract agent type from worker ID like 'worker_honest_0_1'."""
    parts = agent_id.split("_")
    if len(parts) >= 3:
        return parts[1]
    return "unknown"


def _mean_wealth_by_type(
    workers: dict, types: list[str],
) -> Dict[str, float]:
    """Compute mean total wealth per agent type."""
    from swarm.domains.gather_trade_build.entities import ResourceType

    type_wealth: Dict[str, List[float]] = {t: [] for t in types}
    for aid, w in workers.items():
        atype = _agent_type(aid)
        if atype in type_wealth:
            coin = w.get_resource(ResourceType.COIN)
            house_val = w.houses_built * 50
            type_wealth[atype].append(coin + house_val)
    return {
        t: (sum(vals) / len(vals) if vals else 0.0)
        for t, vals in type_wealth.items()
    }


def run_single_seed(
    config: GTBConfig,
    agent_specs: List[Dict[str, Any]],
    seed: int,
    n_epochs: int,
    steps_per_epoch: int,
    output_dir: Path,
) -> SeedResult:
    """Run one seed and export per-seed CSVs."""
    runner = GTBScenarioRunner(
        config=config,
        agent_specs=agent_specs,
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
    )

    epoch_metrics = runner.run()

    # Export per-seed data
    seed_dir = output_dir / f"seed_{seed}"
    runner.export(output_dir=str(seed_dir))

    # Extract final metrics
    final = epoch_metrics[-1] if epoch_metrics else None

    # Tax schedule
    sched = runner.env.tax_schedule.to_dict()
    brackets = sched.get("brackets", [])
    rates = [b["rate"] for b in brackets]
    progressivity = (max(rates) - min(rates)) if rates else 0.0

    # Per-type wealth
    wealth = _mean_wealth_by_type(
        runner.env.workers, ["honest", "gaming", "evasive", "collusive"],
    )

    # Cumulative enforcement
    cum_audits = sum(m.total_audits for m in epoch_metrics)
    cum_catches = sum(m.total_catches for m in epoch_metrics)
    cum_fines = sum(m.total_fines for m in epoch_metrics)

    result = SeedResult(
        seed=seed,
        n_epochs=len(epoch_metrics),
        final_production=final.total_production if final else 0.0,
        final_tax_revenue=final.total_tax_revenue if final else 0.0,
        final_gini=final.gini_coefficient if final else 0.0,
        final_welfare=final.welfare if final else 0.0,
        final_effective_tax_rate=final.mean_effective_tax_rate if final else 0.0,
        final_bunching_intensity=final.bunching_intensity if final else 0.0,
        final_collusion_events=final.collusion_events_detected if final else 0,
        final_collusion_suspicion=final.collusion_suspicion_mean if final else 0.0,
        final_exploit_frequency=final.exploit_frequency if final else 0.0,
        final_variance_amplification=final.variance_amplification if final else 0.0,
        total_audits=cum_audits,
        total_catches=cum_catches,
        total_fines=cum_fines,
        progressivity_index=progressivity,
        honest_mean_wealth=wealth.get("honest", 0.0),
        gaming_mean_wealth=wealth.get("gaming", 0.0),
        evasive_mean_wealth=wealth.get("evasive", 0.0),
        collusive_mean_wealth=wealth.get("collusive", 0.0),
        bracket_rate_0=rates[0] if len(rates) > 0 else 0.0,
        bracket_rate_1=rates[1] if len(rates) > 1 else 0.0,
        bracket_rate_2=rates[2] if len(rates) > 2 else 0.0,
        bracket_rate_3=rates[3] if len(rates) > 3 else 0.0,
    )

    return result


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def write_sweep_results(results: List[SeedResult], path: Path) -> None:
    """Write one-row-per-seed summary CSV."""
    fieldnames = [f.name for f in fields(SeedResult)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {}
            for fname in fieldnames:
                val = getattr(r, fname)
                if isinstance(val, float):
                    row[fname] = f"{val:.6f}"
                else:
                    row[fname] = val
            writer.writerow(row)


def write_all_metrics(output_dir: Path, seeds: List[int]) -> None:
    """Concatenate per-seed metrics CSVs into all_metrics.csv with seed column."""
    import pandas as pd

    dfs = []
    for seed in seeds:
        csv_path = output_dir / f"seed_{seed}" / "csv" / "metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.insert(0, "seed", seed)
            dfs.append(df)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(output_dir / "all_metrics.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Economist GTB multi-seed sweep"
    )
    parser.add_argument(
        "--scenario", type=Path,
        default=Path("scenarios/ai_economist_full.yaml"),
        help="Scenario YAML file",
    )
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds (starting from 42)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    # Load scenario
    with open(args.scenario) as f:
        scenario = yaml.safe_load(f)

    domain_data = scenario.get("domain", {})
    config = GTBConfig.from_dict(domain_data)
    agent_specs = scenario.get("agents", [])

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path(f"runs/{timestamp}_ai_economist_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(42, 42 + args.seeds))

    print("=" * 70)
    print("AI Economist GTB Multi-Seed Sweep")
    print(f"  Scenario: {args.scenario}")
    print(f"  Seeds: {seeds}")
    print(f"  Epochs: {args.epochs}, Steps/epoch: {args.steps}")
    print(f"  Agents: {sum(s.get('count', 1) for s in agent_specs)}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    # Save config for reproducibility
    with open(out_dir / "sweep_config.json", "w") as f:
        json.dump({
            "scenario": str(args.scenario),
            "seeds": seeds,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps,
            "timestamp": timestamp,
        }, f, indent=2)

    all_results: List[SeedResult] = []

    for i, seed in enumerate(seeds, 1):
        print(f"\n  [{i}/{len(seeds)}] seed={seed}...", flush=True)
        result = run_single_seed(
            config, agent_specs, seed, args.epochs, args.steps, out_dir,
        )
        all_results.append(result)
        print(
            f"    production={result.final_production:.1f}, "
            f"gini={result.final_gini:.3f}, "
            f"welfare={result.final_welfare:.2f}, "
            f"progressivity={result.progressivity_index:.3f}"
        )
        print(
            f"    honest_wealth={result.honest_mean_wealth:.1f}, "
            f"collusive_wealth={result.collusive_mean_wealth:.1f}, "
            f"evasive_wealth={result.evasive_mean_wealth:.1f}"
        )

    # Write sweep summary
    sweep_csv = out_dir / "sweep_results.csv"
    write_sweep_results(all_results, sweep_csv)
    print(f"\n  -> {sweep_csv}")

    # Write concatenated time-series
    write_all_metrics(out_dir, seeds)
    print(f"  -> {out_dir / 'all_metrics.csv'}")

    # Print summary table
    import numpy as np

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)

    for metric in [
        "final_production", "final_gini", "final_welfare",
        "progressivity_index", "final_bunching_intensity",
        "honest_mean_wealth", "collusive_mean_wealth",
        "evasive_mean_wealth", "gaming_mean_wealth",
    ]:
        vals = [getattr(r, metric) for r in all_results]
        arr = np.array(vals)
        print(f"  {metric:>30s}: {arr.mean():.3f} ± {arr.std():.3f}")

    print(f"\nAll outputs in: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
