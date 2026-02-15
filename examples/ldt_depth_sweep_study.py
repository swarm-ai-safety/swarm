#!/usr/bin/env python
"""
LDT Depth Sweep Study

Sweep acausality depth 1-5 and measure how deeper acausal reasoning
affects welfare, toxicity, and per-class payoffs.

Hypothesis: "deeper reasoning doesn't always produce deeper cooperation."

Methodology:
  - Fix population at 11 agents: 4 LDT (focal) + 3 honest + 2 opportunistic
    + 2 adversarial
  - Sweep LDT acausality_depth from 1 to 5
  - 3 seeds per depth (default)
  - Compare welfare, toxicity, and LDT payoff across depths

Usage:
    python examples/ldt_depth_sweep_study.py
    python examples/ldt_depth_sweep_study.py --total-agents 11 --epochs 15 --steps 10 --seeds 3

Smoke test:
    python examples/ldt_depth_sweep_study.py --total-agents 4 --epochs 3 --steps 3 --seeds 1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swarm.agents.base import BaseAgent
from swarm.agents.honest import HonestAgent
from swarm.agents.ldt_agent import LDTAgent
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig
from swarm.governance.config import GovernanceConfig


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    """Result from one run (one depth + one seed)."""

    depth: int
    seed: int
    n_epochs: int
    mean_welfare: float = 0.0
    total_welfare: float = 0.0
    mean_toxicity: float = 0.0
    mean_quality_gap: float = 0.0
    mean_avg_payoff: float = 0.0
    ldt_avg_payoff: float = 0.0
    honest_avg_payoff: float = 0.0
    opportunistic_avg_payoff: float = 0.0


@dataclass
class AggResult:
    """Aggregated result across seeds for one depth."""

    depth: int
    n_seeds: int
    welfare_mean: float
    welfare_std: float
    toxicity_mean: float
    toxicity_std: float
    quality_gap_mean: float
    avg_payoff_mean: float
    ldt_payoff_mean: float
    ldt_payoff_std: float
    honest_payoff_mean: float
    opportunistic_payoff_mean: float


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------
def create_agents(total: int, depth: int) -> List[BaseAgent]:
    """Create a mixed population with LDT agents at the given depth."""
    agents: List[BaseAgent] = []

    # Distribute: ~36% LDT, ~27% honest, ~18% opportunistic, ~18% adversarial
    n_ldt = max(1, round(total * 0.36))
    n_honest = max(1, round(total * 0.27))
    n_opp = max(1, round(total * 0.18))
    # Fill remainder as adversarial (use opportunistic with low prior as proxy)
    n_adv = max(0, total - n_ldt - n_honest - n_opp)

    ldt_config = {
        "cooperation_prior": 0.65,
        "similarity_threshold": 0.7,
        "welfare_weight": 0.3,
        "updateless_commitment": 0.8,
        "acausality_depth": depth,
        "decision_theory": "fdt",
        "n_counterfactual_samples": 20,  # keep MC fast
    }

    for i in range(n_ldt):
        agents.append(LDTAgent(agent_id=f"ldt_{i+1}", config=ldt_config))
    for i in range(n_honest):
        agents.append(HonestAgent(agent_id=f"honest_{i+1}"))
    for i in range(n_opp):
        agents.append(OpportunisticAgent(agent_id=f"opp_{i+1}"))
    for i in range(n_adv):
        agents.append(OpportunisticAgent(
            agent_id=f"adv_{i+1}",
            config={"exploit_probability": 0.8},
        ))

    return agents


# ---------------------------------------------------------------------------
# Per-class payoff extraction
# ---------------------------------------------------------------------------
def _extract_per_class_payoffs(
    orchestrator: Orchestrator,
) -> Dict[str, List[float]]:
    payoffs: Dict[str, List[float]] = {
        "ldt": [], "honest": [], "opportunistic": [],
    }
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        if state is None:
            continue
        if isinstance(agent, LDTAgent):
            payoffs["ldt"].append(state.total_payoff)
        elif isinstance(agent, HonestAgent):
            payoffs["honest"].append(state.total_payoff)
        elif isinstance(agent, OpportunisticAgent):
            payoffs["opportunistic"].append(state.total_payoff)
    return payoffs


def _mean_or_zero(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(
    total: int, depth: int, seed: int, n_epochs: int, steps: int,
) -> RunResult:
    payoff_config = PayoffConfig(
        s_plus=2.0, s_minus=1.0, h=1.0, theta=0.5,
        rho_a=0.1, rho_b=0.1, w_rep=0.1,
    )
    governance_config = GovernanceConfig(
        circuit_breaker_enabled=True,
        reputation_decay_rate=0.05,
    )
    orch_config = OrchestratorConfig(
        n_epochs=n_epochs, steps_per_epoch=steps, seed=seed,
        payoff_config=payoff_config, governance_config=governance_config,
    )

    orchestrator = Orchestrator(config=orch_config)
    agents = create_agents(total, depth)
    for agent in agents:
        orchestrator.register_agent(agent)

    epoch_metrics = orchestrator.run()

    welfares = [em.total_welfare for em in epoch_metrics]
    toxicities = [em.toxicity_rate for em in epoch_metrics]
    qgaps = [em.quality_gap for em in epoch_metrics]
    payoffs = [em.avg_payoff for em in epoch_metrics]
    class_payoffs = _extract_per_class_payoffs(orchestrator)

    return RunResult(
        depth=depth, seed=seed, n_epochs=len(epoch_metrics),
        mean_welfare=float(np.mean(welfares)) if welfares else 0.0,
        total_welfare=float(np.sum(welfares)) if welfares else 0.0,
        mean_toxicity=float(np.mean(toxicities)) if toxicities else 0.0,
        mean_quality_gap=float(np.mean(qgaps)) if qgaps else 0.0,
        mean_avg_payoff=float(np.mean(payoffs)) if payoffs else 0.0,
        ldt_avg_payoff=_mean_or_zero(class_payoffs["ldt"]),
        honest_avg_payoff=_mean_or_zero(class_payoffs["honest"]),
        opportunistic_avg_payoff=_mean_or_zero(class_payoffs["opportunistic"]),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(results: List[RunResult]) -> List[AggResult]:
    groups: Dict[int, List[RunResult]] = defaultdict(list)
    for r in results:
        groups[r.depth].append(r)

    aggs: List[AggResult] = []
    for depth in sorted(groups.keys()):
        runs = groups[depth]
        aggs.append(AggResult(
            depth=depth,
            n_seeds=len(runs),
            welfare_mean=float(np.mean([r.total_welfare for r in runs])),
            welfare_std=float(np.std([r.total_welfare for r in runs])),
            toxicity_mean=float(np.mean([r.mean_toxicity for r in runs])),
            toxicity_std=float(np.std([r.mean_toxicity for r in runs])),
            quality_gap_mean=float(np.mean([r.mean_quality_gap for r in runs])),
            avg_payoff_mean=float(np.mean([r.mean_avg_payoff for r in runs])),
            ldt_payoff_mean=float(np.mean([r.ldt_avg_payoff for r in runs])),
            ldt_payoff_std=float(np.std([r.ldt_avg_payoff for r in runs])),
            honest_payoff_mean=float(np.mean([r.honest_avg_payoff for r in runs])),
            opportunistic_payoff_mean=float(np.mean([r.opportunistic_avg_payoff for r in runs])),
        ))
    return aggs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
DPI = 160
DEPTH_COLOR = "#9C27B0"


def _style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_welfare_by_depth(aggs: List[AggResult], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    depths = [a.depth for a in aggs]
    welfares = [a.welfare_mean for a in aggs]
    stds = [a.welfare_std for a in aggs]
    ax.errorbar(
        depths, welfares, yerr=stds,
        color=DEPTH_COLOR, linewidth=2.5, marker="D", markersize=8,
        capsize=5, capthick=1.5,
    )
    _style_ax(ax, "Total Welfare by Acausality Depth", "Acausality Depth", "Total Welfare")
    ax.set_xticks(depths)
    out = out_dir / "welfare_by_depth.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_toxicity_by_depth(aggs: List[AggResult], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    depths = [a.depth for a in aggs]
    tox = [a.toxicity_mean for a in aggs]
    stds = [a.toxicity_std for a in aggs]
    ax.errorbar(
        depths, tox, yerr=stds,
        color="#F44336", linewidth=2.5, marker="o", markersize=8,
        capsize=5, capthick=1.5,
    )
    _style_ax(ax, "Toxicity Rate by Acausality Depth", "Acausality Depth", "Toxicity Rate")
    ax.set_xticks(depths)
    ax.set_ylim(-0.05, 1.05)
    out = out_dir / "toxicity_by_depth.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ldt_payoff_by_depth(aggs: List[AggResult], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    depths = [a.depth for a in aggs]
    payoffs = [a.ldt_payoff_mean for a in aggs]
    stds = [a.ldt_payoff_std for a in aggs]
    ax.errorbar(
        depths, payoffs, yerr=stds,
        color=DEPTH_COLOR, linewidth=2.5, marker="s", markersize=8,
        capsize=5, capthick=1.5,
    )
    _style_ax(ax, "LDT Agent Payoff by Acausality Depth", "Acausality Depth", "LDT Avg Payoff")
    ax.set_xticks(depths)
    out = out_dir / "ldt_payoff_by_depth.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_depth1_vs_5_comparison(aggs: List[AggResult], out_dir: Path) -> Path:
    """Bar chart comparing depth 1 vs depth 5 across key metrics."""
    d1 = next((a for a in aggs if a.depth == 1), None)
    d5 = next((a for a in aggs if a.depth == 5), None)
    if not d1 or not d5:
        # If missing, use first and last
        d1 = aggs[0]
        d5 = aggs[-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["Welfare", "Toxicity", "LDT Payoff", "Honest Payoff"]
    d1_vals = [d1.welfare_mean, d1.toxicity_mean, d1.ldt_payoff_mean, d1.honest_payoff_mean]
    d5_vals = [d5.welfare_mean, d5.toxicity_mean, d5.ldt_payoff_mean, d5.honest_payoff_mean]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, d1_vals, width, label=f"Depth {d1.depth}", color="#2196F3", alpha=0.85)
    ax.bar(x + width / 2, d5_vals, width, label=f"Depth {d5.depth}", color=DEPTH_COLOR, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    _style_ax(
        ax,
        f"Depth {d1.depth} vs Depth {d5.depth}: Key Metrics Comparison",
        "", "Value",
    )
    ax.legend(fontsize=10)

    out = out_dir / "depth1_vs_5_comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def write_results_csv(results: List[RunResult], path: Path) -> None:
    fieldnames = [
        "depth", "seed", "n_epochs", "mean_welfare", "total_welfare",
        "mean_toxicity", "mean_quality_gap", "mean_avg_payoff",
        "ldt_avg_payoff", "honest_avg_payoff", "opportunistic_avg_payoff",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "depth": r.depth, "seed": r.seed, "n_epochs": r.n_epochs,
                "mean_welfare": f"{r.mean_welfare:.6f}",
                "total_welfare": f"{r.total_welfare:.6f}",
                "mean_toxicity": f"{r.mean_toxicity:.6f}",
                "mean_quality_gap": f"{r.mean_quality_gap:.6f}",
                "mean_avg_payoff": f"{r.mean_avg_payoff:.6f}",
                "ldt_avg_payoff": f"{r.ldt_avg_payoff:.6f}",
                "honest_avg_payoff": f"{r.honest_avg_payoff:.6f}",
                "opportunistic_avg_payoff": f"{r.opportunistic_avg_payoff:.6f}",
            })


def write_aggregated_csv(aggs: List[AggResult], path: Path) -> None:
    fieldnames = [
        "depth", "n_seeds", "welfare_mean", "welfare_std",
        "toxicity_mean", "toxicity_std", "quality_gap_mean",
        "avg_payoff_mean", "ldt_payoff_mean", "ldt_payoff_std",
        "honest_payoff_mean", "opportunistic_payoff_mean",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in aggs:
            writer.writerow({
                "depth": a.depth, "n_seeds": a.n_seeds,
                "welfare_mean": f"{a.welfare_mean:.4f}",
                "welfare_std": f"{a.welfare_std:.4f}",
                "toxicity_mean": f"{a.toxicity_mean:.4f}",
                "toxicity_std": f"{a.toxicity_std:.4f}",
                "quality_gap_mean": f"{a.quality_gap_mean:.4f}",
                "avg_payoff_mean": f"{a.avg_payoff_mean:.4f}",
                "ldt_payoff_mean": f"{a.ldt_payoff_mean:.4f}",
                "ldt_payoff_std": f"{a.ldt_payoff_std:.4f}",
                "honest_payoff_mean": f"{a.honest_payoff_mean:.4f}",
                "opportunistic_payoff_mean": f"{a.opportunistic_payoff_mean:.4f}",
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="LDT depth sweep study: deeper reasoning vs cooperation"
    )
    parser.add_argument("--total-agents", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or Path(f"runs/{timestamp}_ldt_depth_sweep")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    depths = [1, 2, 3, 4, 5]
    seeds = list(range(42, 42 + args.seeds))

    print("=" * 70)
    print("LDT Depth Sweep Study")
    print(f"  Agents: {args.total_agents}, Epochs: {args.epochs}, Steps: {args.steps}")
    print(f"  Seeds: {seeds}")
    print(f"  Depths: {depths}")
    print(f"  Total runs: {len(depths) * len(seeds)}")
    print("=" * 70)

    all_results: List[RunResult] = []
    total_runs = len(depths) * len(seeds)
    run_idx = 0

    for depth in depths:
        for seed in seeds:
            run_idx += 1
            print(
                f"  [{run_idx}/{total_runs}] depth={depth}, seed={seed}...",
                end=" ", flush=True,
            )
            result = run_single(args.total_agents, depth, seed, args.epochs, args.steps)
            all_results.append(result)
            print(
                f"welfare={result.total_welfare:.1f}, "
                f"toxicity={result.mean_toxicity:.3f}, "
                f"ldt_payoff={result.ldt_avg_payoff:.2f}"
            )

    aggs = aggregate_results(all_results)

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS BY DEPTH")
    print("=" * 70)
    print(
        f"{'Depth':>5} {'Seeds':>5} {'Welfare':>10} {'WStd':>8} "
        f"{'Toxicity':>9} {'TStd':>7} {'LDT_Pay':>8} {'Hon_Pay':>8} {'Opp_Pay':>8}"
    )
    print("-" * 80)
    for a in aggs:
        print(
            f"{a.depth:>5} {a.n_seeds:>5} {a.welfare_mean:>10.2f} "
            f"{a.welfare_std:>8.2f} {a.toxicity_mean:>9.3f} "
            f"{a.toxicity_std:>7.3f} {a.ldt_payoff_mean:>8.2f} "
            f"{a.honest_payoff_mean:>8.2f} {a.opportunistic_payoff_mean:>8.2f}"
        )

    # Key comparisons
    if len(aggs) >= 2:
        d1 = aggs[0]
        d_max = aggs[-1]
        print(f"\n  Depth {d1.depth} → {d_max.depth}:")
        print(f"    Welfare: {d1.welfare_mean:.2f} → {d_max.welfare_mean:.2f} "
              f"(diff={d_max.welfare_mean - d1.welfare_mean:+.2f})")
        print(f"    Toxicity: {d1.toxicity_mean:.3f} → {d_max.toxicity_mean:.3f} "
              f"(diff={d_max.toxicity_mean - d1.toxicity_mean:+.3f})")
        print(f"    LDT payoff: {d1.ldt_payoff_mean:.2f} → {d_max.ldt_payoff_mean:.2f} "
              f"(diff={d_max.ldt_payoff_mean - d1.ldt_payoff_mean:+.2f})")

    # Generate plots
    print("\nGenerating plots...")
    plots: List[Path] = []
    plots.append(plot_welfare_by_depth(aggs, plots_dir))
    plots.append(plot_toxicity_by_depth(aggs, plots_dir))
    plots.append(plot_ldt_payoff_by_depth(aggs, plots_dir))
    plots.append(plot_depth1_vs_5_comparison(aggs, plots_dir))

    for p in plots:
        print(f"  -> {p}")

    # Export CSVs
    csv_path = out_dir / "results.csv"
    write_results_csv(all_results, csv_path)
    print(f"  -> {csv_path}")

    agg_csv = out_dir / "aggregated_results.csv"
    write_aggregated_csv(aggs, agg_csv)
    print(f"  -> {agg_csv}")

    print(f"\nAll outputs in: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
