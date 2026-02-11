#!/usr/bin/env python
"""
LDT Agent Composition Study

Study whether LDT (Logical Decision Theory) agents produce better welfare/toxicity
outcomes than honest agents when varied in population composition.

Methodology:
  - Fix total population at 10 agents
  - Two parallel sweeps:
    - LDT sweep: Vary LDT agents 0-100% in 10% steps, fill remaining with
      deceptive (60%) + opportunistic (40%)
    - Honest baseline: Same sweep but with honest agents instead of LDT
  - Non-focal slots filled with mix of deceptive + opportunistic agents
  - Use baseline payoff parameters (s_plus=2, s_minus=1, h=1)
  - Run each configuration for 30 epochs, 10 steps/epoch, 3 seeds
  - Compare welfare and toxicity across compositions and sweeps

Usage:
    python examples/ldt_composition_study.py
    python examples/ldt_composition_study.py --total-agents 10 --epochs 30 --steps 10 --seeds 3

Smoke test:
    python examples/ldt_composition_study.py --total-agents 4 --epochs 3 --steps 3 --seeds 1
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
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swarm.agents.base import BaseAgent
from swarm.agents.deceptive import DeceptiveAgent
from swarm.agents.honest import HonestAgent
from swarm.agents.ldt_agent import LDTAgent
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig
from swarm.governance.config import GovernanceConfig


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
@dataclass
class CompositionConfig:
    """A single population composition to test."""

    label: str
    sweep_type: str  # "ldt" or "honest"
    n_ldt: int
    n_honest: int
    n_deceptive: int
    n_opportunistic: int

    @property
    def total(self) -> int:
        return self.n_ldt + self.n_honest + self.n_deceptive + self.n_opportunistic

    @property
    def focal_pct(self) -> float:
        """Percentage of the focal agent type (LDT or honest)."""
        if self.sweep_type == "ldt":
            return self.n_ldt / self.total if self.total else 0.0
        return self.n_honest / self.total if self.total else 0.0


@dataclass
class RunResult:
    """Result from one run (one composition + one seed)."""

    composition: str
    sweep_type: str
    focal_pct: float
    seed: int
    n_epochs: int
    # Aggregates across all epochs
    mean_welfare: float = 0.0
    total_welfare: float = 0.0
    mean_toxicity: float = 0.0
    mean_quality_gap: float = 0.0
    mean_avg_payoff: float = 0.0
    final_welfare: float = 0.0
    final_toxicity: float = 0.0
    # Per-class average payoffs
    ldt_avg_payoff: float = 0.0
    honest_avg_payoff: float = 0.0
    deceptive_avg_payoff: float = 0.0
    opportunistic_avg_payoff: float = 0.0


@dataclass
class AggResult:
    """Aggregated result across seeds for one composition."""

    label: str
    sweep_type: str
    focal_pct: float
    n_seeds: int
    welfare_mean: float
    welfare_std: float
    welfare_total_mean: float
    toxicity_mean: float
    toxicity_std: float
    quality_gap_mean: float
    avg_payoff_mean: float
    # Per-class payoff means
    ldt_payoff_mean: float
    honest_payoff_mean: float
    deceptive_payoff_mean: float
    opportunistic_payoff_mean: float


# ---------------------------------------------------------------------------
# Composition builder
# ---------------------------------------------------------------------------
def build_compositions(total_agents: int) -> List[CompositionConfig]:
    """Build compositions for both LDT and honest sweeps.

    Each sweep varies the focal agent type from 0% to 100% in 10% steps.
    Remaining slots are filled with deceptive (60%) + opportunistic (40%).
    Returns 22 configs (11 LDT + 11 honest).
    """
    compositions: List[CompositionConfig] = []

    for sweep_type in ("ldt", "honest"):
        for focal_pct in range(0, 101, 10):
            n_focal = round(total_agents * focal_pct / 100)
            remaining = total_agents - n_focal
            n_deceptive = round(remaining * 0.6)
            n_opportunistic = remaining - n_deceptive

            if sweep_type == "ldt":
                label = f"LDT {focal_pct}%"
                compositions.append(
                    CompositionConfig(
                        label=label,
                        sweep_type="ldt",
                        n_ldt=max(n_focal, 0),
                        n_honest=0,
                        n_deceptive=max(n_deceptive, 0),
                        n_opportunistic=max(n_opportunistic, 0),
                    )
                )
            else:
                label = f"Honest {focal_pct}%"
                compositions.append(
                    CompositionConfig(
                        label=label,
                        sweep_type="honest",
                        n_ldt=0,
                        n_honest=max(n_focal, 0),
                        n_deceptive=max(n_deceptive, 0),
                        n_opportunistic=max(n_opportunistic, 0),
                    )
                )

    return compositions


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------
def create_agents_from_config(comp: CompositionConfig) -> List[BaseAgent]:
    """Create agent instances from a composition config."""
    agents: List[BaseAgent] = []
    for i in range(comp.n_ldt):
        agents.append(LDTAgent(agent_id=f"ldt_{i + 1}"))
    for i in range(comp.n_honest):
        agents.append(HonestAgent(agent_id=f"honest_{i + 1}"))
    for i in range(comp.n_deceptive):
        agents.append(DeceptiveAgent(agent_id=f"deceptive_{i + 1}"))
    for i in range(comp.n_opportunistic):
        agents.append(OpportunisticAgent(agent_id=f"opportunistic_{i + 1}"))
    return agents


# ---------------------------------------------------------------------------
# Per-class payoff extraction
# ---------------------------------------------------------------------------
def _extract_per_class_payoffs(
    orchestrator: Orchestrator,
) -> Dict[str, List[float]]:
    """Extract per-class payoffs using isinstance to distinguish LDT from honest."""
    payoffs: Dict[str, List[float]] = {
        "ldt": [],
        "honest": [],
        "deceptive": [],
        "opportunistic": [],
    }
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        if state is None:
            continue
        if isinstance(agent, LDTAgent):
            payoffs["ldt"].append(state.total_payoff)
        elif isinstance(agent, HonestAgent):
            payoffs["honest"].append(state.total_payoff)
        elif isinstance(agent, DeceptiveAgent):
            payoffs["deceptive"].append(state.total_payoff)
        elif isinstance(agent, OpportunisticAgent):
            payoffs["opportunistic"].append(state.total_payoff)
    return payoffs


def _mean_or_zero(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(
    comp: CompositionConfig,
    seed: int,
    n_epochs: int,
    steps_per_epoch: int,
) -> RunResult:
    """Run one simulation for a given composition and seed."""
    payoff_config = PayoffConfig(
        s_plus=2.0,
        s_minus=1.0,
        h=1.0,
        theta=0.5,
        rho_a=0.1,
        rho_b=0.1,
        w_rep=0.1,
    )
    governance_config = GovernanceConfig(
        circuit_breaker_enabled=True,
        reputation_decay_rate=0.05,
    )
    orch_config = OrchestratorConfig(
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        payoff_config=payoff_config,
        governance_config=governance_config,
    )

    orchestrator = Orchestrator(config=orch_config)
    agents = create_agents_from_config(comp)
    for agent in agents:
        orchestrator.register_agent(agent)

    epoch_metrics = orchestrator.run()

    # Aggregate epoch-level metrics
    welfares = [em.total_welfare for em in epoch_metrics]
    toxicities = [em.toxicity_rate for em in epoch_metrics]
    qgaps = [em.quality_gap for em in epoch_metrics]
    payoffs = [em.avg_payoff for em in epoch_metrics]

    # Extract per-class payoffs
    class_payoffs = _extract_per_class_payoffs(orchestrator)

    return RunResult(
        composition=comp.label,
        sweep_type=comp.sweep_type,
        focal_pct=comp.focal_pct,
        seed=seed,
        n_epochs=len(epoch_metrics),
        mean_welfare=float(np.mean(welfares)) if welfares else 0.0,
        total_welfare=float(np.sum(welfares)) if welfares else 0.0,
        mean_toxicity=float(np.mean(toxicities)) if toxicities else 0.0,
        mean_quality_gap=float(np.mean(qgaps)) if qgaps else 0.0,
        mean_avg_payoff=float(np.mean(payoffs)) if payoffs else 0.0,
        final_welfare=float(welfares[-1]) if welfares else 0.0,
        final_toxicity=float(toxicities[-1]) if toxicities else 0.0,
        ldt_avg_payoff=_mean_or_zero(class_payoffs["ldt"]),
        honest_avg_payoff=_mean_or_zero(class_payoffs["honest"]),
        deceptive_avg_payoff=_mean_or_zero(class_payoffs["deceptive"]),
        opportunistic_avg_payoff=_mean_or_zero(class_payoffs["opportunistic"]),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(results: List[RunResult]) -> List[AggResult]:
    """Group by composition and compute mean/std across seeds."""
    groups: Dict[str, List[RunResult]] = defaultdict(list)
    for r in results:
        groups[r.composition].append(r)

    aggs: List[AggResult] = []
    for label, runs in sorted(groups.items(), key=lambda x: (x[1][0].sweep_type, x[1][0].focal_pct)):
        welfares = [r.mean_welfare for r in runs]
        welfare_totals = [r.total_welfare for r in runs]
        toxicities = [r.mean_toxicity for r in runs]
        qgaps = [r.mean_quality_gap for r in runs]
        payoffs_all = [r.mean_avg_payoff for r in runs]
        ldt_pays = [r.ldt_avg_payoff for r in runs]
        honest_pays = [r.honest_avg_payoff for r in runs]
        deceptive_pays = [r.deceptive_avg_payoff for r in runs]
        opportunistic_pays = [r.opportunistic_avg_payoff for r in runs]

        aggs.append(
            AggResult(
                label=label,
                sweep_type=runs[0].sweep_type,
                focal_pct=runs[0].focal_pct,
                n_seeds=len(runs),
                welfare_mean=float(np.mean(welfares)),
                welfare_std=float(np.std(welfares)),
                welfare_total_mean=float(np.mean(welfare_totals)),
                toxicity_mean=float(np.mean(toxicities)),
                toxicity_std=float(np.std(toxicities)),
                quality_gap_mean=float(np.mean(qgaps)),
                avg_payoff_mean=float(np.mean(payoffs_all)),
                ldt_payoff_mean=float(np.mean(ldt_pays)),
                honest_payoff_mean=float(np.mean(honest_pays)),
                deceptive_payoff_mean=float(np.mean(deceptive_pays)),
                opportunistic_payoff_mean=float(np.mean(opportunistic_pays)),
            )
        )
    return aggs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
DPI = 160
COLORS = {
    "ldt": "#9C27B0",       # purple
    "honest": "#2196F3",    # blue
    "deceptive": "#F44336", # red
    "opportunistic": "#FF9800",  # orange
    "welfare": "#2196F3",
    "toxicity": "#F44336",
}


def _style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _split_by_sweep(aggs: List[AggResult]) -> Tuple[List[AggResult], List[AggResult]]:
    """Split aggregated results into LDT and honest sweeps."""
    ldt_aggs = sorted([a for a in aggs if a.sweep_type == "ldt"], key=lambda a: a.focal_pct)
    honest_aggs = sorted([a for a in aggs if a.sweep_type == "honest"], key=lambda a: a.focal_pct)
    return ldt_aggs, honest_aggs


# Plot 1: Dual-line welfare comparison
def plot_ldt_vs_honest_welfare(aggs: List[AggResult], out_dir: Path) -> Path:
    """Dual-line: welfare vs focal agent % (LDT purple, honest blue, with error bars)."""
    ldt_aggs, honest_aggs = _split_by_sweep(aggs)
    fig, ax = plt.subplots(figsize=(10, 6))

    # LDT sweep
    ldt_pcts = [a.focal_pct * 100 for a in ldt_aggs]
    ldt_welfares = [a.welfare_total_mean for a in ldt_aggs]
    ldt_stds = [a.welfare_std * a.n_seeds for a in ldt_aggs]
    ax.errorbar(
        ldt_pcts, ldt_welfares, yerr=ldt_stds,
        color=COLORS["ldt"], linewidth=2.5, marker="D", markersize=8,
        capsize=5, capthick=1.5, label="LDT agents", zorder=5,
    )

    # Honest sweep
    honest_pcts = [a.focal_pct * 100 for a in honest_aggs]
    honest_welfares = [a.welfare_total_mean for a in honest_aggs]
    honest_stds = [a.welfare_std * a.n_seeds for a in honest_aggs]
    ax.errorbar(
        honest_pcts, honest_welfares, yerr=honest_stds,
        color=COLORS["honest"], linewidth=2.5, marker="o", markersize=8,
        capsize=5, capthick=1.5, label="Honest agents", zorder=4,
    )

    _style_ax(
        ax,
        "LDT vs Honest: Total Welfare by Focal Agent Proportion",
        "Focal Agent Proportion (%)",
        "Total Welfare (sum over epochs)",
    )
    ax.legend(fontsize=10, loc="best")

    out = out_dir / "ldt_vs_honest_welfare.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# Plot 2: Dual-line toxicity comparison
def plot_ldt_vs_honest_toxicity(aggs: List[AggResult], out_dir: Path) -> Path:
    """Dual-line: toxicity vs focal agent % (LDT purple, honest blue, with error bars)."""
    ldt_aggs, honest_aggs = _split_by_sweep(aggs)
    fig, ax = plt.subplots(figsize=(10, 6))

    # LDT sweep
    ldt_pcts = [a.focal_pct * 100 for a in ldt_aggs]
    ldt_tox = [a.toxicity_mean for a in ldt_aggs]
    ldt_stds = [a.toxicity_std for a in ldt_aggs]
    ax.errorbar(
        ldt_pcts, ldt_tox, yerr=ldt_stds,
        color=COLORS["ldt"], linewidth=2.5, marker="D", markersize=8,
        capsize=5, capthick=1.5, label="LDT agents", zorder=5,
    )

    # Honest sweep
    honest_pcts = [a.focal_pct * 100 for a in honest_aggs]
    honest_tox = [a.toxicity_mean for a in honest_aggs]
    honest_stds = [a.toxicity_std for a in honest_aggs]
    ax.errorbar(
        honest_pcts, honest_tox, yerr=honest_stds,
        color=COLORS["honest"], linewidth=2.5, marker="o", markersize=8,
        capsize=5, capthick=1.5, label="Honest agents", zorder=4,
    )

    _style_ax(
        ax,
        "LDT vs Honest: Toxicity Rate by Focal Agent Proportion",
        "Focal Agent Proportion (%)",
        "Toxicity Rate (mean over epochs)",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc="best")

    out = out_dir / "ldt_vs_honest_toxicity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# Plot 3: Welfare-toxicity scatter
def plot_welfare_toxicity_scatter(aggs: List[AggResult], out_dir: Path) -> Path:
    """All compositions on welfare-vs-toxicity scatter (LDT diamonds, honest circles)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for a in aggs:
        pct = a.focal_pct * 100
        if a.sweep_type == "ldt":
            color = COLORS["ldt"]
            marker = "D"
            alpha = 0.85
        else:
            color = COLORS["honest"]
            marker = "o"
            alpha = 0.85

        ax.scatter(
            a.toxicity_mean, a.welfare_total_mean,
            c=color, s=120, marker=marker, alpha=alpha,
            edgecolors="white", linewidth=1, zorder=5,
        )
        ax.annotate(
            f"{pct:.0f}%",
            (a.toxicity_mean, a.welfare_total_mean),
            textcoords="offset points", xytext=(8, 5), fontsize=8,
        )

    # Legend entries
    ax.scatter([], [], c=COLORS["ldt"], s=100, marker="D", label="LDT sweep")
    ax.scatter([], [], c=COLORS["honest"], s=100, marker="o", label="Honest sweep")

    _style_ax(
        ax,
        "Welfare-Toxicity Trade-off: LDT vs Honest Compositions",
        "Toxicity Rate",
        "Total Welfare",
    )
    ax.legend(fontsize=10, loc="best")

    out = out_dir / "welfare_toxicity_scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# Plot 4: Payoff breakdown (grouped bars)
def plot_payoff_breakdown(aggs: List[AggResult], out_dir: Path) -> Path:
    """Grouped bars at key proportions showing per-class payoffs for each sweep."""
    key_pcts = [0.0, 0.2, 0.5, 0.8, 1.0]
    ldt_aggs, honest_aggs = _split_by_sweep(aggs)

    # Select key compositions
    ldt_selected = [a for a in ldt_aggs if round(a.focal_pct, 1) in key_pcts]
    honest_selected = [a for a in honest_aggs if round(a.focal_pct, 1) in key_pcts]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, selected, title_prefix in [
        (axes[0], ldt_selected, "LDT Sweep"),
        (axes[1], honest_selected, "Honest Sweep"),
    ]:
        if not selected:
            continue
        labels = [a.label for a in selected]
        x = np.arange(len(labels))
        width = 0.18

        # Per-class payoffs
        bars_data = [
            ("LDT", [a.ldt_payoff_mean for a in selected], COLORS["ldt"]),
            ("Honest", [a.honest_payoff_mean for a in selected], COLORS["honest"]),
            ("Deceptive", [a.deceptive_payoff_mean for a in selected], COLORS["deceptive"]),
            ("Opportunistic", [a.opportunistic_payoff_mean for a in selected], COLORS["opportunistic"]),
        ]

        for idx, (name, vals, color) in enumerate(bars_data):
            offset = (idx - 1.5) * width
            bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)
            for bar, val in zip(bars, vals, strict=True):
                if val != 0.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2,
                        f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_title(f"{title_prefix}: Per-Class Payoffs", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Total Payoff" if ax == axes[0] else "", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Per-Class Payoff Breakdown at Key Compositions",
        fontsize=14, fontweight="bold", y=1.02,
    )

    out = out_dir / "payoff_breakdown.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# Plot 5: Pareto frontier
def plot_pareto_frontier(aggs: List[AggResult], out_dir: Path) -> Path:
    """Pareto frontier across both sweeps (shows if LDT shifts the frontier)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    welfares = [a.welfare_total_mean for a in aggs]
    toxicities = [a.toxicity_mean for a in aggs]

    # Find Pareto-optimal points (maximize welfare, minimize toxicity)
    pareto_mask: List[bool] = []
    for i, (w, t) in enumerate(zip(welfares, toxicities, strict=True)):
        dominated = False
        for j, (w2, t2) in enumerate(zip(welfares, toxicities, strict=True)):
            if i != j and w2 >= w and t2 <= t and (w2 > w or t2 < t):
                dominated = True
                break
        pareto_mask.append(not dominated)

    # Plot all points
    for i, a in enumerate(aggs):
        if a.sweep_type == "ldt":
            base_color = COLORS["ldt"]
            marker = "D"
        else:
            base_color = COLORS["honest"]
            marker = "o"

        edge_color = "#FFD700" if pareto_mask[i] else "white"
        edge_width = 2.5 if pareto_mask[i] else 1
        size = 140 if pareto_mask[i] else 80

        ax.scatter(
            a.toxicity_mean, a.welfare_total_mean,
            c=base_color, s=size, marker=marker,
            edgecolors=edge_color, linewidth=edge_width, zorder=5,
        )
        ax.annotate(
            f"{a.focal_pct * 100:.0f}%",
            (a.toxicity_mean, a.welfare_total_mean),
            textcoords="offset points", xytext=(8, 5), fontsize=8,
        )

    # Connect Pareto front
    pareto_points = [
        (toxicities[i], welfares[i]) for i in range(len(aggs)) if pareto_mask[i]
    ]
    pareto_points.sort(key=lambda p: p[0])
    if pareto_points:
        pt, pw = zip(*pareto_points, strict=True)
        ax.plot(pt, pw, color="#4CAF50", linewidth=2, linestyle="--", alpha=0.7, label="Pareto frontier")

    # Legend entries
    ax.scatter([], [], c=COLORS["ldt"], s=100, marker="D", label="LDT sweep")
    ax.scatter([], [], c=COLORS["honest"], s=100, marker="o", label="Honest sweep")
    ax.scatter([], [], c="white", s=80, edgecolors="#FFD700", linewidth=2.5, label="Pareto-optimal")

    _style_ax(
        ax,
        "Pareto Frontier: Welfare vs Toxicity\n(gold border = non-dominated compositions)",
        "Toxicity Rate",
        "Total Welfare",
    )
    ax.legend(fontsize=10, loc="best")

    out = out_dir / "pareto_frontier.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def write_results_csv(results: List[RunResult], path: Path) -> None:
    """Write raw per-run results to CSV."""
    fieldnames = [
        "composition", "sweep_type", "focal_pct", "seed", "n_epochs",
        "mean_welfare", "total_welfare", "mean_toxicity",
        "mean_quality_gap", "mean_avg_payoff",
        "final_welfare", "final_toxicity",
        "ldt_avg_payoff", "honest_avg_payoff",
        "deceptive_avg_payoff", "opportunistic_avg_payoff",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "composition": r.composition,
                "sweep_type": r.sweep_type,
                "focal_pct": f"{r.focal_pct:.2f}",
                "seed": r.seed,
                "n_epochs": r.n_epochs,
                "mean_welfare": f"{r.mean_welfare:.6f}",
                "total_welfare": f"{r.total_welfare:.6f}",
                "mean_toxicity": f"{r.mean_toxicity:.6f}",
                "mean_quality_gap": f"{r.mean_quality_gap:.6f}",
                "mean_avg_payoff": f"{r.mean_avg_payoff:.6f}",
                "final_welfare": f"{r.final_welfare:.6f}",
                "final_toxicity": f"{r.final_toxicity:.6f}",
                "ldt_avg_payoff": f"{r.ldt_avg_payoff:.6f}",
                "honest_avg_payoff": f"{r.honest_avg_payoff:.6f}",
                "deceptive_avg_payoff": f"{r.deceptive_avg_payoff:.6f}",
                "opportunistic_avg_payoff": f"{r.opportunistic_avg_payoff:.6f}",
            })


def write_aggregated_csv(aggs: List[AggResult], path: Path) -> None:
    """Write aggregated results to CSV."""
    fieldnames = [
        "composition", "sweep_type", "focal_pct", "n_seeds",
        "welfare_total_mean", "welfare_std",
        "toxicity_mean", "toxicity_std",
        "quality_gap_mean", "avg_payoff_mean",
        "ldt_payoff_mean", "honest_payoff_mean",
        "deceptive_payoff_mean", "opportunistic_payoff_mean",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in aggs:
            writer.writerow({
                "composition": a.label,
                "sweep_type": a.sweep_type,
                "focal_pct": f"{a.focal_pct:.2f}",
                "n_seeds": a.n_seeds,
                "welfare_total_mean": f"{a.welfare_total_mean:.4f}",
                "welfare_std": f"{a.welfare_std:.4f}",
                "toxicity_mean": f"{a.toxicity_mean:.4f}",
                "toxicity_std": f"{a.toxicity_std:.4f}",
                "quality_gap_mean": f"{a.quality_gap_mean:.4f}",
                "avg_payoff_mean": f"{a.avg_payoff_mean:.4f}",
                "ldt_payoff_mean": f"{a.ldt_payoff_mean:.4f}",
                "honest_payoff_mean": f"{a.honest_payoff_mean:.4f}",
                "deceptive_payoff_mean": f"{a.deceptive_payoff_mean:.4f}",
                "opportunistic_payoff_mean": f"{a.opportunistic_payoff_mean:.4f}",
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="LDT composition study: compare LDT vs honest agent welfare/toxicity"
    )
    parser.add_argument("--total-agents", type=int, default=10, help="Total agents per run")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per run")
    parser.add_argument("--steps", type=int, default=10, help="Steps per epoch")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or Path(f"runs/{timestamp}_ldt_composition_study")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    compositions = build_compositions(args.total_agents)
    seeds = list(range(42, 42 + args.seeds))

    print("=" * 70)
    print("LDT Composition Study")
    print(f"  Agents: {args.total_agents}, Epochs: {args.epochs}, Steps/epoch: {args.steps}")
    print(f"  Seeds: {seeds}")
    print(f"  Compositions: {len(compositions)} (11 LDT + 11 honest)")
    print(f"  Total runs: {len(compositions) * len(seeds)}")
    print("=" * 70)

    # Run all combinations
    all_results: List[RunResult] = []
    total_runs = len(compositions) * len(seeds)
    run_idx = 0

    for comp in compositions:
        for seed in seeds:
            run_idx += 1
            print(
                f"  [{run_idx}/{total_runs}] {comp.label} "
                f"(L={comp.n_ldt}, H={comp.n_honest}, D={comp.n_deceptive}, "
                f"O={comp.n_opportunistic}) seed={seed}...",
                end=" ", flush=True,
            )
            result = run_single(comp, seed, args.epochs, args.steps)
            all_results.append(result)
            print(
                f"welfare={result.total_welfare:.1f}, "
                f"toxicity={result.mean_toxicity:.3f}"
            )

    # Aggregate
    aggs = aggregate_results(all_results)

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(
        f"{'Composition':<16} {'Sweep':<8} {'Focal%':>6} {'TotalWelfare':>13} "
        f"{'WelfStd':>8} {'Toxicity':>9} {'ToxStd':>7} "
        f"{'LDT_Pay':>8} {'Hon_Pay':>8} {'Dec_Pay':>8} {'Opp_Pay':>8}"
    )
    print("-" * 120)
    for a in aggs:
        print(
            f"{a.label:<16} {a.sweep_type:<8} {a.focal_pct*100:>5.0f}% "
            f"{a.welfare_total_mean:>13.2f} {a.welfare_std:>8.2f} "
            f"{a.toxicity_mean:>9.3f} {a.toxicity_std:>7.3f} "
            f"{a.ldt_payoff_mean:>8.2f} {a.honest_payoff_mean:>8.2f} "
            f"{a.deceptive_payoff_mean:>8.2f} {a.opportunistic_payoff_mean:>8.2f}"
        )

    # Key comparisons
    ldt_aggs = [a for a in aggs if a.sweep_type == "ldt"]
    honest_aggs = [a for a in aggs if a.sweep_type == "honest"]

    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)

    # Compare peak welfare across sweeps
    if ldt_aggs:
        ldt_peak = max(ldt_aggs, key=lambda a: a.welfare_total_mean)
        print(f"  LDT peak welfare: {ldt_peak.label} "
              f"(welfare={ldt_peak.welfare_total_mean:.2f}, toxicity={ldt_peak.toxicity_mean:.3f})")
    if honest_aggs:
        honest_peak = max(honest_aggs, key=lambda a: a.welfare_total_mean)
        print(f"  Honest peak welfare: {honest_peak.label} "
              f"(welfare={honest_peak.welfare_total_mean:.2f}, toxicity={honest_peak.toxicity_mean:.3f})")

    # Compare at matching percentages
    for pct in [0.2, 0.5, 1.0]:
        ldt_match = next((a for a in ldt_aggs if round(a.focal_pct, 1) == pct), None)
        hon_match = next((a for a in honest_aggs if round(a.focal_pct, 1) == pct), None)
        if ldt_match and hon_match:
            w_diff = ldt_match.welfare_total_mean - hon_match.welfare_total_mean
            t_diff = ldt_match.toxicity_mean - hon_match.toxicity_mean
            print(f"  At {pct*100:.0f}%: LDT welfare={ldt_match.welfare_total_mean:.2f} vs "
                  f"Honest welfare={hon_match.welfare_total_mean:.2f} "
                  f"(diff={w_diff:+.2f}), "
                  f"toxicity diff={t_diff:+.3f}")

    # Verify: 0% composition should be identical across sweeps
    ldt_0 = next((a for a in ldt_aggs if round(a.focal_pct, 1) == 0.0), None)
    hon_0 = next((a for a in honest_aggs if round(a.focal_pct, 1) == 0.0), None)
    if ldt_0 and hon_0:
        w_match = abs(ldt_0.welfare_total_mean - hon_0.welfare_total_mean) < 0.01
        t_match = abs(ldt_0.toxicity_mean - hon_0.toxicity_mean) < 0.001
        status = "PASS" if (w_match and t_match) else "MISMATCH"
        print(f"\n  Sanity check (0% focal should match): {status}")
        if not (w_match and t_match):
            print(f"    LDT 0%: welfare={ldt_0.welfare_total_mean:.4f}, toxicity={ldt_0.toxicity_mean:.4f}")
            print(f"    Honest 0%: welfare={hon_0.welfare_total_mean:.4f}, toxicity={hon_0.toxicity_mean:.4f}")

    # Generate plots
    print("\nGenerating plots...")
    plots_written: List[Path] = []
    plots_written.append(plot_ldt_vs_honest_welfare(aggs, plots_dir))
    plots_written.append(plot_ldt_vs_honest_toxicity(aggs, plots_dir))
    plots_written.append(plot_welfare_toxicity_scatter(aggs, plots_dir))
    plots_written.append(plot_payoff_breakdown(aggs, plots_dir))
    plots_written.append(plot_pareto_frontier(aggs, plots_dir))

    for p in plots_written:
        print(f"  -> {p}")

    # Export CSVs
    csv_path = out_dir / "results.csv"
    write_results_csv(all_results, csv_path)
    print(f"  -> {csv_path}")

    agg_csv_path = out_dir / "aggregated_results.csv"
    write_aggregated_csv(aggs, agg_csv_path)
    print(f"  -> {agg_csv_path}")

    print(f"\nAll outputs in: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
