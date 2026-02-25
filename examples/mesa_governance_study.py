#!/usr/bin/env python
"""Mesa Bridge Governance Study.

Research question: How does externality internalization (rho) interact with
adaptive acceptance thresholds to shape welfare and toxicity in a
heterogeneous agent population?

Uses the Mesa bridge in protocol mode (no Mesa dependency) to simulate an
agent-based model where agents evolve task_progress and rework_count over
time based on their behavioral archetype.  The bridge converts each step's
agent states into SoftInteractions, enabling governance analysis on the
emergent dynamics.

Agent archetypes:
  - cooperative:   high task_progress, low rework, steady engagement
  - selfish:       moderate task_progress, moderate rework, decaying engagement
  - exploitative:  low task_progress, high rework, volatile engagement

Key mechanism: When rho > 0, agents pay externality costs proportional to
harm.  Under the adaptive regime, agents raise their acceptance threshold
in response — rejecting borderline interactions.  This creates a feedback
loop where governance pressure improves selection quality.

Swept parameters:
  - rho_a: initiator externality internalization (0.0 to 1.0)
  - adaptive: whether acceptance threshold responds to rho (True/False)
Fixed seeds: 5 seeds per configuration for statistical power.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from swarm.bridges.mesa import MesaBridge, MesaBridgeConfig
from swarm.core.payoff import PayoffConfig
from swarm.metrics.reporters import MetricsReporter
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Agent archetypes (simulated ABM population)
# ---------------------------------------------------------------------------


@dataclass
class SimAgent:
    """Simulated Mesa-style agent with behavioral archetype."""

    agent_id: str
    archetype: str  # cooperative, selfish, exploitative
    task_progress: float = 1.0
    rework_count: int = 0
    engagement: float = 0.5

    # Internal state for dynamics
    _rng: random.Random = field(default_factory=lambda: random.Random())

    def step(self) -> None:
        """Advance agent state by one tick (ABM dynamics)."""
        if self.archetype == "cooperative":
            self.task_progress = min(1.0, max(0.0, self.task_progress + self._rng.gauss(0.02, 0.05)))
            self.rework_count = max(0, self.rework_count + (1 if self._rng.random() < 0.05 else 0))
            self.engagement = min(1.0, max(0.0, self.engagement + self._rng.gauss(0.01, 0.03)))
        elif self.archetype == "selfish":
            self.task_progress = min(1.0, max(0.0, self.task_progress + self._rng.gauss(-0.01, 0.08)))
            self.rework_count = max(0, self.rework_count + (1 if self._rng.random() < 0.15 else 0))
            self.engagement = min(1.0, max(0.0, self.engagement + self._rng.gauss(-0.02, 0.05)))
        elif self.archetype == "exploitative":
            self.task_progress = min(1.0, max(0.0, self.task_progress + self._rng.gauss(-0.03, 0.10)))
            self.rework_count = max(0, self.rework_count + (1 if self._rng.random() < 0.30 else 0))
            self.engagement = min(1.0, max(0.0, self.engagement + self._rng.gauss(-0.01, 0.08)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "task_progress": self.task_progress,
            "rework_count": self.rework_count,
            "engagement": self.engagement,
        }


def build_population(
    n_cooperative: int,
    n_selfish: int,
    n_exploitative: int,
    seed: int,
) -> List[SimAgent]:
    """Build a heterogeneous agent population."""
    rng = random.Random(seed)
    agents: List[SimAgent] = []
    idx = 0
    for archetype, count, (tp_mu, eng_mu) in [
        ("cooperative", n_cooperative, (0.85, 0.7)),
        ("selfish", n_selfish, (0.55, 0.45)),
        ("exploitative", n_exploitative, (0.25, 0.3)),
    ]:
        for _ in range(count):
            a = SimAgent(
                agent_id=f"{archetype[:4]}-{idx}",
                archetype=archetype,
                task_progress=max(0.0, min(1.0, rng.gauss(tp_mu, 0.08))),
                rework_count=max(0, int(rng.gauss(1 if archetype == "exploitative" else 0, 0.5))),
                engagement=max(0.0, min(1.0, rng.gauss(eng_mu, 0.1))),
            )
            a._rng = random.Random(seed + idx)
            agents.append(a)
            idx += 1
    return agents


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Aggregate results from a single simulation run."""

    rho_a: float
    adaptive: bool
    seed: int
    n_steps: int
    toxicity: float
    quality_gap: float
    avg_quality: float
    total_welfare: float
    avg_initiator_payoff: float
    avg_counterparty_payoff: float
    acceptance_rate: float
    n_interactions: int
    # Per-archetype mean p
    mean_p_cooperative: float
    mean_p_selfish: float
    mean_p_exploitative: float


def run_single(
    rho_a: float,
    seed: int,
    adaptive: bool = False,
    n_steps: int = 50,
    n_cooperative: int = 15,
    n_selfish: int = 10,
    n_exploitative: int = 5,
) -> RunResult:
    """Run one Mesa bridge simulation with given governance parameters.

    When adaptive=True, the acceptance threshold rises with rho_a:
        threshold = 0.5 + 0.3 * rho_a
    This models agents becoming pickier when externality costs bite.
    """
    payoff_cfg = PayoffConfig(
        s_plus=2.0,
        s_minus=1.0,
        h=2.0,
        theta=0.5,
        rho_a=rho_a,
        rho_b=0.0,
        w_rep=1.0,
    )
    bridge_cfg = MesaBridgeConfig(
        model_id=f"mesa-rho{rho_a:.2f}-s{seed}",
        enable_event_log=False,
    )
    bridge = MesaBridge(config=bridge_cfg, payoff_config=payoff_cfg)

    # Adaptive threshold: agents get pickier under externality pressure
    accept_threshold = 0.5 + 0.3 * rho_a if adaptive else 0.5

    agents = build_population(n_cooperative, n_selfish, n_exploitative, seed)

    # Track per-archetype interactions
    archetype_interactions: Dict[str, List[SoftInteraction]] = {
        "cooperative": [],
        "selfish": [],
        "exploitative": [],
    }
    archetype_map = {a.agent_id: a.archetype for a in agents}

    all_interactions: List[SoftInteraction] = []

    for _step in range(n_steps):
        # Advance ABM dynamics
        for a in agents:
            a.step()

        # Record through bridge
        step_interactions = bridge.record_agent_states([a.to_dict() for a in agents])

        # Apply adaptive acceptance threshold (override bridge default)
        for ix in step_interactions:
            ix.accepted = ix.p >= accept_threshold
            # Recompute payoff with updated acceptance
            bridge._payoff_engine.payoff_initiator(ix)

            all_interactions.append(ix)
            arch = archetype_map.get(ix.counterparty, "unknown")
            if arch in archetype_interactions:
                archetype_interactions[arch].append(ix)

    # Compute summary metrics
    reporter = MetricsReporter(payoff_config=payoff_cfg)
    summary = reporter.summary(all_interactions)

    def mean_p(interactions: List[SoftInteraction]) -> float:
        if not interactions:
            return 0.0
        return statistics.mean(ix.p for ix in interactions)

    return RunResult(
        rho_a=rho_a,
        adaptive=adaptive,
        seed=seed,
        n_steps=n_steps,
        toxicity=summary.toxicity_soft,
        quality_gap=summary.quality_gap,
        avg_quality=summary.average_quality,
        total_welfare=summary.total_welfare,
        avg_initiator_payoff=summary.avg_initiator_payoff,
        avg_counterparty_payoff=summary.avg_counterparty_payoff,
        acceptance_rate=summary.acceptance_rate,
        n_interactions=summary.total_interactions,
        mean_p_cooperative=mean_p(archetype_interactions["cooperative"]),
        mean_p_selfish=mean_p(archetype_interactions["selfish"]),
        mean_p_exploitative=mean_p(archetype_interactions["exploitative"]),
    )


# ---------------------------------------------------------------------------
# Sweep + export
# ---------------------------------------------------------------------------


def run_sweep(
    rho_values: List[float] | None = None,
    seeds: List[int] | None = None,
    n_steps: int = 50,
) -> List[RunResult]:
    """Sweep rho_a × adaptive across values, multiple seeds each."""
    if rho_values is None:
        rho_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if seeds is None:
        seeds = [42, 123, 256, 789, 1024]

    regimes = [("static", False), ("adaptive", True)]
    results: List[RunResult] = []
    total = len(rho_values) * len(seeds) * len(regimes)
    done = 0

    for regime_name, adaptive in regimes:
        print(f"\n--- Regime: {regime_name} threshold ---")
        for rho in rho_values:
            for seed in seeds:
                result = run_single(rho_a=rho, seed=seed, adaptive=adaptive, n_steps=n_steps)
                results.append(result)
                done += 1
                print(f"  [{done}/{total}] {regime_name}  rho_a={rho:.1f}  seed={seed}  "
                      f"tox={result.toxicity:.3f}  welfare={result.total_welfare:.1f}  "
                      f"qgap={result.quality_gap:+.3f}  accept={result.acceptance_rate:.2f}")

    return results


def export_csv(results: List[RunResult], path: Path) -> None:
    """Write results to CSV."""
    fieldnames = [
        "rho_a", "adaptive", "seed", "n_steps", "toxicity", "quality_gap", "avg_quality",
        "total_welfare", "avg_initiator_payoff", "avg_counterparty_payoff",
        "acceptance_rate", "n_interactions",
        "mean_p_cooperative", "mean_p_selfish", "mean_p_exploitative",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})
    print(f"  Exported {len(results)} rows to {path}")


def compute_summary(results: List[RunResult]) -> Dict[str, Any]:
    """Compute per-regime, per-rho aggregate statistics."""
    by_key: Dict[Tuple[str, float], List[RunResult]] = defaultdict(list)
    for r in results:
        regime = "adaptive" if r.adaptive else "static"
        by_key[(regime, r.rho_a)].append(r)

    summary = {}
    for (regime, rho) in sorted(by_key):
        runs = by_key[(regime, rho)]
        n = len(runs)
        tox = [r.toxicity for r in runs]
        wel = [r.total_welfare for r in runs]
        qg = [r.quality_gap for r in runs]
        ar = [r.acceptance_rate for r in runs]

        summary[f"{regime}_rho_{rho:.1f}"] = {
            "regime": regime,
            "rho_a": rho,
            "n_runs": n,
            "toxicity_mean": statistics.mean(tox),
            "toxicity_std": statistics.stdev(tox) if n > 1 else 0.0,
            "welfare_mean": statistics.mean(wel),
            "welfare_std": statistics.stdev(wel) if n > 1 else 0.0,
            "quality_gap_mean": statistics.mean(qg),
            "quality_gap_std": statistics.stdev(qg) if n > 1 else 0.0,
            "acceptance_rate_mean": statistics.mean(ar),
            "mean_p_cooperative": statistics.mean(r.mean_p_cooperative for r in runs),
            "mean_p_selfish": statistics.mean(r.mean_p_selfish for r in runs),
            "mean_p_exploitative": statistics.mean(r.mean_p_exploitative for r in runs),
        }

    return summary


def print_summary_table(summary: Dict[str, Any]) -> None:
    """Pretty-print the summary as an ASCII table."""
    for regime in ["static", "adaptive"]:
        entries = {k: v for k, v in summary.items() if v["regime"] == regime}
        if not entries:
            continue
        print(f"\n{'=' * 105}")
        print(f"  Regime: {regime.upper()} threshold")
        print(f"{'=' * 105}")
        print(f"{'rho_a':>6}  {'toxicity':>16}  {'welfare':>16}  {'quality_gap':>16}  "
              f"{'accept':>6}  {'p_coop':>7}  {'p_self':>7}  {'p_expl':>7}")
        print("-" * 105)
        for key in sorted(entries):
            s = entries[key]
            print(f"{s['rho_a']:>6.1f}  "
                  f"{s['toxicity_mean']:>7.4f}±{s['toxicity_std']:<6.4f}  "
                  f"{s['welfare_mean']:>7.1f}±{s['welfare_std']:<6.1f}  "
                  f"{s['quality_gap_mean']:>+7.4f}±{s['quality_gap_std']:<6.4f}  "
                  f"{s['acceptance_rate_mean']:>6.2f}  "
                  f"{s['mean_p_cooperative']:>7.3f}  "
                  f"{s['mean_p_selfish']:>7.3f}  "
                  f"{s['mean_p_exploitative']:>7.3f}")
        print(f"{'=' * 105}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(f"runs/{timestamp}_mesa_governance_study")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Mesa Governance Study")
    print(f"Run directory: {run_dir}")
    print("Sweeping rho_a from 0.0 to 1.0 (11 values x 5 seeds x 2 regimes = 110 runs)")
    print("Population: 15 cooperative + 10 selfish + 5 exploitative = 30 agents")
    print("Steps per run: 50")
    print("Regimes: static (fixed threshold=0.5) vs adaptive (threshold=0.5+0.3*rho)")
    print()

    results = run_sweep()

    # Export
    csv_path = run_dir / "sweep_results.csv"
    export_csv(results, csv_path)

    summary = compute_summary(results)
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Exported summary to {summary_path}")

    print_summary_table(summary)

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    s_00 = summary.get("static_rho_0.0", {})
    s_10 = summary.get("static_rho_1.0", {})
    a_00 = summary.get("adaptive_rho_0.0", {})
    a_10 = summary.get("adaptive_rho_1.0", {})

    if s_00 and s_10:
        print("\n  Static regime (rho=0.0 -> 1.0):")
        print(f"    Toxicity:     {s_00['toxicity_mean']:.4f} → {s_10['toxicity_mean']:.4f}  "
              f"(Δ = {s_10['toxicity_mean'] - s_00['toxicity_mean']:+.4f})")
        print(f"    Welfare:      {s_00['welfare_mean']:.1f} → {s_10['welfare_mean']:.1f}  "
              f"(Δ = {s_10['welfare_mean'] - s_00['welfare_mean']:+.1f})")
        print(f"    Quality gap:  {s_00['quality_gap_mean']:+.4f} → {s_10['quality_gap_mean']:+.4f}")

    if a_00 and a_10:
        print("\n  Adaptive regime (rho=0.0 -> 1.0):")
        print(f"    Toxicity:     {a_00['toxicity_mean']:.4f} → {a_10['toxicity_mean']:.4f}  "
              f"(Δ = {a_10['toxicity_mean'] - a_00['toxicity_mean']:+.4f})")
        print(f"    Welfare:      {a_00['welfare_mean']:.1f} → {a_10['welfare_mean']:.1f}  "
              f"(Δ = {a_10['welfare_mean'] - a_00['welfare_mean']:+.1f})")
        print(f"    Quality gap:  {a_00['quality_gap_mean']:+.4f} → {a_10['quality_gap_mean']:+.4f}")
        print(f"    Accept rate:  {a_00['acceptance_rate_mean']:.2f} → {a_10['acceptance_rate_mean']:.2f}")

    if s_10 and a_10:
        print("\n  Adaptive vs Static at rho=1.0:")
        print(f"    Toxicity improvement: {s_10['toxicity_mean'] - a_10['toxicity_mean']:+.4f}")
        print(f"    Welfare cost:         {a_10['welfare_mean'] - s_10['welfare_mean']:+.1f}")

    print(f"\nDone. Results in {run_dir}/")


if __name__ == "__main__":
    main()
