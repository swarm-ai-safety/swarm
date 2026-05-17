#!/usr/bin/env python
"""Mesa Bridge Adaptive Agents Study.

Research question: Can agents that learn from rejection (improving their
task_progress when rejected) overcome the welfare collapse at high rho
that we observed in the static-agent study?

Extends the mesa_governance_study with a learning mechanism:
  - When an agent's interaction is REJECTED, it observes the signal and
    adjusts its task_progress upward by a learning rate (lr).
  - When ACCEPTED, no change (already good enough).
  - Different archetypes have different learning capacities:
      cooperative:   lr = 0.03  (already good, small improvement room)
      selfish:       lr = 0.05  (moderate learning, key swing group)
      exploitative:  lr = 0.02  (low capacity, hard to reform)

The hypothesis: learning agents should partially recover welfare at high
rho because rejected agents improve rather than just being excluded.
This would shift the governance sweet spot rightward (higher rho becomes
viable).

Swept parameters:
  - rho_a: externality internalization (0.0 to 1.0)
  - regime: static | adaptive | adaptive+learning
Fixed: 5 seeds, 50 steps, 30 agents (15/10/5)
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
# Learning rates by archetype
# ---------------------------------------------------------------------------

LEARNING_RATES = {
    "cooperative": 0.03,
    "selfish": 0.05,
    "exploitative": 0.02,
}

# ---------------------------------------------------------------------------
# Agent archetypes with optional learning
# ---------------------------------------------------------------------------


@dataclass
class SimAgent:
    """Simulated Mesa-style agent with behavioral archetype and learning."""

    agent_id: str
    archetype: str
    task_progress: float = 1.0
    rework_count: int = 0
    engagement: float = 0.5

    # Learning state
    rejection_count: int = 0
    acceptance_count: int = 0

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

    def learn_from_rejection(self) -> None:
        """Improve task_progress after being rejected."""
        lr = LEARNING_RATES.get(self.archetype, 0.0)
        # Diminishing returns: learning slows as task_progress approaches 1.0
        headroom = 1.0 - self.task_progress
        improvement = lr * headroom
        self.task_progress = min(1.0, self.task_progress + improvement)
        # Also reduce rework tendency slightly
        if self.rework_count > 0 and self._rng.random() < 0.3:
            self.rework_count -= 1
        self.rejection_count += 1

    def observe_acceptance(self) -> None:
        """Record acceptance (no behavioral change needed)."""
        self.acceptance_count += 1

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
    regime: str  # "static", "adaptive", "adaptive_learning"
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
    mean_p_cooperative: float
    mean_p_selfish: float
    mean_p_exploitative: float
    # Learning-specific metrics
    avg_rejections_selfish: float
    avg_final_tp_selfish: float
    avg_final_tp_exploitative: float


def run_single(
    rho_a: float,
    seed: int,
    regime: str = "static",
    n_steps: int = 50,
    n_cooperative: int = 15,
    n_selfish: int = 10,
    n_exploitative: int = 5,
) -> RunResult:
    """Run one Mesa bridge simulation.

    Regimes:
      - static:            fixed threshold = 0.5, no learning
      - adaptive:          threshold = 0.5 + 0.3*rho, no learning
      - adaptive_learning: threshold = 0.5 + 0.3*rho, agents learn from rejection
    """
    payoff_cfg = PayoffConfig(
        s_plus=2.0, s_minus=1.0, h=2.0, theta=0.5,
        rho_a=rho_a, rho_b=0.0, w_rep=1.0,
    )
    bridge_cfg = MesaBridgeConfig(
        model_id=f"mesa-{regime}-rho{rho_a:.2f}-s{seed}",
        enable_event_log=False,
    )
    bridge = MesaBridge(config=bridge_cfg, payoff_config=payoff_cfg)

    use_adaptive = regime in ("adaptive", "adaptive_learning")
    use_learning = regime == "adaptive_learning"
    accept_threshold = 0.5 + 0.3 * rho_a if use_adaptive else 0.5

    agents = build_population(n_cooperative, n_selfish, n_exploitative, seed)
    archetype_map = {a.agent_id: a.archetype for a in agents}
    agent_map = {a.agent_id: a for a in agents}

    archetype_interactions: Dict[str, List[SoftInteraction]] = {
        "cooperative": [], "selfish": [], "exploitative": [],
    }
    all_interactions: List[SoftInteraction] = []

    for _step in range(n_steps):
        for a in agents:
            a.step()

        step_interactions = bridge.record_agent_states([a.to_dict() for a in agents])

        for ix in step_interactions:
            ix.accepted = ix.p >= accept_threshold
            bridge._payoff_engine.payoff_initiator(ix)

            # Learning feedback
            if use_learning:
                agent = agent_map.get(ix.counterparty)
                if agent is not None:
                    if ix.accepted:
                        agent.observe_acceptance()
                    else:
                        agent.learn_from_rejection()

            all_interactions.append(ix)
            arch = archetype_map.get(ix.counterparty, "unknown")
            if arch in archetype_interactions:
                archetype_interactions[arch].append(ix)

    reporter = MetricsReporter(payoff_config=payoff_cfg)
    summary = reporter.summary(all_interactions)

    def mean_p(interactions: List[SoftInteraction]) -> float:
        if not interactions:
            return 0.0
        return statistics.mean(ix.p for ix in interactions)

    # Learning-specific: track selfish/exploitative final state
    selfish_agents = [a for a in agents if a.archetype == "selfish"]
    exploit_agents = [a for a in agents if a.archetype == "exploitative"]

    return RunResult(
        rho_a=rho_a,
        regime=regime,
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
        avg_rejections_selfish=statistics.mean(a.rejection_count for a in selfish_agents),
        avg_final_tp_selfish=statistics.mean(a.task_progress for a in selfish_agents),
        avg_final_tp_exploitative=statistics.mean(a.task_progress for a in exploit_agents),
    )


# ---------------------------------------------------------------------------
# Sweep + export
# ---------------------------------------------------------------------------


def run_sweep(
    rho_values: List[float] | None = None,
    seeds: List[int] | None = None,
    n_steps: int = 50,
) -> List[RunResult]:
    """Sweep rho_a x regime across values, multiple seeds each."""
    if rho_values is None:
        rho_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if seeds is None:
        seeds = [42, 123, 256, 789, 1024]

    regimes = ["static", "adaptive", "adaptive_learning"]
    results: List[RunResult] = []
    total = len(rho_values) * len(seeds) * len(regimes)
    done = 0

    for regime in regimes:
        print(f"\n--- Regime: {regime} ---")
        for rho in rho_values:
            for seed in seeds:
                result = run_single(rho_a=rho, seed=seed, regime=regime, n_steps=n_steps)
                results.append(result)
                done += 1
                print(f"  [{done}/{total}] {regime:<18s}  rho={rho:.1f}  seed={seed}  "
                      f"tox={result.toxicity:.3f}  welfare={result.total_welfare:.1f}  "
                      f"accept={result.acceptance_rate:.2f}  "
                      f"tp_self={result.avg_final_tp_selfish:.3f}  "
                      f"tp_expl={result.avg_final_tp_exploitative:.3f}")

    return results


def export_csv(results: List[RunResult], path: Path) -> None:
    """Write results to CSV."""
    fieldnames = [
        "rho_a", "regime", "seed", "n_steps", "toxicity", "quality_gap", "avg_quality",
        "total_welfare", "avg_initiator_payoff", "avg_counterparty_payoff",
        "acceptance_rate", "n_interactions",
        "mean_p_cooperative", "mean_p_selfish", "mean_p_exploitative",
        "avg_rejections_selfish", "avg_final_tp_selfish", "avg_final_tp_exploitative",
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
        by_key[(r.regime, r.rho_a)].append(r)

    summary: Dict[str, Any] = {}
    for (regime, rho) in sorted(by_key):
        runs = by_key[(regime, rho)]
        n = len(runs)

        def _mean(vals: List[float]) -> float:
            return statistics.mean(vals)

        def _std(vals: List[float], count: int = n) -> float:
            return statistics.stdev(vals) if count > 1 else 0.0

        tox = [r.toxicity for r in runs]
        wel = [r.total_welfare for r in runs]
        qg = [r.quality_gap for r in runs]
        ar = [r.acceptance_rate for r in runs]
        tp_s = [r.avg_final_tp_selfish for r in runs]
        tp_e = [r.avg_final_tp_exploitative for r in runs]

        summary[f"{regime}_rho_{rho:.1f}"] = {
            "regime": regime,
            "rho_a": rho,
            "n_runs": n,
            "toxicity_mean": _mean(tox), "toxicity_std": _std(tox),
            "welfare_mean": _mean(wel), "welfare_std": _std(wel),
            "quality_gap_mean": _mean(qg), "quality_gap_std": _std(qg),
            "acceptance_rate_mean": _mean(ar),
            "mean_p_cooperative": _mean([r.mean_p_cooperative for r in runs]),
            "mean_p_selfish": _mean([r.mean_p_selfish for r in runs]),
            "mean_p_exploitative": _mean([r.mean_p_exploitative for r in runs]),
            "avg_final_tp_selfish": _mean(tp_s),
            "avg_final_tp_exploitative": _mean(tp_e),
        }

    return summary


def print_summary_table(summary: Dict[str, Any]) -> None:
    """Pretty-print the summary."""
    for regime in ["static", "adaptive", "adaptive_learning"]:
        entries = {k: v for k, v in summary.items() if v["regime"] == regime}
        if not entries:
            continue
        label = regime.replace("_", "+").upper()
        print(f"\n{'=' * 120}")
        print(f"  Regime: {label}")
        print(f"{'=' * 120}")
        print(f"{'rho':>5}  {'toxicity':>16}  {'welfare':>16}  {'qgap':>16}  "
              f"{'accept':>6}  {'p_self':>7}  {'p_expl':>7}  {'tp_self':>8}  {'tp_expl':>8}")
        print("-" * 120)
        for key in sorted(entries):
            s = entries[key]
            print(f"{s['rho_a']:>5.1f}  "
                  f"{s['toxicity_mean']:>7.4f}+/-{s['toxicity_std']:<6.4f}  "
                  f"{s['welfare_mean']:>7.1f}+/-{s['welfare_std']:<6.1f}  "
                  f"{s['quality_gap_mean']:>+7.4f}+/-{s['quality_gap_std']:<6.4f}  "
                  f"{s['acceptance_rate_mean']:>6.2f}  "
                  f"{s['mean_p_selfish']:>7.3f}  "
                  f"{s['mean_p_exploitative']:>7.3f}  "
                  f"{s['avg_final_tp_selfish']:>8.3f}  "
                  f"{s['avg_final_tp_exploitative']:>8.3f}")
        print(f"{'=' * 120}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(f"runs/{timestamp}_mesa_adaptive_agents_study")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Mesa Adaptive Agents Study")
    print(f"Run directory: {run_dir}")
    print("Sweeping rho_a 0.0-1.0 (11 values x 5 seeds x 3 regimes = 165 runs)")
    print("Population: 15 cooperative + 10 selfish + 5 exploitative = 30 agents")
    print("Steps per run: 50")
    print("Regimes: static | adaptive | adaptive+learning")
    print(f"Learning rates: coop={LEARNING_RATES['cooperative']}, "
          f"selfish={LEARNING_RATES['selfish']}, "
          f"exploit={LEARNING_RATES['exploitative']}")
    print()

    results = run_sweep()

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

    a_10 = summary.get("adaptive_rho_1.0", {})
    al_10 = summary.get("adaptive_learning_rho_1.0", {})
    a_05 = summary.get("adaptive_rho_0.5", {})
    al_05 = summary.get("adaptive_learning_rho_0.5", {})

    if a_10 and al_10:
        print("\n  At rho=1.0 (maximum governance pressure):")
        print(f"    Adaptive:          tox={a_10['toxicity_mean']:.4f}  "
              f"welfare={a_10['welfare_mean']:.1f}  accept={a_10['acceptance_rate_mean']:.2f}")
        print(f"    Adaptive+Learning: tox={al_10['toxicity_mean']:.4f}  "
              f"welfare={al_10['welfare_mean']:.1f}  accept={al_10['acceptance_rate_mean']:.2f}")
        welfare_recovery = al_10["welfare_mean"] - a_10["welfare_mean"]
        tox_diff = al_10["toxicity_mean"] - a_10["toxicity_mean"]
        print(f"    Welfare recovery:  {welfare_recovery:+.1f} "
              f"({welfare_recovery / max(a_10['welfare_mean'], 1) * 100:+.1f}%)")
        print(f"    Toxicity change:   {tox_diff:+.4f}")

    if a_05 and al_05:
        print("\n  At rho=0.5 (governance sweet spot):")
        print(f"    Adaptive:          tox={a_05['toxicity_mean']:.4f}  "
              f"welfare={a_05['welfare_mean']:.1f}  accept={a_05['acceptance_rate_mean']:.2f}")
        print(f"    Adaptive+Learning: tox={al_05['toxicity_mean']:.4f}  "
              f"welfare={al_05['welfare_mean']:.1f}  accept={al_05['acceptance_rate_mean']:.2f}")

    if al_10:
        print("\n  Selfish agent adaptation at rho=1.0:")
        print(f"    Final task_progress: {al_10['avg_final_tp_selfish']:.3f}")
        print(f"    Mean p (selfish):    {al_10['mean_p_selfish']:.3f}")

    # Find where learning regime welfare matches adaptive regime at rho=0
    a_00 = summary.get("adaptive_rho_0.0", {})
    if a_00:
        baseline_welfare = a_00["welfare_mean"]
        for rho_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            al_entry = summary.get(f"adaptive_learning_rho_{rho_val:.1f}", {})
            if al_entry and al_entry["welfare_mean"] < baseline_welfare * 0.8:
                print("\n  Welfare collapse threshold (< 80% of baseline):")
                print("    Adaptive:          rho >= ~0.7")
                print(f"    Adaptive+Learning: rho >= ~{rho_val}")
                break

    print(f"\nDone. Results in {run_dir}/")


if __name__ == "__main__":
    main()
