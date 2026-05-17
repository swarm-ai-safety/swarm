#!/usr/bin/env python
"""Mesa Bridge Adversarial Learners Study (Study 4).

Research question: Can adversarial learners who game the acceptance
threshold — inflating observable signals without genuine quality
improvement — break the governance mechanism?

Prior findings (studies 1-3, 455 runs):
  - Rho alone is a pure welfare tax (Study 1, 110 runs)
  - Learning agents recover welfare at high rho (Study 2, 165 runs)
  - Results generalize across PD, Stag Hunt, Hawk-Dove (Study 3, 180 runs)

Gaming surface analysis:
  The ProxyComputer computes v_hat from 4 signals with weights:
    task_progress (0.4) — EASY to game (self-reported, highest weight)
    engagement    (0.2) — EASY to game (self-reported activity level)
    rework_count  (0.2) — MEDIUM (requires effort to suppress)
    verifier_rej  (0.2) — HARD  (external check, can't fake)

  Key insight: 60% of v_hat comes from easily-gamed signals.

Agent types:
  - Honest agents: cooperative/selfish with adaptive+learning behavior
  - Adversarial agents: inflate task_progress and engagement without
    genuine quality improvement. Two strategies:
      signal_gaming:     inflate easy signals proportional to gaming_skill
      threshold_hunting: binary-search for minimum p that passes threshold

Detection signals:
  - gaming_gap:       mean(gamed_p - honest_p) across adversarial agents
  - tp_rework_corr:   correlation between task_progress and rework_count
                       (honest agents correlate; gamers don't)

Swept parameters:
  - adversarial_fraction: 6 values (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
  - rho_a: 5 values (0.0, 0.25, 0.5, 0.75, 1.0)
  - gaming_strategy: 2 (signal_gaming, threshold_hunting)
  - seeds: 5
  Total: 6 x 5 x 2 x 5 = 300 runs
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
from swarm.core.proxy import ProxyComputer
from swarm.metrics.reporters import MetricsReporter
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Learning rates by archetype (from Study 2)
# ---------------------------------------------------------------------------

LEARNING_RATES = {
    "cooperative": 0.03,
    "selfish": 0.05,
    "exploitative": 0.02,
}

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


@dataclass
class SimAgent:
    """Simulated Mesa-style agent with behavioral archetype and learning."""

    agent_id: str
    archetype: str  # cooperative, selfish, exploitative
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
        headroom = 1.0 - self.task_progress
        improvement = lr * headroom
        self.task_progress = min(1.0, self.task_progress + improvement)
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


@dataclass
class AdversarialSimAgent:
    """Adversarial agent that games observable signals.

    true_quality stays low while reported signals are inflated.
    Two strategies:
      signal_gaming:     inflate task_progress and engagement proportional to gaming_skill
      threshold_hunting: binary-search for minimum signal level that passes threshold
    """

    agent_id: str
    true_quality: float = 0.3       # actual quality (stays low)
    gaming_skill: float = 0.7       # how good at inflating signals (0.0-1.0)
    gaming_strategy: str = "signal_gaming"  # or "threshold_hunting"
    rework_count: int = 0
    verifier_rejections: int = 0
    engagement: float = 0.3

    # Threshold hunting state
    _threshold_estimate: float = 0.5
    _search_lo: float = 0.0
    _search_hi: float = 1.0

    # Tracking
    rejection_count: int = 0
    acceptance_count: int = 0

    _rng: random.Random = field(default_factory=lambda: random.Random())

    def step(self) -> None:
        """Advance adversarial agent state.

        True quality drifts slightly (adversaries don't genuinely improve).
        Rework and verifier rejections accumulate honestly (hard to fake).
        """
        self.true_quality = min(1.0, max(0.0, self.true_quality + self._rng.gauss(-0.01, 0.05)))
        self.rework_count = max(0, self.rework_count + (1 if self._rng.random() < 0.25 else 0))
        self.verifier_rejections = max(0, self.verifier_rejections + (1 if self._rng.random() < 0.20 else 0))
        self.engagement = min(1.0, max(0.0, self.engagement + self._rng.gauss(-0.01, 0.05)))

    def game_signals(self, threshold: float) -> Dict[str, Any]:
        """Produce gamed observable signals.

        Returns a dict suitable for MesaBridge.record_agent_states().
        """
        if self.gaming_strategy == "signal_gaming":
            return self._signal_gaming()
        else:
            return self._threshold_hunting(threshold)

    def _signal_gaming(self) -> Dict[str, Any]:
        """Inflate easy signals proportional to gaming_skill."""
        inflated_progress = self.true_quality + self.gaming_skill * (1.0 - self.true_quality) * 0.8
        inflated_engagement = 0.3 + self.gaming_skill * 0.6
        return {
            "agent_id": self.agent_id,
            "task_progress": min(1.0, inflated_progress),
            "rework_count": self.rework_count,       # honest
            "engagement": min(1.0, inflated_engagement),
        }

    def _threshold_hunting(self, threshold: float) -> Dict[str, Any]:
        """Binary-search for minimum signal level that passes threshold."""
        # Use feedback from acceptance/rejection to narrow the search
        target_progress = self._threshold_estimate + 0.05  # slight margin
        target_progress = min(1.0, max(0.0, target_progress))
        target_engagement = 0.4 + self.gaming_skill * 0.3  # moderate inflation
        return {
            "agent_id": self.agent_id,
            "task_progress": target_progress,
            "rework_count": self.rework_count,
            "engagement": min(1.0, target_engagement),
        }

    def observe_result(self, accepted: bool, p: float, threshold: float) -> None:
        """Update threshold hunting state based on acceptance result."""
        if accepted:
            self.acceptance_count += 1
            if self.gaming_strategy == "threshold_hunting":
                # Accepted — try to lower the estimate (save effort)
                self._search_hi = self._threshold_estimate
                self._threshold_estimate = (self._search_lo + self._threshold_estimate) / 2
        else:
            self.rejection_count += 1
            if self.gaming_strategy == "threshold_hunting":
                # Rejected — need higher signals
                self._search_lo = self._threshold_estimate
                self._threshold_estimate = (self._threshold_estimate + self._search_hi) / 2

    def honest_dict(self) -> Dict[str, Any]:
        """What the agent's signals would look like without gaming."""
        return {
            "agent_id": self.agent_id,
            "task_progress": self.true_quality,
            "rework_count": self.rework_count,
            "engagement": self.engagement,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Alias for honest_dict (used for honest-p computation)."""
        return self.honest_dict()


# ---------------------------------------------------------------------------
# Population builder
# ---------------------------------------------------------------------------


def build_population(
    n_honest: int,
    n_adversarial: int,
    gaming_strategy: str,
    seed: int,
) -> Tuple[List[SimAgent], List[AdversarialSimAgent]]:
    """Build a mixed honest + adversarial agent population.

    Honest agents: 60% cooperative, 40% selfish (no exploitative in honest pool).
    Adversarial agents: low true_quality, variable gaming_skill.
    """
    rng = random.Random(seed)
    honest_agents: List[SimAgent] = []
    adversarial_agents: List[AdversarialSimAgent] = []
    idx = 0

    # Honest agents: 60/40 cooperative/selfish split
    n_coop = int(n_honest * 0.6)
    n_self = n_honest - n_coop

    for archetype, count, (tp_mu, eng_mu) in [
        ("cooperative", n_coop, (0.85, 0.7)),
        ("selfish", n_self, (0.55, 0.45)),
    ]:
        for _ in range(count):
            a = SimAgent(
                agent_id=f"{archetype[:4]}-{idx}",
                archetype=archetype,
                task_progress=max(0.0, min(1.0, rng.gauss(tp_mu, 0.08))),
                rework_count=max(0, int(rng.gauss(0, 0.5))),
                engagement=max(0.0, min(1.0, rng.gauss(eng_mu, 0.1))),
            )
            a._rng = random.Random(seed + idx)
            honest_agents.append(a)
            idx += 1

    # Adversarial agents
    for _i in range(n_adversarial):
        gaming_skill = max(0.2, min(1.0, rng.gauss(0.7, 0.15)))
        a = AdversarialSimAgent(
            agent_id=f"advr-{idx}",
            true_quality=max(0.1, min(0.5, rng.gauss(0.3, 0.1))),
            gaming_skill=gaming_skill,
            gaming_strategy=gaming_strategy,
            rework_count=max(0, int(rng.gauss(1, 0.5))),
            verifier_rejections=0,
            engagement=max(0.1, min(0.5, rng.gauss(0.3, 0.1))),
        )
        a._rng = random.Random(seed + idx)
        adversarial_agents.append(a)
        idx += 1

    return honest_agents, adversarial_agents


# ---------------------------------------------------------------------------
# Correlation helper
# ---------------------------------------------------------------------------


def pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation between two lists. Returns 0.0 if degenerate."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    sx = statistics.stdev(xs)
    sy = statistics.stdev(ys)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (n - 1)
    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Aggregate results from a single adversarial simulation run."""

    rho_a: float
    adversarial_fraction: float
    gaming_strategy: str
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
    # Adversarial-specific
    gaming_gap: float        # mean(gamed_p - honest_p) for adversarial agents
    tp_rework_corr: float    # task_progress vs rework correlation across all agents
    adv_acceptance_rate: float  # acceptance rate of adversarial agents
    honest_acceptance_rate: float  # acceptance rate of honest agents
    mean_p_honest: float
    mean_p_adversarial: float


def run_single(
    rho_a: float,
    adversarial_fraction: float,
    gaming_strategy: str,
    seed: int,
    n_steps: int = 50,
    n_total: int = 30,
) -> RunResult:
    """Run one Mesa bridge simulation with adversarial agents."""
    n_adversarial = int(n_total * adversarial_fraction)
    n_honest = n_total - n_adversarial

    payoff_cfg = PayoffConfig(
        s_plus=2.0, s_minus=1.0, h=2.0, theta=0.5,
        rho_a=rho_a, rho_b=0.0, w_rep=1.0,
    )
    bridge_cfg = MesaBridgeConfig(
        model_id=f"mesa-adv-rho{rho_a:.2f}-af{adversarial_fraction:.1f}-{gaming_strategy}-s{seed}",
        enable_event_log=False,
    )
    bridge = MesaBridge(config=bridge_cfg, payoff_config=payoff_cfg)
    proxy = ProxyComputer()  # for computing honest_p

    accept_threshold = 0.5 + 0.3 * rho_a

    honest_agents, adversarial_agents = build_population(
        n_honest, n_adversarial, gaming_strategy, seed,
    )

    all_interactions: List[SoftInteraction] = []
    honest_interactions: List[SoftInteraction] = []
    adversarial_interactions: List[SoftInteraction] = []

    # Track gaming gaps per step
    gaming_gaps: List[float] = []
    # Track task_progress and rework for correlation
    all_tp: List[float] = []
    all_rework: List[float] = []

    for _step in range(n_steps):
        # Step all agents
        for a in honest_agents:
            a.step()
        for a in adversarial_agents:
            a.step()

        # Build state dicts: honest agents report truthfully
        honest_states = [a.to_dict() for a in honest_agents]
        # Adversarial agents produce gamed signals
        adversarial_gamed_states = [a.game_signals(accept_threshold) for a in adversarial_agents]
        # Also compute honest signals for gaming gap
        adversarial_honest_states = [a.honest_dict() for a in adversarial_agents]

        all_states = honest_states + adversarial_gamed_states

        # Record interactions through the bridge
        step_interactions = bridge.record_agent_states(all_states)

        # Build agent_id -> agent mapping for feedback
        honest_map = {a.agent_id: a for a in honest_agents}
        adversarial_map = {a.agent_id: a for a in adversarial_agents}

        # Compute honest_p for adversarial agents to measure gaming gap
        for adv_agent, honest_state in zip(adversarial_agents, adversarial_honest_states, strict=True):
            honest_obs = bridge._extract_observables(honest_state)
            _, honest_p = proxy.compute_labels(honest_obs)
            # Find the corresponding gamed interaction
            for ix in step_interactions:
                if ix.counterparty == adv_agent.agent_id:
                    gap = ix.p - honest_p
                    gaming_gaps.append(gap)
                    break

        # Apply acceptance and learning feedback
        for ix in step_interactions:
            ix.accepted = ix.p >= accept_threshold
            bridge._payoff_engine.payoff_initiator(ix)

            agent_id = ix.counterparty
            if agent_id in honest_map:
                agent = honest_map[agent_id]
                if ix.accepted:
                    agent.observe_acceptance()
                else:
                    agent.learn_from_rejection()
                honest_interactions.append(ix)
            elif agent_id in adversarial_map:
                adv = adversarial_map[agent_id]
                adv.observe_result(ix.accepted, ix.p, accept_threshold)
                adversarial_interactions.append(ix)

            all_interactions.append(ix)

        # Collect task_progress and rework for correlation
        for state in all_states:
            all_tp.append(state["task_progress"])
            all_rework.append(float(state["rework_count"]))

    # Compute summary metrics
    reporter = MetricsReporter(payoff_config=payoff_cfg)
    summary = reporter.summary(all_interactions)

    # Gaming gap
    mean_gaming_gap = statistics.mean(gaming_gaps) if gaming_gaps else 0.0

    # Task_progress vs rework correlation
    tp_rework_corr = pearson_correlation(all_tp, all_rework)

    # Per-group acceptance rates and mean p
    def acceptance_rate(ixs: List[SoftInteraction]) -> float:
        if not ixs:
            return 0.0
        return sum(1 for ix in ixs if ix.accepted) / len(ixs)

    def mean_p(ixs: List[SoftInteraction]) -> float:
        if not ixs:
            return 0.0
        return statistics.mean(ix.p for ix in ixs)

    return RunResult(
        rho_a=rho_a,
        adversarial_fraction=adversarial_fraction,
        gaming_strategy=gaming_strategy,
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
        gaming_gap=mean_gaming_gap,
        tp_rework_corr=tp_rework_corr,
        adv_acceptance_rate=acceptance_rate(adversarial_interactions),
        honest_acceptance_rate=acceptance_rate(honest_interactions),
        mean_p_honest=mean_p(honest_interactions),
        mean_p_adversarial=mean_p(adversarial_interactions),
    )


# ---------------------------------------------------------------------------
# Sweep + export
# ---------------------------------------------------------------------------


def run_sweep(
    adversarial_fractions: List[float] | None = None,
    rho_values: List[float] | None = None,
    strategies: List[str] | None = None,
    seeds: List[int] | None = None,
    n_steps: int = 50,
) -> List[RunResult]:
    """Sweep adversarial_fraction x rho_a x gaming_strategy x seeds."""
    if adversarial_fractions is None:
        adversarial_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    if rho_values is None:
        rho_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    if strategies is None:
        strategies = ["signal_gaming", "threshold_hunting"]
    if seeds is None:
        seeds = [42, 123, 256, 789, 1024]

    results: List[RunResult] = []
    total = len(adversarial_fractions) * len(rho_values) * len(strategies) * len(seeds)
    done = 0

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        for af in adversarial_fractions:
            for rho in rho_values:
                for seed in seeds:
                    result = run_single(
                        rho_a=rho,
                        adversarial_fraction=af,
                        gaming_strategy=strategy,
                        seed=seed,
                        n_steps=n_steps,
                    )
                    results.append(result)
                    done += 1
                    print(
                        f"  [{done}/{total}] {strategy:<20s}  af={af:.1f}  rho={rho:.2f}  "
                        f"seed={seed}  tox={result.toxicity:.3f}  "
                        f"welfare={result.total_welfare:.1f}  "
                        f"gap={result.gaming_gap:.3f}  "
                        f"adv_ar={result.adv_acceptance_rate:.2f}"
                    )

    return results


FIELDNAMES = [
    "rho_a", "adversarial_fraction", "gaming_strategy", "seed", "n_steps",
    "toxicity", "quality_gap", "avg_quality",
    "total_welfare", "avg_initiator_payoff", "avg_counterparty_payoff",
    "acceptance_rate", "n_interactions",
    "gaming_gap", "tp_rework_corr",
    "adv_acceptance_rate", "honest_acceptance_rate",
    "mean_p_honest", "mean_p_adversarial",
]


def export_csv(results: List[RunResult], path: Path) -> None:
    """Write results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in FIELDNAMES})
    print(f"  Exported {len(results)} rows to {path}")


def compute_summary(results: List[RunResult]) -> Dict[str, Any]:
    """Compute per-strategy, per-af, per-rho aggregate statistics."""
    by_key: Dict[Tuple[str, float, float], List[RunResult]] = defaultdict(list)
    for r in results:
        by_key[(r.gaming_strategy, r.adversarial_fraction, r.rho_a)].append(r)

    summary: Dict[str, Any] = {}
    for (strategy, af, rho) in sorted(by_key):
        runs = by_key[(strategy, af, rho)]
        n = len(runs)

        def _mean(vals: List[float]) -> float:
            return statistics.mean(vals)

        def _std(vals: List[float]) -> float:
            return statistics.stdev(vals) if len(vals) > 1 else 0.0

        tox = [r.toxicity for r in runs]
        wel = [r.total_welfare for r in runs]
        gg = [r.gaming_gap for r in runs]
        adv_ar = [r.adv_acceptance_rate for r in runs]
        tp_rc = [r.tp_rework_corr for r in runs]

        key = f"{strategy}_af{af:.1f}_rho{rho:.2f}"
        summary[key] = {
            "gaming_strategy": strategy,
            "adversarial_fraction": af,
            "rho_a": rho,
            "n_runs": n,
            "toxicity_mean": _mean(tox),
            "toxicity_std": _std(tox),
            "welfare_mean": _mean(wel),
            "welfare_std": _std(wel),
            "gaming_gap_mean": _mean(gg),
            "gaming_gap_std": _std(gg),
            "adv_acceptance_rate_mean": _mean(adv_ar),
            "tp_rework_corr_mean": _mean(tp_rc),
            "mean_p_honest": _mean([r.mean_p_honest for r in runs]),
            "mean_p_adversarial": _mean([r.mean_p_adversarial for r in runs]),
        }

    return summary


def print_summary_table(summary: Dict[str, Any]) -> None:
    """Pretty-print the summary grouped by strategy."""
    for strategy in ["signal_gaming", "threshold_hunting"]:
        entries = {k: v for k, v in summary.items() if v["gaming_strategy"] == strategy}
        if not entries:
            continue
        print(f"\n{'=' * 130}")
        print(f"  Strategy: {strategy.upper().replace('_', ' ')}")
        print(f"{'=' * 130}")
        print(
            f"{'af':>4}  {'rho':>5}  {'toxicity':>16}  {'welfare':>16}  "
            f"{'gaming_gap':>14}  {'adv_accept':>10}  {'tp_rw_corr':>10}  "
            f"{'p_honest':>8}  {'p_adv':>8}"
        )
        print("-" * 130)
        for key in sorted(entries):
            s = entries[key]
            print(
                f"{s['adversarial_fraction']:>4.1f}  {s['rho_a']:>5.2f}  "
                f"{s['toxicity_mean']:>7.4f}+/-{s['toxicity_std']:<6.4f}  "
                f"{s['welfare_mean']:>7.1f}+/-{s['welfare_std']:<6.1f}  "
                f"{s['gaming_gap_mean']:>+7.4f}+/-{s['gaming_gap_std']:<4.4f}  "
                f"{s['adv_acceptance_rate_mean']:>10.3f}  "
                f"{s['tp_rework_corr_mean']:>10.3f}  "
                f"{s['mean_p_honest']:>8.3f}  "
                f"{s['mean_p_adversarial']:>8.3f}"
            )
        print(f"{'=' * 130}")


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def generate_plots(results: List[RunResult], run_dir: Path) -> None:
    """Generate 6 analysis figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib/numpy not available — skipping plots")
        return

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Helper to aggregate results
    def agg(results: List[RunResult], key: str) -> float:
        return statistics.mean(getattr(r, key) for r in results)

    # Group results
    by_key: Dict[Tuple[str, float, float], List[RunResult]] = defaultdict(list)
    for r in results:
        by_key[(r.gaming_strategy, r.adversarial_fraction, r.rho_a)].append(r)

    strategies = ["signal_gaming", "threshold_hunting"]
    afs = sorted({r.adversarial_fraction for r in results})
    rhos = sorted({r.rho_a for r in results})

    # --- Plot 1: Breakdown curve — welfare vs adversarial_fraction ---
    fig, axes = plt.subplots(1, len(rhos), figsize=(4 * len(rhos), 4), sharey=True)
    if len(rhos) == 1:
        axes = [axes]
    for ax, rho in zip(axes, rhos, strict=True):
        for strategy in strategies:
            wels = [agg(by_key[(strategy, af, rho)], "total_welfare")
                    for af in afs if by_key.get((strategy, af, rho))]
            valid_afs = [af for af in afs if by_key.get((strategy, af, rho))]
            ax.plot(valid_afs, wels, marker="o", label=strategy.replace("_", " "))
        ax.set_title(f"rho={rho:.2f}")
        ax.set_xlabel("Adversarial fraction")
        if ax == axes[0]:
            ax.set_ylabel("Total welfare")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Fig 1: Welfare Breakdown by Adversarial Fraction", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "01_breakdown_curve.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Gaming gap heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, strategy in zip(axes, strategies, strict=True):
        data = np.zeros((len(afs), len(rhos)))
        for i, af in enumerate(afs):
            for j, rho in enumerate(rhos):
                runs = by_key.get((strategy, af, rho), [])
                data[i, j] = agg(runs, "gaming_gap") if runs else 0.0
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="YlOrRd")
        ax.set_xticks(range(len(rhos)))
        ax.set_xticklabels([f"{r:.2f}" for r in rhos])
        ax.set_yticks(range(len(afs)))
        ax.set_yticklabels([f"{a:.1f}" for a in afs])
        ax.set_xlabel("rho_a")
        ax.set_ylabel("Adversarial fraction")
        ax.set_title(strategy.replace("_", " ").title())
        fig.colorbar(im, ax=ax, label="Gaming gap")
    fig.suptitle("Fig 2: Gaming Gap Heatmap", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "02_gaming_gap_heatmap.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: Toxicity vs adversarial fraction ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy in strategies:
        for rho in [0.0, 0.5, 1.0]:
            if rho not in rhos:
                continue
            toxs = [agg(by_key[(strategy, af, rho)], "toxicity")
                    for af in afs if by_key.get((strategy, af, rho))]
            valid_afs = [af for af in afs if by_key.get((strategy, af, rho))]
            label = f"{strategy.replace('_', ' ')} rho={rho:.1f}"
            ax.plot(valid_afs, toxs, marker="o", label=label)
    ax.set_xlabel("Adversarial fraction")
    ax.set_ylabel("Toxicity")
    ax.set_title("Fig 3: Toxicity vs Adversarial Fraction")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "03_toxicity_vs_adversarial.png", dpi=150)
    plt.close(fig)

    # --- Plot 4: Detection signal — tp_rework_corr by agent type ---
    fig, ax = plt.subplots(figsize=(8, 5))
    # For each strategy and af, show tp_rework_corr
    for strategy in strategies:
        corrs = [agg(by_key[(strategy, af, 0.5)], "tp_rework_corr")
                 for af in afs if by_key.get((strategy, af, 0.5))]
        valid_afs = [af for af in afs if by_key.get((strategy, af, 0.5))]
        ax.plot(valid_afs, corrs, marker="s", label=f"{strategy.replace('_', ' ')} (rho=0.5)")
    ax.set_xlabel("Adversarial fraction")
    ax.set_ylabel("Task-progress / rework correlation")
    ax.set_title("Fig 4: Detection Signal (tp-rework correlation)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "04_detection_signal.png", dpi=150)
    plt.close(fig)

    # --- Plot 5: Strategy comparison — welfare impact ---
    fig, ax = plt.subplots(figsize=(8, 5))
    # Welfare at af=0 as baseline for each rho
    for rho in rhos:
        baseline_runs = by_key.get(("signal_gaming", 0.0, rho), [])
        if not baseline_runs:
            continue
        baseline = agg(baseline_runs, "total_welfare")
        for strategy in strategies:
            deltas = []
            valid_afs = []
            for af in afs:
                if af == 0.0:
                    continue
                runs = by_key.get((strategy, af, rho), [])
                if runs:
                    deltas.append(agg(runs, "total_welfare") - baseline)
                    valid_afs.append(af)
            if deltas:
                ax.plot(valid_afs, deltas, marker="o",
                        label=f"{strategy.replace('_', ' ')} rho={rho:.2f}")
    ax.set_xlabel("Adversarial fraction")
    ax.set_ylabel("Welfare delta from baseline")
    ax.set_title("Fig 5: Strategy Comparison — Welfare Impact")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "05_strategy_comparison.png", dpi=150)
    plt.close(fig)

    # --- Plot 6: Governance robustness contour ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, strategy in zip(axes, strategies, strict=True):
        data = np.zeros((len(afs), len(rhos)))
        for i, af in enumerate(afs):
            for j, rho in enumerate(rhos):
                runs = by_key.get((strategy, af, rho), [])
                data[i, j] = agg(runs, "toxicity") if runs else 0.0
        cf = ax.contourf(
            np.array(rhos), np.array(afs), data,
            levels=15, cmap="RdYlGn_r",
        )
        ax.set_xlabel("rho_a")
        ax.set_ylabel("Adversarial fraction")
        ax.set_title(strategy.replace("_", " ").title())
        fig.colorbar(cf, ax=ax, label="Toxicity")
    fig.suptitle("Fig 6: Governance Robustness (Toxicity Contour)", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "06_governance_robustness.png", dpi=150)
    plt.close(fig)

    print(f"  Generated 6 plots in {plots_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(f"runs/{timestamp}_mesa_adversarial_learners_study")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Mesa Adversarial Learners Study (Study 4)")
    print(f"Run directory: {run_dir}")
    print("Sweeping: 6 adversarial_fractions x 5 rho x 2 strategies x 5 seeds = 300 runs")
    print("Population: 30 agents (honest: 60% cooperative + 40% selfish; adversarial fill remainder)")
    print("Steps per run: 50")
    print("Strategies: signal_gaming | threshold_hunting")
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

    # Generate plots
    generate_plots(results, run_dir)

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # 1. Breakdown threshold: at what af does welfare drop significantly?
    for strategy in ["signal_gaming", "threshold_hunting"]:
        baseline_key = f"{strategy}_af0.0_rho0.50"
        baseline = summary.get(baseline_key, {})
        if not baseline:
            continue
        baseline_welfare = baseline.get("welfare_mean", 0)
        print(f"\n  {strategy.upper().replace('_', ' ')}:")
        for af in [0.1, 0.2, 0.3, 0.4, 0.5]:
            entry = summary.get(f"{strategy}_af{af:.1f}_rho0.50", {})
            if entry:
                welfare = entry.get("welfare_mean", 0)
                pct = ((welfare - baseline_welfare) / max(abs(baseline_welfare), 1)) * 100
                gg = entry.get("gaming_gap_mean", 0)
                print(f"    af={af:.1f}: welfare={welfare:.1f} ({pct:+.1f}%)  "
                      f"gaming_gap={gg:+.4f}  "
                      f"adv_accept={entry.get('adv_acceptance_rate_mean', 0):.3f}")

    # 2. Gaming gap at high adversarial fraction
    for strategy in ["signal_gaming", "threshold_hunting"]:
        entry = summary.get(f"{strategy}_af0.5_rho0.50", {})
        if entry:
            print(f"\n  At 50% adversarial, rho=0.5 ({strategy}):")
            print(f"    Gaming gap:     {entry.get('gaming_gap_mean', 0):+.4f}")
            print(f"    Toxicity:       {entry.get('toxicity_mean', 0):.4f}")
            print(f"    Adv acceptance: {entry.get('adv_acceptance_rate_mean', 0):.3f}")

    # 3. Does higher rho help against adversaries?
    print("\n  Rho effectiveness against adversaries (signal_gaming, af=0.3):")
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        entry = summary.get(f"signal_gaming_af0.3_rho{rho:.2f}", {})
        if entry:
            print(f"    rho={rho:.2f}: tox={entry.get('toxicity_mean', 0):.4f}  "
                  f"adv_accept={entry.get('adv_acceptance_rate_mean', 0):.3f}  "
                  f"welfare={entry.get('welfare_mean', 0):.1f}")

    print(f"\nDone. Results in {run_dir}/")


if __name__ == "__main__":
    main()
