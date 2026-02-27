#!/usr/bin/env python
"""Mesa Bridge Game Structure Study.

Research question: Does the governance sweet spot generalize across
different game-theoretic structures, or is it an artifact of the
Prisoner's Dilemma payoff matrix used in prior studies?

Prior findings (studies 1 & 2):
  - rho is a pure welfare tax without adaptive acceptance
  - Learning agents recover welfare collapse at high rho
  - Both used s_plus=2.0, s_minus=1.0, h=2.0 (PD-like)

This study sweeps three game types defined by PayoffConfig:

  Prisoner's Dilemma:  s+=2.0, s-=1.0, h=2.0  (temptation to defect)
  Stag Hunt:           s+=4.0, s-=0.5, h=1.0  (coordination game)
  Hawk-Dove:           s+=1.5, s-=2.0, h=3.0  (costly conflict)

Key questions:
  1. Does the governance sweet spot shift across game structures?
  2. Is rho more/less effective in coordination vs conflict games?
  3. Do learning agents still Pareto-dominate across all game types?
  4. Does break-even probability predict where governance becomes effective?

Swept parameters:
  - game_type: 3 (prisoners_dilemma, stag_hunt, hawk_dove)
  - rho_a: 6 values (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
  - regime: 2 (adaptive, adaptive_learning)
  - seeds: 5
  Total: 3 x 6 x 2 x 5 = 180 runs
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
from typing import Any, Dict, List, NamedTuple, Tuple

from swarm.bridges.mesa import MesaBridge, MesaBridgeConfig
from swarm.core.payoff import PayoffConfig
from swarm.metrics.reporters import MetricsReporter
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Game type definitions
# ---------------------------------------------------------------------------


class GameType(NamedTuple):
    """Payoff structure defining a game type."""

    name: str
    s_plus: float
    s_minus: float
    h: float
    description: str


GAME_TYPES: Dict[str, GameType] = {
    "prisoners_dilemma": GameType(
        name="prisoners_dilemma",
        s_plus=2.0, s_minus=1.0, h=2.0,
        description="Temptation to defect, high externality",
    ),
    "stag_hunt": GameType(
        name="stag_hunt",
        s_plus=4.0, s_minus=0.5, h=1.0,
        description="Coordination game — low threshold, high reward for cooperation",
    ),
    "hawk_dove": GameType(
        name="hawk_dove",
        s_plus=1.5, s_minus=2.0, h=3.0,
        description="Conflict costly — high threshold, severe externalities",
    ),
}


def compute_break_even(game: GameType, theta: float = 0.5) -> Tuple[float, float]:
    """Compute break-even probabilities for a game type.

    Returns (p_surplus_break_even, p_with_externality_break_even).
    p_surplus: where expected surplus crosses zero: p*s+ = (1-p)*s-
    p_extern: where surplus minus externality crosses zero
    """
    # p * s_plus = (1-p) * s_minus  =>  p = s_minus / (s_plus + s_minus)
    p_surplus = game.s_minus / (game.s_plus + game.s_minus)
    # Including externality: p*s+ - (1-p)*s- - (1-p)*h >= 0
    # p*(s+ + s- + h) >= s- + h  =>  p >= (s- + h)/(s+ + s- + h)
    p_extern = (game.s_minus + game.h) / (game.s_plus + game.s_minus + game.h)
    return p_surplus, p_extern


# ---------------------------------------------------------------------------
# Learning rates by archetype (same as adaptive agents study)
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

    game_type: str
    rho_a: float
    regime: str  # "adaptive", "adaptive_learning"
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
    # Game-specific context
    p_break_even_surplus: float
    p_break_even_extern: float


def run_single(
    game_type: str,
    rho_a: float,
    seed: int,
    regime: str = "adaptive",
    n_steps: int = 50,
    n_cooperative: int = 15,
    n_selfish: int = 10,
    n_exploitative: int = 5,
) -> RunResult:
    """Run one Mesa bridge simulation with a specific game type.

    Regimes:
      - adaptive:          threshold = 0.5 + 0.3*rho, no learning
      - adaptive_learning: threshold = 0.5 + 0.3*rho, agents learn from rejection
    """
    game = GAME_TYPES[game_type]
    p_be_surplus, p_be_extern = compute_break_even(game)

    payoff_cfg = PayoffConfig(
        s_plus=game.s_plus, s_minus=game.s_minus, h=game.h, theta=0.5,
        rho_a=rho_a, rho_b=0.0, w_rep=1.0,
    )
    bridge_cfg = MesaBridgeConfig(
        model_id=f"mesa-{game_type}-{regime}-rho{rho_a:.2f}-s{seed}",
        enable_event_log=False,
    )
    bridge = MesaBridge(config=bridge_cfg, payoff_config=payoff_cfg)

    use_learning = regime == "adaptive_learning"
    accept_threshold = 0.5 + 0.3 * rho_a

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

    selfish_agents = [a for a in agents if a.archetype == "selfish"]
    exploit_agents = [a for a in agents if a.archetype == "exploitative"]

    return RunResult(
        game_type=game_type,
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
        p_break_even_surplus=p_be_surplus,
        p_break_even_extern=p_be_extern,
    )


# ---------------------------------------------------------------------------
# Sweep + export
# ---------------------------------------------------------------------------


def run_sweep(
    rho_values: List[float] | None = None,
    seeds: List[int] | None = None,
    n_steps: int = 50,
) -> List[RunResult]:
    """Sweep game_type x rho_a x regime across values, multiple seeds each."""
    if rho_values is None:
        rho_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    if seeds is None:
        seeds = [42, 123, 256, 789, 1024]

    regimes = ["adaptive", "adaptive_learning"]
    game_names = list(GAME_TYPES.keys())
    results: List[RunResult] = []
    total = len(game_names) * len(rho_values) * len(seeds) * len(regimes)
    done = 0

    for game_name in game_names:
        game = GAME_TYPES[game_name]
        p_be_s, p_be_e = compute_break_even(game)
        print(f"\n{'=' * 80}")
        print(f"  Game: {game_name}  ({game.description})")
        print(f"  s+={game.s_plus}  s-={game.s_minus}  h={game.h}")
        print(f"  Break-even: p_surplus={p_be_s:.2f}  p_extern={p_be_e:.2f}")
        print(f"{'=' * 80}")

        for regime in regimes:
            print(f"\n  --- Regime: {regime} ---")
            for rho in rho_values:
                for seed in seeds:
                    result = run_single(
                        game_type=game_name, rho_a=rho, seed=seed,
                        regime=regime, n_steps=n_steps,
                    )
                    results.append(result)
                    done += 1
                    print(f"    [{done}/{total}] rho={rho:.1f}  seed={seed}  "
                          f"tox={result.toxicity:.3f}  welfare={result.total_welfare:.1f}  "
                          f"accept={result.acceptance_rate:.2f}  "
                          f"tp_self={result.avg_final_tp_selfish:.3f}")

    return results


CSV_FIELDNAMES = [
    "game_type", "rho_a", "regime", "seed", "n_steps",
    "toxicity", "quality_gap", "avg_quality",
    "total_welfare", "avg_initiator_payoff", "avg_counterparty_payoff",
    "acceptance_rate", "n_interactions",
    "mean_p_cooperative", "mean_p_selfish", "mean_p_exploitative",
    "avg_rejections_selfish", "avg_final_tp_selfish", "avg_final_tp_exploitative",
    "p_break_even_surplus", "p_break_even_extern",
]


def export_csv(results: List[RunResult], path: Path) -> None:
    """Write results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in CSV_FIELDNAMES})
    print(f"  Exported {len(results)} rows to {path}")


def compute_summary(results: List[RunResult]) -> Dict[str, Any]:
    """Compute per-game, per-regime, per-rho aggregate statistics."""
    by_key: Dict[Tuple[str, str, float], List[RunResult]] = defaultdict(list)
    for r in results:
        by_key[(r.game_type, r.regime, r.rho_a)].append(r)

    summary: Dict[str, Any] = {}
    for (game, regime, rho) in sorted(by_key):
        runs = by_key[(game, regime, rho)]
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

        key = f"{game}_{regime}_rho_{rho:.1f}"
        summary[key] = {
            "game_type": game,
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
            "p_break_even_surplus": runs[0].p_break_even_surplus,
            "p_break_even_extern": runs[0].p_break_even_extern,
        }

    return summary


def print_summary_table(summary: Dict[str, Any]) -> None:
    """Pretty-print the summary, grouped by game type and regime."""
    for game_name in GAME_TYPES:
        game = GAME_TYPES[game_name]
        p_be_s, p_be_e = compute_break_even(game)

        print(f"\n{'#' * 120}")
        print(f"  Game: {game_name}  |  s+={game.s_plus}  s-={game.s_minus}  h={game.h}  "
              f"|  p_break_even: surplus={p_be_s:.2f}  extern={p_be_e:.2f}")
        print(f"{'#' * 120}")

        for regime in ["adaptive", "adaptive_learning"]:
            entries = {k: v for k, v in summary.items()
                       if v["game_type"] == game_name and v["regime"] == regime}
            if not entries:
                continue
            label = regime.replace("_", "+").upper()
            print(f"\n  {'=' * 110}")
            print(f"    Regime: {label}")
            print(f"  {'=' * 110}")
            print(f"  {'rho':>5}  {'toxicity':>16}  {'welfare':>16}  {'qgap':>16}  "
                  f"{'accept':>6}  {'p_self':>7}  {'p_expl':>7}  {'tp_self':>8}  {'tp_expl':>8}")
            print(f"  {'-' * 110}")
            for key in sorted(entries):
                s = entries[key]
                print(f"  {s['rho_a']:>5.1f}  "
                      f"{s['toxicity_mean']:>7.4f}+/-{s['toxicity_std']:<6.4f}  "
                      f"{s['welfare_mean']:>7.1f}+/-{s['welfare_std']:<6.1f}  "
                      f"{s['quality_gap_mean']:>+7.4f}+/-{s['quality_gap_std']:<6.4f}  "
                      f"{s['acceptance_rate_mean']:>6.2f}  "
                      f"{s['mean_p_selfish']:>7.3f}  "
                      f"{s['mean_p_exploitative']:>7.3f}  "
                      f"{s['avg_final_tp_selfish']:>8.3f}  "
                      f"{s['avg_final_tp_exploitative']:>8.3f}")


def print_cross_game_comparison(summary: Dict[str, Any]) -> None:
    """Print cross-game comparison of key metrics."""
    print(f"\n{'=' * 90}")
    print("  CROSS-GAME COMPARISON")
    print(f"{'=' * 90}")

    for rho_val in [0.0, 0.4, 0.8, 1.0]:
        print(f"\n  --- rho = {rho_val:.1f} ---")
        print(f"  {'game':<20s}  {'regime':<20s}  {'tox':>7}  {'welfare':>8}  "
              f"{'accept':>6}  {'qgap':>8}")
        print(f"  {'-' * 80}")
        for game_name in GAME_TYPES:
            for regime in ["adaptive", "adaptive_learning"]:
                key = f"{game_name}_{regime}_rho_{rho_val:.1f}"
                s = summary.get(key)
                if s:
                    print(f"  {game_name:<20s}  {regime:<20s}  "
                          f"{s['toxicity_mean']:>7.4f}  {s['welfare_mean']:>8.1f}  "
                          f"{s['acceptance_rate_mean']:>6.2f}  "
                          f"{s['quality_gap_mean']:>+8.4f}")


def find_sweet_spots(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Find the rho value that maximizes welfare for each game+regime combo."""
    sweet_spots: Dict[str, Any] = {}

    for game_name in GAME_TYPES:
        for regime in ["adaptive", "adaptive_learning"]:
            entries = {k: v for k, v in summary.items()
                       if v["game_type"] == game_name and v["regime"] == regime}
            if not entries:
                continue

            # Sweet spot = rho with highest welfare while toxicity < 0.15
            best_key = None
            best_welfare = float("-inf")
            for key, s in entries.items():
                if s["toxicity_mean"] < 0.15 and s["welfare_mean"] > best_welfare:
                    best_welfare = s["welfare_mean"]
                    best_key = key

            # Fallback: just max welfare
            if best_key is None:
                best_key = max(entries, key=lambda k: entries[k]["welfare_mean"])

            s = entries[best_key]
            sweet_spots[f"{game_name}_{regime}"] = {
                "game_type": game_name,
                "regime": regime,
                "optimal_rho": s["rho_a"],
                "welfare": s["welfare_mean"],
                "toxicity": s["toxicity_mean"],
                "acceptance_rate": s["acceptance_rate_mean"],
            }

    return sweet_spots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(f"runs/{timestamp}_mesa_game_structures_study")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Mesa Game Structure Study")
    print(f"Run directory: {run_dir}")
    print("Sweeping 3 game types x 6 rho values x 2 regimes x 5 seeds = 180 runs")
    print("Population: 15 cooperative + 10 selfish + 5 exploitative = 30 agents")
    print("Steps per run: 50")
    print("Regimes: adaptive | adaptive+learning")
    print()

    # Print game type summary
    print("Game Types:")
    for name, game in GAME_TYPES.items():
        p_s, p_e = compute_break_even(game)
        print(f"  {name:<20s}  s+={game.s_plus}  s-={game.s_minus}  h={game.h}  "
              f"p_be(surplus)={p_s:.2f}  p_be(extern)={p_e:.2f}  — {game.description}")
    print()

    results = run_sweep()

    # Export CSV
    csv_path = run_dir / "sweep_results.csv"
    export_csv(results, csv_path)

    # Export summary
    summary = compute_summary(results)
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Exported summary to {summary_path}")

    # Print tables
    print_summary_table(summary)
    print_cross_game_comparison(summary)

    # Find and report sweet spots
    sweet_spots = find_sweet_spots(summary)
    sweet_spots_path = run_dir / "sweet_spots.json"
    with open(sweet_spots_path, "w") as f:
        json.dump(sweet_spots, f, indent=2)

    print(f"\n{'=' * 90}")
    print("  GOVERNANCE SWEET SPOTS (max welfare with toxicity < 0.15)")
    print(f"{'=' * 90}")
    print(f"  {'game':<20s}  {'regime':<20s}  {'rho*':>5}  {'welfare':>8}  "
          f"{'tox':>7}  {'accept':>6}")
    print(f"  {'-' * 80}")
    for key in sorted(sweet_spots):
        ss = sweet_spots[key]
        print(f"  {ss['game_type']:<20s}  {ss['regime']:<20s}  "
              f"{ss['optimal_rho']:>5.1f}  {ss['welfare']:>8.1f}  "
              f"{ss['toxicity']:>7.4f}  {ss['acceptance_rate']:>6.2f}")

    # Key findings
    print(f"\n{'=' * 90}")
    print("  KEY FINDINGS")
    print(f"{'=' * 90}")

    # Compare learning benefit across games at rho=0.8
    for game_name in GAME_TYPES:
        a_key = f"{game_name}_adaptive_rho_0.8"
        al_key = f"{game_name}_adaptive_learning_rho_0.8"
        a = summary.get(a_key)
        al = summary.get(al_key)
        if a and al:
            welfare_recovery = al["welfare_mean"] - a["welfare_mean"]
            pct = welfare_recovery / max(abs(a["welfare_mean"]), 1) * 100
            print(f"\n  {game_name} at rho=0.8:")
            print(f"    Adaptive:          welfare={a['welfare_mean']:.1f}  tox={a['toxicity_mean']:.4f}")
            print(f"    Adaptive+Learning: welfare={al['welfare_mean']:.1f}  tox={al['toxicity_mean']:.4f}")
            print(f"    Learning benefit:  {welfare_recovery:+.1f} ({pct:+.1f}%)")

    # Break-even vs sweet spot analysis
    print("\n  Break-even vs Sweet Spot:")
    for game_name in GAME_TYPES:
        game = GAME_TYPES[game_name]
        _, p_be = compute_break_even(game)
        ss = sweet_spots.get(f"{game_name}_adaptive_learning")
        if ss:
            print(f"    {game_name:<20s}  p_break_even={p_be:.2f}  sweet_rho={ss['optimal_rho']:.1f}")

    print(f"\nDone. Results in {run_dir}/")


if __name__ == "__main__":
    main()
