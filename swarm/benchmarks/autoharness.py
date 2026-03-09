"""AutoHarness: closed-loop benchmark generation, execution, and selection.

This module wires together a deterministic loop:

1) generate eval candidates from a parameter search space,
2) run each candidate across fixed seeds,
3) score aggregate performance,
4) mark candidates for promote/demote/hold vs baseline.

Design goals:
- deterministic generation (explicit RNG seed),
- replayable decisions (stores mean/std and confidence-aware thresholds),
- governance-friendly outputs (explicit decision reason per candidate).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from swarm.benchmarks.base import BenchmarkTask
from swarm.benchmarks.runner import BenchmarkRunner, RunFn


@dataclass(frozen=True)
class HarnessCandidate:
    """A single governance candidate to evaluate."""

    candidate_id: str
    config: dict[str, Any]


@dataclass(frozen=True)
class HarnessDecision:
    """Promotion decision for one candidate."""

    candidate_id: str
    mean_score: float
    std_score: float
    baseline_mean: float
    baseline_std: float
    decision: str  # "promote" | "demote" | "hold" | "baseline"
    reason: str


@dataclass(frozen=True)
class AutoHarnessConfig:
    """Configuration for one AutoHarness cycle."""

    n_seeds: int = 10
    max_candidates: int = 12
    score_column: str = "p"
    promotion_margin: float = 0.02
    demotion_margin: float = 0.02
    confidence_k: float = 1.0
    random_seed: int = 0


@dataclass
class AutoHarnessReport:
    """Structured output of a complete harness cycle."""

    candidates: list[HarnessCandidate]
    raw_results: pd.DataFrame
    summary: pd.DataFrame
    decisions: list[HarnessDecision]


class AutoHarness:
    """Closed-loop automated evaluator for governance candidate selection."""

    def __init__(
        self,
        benchmark: BenchmarkTask,
        run_fn: RunFn,
        baseline_config: dict[str, Any],
        parameter_space: dict[str, list[Any]],
        *,
        runner: BenchmarkRunner | None = None,
        config: AutoHarnessConfig | None = None,
    ):
        if "id" not in baseline_config:
            raise ValueError("baseline_config must include an 'id' field")
        if not parameter_space:
            raise ValueError("parameter_space must not be empty")

        self.benchmark = benchmark
        self.run_fn = run_fn
        self.baseline_config = dict(baseline_config)
        self.parameter_space = {k: list(v) for k, v in parameter_space.items()}
        self.config = config or AutoHarnessConfig()
        self.runner = runner or BenchmarkRunner()

    def generate_candidates(self) -> list[HarnessCandidate]:
        """Deterministically generate candidate configs from parameter space."""
        keys = sorted(self.parameter_space.keys())
        combos = list(itertools.product(*(self.parameter_space[k] for k in keys)))

        rng = random.Random(self.config.random_seed)
        rng.shuffle(combos)

        # Reserve one slot for baseline so total remains bounded.
        selected = combos[: max(0, self.config.max_candidates - 1)]
        candidates = [
            HarnessCandidate(candidate_id=self.baseline_config["id"], config=dict(self.baseline_config))
        ]

        for idx, values in enumerate(selected):
            cfg = {"id": f"auto_{idx:03d}"}
            cfg.update(dict(zip(keys, values, strict=True)))
            candidates.append(HarnessCandidate(candidate_id=cfg["id"], config=cfg))

        return candidates

    def execute_cycle(self) -> AutoHarnessReport:
        """Run generate -> run -> score -> promote/demote for one cycle."""
        candidates = self.generate_candidates()
        configs = [c.config for c in candidates]

        raw = self.runner.run_frontier(
            self.benchmark,
            configs,
            n_seeds=self.config.n_seeds,
            run_fn=self.run_fn,
        )
        summary = self._summarize(raw)
        decisions = self._decide(summary)

        return AutoHarnessReport(
            candidates=candidates,
            raw_results=raw,
            summary=summary,
            decisions=decisions,
        )

    def _summarize(self, raw_results: pd.DataFrame) -> pd.DataFrame:
        score_col = self.config.score_column
        if score_col not in raw_results.columns:
            raise ValueError(f"score column '{score_col}' missing from raw results")

        grouped = (
            raw_results.groupby("gov_config", as_index=False)[score_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped = grouped.rename(
            columns={
                "gov_config": "candidate_id",
                "mean": "mean_score",
                "std": "std_score",
                "count": "n_runs",
            }
        )
        grouped["std_score"] = grouped["std_score"].fillna(0.0)
        return grouped

    def _decide(self, summary: pd.DataFrame) -> list[HarnessDecision]:
        baseline_id = self.baseline_config["id"]
        baseline_row = summary[summary["candidate_id"] == baseline_id]
        if baseline_row.empty:
            raise ValueError("baseline candidate missing from summary")

        b_mean = float(baseline_row.iloc[0]["mean_score"])
        b_std = float(baseline_row.iloc[0]["std_score"])
        k = self.config.confidence_k

        decisions: list[HarnessDecision] = []
        for row in summary.itertuples(index=False):
            candidate_id = str(row.candidate_id)
            mean_score = float(row.mean_score)
            std_score = float(row.std_score)

            if candidate_id == baseline_id:
                decisions.append(
                    HarnessDecision(
                        candidate_id=candidate_id,
                        mean_score=mean_score,
                        std_score=std_score,
                        baseline_mean=b_mean,
                        baseline_std=b_std,
                        decision="baseline",
                        reason="reference candidate",
                    )
                )
                continue

            lower = mean_score - k * std_score
            upper = mean_score + k * std_score
            baseline_upper = b_mean + k * b_std
            baseline_lower = b_mean - k * b_std

            if lower > baseline_upper + self.config.promotion_margin:
                decision = "promote"
                reason = (
                    f"confidence-adjusted lower bound {lower:.4f} exceeds baseline upper "
                    f"bound {baseline_upper:.4f} + margin {self.config.promotion_margin:.4f}"
                )
            elif upper < baseline_lower - self.config.demotion_margin:
                decision = "demote"
                reason = (
                    f"confidence-adjusted upper bound {upper:.4f} is below baseline lower "
                    f"bound {baseline_lower:.4f} - margin {self.config.demotion_margin:.4f}"
                )
            else:
                decision = "hold"
                reason = "insufficient confidence-adjusted separation from baseline"

            decisions.append(
                HarnessDecision(
                    candidate_id=candidate_id,
                    mean_score=mean_score,
                    std_score=std_score,
                    baseline_mean=b_mean,
                    baseline_std=b_std,
                    decision=decision,
                    reason=reason,
                )
            )

        return decisions
