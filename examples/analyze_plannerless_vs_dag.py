#!/usr/bin/env python
"""Paired statistical test on the plannerless-vs-dag comparison summary.

Reads a ``summary.csv`` produced by ``examples/compare_plannerless_vs_dag.py``
(one row per scenario/seed) and runs Wilcoxon signed-rank tests on the
paired per-seed deltas for welfare and toxicity. The per-seed design
is exactly paired (same seed, same payoff, different scenario) so a
paired non-parametric test is appropriate.

Usage:
    python examples/analyze_plannerless_vs_dag.py runs/compare_plannerless_vs_dag_XXX/summary.csv
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None  # type: ignore[assignment]


def load_summary(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    """Return ``{scenario: {seed: {metric: value, ...}}}``."""
    by_sc: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    with path.open() as f:
        for row in csv.DictReader(f):
            sc = row["scenario"]
            seed = int(row["seed"])
            by_sc[sc][seed] = {
                "acc_rate": float(row["acceptance_rate"]),
                "avg_tox": float(row["avg_toxicity"]),
                "welfare": float(row["total_welfare"]),
            }
    return by_sc


def paired_deltas(a: dict[int, dict[str, float]], b: dict[int, dict[str, float]], metric: str):
    """Return per-seed (a - b) deltas for a shared seed set."""
    seeds = sorted(set(a) & set(b))
    return seeds, [a[s][metric] - b[s][metric] for s in seeds]


def report(label: str, deltas: list[float]) -> None:
    mean = statistics.fmean(deltas)
    stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    wins_pos = sum(1 for d in deltas if d > 0)
    wins_neg = sum(1 for d in deltas if d < 0)
    print(f"  {label}")
    print(f"    n={len(deltas)}  mean={mean:+.4f}  stdev={stdev:.4f}")
    print(f"    wins: +{wins_pos} / -{wins_neg} (ties {len(deltas) - wins_pos - wins_neg})")
    if wilcoxon is not None and len(deltas) >= 6:
        try:
            stat, pval = wilcoxon(deltas)
            print(f"    wilcoxon signed-rank: W={stat:.3f}  p={pval:.4f}")
        except ValueError as exc:
            print(f"    wilcoxon failed: {exc}")
    elif wilcoxon is None:
        print("    (scipy not installed — skipping wilcoxon)")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"error: {path} not found")
        return 1

    by_sc = load_summary(path)
    for pair in [("plannerless", "dag_planner_lightgov"),
                 ("plannerless", "dag_planner"),
                 ("dag_planner_lightgov", "dag_planner")]:
        a, b = pair
        if a not in by_sc or b not in by_sc:
            continue
        print(f"\n=== {a}  vs  {b}  (paired per-seed deltas = {a} − {b}) ===")
        for metric in ("welfare", "avg_tox", "acc_rate"):
            _, deltas = paired_deltas(by_sc[a], by_sc[b], metric)
            report(f"{metric}:", deltas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
