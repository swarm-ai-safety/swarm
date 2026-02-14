#!/usr/bin/env python
"""Phase 2: Statistical analysis for Moltbook CAPTCHA full study.

Runs pairwise comparisons, multiple comparisons correction,
normality checks, and agent-type stratification.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = Path(__file__).resolve().parent / "sweep_results.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

METRICS = [
    "welfare",
    "toxicity_rate",
    "quality_gap",
    "honest_payoff",
    "opportunistic_payoff",
    "deceptive_payoff",
    "adversarial_payoff",
]

AGENT_PAYOFF_COLS = [
    "honest_payoff",
    "opportunistic_payoff",
    "deceptive_payoff",
    "adversarial_payoff",
]


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = np.sqrt((s1**2 + s2**2) / 2)
    if pooled == 0:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled)


def main():
    print("=" * 60)
    print("Phase 2: Statistical Analysis — Moltbook CAPTCHA")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH.name}")

    # Identify swept parameters
    param_cols = [c for c in df.columns if "." in c]
    metric_cols = [c for c in METRICS if c in df.columns]

    print(f"Swept parameters: {param_cols}")
    print(f"Metrics: {metric_cols}")
    print()

    # ---------------------------------------------------------------
    # Pairwise comparisons
    # ---------------------------------------------------------------
    results = []
    for param in param_cols:
        values = sorted(df[param].unique())
        for v1, v2 in combinations(values, 2):
            for metric in metric_cols:
                g1 = df[df[param] == v1][metric].dropna().values
                g2 = df[df[param] == v2][metric].dropna().values
                if len(g1) < 3 or len(g2) < 3:
                    continue

                t_stat, t_p = stats.ttest_ind(g1, g2, equal_var=False)
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                d = cohens_d(g1, g2)

                results.append({
                    "parameter": param,
                    "value_1": v1,
                    "value_2": v2,
                    "metric": metric,
                    "n1": len(g1),
                    "n2": len(g2),
                    "mean_1": float(np.mean(g1)),
                    "mean_2": float(np.mean(g2)),
                    "std_1": float(np.std(g1, ddof=1)),
                    "std_2": float(np.std(g2, ddof=1)),
                    "t_stat": float(t_stat),
                    "t_p": float(t_p),
                    "u_stat": float(u_stat),
                    "u_p": float(u_p),
                    "cohens_d": float(d),
                })

    n_tests = len(results)
    print(f"Total hypotheses tested: {n_tests}")

    # ---------------------------------------------------------------
    # Multiple comparisons correction
    # ---------------------------------------------------------------
    alpha = 0.05
    bonferroni_threshold = alpha / n_tests if n_tests > 0 else alpha

    # Benjamini-Hochberg
    sorted_by_p = sorted(range(n_tests), key=lambda i: results[i]["t_p"])
    bh_reject = [False] * n_tests
    for rank_idx, orig_idx in enumerate(sorted_by_p):
        bh_threshold = (rank_idx + 1) / n_tests * alpha
        if results[orig_idx]["t_p"] <= bh_threshold:
            bh_reject[orig_idx] = True
        else:
            break  # All subsequent are also non-significant

    for i, r in enumerate(results):
        r["bonferroni_sig"] = r["t_p"] < bonferroni_threshold
        r["bh_sig"] = bh_reject[i]

    n_bonf = sum(1 for r in results if r["bonferroni_sig"])
    n_bh = sum(1 for r in results if r["bh_sig"])

    print(f"Bonferroni threshold: {bonferroni_threshold:.6f}")
    print(f"Bonferroni significant: {n_bonf}/{n_tests}")
    print(f"Benjamini-Hochberg significant: {n_bh}/{n_tests}")
    print()

    # ---------------------------------------------------------------
    # Print significant results
    # ---------------------------------------------------------------
    print("=== Significant Results (Bonferroni) ===")
    sig_results = [r for r in results if r["bonferroni_sig"]]
    sig_results.sort(key=lambda r: r["t_p"])
    if not sig_results:
        print("  (none)")
    for r in sig_results[:20]:
        stars = "***" if abs(r["cohens_d"]) > 0.8 else "**" if abs(r["cohens_d"]) > 0.5 else "*"
        print(
            f"  {r['metric']}: {r['parameter']}={r['value_1']} vs {r['value_2']} "
            f"— p={r['t_p']:.6f}, d={r['cohens_d']:.2f}{stars} "
            f"(mean {r['mean_1']:.3f} vs {r['mean_2']:.3f})"
        )
    print()

    # ---------------------------------------------------------------
    # Normality checks (Shapiro-Wilk on welfare for first param)
    # ---------------------------------------------------------------
    print("=== Normality (Shapiro-Wilk on welfare) ===")
    normality_results = []
    first_param = param_cols[0] if param_cols else None
    if first_param and "welfare" in df.columns:
        for val in sorted(df[first_param].unique()):
            group = df[df[first_param] == val]["welfare"].dropna().values
            if len(group) >= 3:
                w_stat, w_p = stats.shapiro(group)
                label = "NORMAL" if w_p > 0.05 else "NON-NORMAL"
                normality_results.append({
                    "parameter": first_param,
                    "value": float(val),
                    "W": float(w_stat),
                    "p": float(w_p),
                    "label": label,
                })
                print(f"  {first_param}={val}: W={w_stat:.4f}, p={w_p:.4f} [{label}]")
    print()

    # ---------------------------------------------------------------
    # Agent-type stratification
    # ---------------------------------------------------------------
    print("=== Agent-Type Stratification ===")
    present_agent_cols = [c for c in AGENT_PAYOFF_COLS if c in df.columns]
    agent_strat = []

    # Overall means
    for col in present_agent_cols:
        label = col.replace("_payoff", "")
        print(f"  {label}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")

    print()
    # Pairwise paired t-tests between agent types
    agent_pairs = list(combinations(present_agent_cols, 2))
    n_agent_tests = len(agent_pairs)
    agent_bonf = alpha / n_agent_tests if n_agent_tests > 0 else alpha

    for c1, c2 in agent_pairs:
        g1 = df[c1].dropna().values
        g2 = df[c2].dropna().values
        min_len = min(len(g1), len(g2))
        if min_len < 3:
            continue
        g1, g2 = g1[:min_len], g2[:min_len]
        t_stat, t_p = stats.ttest_rel(g1, g2)
        d = cohens_d(g1, g2)
        sig = t_p < agent_bonf
        label1 = c1.replace("_payoff", "")
        label2 = c2.replace("_payoff", "")
        stars = "***" if sig and abs(d) > 0.8 else "**" if sig else ""
        agent_strat.append({
            "type_1": label1,
            "type_2": label2,
            "t_stat": float(t_stat),
            "t_p": float(t_p),
            "cohens_d": float(d),
            "bonferroni_sig": sig,
        })
        print(f"  {label1} vs {label2}: d={d:.2f}{stars}, p={t_p:.6f}")

    print()

    # ---------------------------------------------------------------
    # Summary JSON
    # ---------------------------------------------------------------
    summary = {
        "csv": str(CSV_PATH),
        "total_runs": len(df),
        "total_hypotheses": n_tests,
        "n_bonferroni_significant": n_bonf,
        "n_bh_significant": n_bh,
        "bonferroni_threshold": bonferroni_threshold,
        "swept_parameters": {
            p: sorted([float(v) for v in df[p].unique()])
            for p in param_cols
        },
        "results": results,
        "agent_stratification": agent_strat,
        "normality": normality_results,
    }

    summary_path = OUTPUT_DIR / "summary.json"

    def _default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_default)
    print(f"Summary written to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
