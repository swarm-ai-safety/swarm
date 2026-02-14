#!/usr/bin/env python
"""Statistical analysis for collusion tax effect parameter sweep."""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUN_DIR = Path(__file__).resolve().parent
CSV_PATH = RUN_DIR / "sweep_results.csv"


def _default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_sd = np.sqrt((np.std(g1, ddof=1) ** 2 + np.std(g2, ddof=1) ** 2) / 2)
    if pooled_sd == 0:
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / pooled_sd


def main() -> int:
    print("=" * 60)
    print("Statistical Analysis: collusion_tax_effect")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df)} runs")

    # Compute RLM payoff: (welfare - honest_payoff * 3) / 9
    df["rlm_payoff"] = (df["welfare"] - df["honest_payoff"] * 3) / 9

    # Identify swept parameters and metrics
    sweep_params = [c for c in df.columns if "." in c]
    metrics = [
        "welfare",
        "toxicity_rate",
        "quality_gap",
        "honest_payoff",
        "rlm_payoff",
    ]
    metrics = [m for m in metrics if m in df.columns]

    print(f"Swept parameters: {sweep_params}")
    print(f"Metrics: {metrics}")

    all_results = []
    total_hypotheses = 0

    # Pairwise comparisons for each swept parameter
    for param in sweep_params:
        values = sorted(df[param].unique())
        print(f"\n--- {param}: {values} ---")

        for metric in metrics:
            for v1, v2 in combinations(values, 2):
                g1 = df[df[param] == v1][metric].dropna().values
                g2 = df[df[param] == v2][metric].dropna().values

                if len(g1) < 5 or len(g2) < 5:
                    print(f"  SKIP {metric} {v1} vs {v2}: insufficient data")
                    continue

                total_hypotheses += 1

                t_stat, t_p = stats.ttest_ind(g1, g2, equal_var=False)
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                d = cohens_d(g1, g2)

                result = {
                    "parameter": param,
                    "metric": metric,
                    "value_1": v1,
                    "value_2": v2,
                    "n_1": len(g1),
                    "n_2": len(g2),
                    "mean_1": float(np.mean(g1)),
                    "mean_2": float(np.mean(g2)),
                    "sd_1": float(np.std(g1, ddof=1)),
                    "sd_2": float(np.std(g2, ddof=1)),
                    "t_stat": float(t_stat),
                    "t_p": float(t_p),
                    "u_stat": float(u_stat),
                    "u_p": float(u_p),
                    "cohens_d": float(d),
                }
                all_results.append(result)

    # Multiple comparisons correction
    print(f"\nTotal hypotheses tested: {total_hypotheses}")
    bonferroni_threshold = 0.05 / total_hypotheses if total_hypotheses > 0 else 0.05

    # Sort by p-value for BH correction
    all_results.sort(key=lambda r: r["t_p"])
    for i, r in enumerate(all_results):
        r["bonferroni_sig"] = r["t_p"] < bonferroni_threshold
        bh_threshold = ((i + 1) / total_hypotheses) * 0.05 if total_hypotheses > 0 else 0.05
        r["bh_sig"] = r["t_p"] <= bh_threshold

    # Report significant results
    bonferroni_sig = [r for r in all_results if r["bonferroni_sig"]]
    bh_sig = [r for r in all_results if r["bh_sig"]]

    print(f"\n=== Significant Results (Bonferroni, threshold={bonferroni_threshold:.6f}) ===")
    if bonferroni_sig:
        for r in bonferroni_sig:
            print(
                f"  {r['metric']}: {r['parameter']}={r['value_1']} vs {r['value_2']}"
                f" — p={r['t_p']:.6f}, d={r['cohens_d']:.2f}"
            )
    else:
        print("  No results survive Bonferroni correction")

    print("\n=== Significant Results (Benjamini-Hochberg) ===")
    if bh_sig:
        for r in bh_sig:
            print(
                f"  {r['metric']}: {r['parameter']}={r['value_1']} vs {r['value_2']}"
                f" — p={r['t_p']:.6f}, d={r['cohens_d']:.2f}"
            )
    else:
        print("  No results survive BH correction")

    # Normality checks (Shapiro-Wilk on welfare for each tax rate group)
    print("\n=== Normality (Shapiro-Wilk on welfare) ===")
    normality_results = []
    tax_param = "governance.transaction_tax_rate"
    if tax_param in sweep_params:
        for val in sorted(df[tax_param].unique()):
            group = df[df[tax_param] == val]["welfare"].dropna().values
            if len(group) >= 3:
                w_stat, w_p = stats.shapiro(group)
                is_normal = w_p > 0.05
                label = "NORMAL" if is_normal else "NON-NORMAL"
                print(f"  tax_rate={val}: W={w_stat:.4f}, p={w_p:.4f} — {label}")
                normality_results.append({
                    "group": f"tax_rate={val}",
                    "W": float(w_stat),
                    "p": float(w_p),
                    "normal": is_normal,
                })

    # Agent-type stratification: honest vs RLM
    print("\n=== Agent-Type Stratification (Honest vs RLM) ===")
    agent_strat = []
    honest_vals = df["honest_payoff"].dropna().values
    rlm_vals = df["rlm_payoff"].dropna().values
    print(f"  Honest: mean={np.mean(honest_vals):.2f}, sd={np.std(honest_vals, ddof=1):.2f}")
    print(f"  RLM:    mean={np.mean(rlm_vals):.2f}, sd={np.std(rlm_vals, ddof=1):.2f}")

    t_stat, t_p = stats.ttest_rel(honest_vals, rlm_vals)
    d = cohens_d(honest_vals, rlm_vals)
    sig_marker = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else ""
    print(f"  Honest vs RLM: d={d:.2f}{sig_marker}, p={t_p:.6f}")
    agent_strat.append({
        "type_1": "honest",
        "type_2": "rlm",
        "mean_1": float(np.mean(honest_vals)),
        "mean_2": float(np.mean(rlm_vals)),
        "cohens_d": float(d),
        "t_p": float(t_p),
    })

    # Honest vs RLM payoff ratio by tax rate
    print("\n=== Honest/RLM Payoff Ratio by Tax Rate ===")
    for tax in sorted(df[tax_param].unique()):
        subset = df[df[tax_param] == tax]
        h = subset["honest_payoff"].mean()
        r = subset["rlm_payoff"].mean()
        ratio = h / r if r != 0 else float("inf")
        print(f"  tax={tax}: honest={h:.1f}, rlm={r:.1f}, ratio={ratio:.2f}")

    # Top 10 results by effect size
    print("\n=== Top 10 by Effect Size ===")
    by_effect = sorted(all_results, key=lambda r: abs(r["cohens_d"]), reverse=True)
    for r in by_effect[:10]:
        sig = ""
        if r["bonferroni_sig"]:
            sig = " [BONF]"
        elif r["bh_sig"]:
            sig = " [BH]"
        print(
            f"  d={r['cohens_d']:+.2f} | {r['metric']}: "
            f"{r['parameter']}={r['value_1']} vs {r['value_2']} "
            f"(p={r['t_p']:.4f}){sig}"
        )

    # Summary JSON
    summary = {
        "csv": str(CSV_PATH),
        "total_runs": len(df),
        "total_hypotheses": total_hypotheses,
        "n_bonferroni_significant": len(bonferroni_sig),
        "n_bh_significant": len(bh_sig),
        "swept_parameters": {
            p: sorted(df[p].unique().tolist()) for p in sweep_params
        },
        "results": all_results,
        "agent_stratification": agent_strat,
        "normality": normality_results,
    }

    summary_path = RUN_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_default)
    print(f"\nSummary written to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
