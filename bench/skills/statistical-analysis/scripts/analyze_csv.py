#!/usr/bin/env python3
"""Standalone statistical analysis script for SWARM sweep CSVs.

Usage:
    python analyze_csv.py <input_csv> <output_dir> [--metric welfare] [--param transaction_tax_rate]
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(x, y):
    """Compute Cohen's d effect size."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def analyze(csv_path: str, output_dir: str, metric: str = "welfare", param_col: str = "transaction_tax_rate"):
    """Run full statistical analysis pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Normalize aliases
    alias_map = {"tax_rate": "transaction_tax_rate", "tax": "transaction_tax_rate",
                 "tox": "toxicity_rate", "toxicity": "toxicity_rate"}
    df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns}, inplace=True)

    # If agent_type column exists, aggregate first
    if "agent_type" in df.columns:
        df = df.groupby([param_col, "seed"])[metric].mean().reset_index()

    groups = {val: grp[metric].values for val, grp in df.groupby(param_col)}
    pairs = list(combinations(sorted(groups.keys()), 2))

    # Pairwise tests
    results = []
    for a, b in pairs:
        t_stat, p_val = stats.ttest_ind(groups[a], groups[b], equal_var=False)
        d = cohens_d(groups[a], groups[b])
        results.append({
            "group_a": float(a), "group_b": float(b),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "cohens_d": round(d, 4),
            "effect_magnitude": (
                "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else
                "small" if abs(d) >= 0.2 else "negligible"
            ),
        })

    n_tests = len(results)
    bonferroni_threshold = 0.05 / n_tests if n_tests > 0 else 0.05

    for r in results:
        r["bonferroni_significant"] = r["p_value"] < bonferroni_threshold

    # Normality tests
    normality = {}
    for val, data in groups.items():
        if len(data) >= 3:
            w_stat, p_val = stats.shapiro(data)
            normality[str(float(val))] = {
                "W_statistic": round(float(w_stat), 4),
                "p_value": round(float(p_val), 6),
                "normal_at_0.05": bool(p_val > 0.05),
            }

    summary = {
        "metric_analyzed": metric,
        "parameter_column": param_col,
        "total_hypotheses": n_tests,
        "bonferroni_threshold": round(bonferroni_threshold, 6),
        "n_bonferroni_significant": sum(1 for r in results if r["bonferroni_significant"]),
        "n_nominal_significant": sum(1 for r in results if r["p_value"] < 0.05),
        "results": results,
        "normality_tests": normality,
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable report
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write("Statistical Analysis Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Metric: {metric}\n")
        f.write(f"Parameter: {param_col}\n")
        f.write(f"Total comparisons: {n_tests}\n")
        f.write(f"Bonferroni threshold: {bonferroni_threshold:.4f}\n\n")
        for r in results:
            sig = " *" if r["bonferroni_significant"] else ""
            f.write(f"{r['group_a']} vs {r['group_b']}: "
                    f"t={r['t_statistic']:.3f}, p={r['p_value']:.4f}{sig}, "
                    f"d={r['cohens_d']:.3f} ({r['effect_magnitude']})\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SWARM sweep CSV")
    parser.add_argument("csv_path", help="Path to sweep CSV")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--metric", default="welfare", help="Metric to analyze")
    parser.add_argument("--param", default="transaction_tax_rate", help="Parameter column")
    args = parser.parse_args()

    result = analyze(args.csv_path, args.output_dir, args.metric, args.param)
    print(f"Analysis complete: {result['n_bonferroni_significant']}/{result['total_hypotheses']} significant after Bonferroni")
