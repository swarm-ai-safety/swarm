#!/bin/bash

python3 << 'EOF'
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import json
import os

# Create output directory
os.makedirs("/root/output", exist_ok=True)

# Load data
df = pd.read_csv("/root/data/sweep_results.csv")

# Normalize column aliases
alias_map = {
    "tax_rate": "transaction_tax_rate",
    "tax": "transaction_tax_rate",
    "tox": "toxicity_rate",
    "toxicity": "toxicity_rate",
}
df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns}, inplace=True)

# Aggregate by (transaction_tax_rate, seed) if agent_type column exists
if "agent_type" in df.columns:
    df = df.groupby(["transaction_tax_rate", "seed"])["welfare"].mean().reset_index()

param_col = "transaction_tax_rate"
metric = "welfare"

# Group data by parameter values
groups = {val: grp[metric].values for val, grp in df.groupby(param_col)}
pairs = list(combinations(sorted(groups.keys()), 2))

# Cohen's d function
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx <= 1 or ny <= 1:
        return 0.0
    pooled_std = np.sqrt(((nx-1)*np.std(x,ddof=1)**2 + (ny-1)*np.std(y,ddof=1)**2) / (nx+ny-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std

# Perform pairwise Welch's t-tests
results = []
for a, b in pairs:
    t_stat, p_val = stats.ttest_ind(groups[a], groups[b], equal_var=False)
    d = cohens_d(groups[a], groups[b])
    
    effect_magnitude = (
        "large" if abs(d) >= 0.8 else
        "medium" if abs(d) >= 0.5 else
        "small" if abs(d) >= 0.2 else
        "negligible"
    )
    
    results.append({
        "group_a": float(a),
        "group_b": float(b),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(d),
        "effect_magnitude": effect_magnitude
    })

# Bonferroni correction
n_tests = len(results)
bonferroni_threshold = 0.05 / n_tests

for r in results:
    r["bonferroni_significant"] = r["p_value"] < bonferroni_threshold

# Shapiro-Wilk normality tests
normality = {}
for val, data in groups.items():
    if len(data) >= 3:
        w_stat, p_val = stats.shapiro(data)
        normality[float(val)] = {
            "W_statistic": float(w_stat),
            "p_value": float(p_val),
            "normal_at_0.05": p_val > 0.05,
        }

# Create summary
summary = {
    "metric_analyzed": metric,
    "parameter_column": param_col,
    "total_hypotheses": n_tests,
    "bonferroni_threshold": bonferroni_threshold,
    "n_bonferroni_significant": sum(1 for r in results if r["bonferroni_significant"]),
    "n_nominal_significant": sum(1 for r in results if r["p_value"] < 0.05),
    "results": results,
    "normality_tests": normality,
}

# Save JSON summary
with open("/root/output/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Create human-readable report
with open("/root/output/results.txt", "w") as f:
    f.write(f"Statistical Analysis Report\n")
    f.write(f"={'='*40}\n")
    f.write(f"Metric: {metric}\n")
    f.write(f"Parameter: {param_col}\n")
    f.write(f"Total comparisons: {n_tests}\n")
    f.write(f"Bonferroni threshold: {bonferroni_threshold:.6f}\n")
    f.write(f"Nominal significant (p<0.05): {sum(1 for r in results if r['p_value'] < 0.05)}\n")
    f.write(f"Bonferroni significant: {sum(1 for r in results if r['bonferroni_significant'])}\n\n")
    
    f.write("Pairwise Comparisons:\n")
    f.write("-" * 20 + "\n")
    for r in results:
        sig = "*" if r["bonferroni_significant"] else ""
        f.write(f"{r['group_a']} vs {r['group_b']}: "
                f"t={r['t_statistic']:.3f}, p={r['p_value']:.6f}{sig}, "
                f"d={r['cohens_d']:.3f} ({r['effect_magnitude']})\n")
    
    f.write(f"\nNormality Tests (Shapiro-Wilk):\n")
    f.write("-" * 30 + "\n")
    for val, test in normality.items():
        normal_str = "Normal" if test["normal_at_0.05"] else "Non-normal"
        f.write(f"Group {val}: W={test['W_statistic']:.4f}, p={test['p_value']:.6f} ({normal_str})\n")
    
    f.write(f"\nSample Sizes:\n")
    f.write("-" * 13 + "\n")
    for val, data in groups.items():
        f.write(f"Group {val}: n={len(data)}\n")

print("Analysis complete. Results saved to /root/output/")
EOF