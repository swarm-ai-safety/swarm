---
name: statistical-analysis
description: Perform rigorous statistical analysis on SWARM experiment data with multiple-comparison corrections
version: "1.0"
domain: swarm-safety
triggers:
  - statistical analysis
  - analyze results
  - compute significance
  - effect size
---

# Statistical Analysis Skill

Perform rigorous statistical analysis on SWARM sweep or multi-seed data, including hypothesis tests, effect sizes, and multiple-comparison corrections.

## Prerequisites

- `scipy>=1.10` for statistical tests
- `pandas>=2.0` for data manipulation
- `numpy>=1.24` for numerical computation

## Procedure

### 1. Load and normalize data

```python
import pandas as pd

df = pd.read_csv(csv_path)

# Normalize column aliases
alias_map = {
    "tax_rate": "transaction_tax_rate",
    "tax": "transaction_tax_rate",
    "tox": "toxicity_rate",
    "toxicity": "toxicity_rate",
}
df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns}, inplace=True)
```

### 2. Pairwise Welch's t-tests

```python
from scipy import stats
from itertools import combinations

param_col = "transaction_tax_rate"  # or detect automatically
metric = "welfare"

groups = {val: grp[metric].values for val, grp in df.groupby(param_col)}
pairs = list(combinations(sorted(groups.keys()), 2))

results = []
for a, b in pairs:
    t_stat, p_val = stats.ttest_ind(groups[a], groups[b], equal_var=False)
    results.append({
        "group_a": float(a),
        "group_b": float(b),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
    })
```

### 3. Effect sizes (Cohen's d)

```python
import numpy as np

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.std(x,ddof=1)**2 + (ny-1)*np.std(y,ddof=1)**2) / (nx+ny-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std

for r, (a, b) in zip(results, pairs):
    r["cohens_d"] = float(cohens_d(groups[a], groups[b]))
    r["effect_magnitude"] = (
        "large" if abs(r["cohens_d"]) >= 0.8 else
        "medium" if abs(r["cohens_d"]) >= 0.5 else
        "small" if abs(r["cohens_d"]) >= 0.2 else
        "negligible"
    )
```

### 4. Bonferroni correction

```python
n_tests = len(results)
bonferroni_threshold = 0.05 / n_tests

for r in results:
    r["bonferroni_significant"] = r["p_value"] < bonferroni_threshold
```

### 5. Shapiro-Wilk normality tests

```python
normality = {}
for val, data in groups.items():
    if len(data) >= 3:
        w_stat, p_val = stats.shapiro(data)
        normality[float(val)] = {
            "W_statistic": float(w_stat),
            "p_value": float(p_val),
            "normal_at_0.05": p_val > 0.05,
        }
```

### 6. Save results

```python
import json

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

with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Human-readable report
with open(os.path.join(output_dir, "results.txt"), "w") as f:
    f.write(f"Statistical Analysis Report\n")
    f.write(f"={'='*40}\n")
    f.write(f"Metric: {metric}\n")
    f.write(f"Parameter: {param_col}\n")
    f.write(f"Total comparisons: {n_tests}\n")
    f.write(f"Bonferroni threshold: {bonferroni_threshold:.4f}\n\n")
    for r in results:
        sig = "*" if r["bonferroni_significant"] else ""
        f.write(f"{r['group_a']} vs {r['group_b']}: "
                f"t={r['t_statistic']:.3f}, p={r['p_value']:.4f}{sig}, "
                f"d={r['cohens_d']:.3f} ({r['effect_magnitude']})\n")
```

## Statistical Rigor Checklist

1. Report exact p-values (not just < 0.05)
2. Always apply multiple-comparison correction (Bonferroni minimum)
3. Report effect sizes alongside p-values
4. Check normality assumptions (Shapiro-Wilk)
5. Use Welch's t-test (not Student's) — does not assume equal variance
6. Report sample sizes per group
7. Distinguish nominal significance from corrected significance

## References

See `references/stat-checklist.md` for the full rigor checklist.
