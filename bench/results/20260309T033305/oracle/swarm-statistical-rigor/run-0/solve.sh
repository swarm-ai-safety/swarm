#!/bin/bash
set -e

python3 << 'PYEOF'
import json, os
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

df = pd.read_csv('/root/data/sweep_results.csv')
param_col = 'transaction_tax_rate'
metric = 'welfare'

# Aggregate if agent_type column exists
if 'agent_type' in df.columns:
    df = df.groupby([param_col, 'seed'])[metric].mean().reset_index()

groups = {val: grp[metric].values for val, grp in df.groupby(param_col)}
pairs = list(combinations(sorted(groups.keys()), 2))

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*np.std(x,ddof=1)**2 + (ny-1)*np.std(y,ddof=1)**2) / (nx+ny-2))
    return float((np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else 0.0

results = []
for a, b in pairs:
    t_stat, p_val = stats.ttest_ind(groups[a], groups[b], equal_var=False)
    d = cohens_d(groups[a], groups[b])
    results.append({
        'group_a': float(a), 'group_b': float(b),
        't_statistic': round(float(t_stat), 4),
        'p_value': round(float(p_val), 6),
        'cohens_d': round(d, 4),
        'effect_magnitude': (
            'large' if abs(d) >= 0.8 else
            'medium' if abs(d) >= 0.5 else
            'small' if abs(d) >= 0.2 else 'negligible'
        ),
    })

n_tests = len(results)
bonf = 0.05 / n_tests if n_tests > 0 else 0.05
for r in results:
    r['bonferroni_significant'] = r['p_value'] < bonf

normality = {}
for val, data in groups.items():
    if len(data) >= 3:
        w, p = stats.shapiro(data)
        normality[str(float(val))] = {
            'W_statistic': round(float(w), 4),
            'p_value': round(float(p), 6),
            'normal_at_0.05': bool(p > 0.05),
        }

summary = {
    'metric_analyzed': metric,
    'parameter_column': param_col,
    'total_hypotheses': n_tests,
    'bonferroni_threshold': round(bonf, 6),
    'n_bonferroni_significant': sum(1 for r in results if r['bonferroni_significant']),
    'n_nominal_significant': sum(1 for r in results if r['p_value'] < 0.05),
    'results': results,
    'normality_tests': normality,
}

os.makedirs('/root/output', exist_ok=True)
with open('/root/output/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

with open('/root/output/results.txt', 'w') as f:
    f.write('Statistical Analysis Report\n')
    f.write('=' * 40 + '\n')
    f.write(f'Metric: {metric}\n')
    f.write(f'Parameter: {param_col}\n')
    f.write(f'Total comparisons: {n_tests}\n')
    f.write(f'Bonferroni threshold: {bonf:.4f}\n\n')
    for r in results:
        sig = ' *' if r['bonferroni_significant'] else ''
        f.write(
            f"{r['group_a']} vs {r['group_b']}: "
            f"t={r['t_statistic']:.3f}, p={r['p_value']:.4f}{sig}, "
            f"d={r['cohens_d']:.3f} ({r['effect_magnitude']})\n"
        )

print(f"Done: {summary['n_bonferroni_significant']}/{n_tests} significant after Bonferroni")
PYEOF
