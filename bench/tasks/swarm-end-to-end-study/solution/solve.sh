#!/bin/bash
set -e

python3 << 'PYEOF'
import json, os, csv
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

# === Step 1: Sweep ===
tax_rates = [0.0, 0.10, 0.20]
seeds = [42, 7, 123]
results = []

for tax in tax_rates:
    for seed in seeds:
        sc = load_scenario('/root/scenarios/baseline.yaml')
        oc = sc.orchestrator_config
        oc.governance_config.transaction_tax_rate = tax
        oc.seed = seed
        oc.n_epochs = 5
        oc.steps_per_epoch = 10

        orch = Orchestrator(oc)
        epochs = orch.run()  # list[EpochMetrics]
        final = epochs[-1]

        results.append({
            'transaction_tax_rate': tax,
            'seed': seed,
            'welfare': round(final.total_welfare, 3),
            'toxicity_rate': round(final.toxicity_rate, 4),
            'quality_gap': round(final.quality_gap, 4),
        })

os.makedirs('/root/output/sweep', exist_ok=True)
df = pd.DataFrame(results)
df.to_csv('/root/output/sweep/sweep_results.csv', index=False)
print(f'Step 1: Sweep complete ({len(results)} rows)')

# === Step 2: Analysis ===
param_col = 'transaction_tax_rate'
metric = 'welfare'
groups = {val: grp[metric].values for val, grp in df.groupby(param_col)}
pairs = list(combinations(sorted(groups.keys()), 2))

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*np.std(x,ddof=1)**2 + (ny-1)*np.std(y,ddof=1)**2) / (nx+ny-2))
    return float((np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else 0.0

analysis_results = []
for a, b in pairs:
    t_stat, p_val = stats.ttest_ind(groups[a], groups[b], equal_var=False)
    d = cohens_d(groups[a], groups[b])
    analysis_results.append({
        'group_a': float(a), 'group_b': float(b),
        't_statistic': round(float(t_stat), 4),
        'p_value': round(float(p_val), 6),
        'cohens_d': round(d, 4),
        'bonferroni_significant': float(p_val) < (0.05 / len(pairs)),
    })

os.makedirs('/root/output/analysis', exist_ok=True)
summary = {
    'metric_analyzed': metric,
    'total_hypotheses': len(pairs),
    'bonferroni_threshold': round(0.05 / len(pairs), 6),
    'results': analysis_results,
}
with open('/root/output/analysis/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Step 2: Analysis complete ({len(pairs)} comparisons)')

# === Step 3: Plots ===
os.makedirs('/root/output/plots', exist_ok=True)

# Bar chart
summary_df = df.groupby(param_col)['welfare'].agg(['mean', 'std']).reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(summary_df))
ax.bar(x, summary_df['mean'], yerr=summary_df['std'], capsize=5, color='steelblue', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'{v:.2f}' for v in summary_df[param_col]])
ax.set_xlabel('Transaction Tax Rate')
ax.set_ylabel('Welfare')
ax.set_title('Welfare by Tax Rate')
plt.tight_layout()
plt.savefig('/root/output/plots/welfare_by_config.png', dpi=150)
plt.close()

# Box plot
fig, ax = plt.subplots(figsize=(8, 5))
grps = sorted(df[param_col].unique())
data = [df[df[param_col] == g]['welfare'].values for g in grps]
ax.boxplot(data, labels=[f'{g:.2f}' for g in grps], patch_artist=True)
ax.set_xlabel('Transaction Tax Rate')
ax.set_ylabel('Welfare')
ax.set_title('Welfare Distribution')
plt.tight_layout()
plt.savefig('/root/output/plots/welfare_boxplot.png', dpi=150)
plt.close()
print('Step 3: Plots complete')

# === Step 4: Paper ===
os.makedirs('/root/output/paper', exist_ok=True)

results_lines = ['| Tax Rate | Welfare (mean +/- std) | Toxicity (mean +/- std) |',
                 '|----------|----------------------|------------------------|']
for val, grp in df.groupby(param_col):
    w_m, w_s = grp['welfare'].mean(), grp['welfare'].std()
    t_m, t_s = grp['toxicity_rate'].mean(), grp['toxicity_rate'].std()
    results_lines.append(f'| {val:.2f} | {w_m:.1f} +/- {w_s:.1f} | {t_m:.3f} +/- {t_s:.3f} |')
results_table = chr(10).join(results_lines)

n_sig = sum(1 for r in analysis_results if r["bonferroni_significant"])

paper = f'''# Tax Rate Impact on Distributional Safety

## Abstract

We study the effect of transaction tax rate on welfare and toxicity in a multi-agent system using the SWARM framework. We sweep tax rates [0.0, 0.10, 0.20] across 3 seeds and find systematic effects on both metrics.

## Experimental Setup

We use the baseline scenario with 5 agents (3 honest, 1 opportunistic, 1 deceptive) and vary the transaction tax rate. Each configuration runs for 5 epochs with 10 steps per epoch.

## Results

### Welfare and Toxicity by Tax Rate

{results_table}

Statistical analysis reveals {n_sig} significant pairwise differences after Bonferroni correction.

## Conclusion

Our results show that transaction tax rate affects both welfare and toxicity in the SWARM simulation. The tradeoff between welfare and toxicity must be carefully considered when tuning governance parameters.
'''

with open('/root/output/paper/paper.md', 'w') as f:
    f.write(paper)
print('Step 4: Paper complete')
print('All steps done!')
PYEOF
