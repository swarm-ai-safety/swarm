#!/bin/bash
set -e

python3 << 'PYEOF'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

df = pd.read_csv('/root/data/sweep_results.csv')
param_col = 'transaction_tax_rate'
out = '/root/output/plots'
os.makedirs(out, exist_ok=True)

# 1. Grouped bar chart
summary = df.groupby(param_col)['welfare'].agg(['mean', 'std']).reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(summary))
ax.bar(x, summary['mean'], yerr=summary['std'], capsize=5, color='steelblue', edgecolor='black', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'{v:.2f}' for v in summary[param_col]])
ax.set_xlabel('Transaction Tax Rate')
ax.set_ylabel('Welfare (mean +/- SD)')
ax.set_title('Welfare by Governance Configuration')
plt.tight_layout()
plt.savefig(os.path.join(out, 'welfare_by_config.png'), dpi=150)
plt.close()

# 2. Box plot
fig, ax = plt.subplots(figsize=(8, 5))
groups = sorted(df[param_col].unique())
data = [df[df[param_col] == g]['welfare'].values for g in groups]
bp = ax.boxplot(data, labels=[f'{g:.2f}' for g in groups], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_xlabel('Transaction Tax Rate')
ax.set_ylabel('Welfare')
ax.set_title('Welfare Distribution by Configuration')
plt.tight_layout()
plt.savefig(os.path.join(out, 'welfare_boxplot.png'), dpi=150)
plt.close()

# 3. Agent payoff comparison
payoff_cols = [c for c in df.columns if c.startswith('mean_payoff_')]
if payoff_cols:
    summary2 = df.groupby(param_col)[payoff_cols].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary2))
    width = 0.25
    for i, col in enumerate(payoff_cols):
        label = col.replace('mean_payoff_', '').title()
        ax.bar(x + i * width, summary2[col], width, label=label, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{v:.2f}' for v in summary2[param_col]])
    ax.set_xlabel('Transaction Tax Rate')
    ax.set_ylabel('Mean Payoff')
    ax.set_title('Agent Payoff Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'agent_payoff_comparison.png'), dpi=150)
    plt.close()

print(f'Generated {len(os.listdir(out))} plots')
PYEOF
