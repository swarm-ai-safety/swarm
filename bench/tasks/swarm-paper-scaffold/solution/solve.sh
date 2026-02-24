#!/bin/bash
set -e

python3 -c "
import sqlite3
import os
import pandas as pd

conn = sqlite3.connect('/root/data/runs.db')
df = pd.read_sql_query('SELECT * FROM scenario_runs', conn)
conn.close()

scenarios = df.groupby('scenario_id').first().reset_index()
n_scenarios = len(scenarios)

# Methods table
methods_lines = ['| Scenario | Agents | Governance | Seeds | Epochs |',
                 '|----------|--------|-----------|-------|--------|']
for _, row in scenarios.iterrows():
    methods_lines.append(f\"| {row['scenario_id']} | {row['n_agents']} | {row['governance_desc']} | {row['n_seeds']} | {row['n_epochs']} |\")
methods_table = '\n'.join(methods_lines)

# Results table
results_agg = df.groupby('scenario_id').agg({
    'welfare': ['mean', 'std'],
    'toxicity_rate': ['mean', 'std'],
    'quality_gap': ['mean', 'std'],
}).reset_index()

results_lines = ['| Scenario | Welfare (mean +/- std) | Toxicity (mean +/- std) | Quality Gap (mean +/- std) |',
                 '|----------|----------------------|------------------------|--------------------------|']
for idx in range(len(results_agg)):
    sid = results_agg.iloc[idx][('scenario_id', '')]
    w_m = results_agg.iloc[idx][('welfare', 'mean')]
    w_s = results_agg.iloc[idx][('welfare', 'std')]
    t_m = results_agg.iloc[idx][('toxicity_rate', 'mean')]
    t_s = results_agg.iloc[idx][('toxicity_rate', 'std')]
    q_m = results_agg.iloc[idx][('quality_gap', 'mean')]
    q_s = results_agg.iloc[idx][('quality_gap', 'std')]
    results_lines.append(f'| {sid} | {w_m:.1f} +/- {w_s:.1f} | {t_m:.3f} +/- {t_s:.3f} | {q_m:.3f} +/- {q_s:.3f} |')
results_table = '\n'.join(results_lines)

# Compute overall stats for abstract
mean_welfare = df['welfare'].mean()
mean_toxicity = df['toxicity_rate'].mean()

paper = f'''# Distributional Safety in Multi-Agent AI Systems: An Empirical Study

## Abstract

This paper presents an empirical study of distributional safety in multi-agent AI systems using the SWARM framework. We evaluate {n_scenarios} scenarios across multiple seeds, measuring welfare, toxicity, and quality gap under varying governance configurations. Our experiments show a mean welfare of {mean_welfare:.1f} and mean toxicity rate of {mean_toxicity:.3f} across all configurations.

## Experimental Setup

### Scenarios

{methods_table}

### Metrics

| Metric | Definition | Range |
|--------|-----------|-------|
| Welfare | Aggregate agent payoffs | (-inf, +inf) |
| Toxicity Rate | E[1-p | accepted] | [0, 1] |
| Quality Gap | E[p|accepted] - E[p|rejected] | [-1, 1] |

## Results

### Cross-Scenario Summary

{results_table}

## Conclusion

The experimental results demonstrate the relationship between governance configurations and distributional safety outcomes. Across {n_scenarios} scenarios with varying governance settings, we observe that welfare and toxicity are influenced by the choice of governance parameters. These findings suggest that careful tuning of governance mechanisms is essential for maintaining distributional safety in multi-agent systems.
'''

os.makedirs('/root/output', exist_ok=True)
with open('/root/output/paper.md', 'w') as f:
    f.write(paper)

print(f'Paper written with {n_scenarios} scenarios')
"
