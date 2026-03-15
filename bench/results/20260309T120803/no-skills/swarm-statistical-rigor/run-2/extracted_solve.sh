#!/bin/bash

python3 << 'EOF'
import pandas as pd
import numpy as np
from scipy import stats
import json
import itertools
import os

# Load data
df = pd.read_csv('/root/data/sweep_results.csv')

# Check if agent_type column exists and aggregate if needed
if 'agent_type' in df.columns:
    # Aggregate by (transaction_tax_rate, seed) taking mean welfare
    df_agg = df.groupby(['transaction_tax_rate', 'seed'])['welfare'].mean().reset_index()
else:
    df_agg = df.copy()

# Group data by tax rate
tax_rates = sorted(df_agg['transaction_tax_rate'].unique())
groups = {}
for rate in tax_rates:
    groups[rate] = df_agg[df_agg['transaction_tax_rate'] == rate]['welfare'].values

# Perform pairwise Welch's t-tests
comparisons = list(itertools.combinations(tax_rates, 2))
results = []

for group_a, group_b in comparisons:
    data_a = groups[group_a]
    data_b = groups[group_b]
    
    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
    
    # Cohen's d effect size
    pooled_std = np.sqrt(((len(data_a) - 1) * np.var(data_a, ddof=1) + 
                         (len(data_b) - 1) * np.var(data_b, ddof=1)) / 
                        (len(data_a) + len(data_b) - 2))
    cohens_d = (np.mean(data_a) - np.mean(data_b)) / pooled_std
    
    results.append({
        'group_a': float(group_a),
        'group_b': float(group_b),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d)
    })

# Bonferroni correction
total_hypotheses = len(results)
bonferroni_threshold = 0.05 / total_hypotheses
n_bonferroni_significant = sum(1 for r in results if r['p_value'] < bonferroni_threshold)

# Add bonferroni significance to results
for r in results:
    r['bonferroni_significant'] = r['p_value'] < bonferroni_threshold

# Shapiro-Wilk normality tests
normality_tests = {}
for rate in tax_rates:
    data = groups[rate]
    if len(data) >= 3:  # Shapiro-Wilk requires at least 3 observations
        stat, p_value = stats.shapiro(data)
        normality_tests[str(float(rate))] = {
            'statistic': float(stat),
            'p_value': float(p_value),
            'normal': p_value > 0.05
        }
    else:
        normality_tests[str(float(rate))] = {
            'statistic': None,
            'p_value': None,
            'normal': None
        }

# Create output directory
os.makedirs('/root/output', exist_ok=True)

# Save summary.json
summary = {
    'total_hypotheses': total_hypotheses,
    'bonferroni_threshold': bonferroni_threshold,
    'n_bonferroni_significant': n_bonferroni_significant,
    'results': results,
    'normality_tests': normality_tests
}

with open('/root/output/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Create human-readable report
report_lines = []
report_lines.append("Statistical Analysis Report")
report_lines.append("=" * 50)
report_lines.append(f"Total number of pairwise comparisons: {total_hypotheses}")
report_lines.append(f"Bonferroni corrected significance threshold: {bonferroni_threshold:.6f}")
report_lines.append(f"Number of significant comparisons after correction: {n_bonferroni_significant}")
report_lines.append("")

report_lines.append("Pairwise Comparisons (Welch's t-tests):")
report_lines.append("-" * 40)
for r in results:
    sig_marker = "*" if r['bonferroni_significant'] else ""
    report_lines.append(f"Tax rate {r['group_a']:.3f} vs {r['group_b']:.3f}:")
    report_lines.append(f"  p-value: {r['p_value']:.6f}{sig_marker}")
    report_lines.append(f"  Cohen's d: {r['cohens_d']:.4f}")
    report_lines.append("")

report_lines.append("Normality Tests (Shapiro-Wilk):")
report_lines.append("-" * 35)
for rate_str, test_result in normality_tests.items():
    rate = float(rate_str)
    if test_result['p_value'] is not None:
        normal_str = "Normal" if test_result['normal'] else "Not Normal"
        report_lines.append(f"Tax rate {rate:.3f}: p-value = {test_result['p_value']:.6f} ({normal_str})")
    else:
        report_lines.append(f"Tax rate {rate:.3f}: Insufficient data for test")

with open('/root/output/results.txt', 'w') as f:
    f.write('\n'.join(report_lines))

EOF