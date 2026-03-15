#!/bin/bash

python3 << 'EOF'
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import json
import os

# Create output directory
os.makedirs('/root/output', exist_ok=True)

# Load data
df = pd.read_csv('/root/data/sweep_results.csv')

# Check if agent_type column exists and aggregate if needed
if 'agent_type' in df.columns:
    # Aggregate by taking mean welfare per (transaction_tax_rate, seed) combination
    df_agg = df.groupby(['transaction_tax_rate', 'seed'])['welfare'].mean().reset_index()
else:
    df_agg = df.copy()

# Get unique tax rates
tax_rates = sorted(df_agg['transaction_tax_rate'].unique())

# Perform Shapiro-Wilk normality tests for each group
normality_tests = {}
for tax_rate in tax_rates:
    group_data = df_agg[df_agg['transaction_tax_rate'] == tax_rate]['welfare']
    stat, p_value = stats.shapiro(group_data)
    normality_tests[str(tax_rate)] = {
        'statistic': float(stat),
        'p_value': float(p_value),
        'is_normal': bool(p_value > 0.05)
    }

# Perform Welch's t-tests between all pairs
results = []
for tax_rate_a, tax_rate_b in combinations(tax_rates, 2):
    group_a = df_agg[df_agg['transaction_tax_rate'] == tax_rate_a]['welfare']
    group_b = df_agg[df_agg['transaction_tax_rate'] == tax_rate_b]['welfare']
    
    # Welch's t-test (equal_var=False)
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    
    # Cohen's d effect size
    pooled_std = np.sqrt(((len(group_a) - 1) * np.var(group_a, ddof=1) + 
                         (len(group_b) - 1) * np.var(group_b, ddof=1)) / 
                        (len(group_a) + len(group_b) - 2))
    cohens_d = (np.mean(group_a) - np.mean(group_b)) / pooled_std
    
    results.append({
        'group_a': str(tax_rate_a),
        'group_b': str(tax_rate_b),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        't_statistic': float(t_stat)
    })

# Calculate Bonferroni correction
total_hypotheses = len(results)
bonferroni_threshold = 0.05 / total_hypotheses

# Apply Bonferroni correction
n_bonferroni_significant = 0
for result in results:
    result['bonferroni_significant'] = bool(result['p_value'] < bonferroni_threshold)
    if result['bonferroni_significant']:
        n_bonferroni_significant += 1

# Create summary JSON
summary = {
    'total_hypotheses': total_hypotheses,
    'bonferroni_threshold': bonferroni_threshold,
    'n_bonferroni_significant': n_bonferroni_significant,
    'results': results,
    'normality_tests': normality_tests
}

# Save summary JSON
with open('/root/output/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Create human-readable report
with open('/root/output/results.txt', 'w') as f:
    f.write("FULL STATISTICAL ANALYSIS REPORT\n")
    f.write("=" * 50 + "\n\n")
    
    f.write(f"Dataset Overview:\n")
    f.write(f"- Total observations after aggregation: {len(df_agg)}\n")
    f.write(f"- Tax rates analyzed: {tax_rates}\n")
    f.write(f"- Total pairwise comparisons: {total_hypotheses}\n\n")
    
    f.write("NORMALITY TESTS (Shapiro-Wilk)\n")
    f.write("-" * 30 + "\n")
    for tax_rate, test in normality_tests.items():
        f.write(f"Tax Rate {tax_rate}: p={test['p_value']:.6f}, Normal: {test['is_normal']}\n")
    
    f.write(f"\nMULTIPLE COMPARISONS CORRECTION\n")
    f.write("-" * 30 + "\n")
    f.write(f"Bonferroni threshold: {bonferroni_threshold:.6f}\n")
    f.write(f"Significant after correction: {n_bonferroni_significant}/{total_hypotheses}\n\n")
    
    f.write("PAIRWISE COMPARISONS (Welch's t-test)\n")
    f.write("-" * 40 + "\n")
    for result in results:
        sig_marker = "***" if result['bonferroni_significant'] else ""
        f.write(f"Tax {result['group_a']} vs {result['group_b']}: ")
        f.write(f"p={result['p_value']:.6f}, Cohen's d={result['cohens_d']:.3f} {sig_marker}\n")
    
    f.write(f"\n*** = Significant after Bonferroni correction (α = {bonferroni_threshold:.6f})\n")

print("Analysis completed successfully!")
EOF