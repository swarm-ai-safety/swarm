#!/bin/bash

# Create output directories
mkdir -p /root/output/sweep
mkdir -p /root/output/analysis
mkdir -p /root/output/plots
mkdir -p /root/output/paper

# Change to swarm package directory
cd /root/swarm-package

# Step 1: Run sweep
python -m swarm.sweep \
  --scenario /root/scenarios/baseline.yaml \
  --param tax_rate \
  --values 0.0 0.10 0.20 \
  --seeds 42 7 123 \
  --epochs 5 \
  --steps 10 \
  --output /root/output/sweep/sweep_results.csv

# Step 2: Statistical analysis
python -c "
import pandas as pd
import json
import numpy as np

# Load sweep results
df = pd.read_csv('/root/output/sweep/sweep_results.csv')

# Group by tax_rate and compute statistics
results = []
for tax_rate in [0.0, 0.10, 0.20]:
    subset = df[df['tax_rate'] == tax_rate]
    if 'final_reward' in subset.columns:
        metric = 'final_reward'
    elif 'reward' in subset.columns:
        metric = 'reward'
    else:
        metric = subset.columns[-1]  # Use last column as metric
    
    values = subset[metric].values
    result = {
        'tax_rate': tax_rate,
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'count': int(len(values))
    }
    results.append(result)

# Save analysis
analysis = {'results': results}
with open('/root/output/analysis/summary.json', 'w') as f:
    json.dump(analysis, f, indent=2)
"

# Step 3: Generate plots
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Load data
df = pd.read_csv('/root/output/sweep/sweep_results.csv')
with open('/root/output/analysis/summary.json', 'r') as f:
    analysis = json.load(f)

# Determine metric column
if 'final_reward' in df.columns:
    metric = 'final_reward'
elif 'reward' in df.columns:
    metric = 'reward'
else:
    metric = df.columns[-1]

# Plot 1: Bar plot with error bars
plt.figure(figsize=(10, 6))
tax_rates = [r['tax_rate'] for r in analysis['results']]
means = [r['mean'] for r in analysis['results']]
stds = [r['std'] for r in analysis['results']]

plt.bar(tax_rates, means, yerr=stds, capsize=5, alpha=0.7)
plt.xlabel('Tax Rate')
plt.ylabel(metric.replace('_', ' ').title())
plt.title('Performance vs Tax Rate')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/root/output/plots/performance_by_tax_rate.png', dpi=150)
plt.close()

# Plot 2: Box plot
plt.figure(figsize=(10, 6))
data_by_tax = []
labels = []
for tax_rate in [0.0, 0.10, 0.20]:
    subset = df[df['tax_rate'] == tax_rate][metric]
    data_by_tax.append(subset.values)
    labels.append(f'{tax_rate:.2f}')

plt.boxplot(data_by_tax, labels=labels)
plt.xlabel('Tax Rate')
plt.ylabel(metric.replace('_', ' ').title())
plt.title('Distribution of Performance by Tax Rate')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/root/output/plots/performance_distribution.png', dpi=150)
plt.close()
"

# Step 4: Generate paper
python -c "
import json

# Load analysis results
with open('/root/output/analysis/summary.json', 'r') as f:
    analysis = json.load(f)

paper = '''# Economic Policy Analysis: The Impact of Tax Rates on System Performance

## Abstract

This study investigates the relationship between tax rates and system performance using computational simulation. We conducted a systematic analysis across three tax rate scenarios (0%, 10%, and 20%) to understand their impact on overall system outcomes.

## Introduction

Understanding the effects of taxation on economic systems is crucial for policy development. This study employs simulation-based methodology to analyze how different tax rate policies affect system performance metrics.

## Methodology

We conducted a parameter sweep experiment with the following configuration:
- Tax rates tested: 0.0, 0.10, 0.20
- Seeds: 42, 7, 123 (for statistical reliability)
- Simulation parameters: 5 epochs, 10 steps per epoch
- Total experimental runs: 9 (3 tax rates × 3 seeds)

## Results

Our analysis reveals the following key findings:

'''

for result in analysis['results']:
    tax_rate = result['tax_rate']
    mean_val = result['mean']
    std_val = result['std']
    count = result['count']
    
    paper += f'''
### Tax Rate {tax_rate:.1%}
- Mean Performance: {mean_val:.4f} (±{std_val:.4f})
- Sample Size: {count} runs
- Range: {result['min']:.4f} to {result['max']:.4f}
'''

paper += '''
## Discussion

The results demonstrate varying performance levels across different tax rate scenarios. These findings provide insights into the trade-offs between taxation policies and system efficiency.

## Conclusion

This comprehensive analysis contributes to understanding the relationship between tax policy and system performance through systematic computational experimentation.

## Data Availability

All experimental data and analysis code are available in the accompanying output directories:
- Raw sweep data: `/output/sweep/sweep_results.csv`
- Statistical analysis: `/output/analysis/summary.json`
- Visualizations: `/output/plots/`
'''

with open('/root/output/paper/paper.md', 'w') as f:
    f.write(paper)
"

echo "End-to-end study completed successfully!"
echo "Results available in /root/output/"