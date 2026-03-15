#!/bin/bash

python3 << 'EOF'
import pandas as pd
import numpy as np
import json
import os

# Create output directory if it doesn't exist
os.makedirs('/root/output', exist_ok=True)

# Load the sweep results
df = pd.read_csv('/root/data/sweep_results.csv')

# Normalize column names
alias_map = {
    "tax_rate": "transaction_tax_rate",
    "tax": "transaction_tax_rate",
    "tox": "toxicity_rate",
    "toxicity": "toxicity_rate",
}
df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns}, inplace=True)

# Determine the correct toxicity column name
toxicity_col = None
for col in ['toxicity_rate', 'toxicity', 'tox']:
    if col in df.columns:
        toxicity_col = col
        break

if toxicity_col is None:
    raise ValueError("No toxicity column found in the data")

# Group by transaction_tax_rate and calculate mean toxicity
grouped = df.groupby('transaction_tax_rate')[toxicity_col].agg(['mean', 'count']).reset_index()
grouped.columns = ['tax_rate', 'mean_toxicity', 'count']

# Find worst and best configurations
worst_idx = grouped['mean_toxicity'].idxmax()
best_idx = grouped['mean_toxicity'].idxmin()

worst_config = float(grouped.loc[worst_idx, 'tax_rate'])
best_config = float(grouped.loc[best_idx, 'tax_rate'])
worst_toxicity = float(grouped.loc[worst_idx, 'mean_toxicity'])
best_toxicity = float(grouped.loc[best_idx, 'mean_toxicity'])

# Generate recommendation based on findings
toxicity_reduction = ((worst_toxicity - best_toxicity) / worst_toxicity) * 100
recommendation = f"Set transaction_tax_rate to {best_config} to minimize toxicity. This configuration reduces mean toxicity by {toxicity_reduction:.1f}% compared to the worst performing rate of {worst_config}."

# Create the diagnosis JSON
diagnosis = {
    "worst_config": worst_config,
    "best_config": best_config,
    "worst_toxicity": worst_toxicity,
    "best_toxicity": best_toxicity,
    "recommendation": recommendation
}

# Write to output file
with open('/root/output/diagnosis.json', 'w') as f:
    json.dump(diagnosis, f, indent=2)

print("Analysis complete. Results written to /root/output/diagnosis.json")
EOF