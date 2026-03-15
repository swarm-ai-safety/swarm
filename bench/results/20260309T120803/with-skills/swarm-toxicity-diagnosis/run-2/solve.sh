#!/bin/bash

python3 << 'EOF'
import pandas as pd
import numpy as np
import json
import os

# Read the CSV data
df = pd.read_csv('/root/data/sweep_results.csv')

# Normalize column names if needed
alias_map = {
    "tax_rate": "transaction_tax_rate",
    "tax": "transaction_tax_rate", 
    "tox": "toxicity_rate",
    "toxicity": "toxicity_rate",
}
df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns}, inplace=True)

# Determine toxicity column name
toxicity_col = None
for col in ['toxicity_rate', 'toxicity', 'tox']:
    if col in df.columns:
        toxicity_col = col
        break

# Group by transaction_tax_rate and calculate mean toxicity
grouped = df.groupby('transaction_tax_rate')[toxicity_col].agg(['mean', 'count']).reset_index()

# Find best and worst configurations
worst_idx = grouped['mean'].idxmax()
best_idx = grouped['mean'].idxmin()

worst_config = float(grouped.loc[worst_idx, 'transaction_tax_rate'])
best_config = float(grouped.loc[best_idx, 'transaction_tax_rate'])
worst_toxicity = float(grouped.loc[worst_idx, 'mean'])
best_toxicity = float(grouped.loc[best_idx, 'mean'])

# Generate recommendation
toxicity_reduction = ((worst_toxicity - best_toxicity) / worst_toxicity) * 100
recommendation = f"Set transaction tax rate to {best_config} to minimize toxicity. This reduces mean toxicity by {toxicity_reduction:.1f}% compared to the worst configuration ({worst_config})."

# Create output directory if it doesn't exist
os.makedirs('/root/output', exist_ok=True)

# Prepare results
results = {
    "worst_config": worst_config,
    "best_config": best_config,
    "worst_toxicity": worst_toxicity,
    "best_toxicity": best_toxicity,
    "recommendation": recommendation
}

# Write to JSON file
with open('/root/output/diagnosis.json', 'w') as f:
    json.dump(results, f, indent=2)

EOF