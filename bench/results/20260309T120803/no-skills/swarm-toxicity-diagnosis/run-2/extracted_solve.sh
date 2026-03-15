#!/bin/bash

python3 << 'EOF'
import pandas as pd
import json
import os

# Read the sweep results
df = pd.read_csv('/root/data/sweep_results.csv')

# Group by transaction_tax_rate and calculate mean toxicity
toxicity_by_config = df.groupby('transaction_tax_rate')['toxicity'].mean()

# Find best and worst configurations
worst_config = toxicity_by_config.idxmax()
best_config = toxicity_by_config.idxmin()
worst_toxicity = toxicity_by_config.max()
best_toxicity = toxicity_by_config.min()

# Create recommendation
recommendation = f"Set transaction tax rate to {best_config} to minimize toxicity. Avoid rates around {worst_config} which produce {worst_toxicity:.3f} mean toxicity vs {best_toxicity:.3f} at optimal rate."

# Create output dictionary
output = {
    "worst_config": float(worst_config),
    "best_config": float(best_config), 
    "worst_toxicity": float(worst_toxicity),
    "best_toxicity": float(best_toxicity),
    "recommendation": recommendation
}

# Ensure output directory exists
os.makedirs('/root/output', exist_ok=True)

# Write results to JSON
with open('/root/output/diagnosis.json', 'w') as f:
    json.dump(output, f, indent=2)

EOF