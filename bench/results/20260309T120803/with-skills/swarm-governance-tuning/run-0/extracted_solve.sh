#!/bin/bash

# Create output directory
mkdir -p /root/output

# Create Python script for the sweep
cat > /root/sweep_script.py << 'EOF'
import sys
sys.path.insert(0, '/root/swarm-package')

import os
import json
import itertools
import pandas as pd
import numpy as np
from scipy import stats
from swarm.scenarios.loader import load_scenario, build_orchestrator

# Parameters for sweep
scenario_path = "/root/scenarios/baseline.yaml"
tax_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
seeds = [42, 7, 123, 99, 256]
epochs = 8
steps = 10

# Run sweep
results = []
configs = list(itertools.product(tax_rates, seeds))

for tax_rate, seed in configs:
    sc = load_scenario(scenario_path)
    
    # Override parameters
    sc.orchestrator_config.governance_config.transaction_tax_rate = tax_rate
    sc.orchestrator_config.seed = seed
    sc.orchestrator_config.n_epochs = epochs
    sc.orchestrator_config.steps_per_epoch = steps
    
    orch = build_orchestrator(sc)
    epochs_result = orch.run()
    
    final = epochs_result[-1]
    results.append({
        "transaction_tax_rate": tax_rate,
        "seed": seed,
        "welfare": final.welfare,
        "toxicity_rate": final.toxicity_rate,
    })

# Create DataFrame
df = pd.DataFrame(results)

# Group by tax rate and compute means
summary_configs = []
for tax_rate, group in df.groupby("transaction_tax_rate"):
    mean_welfare = group["welfare"].mean()
    mean_toxicity = group["toxicity_rate"].mean()
    
    summary_configs.append({
        "tax_rate": tax_rate,
        "mean_welfare": mean_welfare,
        "mean_toxicity": mean_toxicity,
        "welfare_values": group["welfare"].values.tolist()
    })

# Filter configs with acceptable toxicity (< 0.50)
acceptable_configs = [c for c in summary_configs if c["mean_toxicity"] < 0.50]

if not acceptable_configs:
    raise ValueError("No configurations with toxicity < 0.50")

# Find optimal configuration (max welfare among acceptable)
optimal_config = max(acceptable_configs, key=lambda x: x["mean_welfare"])

# Get baseline (tax_rate = 0.0) for statistical comparison
baseline_config = next(c for c in summary_configs if c["tax_rate"] == 0.0)

# Perform t-test between optimal and baseline
optimal_welfare_values = optimal_config["welfare_values"]
baseline_welfare_values = baseline_config["welfare_values"]

t_stat, p_value = stats.ttest_ind(
    optimal_welfare_values, 
    baseline_welfare_values, 
    equal_var=False
)

# Create output
output = {
    "optimal_tax_rate": float(optimal_config["tax_rate"]),
    "mean_welfare": float(optimal_config["mean_welfare"]),
    "mean_toxicity": float(optimal_config["mean_toxicity"]),
    "n_configs_tested": len([c for c in summary_configs if c["tax_rate"] != 0.0]),
    "statistical_confidence": float(p_value)
}

# Save result
with open("/root/output/optimal.json", "w") as f:
    json.dump(output, f, indent=2)

print("Sweep completed successfully")
print(f"Optimal tax rate: {output['optimal_tax_rate']}")
print(f"Mean welfare: {output['mean_welfare']:.4f}")
print(f"Mean toxicity: {output['mean_toxicity']:.4f}")
print(f"Statistical confidence (p-value): {output['statistical_confidence']:.4f}")
EOF

# Run the sweep
cd /root
python sweep_script.py