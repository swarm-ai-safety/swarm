#!/bin/bash

cd /root

mkdir -p output

python3 << 'EOF'
import sys
import os
import itertools
import pandas as pd
import json

sys.path.insert(0, '/root/swarm-package')

from swarm.scenarios.loader import load_scenario, build_orchestrator

# Define sweep parameters
param_name = "governance.transaction_tax_rate"
param_values = [0.0, 0.05, 0.10, 0.15]
seeds = [42, 7, 123]
epochs = 5
steps = 10
scenario_path = "/root/scenarios/baseline.yaml"
output_dir = "/root/output"

# Generate all configurations
configs = list(itertools.product(param_values, seeds))

results = []

for param_val, seed in configs:
    print(f"Running config: tax_rate={param_val}, seed={seed}")
    
    # Load scenario
    sc = load_scenario(scenario_path)
    
    # Override the swept parameter
    sc.orchestrator_config.governance_config.transaction_tax_rate = param_val
    
    # Override seed and epoch count
    sc.orchestrator_config.seed = seed
    sc.orchestrator_config.n_epochs = epochs
    sc.orchestrator_config.steps_per_epoch = steps
    
    # Build and run orchestrator
    orch = build_orchestrator(sc)
    epochs_result = orch.run()
    
    # Extract final epoch metrics
    final = epochs_result[-1]
    results.append({
        "transaction_tax_rate": param_val,
        "seed": seed,
        "welfare": final.welfare,
        "toxicity_rate": final.toxicity_rate,
        "quality_gap": final.quality_gap,
        "mean_payoff_honest": getattr(final, "mean_payoff_honest", 0.0),
        "mean_payoff_opportunistic": getattr(final, "mean_payoff_opportunistic", 0.0),
        "mean_payoff_deceptive": getattr(final, "mean_payoff_deceptive", 0.0),
    })

# Create DataFrame and save CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "sweep_results.csv"), index=False)

# Generate summary statistics
param_col = "transaction_tax_rate"
summary_configs = []

for val, group in df.groupby(param_col):
    summary_configs.append({
        param_col: float(val),
        "n_seeds": len(group),
        "mean_welfare": float(group["welfare"].mean()),
        "std_welfare": float(group["welfare"].std()),
        "mean_toxicity": float(group["toxicity_rate"].mean()),
        "std_toxicity": float(group["toxicity_rate"].std()),
        "mean_quality_gap": float(group["quality_gap"].mean()),
    })

# Create summary JSON
summary = {
    "scenario": scenario_path,
    "swept_parameter": param_name,
    "n_configs": len(summary_configs),
    "n_seeds_per_config": len(seeds),
    "configs": summary_configs,
    "best_welfare": max(summary_configs, key=lambda x: x["mean_welfare"]),
    "lowest_toxicity": min(summary_configs, key=lambda x: x["mean_toxicity"]),
}

with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Parameter sweep completed successfully!")
print(f"Results saved to {output_dir}/sweep_results.csv")
print(f"Summary saved to {output_dir}/summary.json")

EOF