#!/bin/bash

# Create output directory structure
mkdir -p /root/output/seed_42
mkdir -p /root/output/seed_7
mkdir -p /root/output/seed_123

# Install swarm-safety package
cd /root/swarm-package
pip install -e .

# Run scenarios with different seeds
python3 << 'EOF'
import os
import json
import csv
from swarm.scenarios.loader import load_scenario, build_orchestrator

def run_seed(seed, output_dir):
    scenario_path = "/root/scenarios/baseline.yaml"
    sc = load_scenario(scenario_path)
    
    # Set parameters
    sc.orchestrator_config.seed = seed
    sc.orchestrator_config.n_epochs = 5
    sc.orchestrator_config.steps_per_epoch = 10
    
    # Run simulation
    orch = build_orchestrator(sc)
    result = orch.run()
    
    # Export history.json
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    # Extract metrics from final epoch
    history = result.to_dict()
    final = history["epoch_snapshots"][-1]
    welfare = final["welfare"]
    toxicity = final["toxicity_rate"]
    
    return welfare, toxicity

# Run all seeds
seeds = [42, 7, 123]
results = []

for seed in seeds:
    output_dir = f"/root/output/seed_{seed}"
    welfare, toxicity = run_seed(seed, output_dir)
    results.append((seed, welfare, toxicity))
    print(f"Seed {seed}: welfare={welfare:.6f}, toxicity={toxicity:.6f}")

# Write summary CSV
with open("/root/output/summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "welfare", "toxicity_rate"])
    for seed, welfare, toxicity in results:
        writer.writerow([seed, welfare, toxicity])

EOF