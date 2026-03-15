#!/bin/bash

# Create output directory structure
mkdir -p /root/output/seed_42
mkdir -p /root/output/seed_7
mkdir -p /root/output/seed_123

# Create summary CSV header
echo "seed,welfare,toxicity_rate" > /root/output/summary.csv

# Function to run scenario and extract metrics
run_seed() {
    local seed=$1
    local output_dir="/root/output/seed_${seed}"
    
    # Run the scenario
    cd /root
    python -c "
import os
import json
from swarm.scenarios.loader import load_scenario, build_orchestrator

# Load and configure scenario
sc = load_scenario('/root/scenarios/baseline.yaml')
sc.orchestrator_config.seed = ${seed}
sc.orchestrator_config.n_epochs = 5
sc.orchestrator_config.steps_per_epoch = 10

# Run simulation
orch = build_orchestrator(sc)
result = orch.run()

# Export history.json
with open('${output_dir}/history.json', 'w') as f:
    json.dump(result.to_dict(), f, indent=2)

# Extract final metrics
history = result.to_dict()
final = history['epoch_snapshots'][-1]
welfare = final['welfare']
toxicity = final['toxicity_rate']

print(f'${seed},{welfare},{toxicity}')
" >> /root/output/summary.csv
}

# Run scenarios for each seed
run_seed 42
run_seed 7
run_seed 123