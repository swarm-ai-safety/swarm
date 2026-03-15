#!/bin/bash

# Create output directory
mkdir -p /root/output

# Initialize summary CSV with header
echo "seed,welfare,toxicity_rate" > /root/output/summary.csv

# Array of seeds to test
seeds=(42 7 123)

# Run simulation for each seed
for seed in "${seeds[@]}"; do
    # Create seed-specific output directory
    mkdir -p /root/output/seed_${seed}
    
    # Run SWARM simulation with current seed
    cd /root/swarm-package
    python -m swarm.main \
        --config /root/scenarios/baseline.yaml \
        --seed ${seed} \
        --epochs 5 \
        --steps 10 \
        --output /root/output/seed_${seed}/
    
    # Extract welfare and toxicity_rate from history.json
    welfare=$(python -c "
import json
with open('/root/output/seed_${seed}/history.json', 'r') as f:
    data = json.load(f)
    print(data['final_metrics']['welfare'])
")
    
    toxicity_rate=$(python -c "
import json
with open('/root/output/seed_${seed}/history.json', 'r') as f:
    data = json.load(f)
    print(data['final_metrics']['toxicity_rate'])
")
    
    # Append to summary CSV
    echo "${seed},${welfare},${toxicity_rate}" >> /root/output/summary.csv
done