#!/bin/bash

# Create output directory
mkdir -p /root/output/csv

# Change to swarm package directory
cd /root/swarm-package/

# Run the baseline scenario
python -m swarm.cli run \
    --scenario /root/scenarios/baseline.yaml \
    --seed 42 \
    --epochs 10 \
    --steps 10 \
    --output-dir /root/output \
    --export-history \
    --export-csv

# Extract and print final welfare and toxicity values
python3 << 'EOF'
import json
import os

# Load history.json
with open('/root/output/history.json', 'r') as f:
    history = json.load(f)

# Get final values (last epoch, last step)
final_epoch = history['epochs'][-1]
final_step = final_epoch['steps'][-1]
final_metrics = final_step['metrics']

# Print final welfare and toxicity
welfare = final_metrics.get('welfare', 'N/A')
toxicity = final_metrics.get('toxicity', 'N/A')

print(f"Final welfare: {welfare}")
print(f"Final toxicity: {toxicity}")
EOF