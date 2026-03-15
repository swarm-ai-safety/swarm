#!/bin/bash

# Create output directory
mkdir -p /root/output

# Change to swarm package directory
cd /root/swarm-package

# Run the kernel market scenario with specified parameters
python -m swarm.run_scenario /root/scenarios/kernel_market/baseline.yaml --seed 42 --epochs 15 --steps 10 --output /tmp/swarm_output

# Process the results to compute quality gap
python3 << 'EOF'
import json
import os
import glob
import numpy as np

# Find the output file
output_files = glob.glob('/tmp/swarm_output/*/results.json')
if not output_files:
    output_files = glob.glob('/tmp/swarm_output/*.json')
if not output_files:
    # Try alternative output locations
    output_files = glob.glob('/root/swarm-package/outputs/*/results.json')

if not output_files:
    print("No output files found")
    exit(1)

# Load the results
with open(output_files[0], 'r') as f:
    data = json.load(f)

# Extract accepted and rejected proposals with their quality values
accepted_qualities = []
rejected_qualities = []
epoch_quality_gaps = []

# Process each epoch
for epoch_data in data.get('epochs', []):
    epoch_accepted = []
    epoch_rejected = []
    
    # Look for proposal data in different possible structures
    proposals = epoch_data.get('proposals', [])
    if not proposals:
        # Try alternative structure
        for agent_data in epoch_data.get('agents', {}).values():
            proposals.extend(agent_data.get('proposals', []))
    
    for proposal in proposals:
        quality = proposal.get('quality', proposal.get('p', 0))
        if proposal.get('accepted', False):
            epoch_accepted.append(quality)
            accepted_qualities.append(quality)
        else:
            epoch_rejected.append(quality)
            rejected_qualities.append(quality)
    
    # Calculate quality gap for this epoch if there are rejections
    if len(epoch_accepted) > 0 and len(epoch_rejected) > 0:
        accepted_mean = np.mean(epoch_accepted)
        rejected_mean = np.mean(epoch_rejected)
        quality_gap = accepted_mean - rejected_mean
        epoch_quality_gaps.append(quality_gap)

# Calculate overall metrics
if len(accepted_qualities) == 0 or len(rejected_qualities) == 0:
    print("No accepted or rejected proposals found")
    exit(1)

accepted_mean_p = float(np.mean(accepted_qualities))
rejected_mean_p = float(np.mean(rejected_qualities))
quality_gap_value = float(np.mean(epoch_quality_gaps)) if epoch_quality_gaps else accepted_mean_p - rejected_mean_p

# Detect adverse selection
adverse_selection_detected = quality_gap_value < 0

# Generate recommendation
if adverse_selection_detected:
    recommendation = "Adverse selection detected. The market is accepting lower quality proposals on average than it rejects, indicating information asymmetry or poor screening mechanisms. Consider improving proposal evaluation criteria or implementing better signaling mechanisms."
else:
    recommendation = "No adverse selection detected. The market is functioning well, accepting higher quality proposals on average than it rejects. Current screening mechanisms appear effective."

# Create output
result = {
    "quality_gap_value": quality_gap_value,
    "adverse_selection_detected": adverse_selection_detected,
    "accepted_mean_p": accepted_mean_p,
    "rejected_mean_p": rejected_mean_p,
    "recommendation": recommendation
}

# Write to output file
with open('/root/output/adverse_selection.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"Quality gap: {quality_gap_value}")
print(f"Adverse selection detected: {adverse_selection_detected}")
EOF