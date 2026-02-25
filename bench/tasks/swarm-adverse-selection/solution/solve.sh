#!/bin/bash
set -e

python3 << 'PYEOF'
import json, os
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

config = load_scenario('/root/scenarios/baseline.yaml')
config['simulation']['seed'] = 42
config['simulation']['n_epochs'] = 10
config['simulation']['steps_per_epoch'] = 10

orch = Orchestrator(config)
result = orch.run()

history = result.to_dict()
final = history['epoch_snapshots'][-1]

# Extract quality gap from metrics
quality_gap = final.get('quality_gap', 0.0)
accepted_p = final.get('accepted_mean_p', 0.5)
rejected_p = final.get('rejected_mean_p', 0.5)

# If quality_gap not directly available, compute from accepted/rejected
if 'quality_gap' not in final:
    quality_gap = accepted_p - rejected_p

if quality_gap < 0:
    rec = (
        'Adverse selection detected - low-quality interactions are preferentially '
        'accepted. Consider increasing governance controls (transaction tax, '
        'reputation decay) to improve selection quality.'
    )
else:
    rec = (
        'No adverse selection detected - the selection mechanism is working as '
        'intended. Quality gap is positive, meaning higher-quality interactions '
        'are preferentially accepted.'
    )

report = {
    'quality_gap_value': round(float(quality_gap), 4),
    'adverse_selection_detected': bool(quality_gap < 0),
    'accepted_mean_p': round(float(accepted_p), 4),
    'rejected_mean_p': round(float(rejected_p), 4),
    'recommendation': rec,
}

os.makedirs('/root/output', exist_ok=True)
with open('/root/output/adverse_selection.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f'Quality gap: {quality_gap:.4f}')
print(f'Adverse selection: {quality_gap < 0}')
PYEOF
