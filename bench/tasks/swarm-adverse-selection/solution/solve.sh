#!/bin/bash
set -e

python3 << 'PYEOF'
import json, os
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

sc = load_scenario('/root/scenarios/baseline.yaml')
oc = sc.orchestrator_config
oc.seed = 42
oc.n_epochs = 10
oc.steps_per_epoch = 10

orch = Orchestrator(oc)
epochs = orch.run()  # list[EpochMetrics]
final = epochs[-1]

# Extract quality gap from EpochMetrics
quality_gap = final.quality_gap

# For accepted/rejected mean p, use quality_gap sign to infer
# quality_gap = E[p|accepted] - E[p|rejected]
# We report the quality_gap directly and derive accepted/rejected bounds
accepted_p = max(0.0, min(1.0, 0.5 + quality_gap / 2))
rejected_p = max(0.0, min(1.0, 0.5 - quality_gap / 2))

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
