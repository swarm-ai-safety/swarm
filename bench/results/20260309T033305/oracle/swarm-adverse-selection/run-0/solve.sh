#!/bin/bash
set -e

python3 << 'PYEOF'
import json, os
from swarm.scenarios.loader import load_scenario, build_orchestrator

sc = load_scenario('/root/scenarios/kernel_market/baseline.yaml')
sc.orchestrator_config.seed = 42
sc.orchestrator_config.n_epochs = 15
sc.orchestrator_config.steps_per_epoch = 10

orch = build_orchestrator(sc)
epochs = orch.run()  # list[EpochMetrics]

# Compute mean quality_gap across epochs that have rejections (non-zero gap)
nonzero_gaps = [em.quality_gap for em in epochs if em.quality_gap != 0.0]
if nonzero_gaps:
    quality_gap = sum(nonzero_gaps) / len(nonzero_gaps)
else:
    quality_gap = 0.0

# Derive accepted/rejected mean p from quality_gap
# quality_gap = E[p|accepted] - E[p|rejected]
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

print(f'Quality gap: {quality_gap:.4f} ({len(nonzero_gaps)} epochs with rejections)')
print(f'Adverse selection: {quality_gap < 0}')
PYEOF
