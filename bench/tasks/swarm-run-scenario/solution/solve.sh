#!/bin/bash
set -e

python3 << 'PYEOF'
import json, os, csv
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

sc = load_scenario('/root/scenarios/baseline.yaml')
oc = sc.orchestrator_config
oc.seed = 42
oc.n_epochs = 10
oc.steps_per_epoch = 10

orch = Orchestrator(oc)
epochs = orch.run()  # list[EpochMetrics]

os.makedirs('/root/output/csv', exist_ok=True)

# Build history.json with epoch_snapshots
snapshots = []
for em in epochs:
    snapshots.append({
        'epoch': em.epoch,
        'total_interactions': em.total_interactions,
        'accepted_interactions': em.accepted_interactions,
        'toxicity_rate': em.toxicity_rate,
        'quality_gap': em.quality_gap,
        'welfare': em.total_welfare,
        'avg_payoff': em.avg_payoff,
    })

history = {'epoch_snapshots': snapshots}
with open('/root/output/history.json', 'w') as f:
    json.dump(history, f, indent=2)

# Export CSV
with open('/root/output/csv/epoch_metrics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=snapshots[0].keys())
    writer.writeheader()
    writer.writerows(snapshots)

final = snapshots[-1]
print(f"Final welfare: {final['welfare']:.3f}")
print(f"Final toxicity: {final['toxicity_rate']:.3f}")
PYEOF
