#!/bin/bash
set -e

python3 -c "
import json, os
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

config = load_scenario('/root/scenarios/baseline.yaml')
config['simulation']['seed'] = 42
config['simulation']['n_epochs'] = 10
config['simulation']['steps_per_epoch'] = 10

orch = Orchestrator(config)
result = orch.run()

os.makedirs('/root/output/csv', exist_ok=True)

with open('/root/output/history.json', 'w') as f:
    json.dump(result.to_dict(), f, indent=2)

result.export_csv('/root/output/csv')

final = result.to_dict()['epoch_snapshots'][-1]
print(f'Final welfare: {final[\"welfare\"]:.3f}')
print(f'Final toxicity: {final[\"toxicity_rate\"]:.3f}')
"
