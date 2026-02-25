#!/bin/bash
set -e

python3 -c "
import json, os, csv
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

seeds = [42, 7, 123]
rows = []

for seed in seeds:
    config = load_scenario('/root/scenarios/baseline.yaml')
    config['simulation']['seed'] = seed
    config['simulation']['n_epochs'] = 5
    config['simulation']['steps_per_epoch'] = 10

    orch = Orchestrator(config)
    result = orch.run()

    out_dir = f'/root/output/seed_{seed}'
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(result.to_dict(), f, indent=2)

    final = result.to_dict()['epoch_snapshots'][-1]
    rows.append({'seed': seed, 'welfare': final['welfare'], 'toxicity_rate': final['toxicity_rate']})
    print(f'Seed {seed}: welfare={final[\"welfare\"]:.3f}, toxicity={final[\"toxicity_rate\"]:.3f}')

with open('/root/output/summary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['seed', 'welfare', 'toxicity_rate'])
    writer.writeheader()
    writer.writerows(rows)
"
