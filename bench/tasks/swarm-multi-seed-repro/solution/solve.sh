#!/bin/bash
set -e

python3 << 'PYEOF'
import json, os, csv
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

seeds = [42, 7, 123]
rows = []

for seed in seeds:
    sc = load_scenario('/root/scenarios/baseline.yaml')
    oc = sc.orchestrator_config
    oc.seed = seed
    oc.n_epochs = 5
    oc.steps_per_epoch = 10

    orch = Orchestrator(oc)
    epochs = orch.run()  # list[EpochMetrics]

    out_dir = f'/root/output/seed_{seed}'
    os.makedirs(out_dir, exist_ok=True)

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
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    final = snapshots[-1]
    rows.append({'seed': seed, 'welfare': final['welfare'], 'toxicity_rate': final['toxicity_rate']})
    print(f'Seed {seed}: welfare={final["welfare"]:.3f}, toxicity={final["toxicity_rate"]:.3f}')

with open('/root/output/summary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['seed', 'welfare', 'toxicity_rate'])
    writer.writeheader()
    writer.writerows(rows)
PYEOF
