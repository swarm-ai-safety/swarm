#!/bin/bash
set -e

python3 << 'PYEOF'
import json, os, csv
import pandas as pd
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

tax_rates = [0.0, 0.05, 0.10, 0.15]
seeds = [42, 7, 123]
results = []

for tax in tax_rates:
    for seed in seeds:
        config = load_scenario('/root/scenarios/baseline.yaml')
        config['governance']['transaction_tax_rate'] = tax
        config['simulation']['seed'] = seed
        config['simulation']['n_epochs'] = 5
        config['simulation']['steps_per_epoch'] = 10

        orch = Orchestrator(config)
        result = orch.run()
        final = result.to_dict()['epoch_snapshots'][-1]

        results.append({
            'transaction_tax_rate': tax,
            'seed': seed,
            'welfare': round(final['welfare'], 3),
            'toxicity_rate': round(final['toxicity_rate'], 4),
            'quality_gap': round(final.get('quality_gap', 0.0), 4),
        })

with open('/root/output/sweep_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

df = pd.DataFrame(results)
configs = []
for val, grp in df.groupby('transaction_tax_rate'):
    configs.append({
        'transaction_tax_rate': float(val),
        'n_seeds': len(grp),
        'mean_welfare': round(float(grp['welfare'].mean()), 3),
        'std_welfare': round(float(grp['welfare'].std()), 3),
        'mean_toxicity': round(float(grp['toxicity_rate'].mean()), 4),
    })

summary = {
    'swept_parameter': 'governance.transaction_tax_rate',
    'n_configs': len(configs),
    'configs': configs,
}

with open('/root/output/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Sweep complete: {len(results)} rows, {len(configs)} configs')
PYEOF
