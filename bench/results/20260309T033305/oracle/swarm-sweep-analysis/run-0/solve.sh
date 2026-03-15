#!/bin/bash
set -e

python3 << 'PYEOF'
import json, os, csv
import pandas as pd
from swarm.scenarios.loader import load_scenario, build_orchestrator

tax_rates = [0.0, 0.05, 0.10, 0.15]
seeds = [42, 7, 123]
results = []

for tax in tax_rates:
    for seed in seeds:
        sc = load_scenario('/root/scenarios/baseline.yaml')
        sc.orchestrator_config.governance_config.transaction_tax_rate = tax
        sc.orchestrator_config.seed = seed
        sc.orchestrator_config.n_epochs = 5
        sc.orchestrator_config.steps_per_epoch = 10

        orch = build_orchestrator(sc)
        epochs = orch.run()  # list[EpochMetrics]
        final = epochs[-1]

        results.append({
            'transaction_tax_rate': tax,
            'seed': seed,
            'welfare': round(final.total_welfare, 3),
            'toxicity_rate': round(final.toxicity_rate, 4),
            'quality_gap': round(final.quality_gap, 4),
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
