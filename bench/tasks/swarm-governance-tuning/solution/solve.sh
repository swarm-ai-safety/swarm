#!/bin/bash
set -e

python3 << 'PYEOF'
import json, os
import numpy as np
from scipy import stats
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

tax_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
seeds = [42, 7, 123, 99, 256]
all_results = {}

for tax in tax_rates:
    all_results[tax] = {'welfare': [], 'toxicity': []}
    for seed in seeds:
        sc = load_scenario('/root/scenarios/baseline.yaml')
        oc = sc.orchestrator_config
        oc.governance_config.transaction_tax_rate = tax
        oc.seed = seed
        oc.n_epochs = 8
        oc.steps_per_epoch = 10

        orch = Orchestrator(oc)
        epochs = orch.run()  # list[EpochMetrics]
        final = epochs[-1]

        all_results[tax]['welfare'].append(final.total_welfare)
        all_results[tax]['toxicity'].append(final.toxicity_rate)

# Find optimal: max welfare with toxicity < 0.15
candidates = []
for tax, data in all_results.items():
    mean_tox = np.mean(data['toxicity'])
    mean_wel = np.mean(data['welfare'])
    if mean_tox < 0.15:
        candidates.append((tax, mean_wel, mean_tox))

if not candidates:
    # Fallback: pick lowest toxicity
    candidates = [
        (tax, np.mean(d['welfare']), np.mean(d['toxicity']))
        for tax, d in all_results.items()
    ]
    candidates.sort(key=lambda x: x[2])

best = max(candidates, key=lambda x: x[1])
optimal_tax, opt_welfare, opt_toxicity = best

# Statistical confidence vs baseline (tax=0.0)
baseline_welfare = all_results[0.0]['welfare']
optimal_welfare_arr = all_results[optimal_tax]['welfare']
if optimal_tax != 0.0:
    _, p_val = stats.ttest_ind(optimal_welfare_arr, baseline_welfare, equal_var=False)
else:
    p_val = 1.0

optimal = {
    'optimal_tax_rate': float(optimal_tax),
    'mean_welfare': round(float(opt_welfare), 3),
    'mean_toxicity': round(float(opt_toxicity), 4),
    'n_configs_tested': len(tax_rates),
    'statistical_confidence': round(float(p_val), 6),
}

os.makedirs('/root/output', exist_ok=True)
with open('/root/output/optimal.json', 'w') as f:
    json.dump(optimal, f, indent=2)

print(f'Optimal: tax={optimal_tax}, welfare={opt_welfare:.3f}, toxicity={opt_toxicity:.4f}, p={p_val:.4f}')
PYEOF
