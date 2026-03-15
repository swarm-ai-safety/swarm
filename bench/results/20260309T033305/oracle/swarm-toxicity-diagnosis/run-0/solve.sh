#!/bin/bash
set -e

python3 << 'PYEOF'
import json
import pandas as pd

df = pd.read_csv('/root/data/sweep_results.csv')
tox_by_config = df.groupby('transaction_tax_rate')['toxicity_rate'].mean()

worst_config = float(tox_by_config.idxmax())
best_config = float(tox_by_config.idxmin())

diagnosis = {
    'worst_config': worst_config,
    'best_config': best_config,
    'worst_toxicity': round(float(tox_by_config.max()), 4),
    'best_toxicity': round(float(tox_by_config.min()), 4),
    'recommendation': (
        f'Use transaction_tax_rate={best_config} to minimize toxicity. '
        f'Avoid tax_rate={worst_config} which produces '
        f'{tox_by_config.max():.3f} mean toxicity.'
    ),
}

with open('/root/output/diagnosis.json', 'w') as f:
    json.dump(diagnosis, f, indent=2)

print(f'Worst: tax={worst_config} (tox={tox_by_config.max():.4f})')
print(f'Best:  tax={best_config} (tox={tox_by_config.min():.4f})')
PYEOF
