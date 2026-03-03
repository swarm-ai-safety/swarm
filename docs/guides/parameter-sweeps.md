---
description: "Systematically explore how parameters affect SWARM metrics."
---

# Parameter Sweeps

Systematically explore how parameters affect SWARM metrics.

## Overview

[Parameter sweeps](../getting-started/first-scenario.md) run multiple simulations varying one or more parameters, enabling:

- Sensitivity analysis
- Optimal parameter discovery
- Trade-off visualization

## CLI Usage

### Single Parameter Sweep

```bash
swarm sweep scenarios/baseline.yaml \
  --param governance.transaction_tax \
  --values 0.0,0.01,0.02,0.05,0.1 \
  --replications 5 \
  --output results/tax_sweep.csv
```

### Multi-Parameter Sweep

```bash
swarm sweep scenarios/baseline.yaml \
  --param governance.transaction_tax:0.0,0.02,0.05 \
  --param governance.reputation_decay:0.0,0.1,0.2 \
  --replications 3 \
  --output results/multi_sweep.csv
```

## YAML Configuration

```yaml
# scenarios/sweep_config.yaml
name: governance_sweep
base_scenario: baseline.yaml

sweep:
  parameters:
    - name: governance.transaction_tax
      values: [0.0, 0.01, 0.02, 0.05, 0.1]
    - name: governance.reputation_decay
      values: [0.0, 0.1, 0.2]

  replications: 5
  seeds: auto  # Generate unique seeds per replication
```

```bash
swarm sweep scenarios/sweep_config.yaml --output results/
```

## Programmatic API

```python
from swarm.analysis.sweep import SweepRunner, SweepConfig

config = SweepConfig(
    base_scenario="scenarios/baseline.yaml",
    parameters={
        "governance.transaction_tax": [0.0, 0.01, 0.02, 0.05, 0.1],
    },
    replications=5,
)

runner = SweepRunner(config)
results = runner.run(progress_callback=print)

# Export results
results.to_csv("results/sweep.csv")
```

## Analyzing Results

### Summary Statistics

```python
# Get summary by parameter value
summary = results.summary()
print(summary)
```

Output:
```
transaction_tax  toxicity_mean  toxicity_std  quality_gap_mean
0.00             0.342          0.045         -0.123
0.01             0.298          0.038         -0.067
0.02             0.251          0.042          0.012
0.05             0.187          0.051          0.089
0.10             0.145          0.063          0.134
```

### Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Toxicity vs Tax
axes[0].errorbar(
    summary['transaction_tax'],
    summary['toxicity_mean'],
    yerr=summary['toxicity_std'],
    capsize=5
)
axes[0].set_xlabel('Transaction Tax')
axes[0].set_ylabel('Toxicity Rate')
axes[0].set_title('Toxicity vs Tax Rate')

# Quality Gap vs Tax
axes[1].errorbar(
    summary['transaction_tax'],
    summary['quality_gap_mean'],
    yerr=summary['quality_gap_std'],
    capsize=5
)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Transaction Tax')
axes[1].set_ylabel('Quality Gap')
axes[1].set_title('Quality Gap vs Tax Rate')

plt.tight_layout()
plt.savefig('sweep_results.png')
```

## Advanced Features

### Parallel Execution

```python
runner = SweepRunner(config, n_workers=4)
results = runner.run()
```

### Conditional Parameters

```yaml
sweep:
  parameters:
    - name: governance.circuit_breaker_threshold
      values: [0.2, 0.3, 0.4]
    - name: governance.circuit_breaker_window
      values: [5, 10, 20]
      depends_on: governance.circuit_breaker_threshold
      # Only varies window when threshold is not None
```

### Custom Metrics

```python
def custom_metric(interactions):
    """Custom analysis on sweep results."""
    return sum(i.p for i in interactions if i.accepted) / len(interactions)

results = runner.run(extra_metrics={'custom': custom_metric})
```

## Best Practices

!!! tip "Start Coarse, Refine"
    Begin with wide parameter ranges, then zoom in on interesting regions.

!!! tip "Use Sufficient Replications"
    At least 5 replications for statistical significance.

!!! tip "Watch for Interactions"
    Multi-parameter sweeps reveal interaction effects.

!!! tip "Save Raw Data"
    Keep full results, not just summaries, for later analysis.

## Common Patterns

### Finding Optimal Governance

```python
# Find tax rate that minimizes toxicity while keeping welfare positive
optimal = results.query('total_welfare > 0').sort_values('toxicity_mean').iloc[0]
print(f"Optimal tax: {optimal['transaction_tax']}")
```

### Identifying Phase Transitions

```python
# Look for sudden changes in quality gap sign
transitions = results[results['quality_gap_mean'].diff().abs() > 0.1]
print("Phase transitions at:", transitions['transaction_tax'].values)
```
