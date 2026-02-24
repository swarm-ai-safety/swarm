---
name: parameter-sweep
description: Run parameter grid sweeps across SWARM scenarios and generate summary statistics
version: "1.0"
domain: swarm-safety
triggers:
  - parameter sweep
  - sweep tax rate
  - grid search governance
---

# Parameter Sweep Skill

Run a parameter sweep over governance configurations, collect results across multiple seeds, and generate summary statistics.

## Prerequisites

- `swarm-safety` package installed
- `pandas` and `numpy` available
- Scenario YAML file

## Procedure

### 1. Define the sweep grid

```python
import itertools

# Example: sweep a single parameter
param_name = "governance.transaction_tax_rate"
param_values = [0.0, 0.05, 0.10, 0.15]
seeds = [42, 7, 123]

# For multi-parameter sweeps, use itertools.product
configs = list(itertools.product(param_values, seeds))
```

### 2. Run each configuration

```python
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario
import copy

results = []

for param_val, seed in configs:
    config = load_scenario(scenario_path)
    
    # Override the swept parameter (supports nested keys)
    keys = param_name.split(".")
    target = config
    for k in keys[:-1]:
        target = target[k]
    target[keys[-1]] = param_val
    
    # Override seed and epoch count
    config["simulation"]["seed"] = seed
    config["simulation"]["n_epochs"] = epochs
    config["simulation"]["steps_per_epoch"] = steps
    
    orch = Orchestrator(config)
    result = orch.run()
    
    final = result.to_dict()["epoch_snapshots"][-1]
    results.append({
        param_name.split(".")[-1]: param_val,
        "seed": seed,
        "welfare": final["welfare"],
        "toxicity_rate": final["toxicity_rate"],
        "quality_gap": final.get("quality_gap", 0.0),
        "mean_payoff_honest": final.get("mean_payoff_honest", 0.0),
        "mean_payoff_opportunistic": final.get("mean_payoff_opportunistic", 0.0),
        "mean_payoff_deceptive": final.get("mean_payoff_deceptive", 0.0),
    })
```

### 3. Create sweep CSV

```python
import pandas as pd

df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "sweep_results.csv"), index=False)
```

### 4. Generate summary.json

```python
import json

param_col = param_name.split(".")[-1]
summary_configs = []

for val, group in df.groupby(param_col):
    summary_configs.append({
        param_col: float(val),
        "n_seeds": len(group),
        "mean_welfare": float(group["welfare"].mean()),
        "std_welfare": float(group["welfare"].std()),
        "mean_toxicity": float(group["toxicity_rate"].mean()),
        "std_toxicity": float(group["toxicity_rate"].std()),
        "mean_quality_gap": float(group["quality_gap"].mean()),
    })

summary = {
    "scenario": scenario_path,
    "swept_parameter": param_name,
    "n_configs": len(summary_configs),
    "n_seeds_per_config": len(seeds),
    "configs": summary_configs,
    "best_welfare": max(summary_configs, key=lambda x: x["mean_welfare"]),
    "lowest_toxicity": min(summary_configs, key=lambda x: x["mean_toxicity"]),
}

with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
```

## Output Structure

```
<output_dir>/
├── sweep_results.csv     # One row per (config, seed) combination
└── summary.json          # Aggregated stats per config
```

## Summary JSON Schema

```json
{
  "scenario": "string",
  "swept_parameter": "string",
  "n_configs": "int",
  "n_seeds_per_config": "int",
  "configs": [
    {
      "<param_col>": "float",
      "n_seeds": "int",
      "mean_welfare": "float",
      "std_welfare": "float",
      "mean_toxicity": "float",
      "std_toxicity": "float"
    }
  ],
  "best_welfare": { "...config object" },
  "lowest_toxicity": { "...config object" }
}
```
