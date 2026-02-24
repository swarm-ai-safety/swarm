---
name: run-scenario
description: Execute a SWARM simulation scenario and export standardized artifacts
version: "1.0"
domain: swarm-safety
triggers:
  - run scenario
  - execute simulation
  - baseline run
---

# Run Scenario Skill

Execute a single SWARM scenario with a given seed and export all artifacts to a standardized output directory.

## Prerequisites

- `swarm-safety` package installed (`pip install swarm-safety` or `pip install -e /root/swarm-package/`)
- Scenario YAML file available (typically in `/root/scenarios/`)

## Procedure

### 1. Resolve the scenario path

Scenario references can be shorthand or full paths:
- `baseline` → `scenarios/baseline.yaml`
- `scenarios/baseline.yaml` → use as-is
- `/root/scenarios/baseline.yaml` → use as-is

```python
import os

def resolve_scenario(ref: str) -> str:
    """Resolve a scenario reference to a full path."""
    candidates = [
        ref,
        f"scenarios/{ref}.yaml",
        f"/root/scenarios/{ref}.yaml",
        f"scenarios/{ref}",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Cannot find scenario: {ref}")
```

### 2. Run the simulation

Use the SWARM CLI to execute:

```bash
python -m swarm run <scenario_path> --seed <seed> --epochs <N> --steps <M>
```

Or programmatically:

```python
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario

config = load_scenario(scenario_path)
# Override simulation parameters if needed
config["simulation"]["seed"] = seed
config["simulation"]["n_epochs"] = epochs
config["simulation"]["steps_per_epoch"] = steps

orch = Orchestrator(config)
result = orch.run()
```

### 3. Export artifacts

After the run completes, export to the output directory:

```python
import json
import os

os.makedirs(output_dir, exist_ok=True)

# Export history.json
with open(os.path.join(output_dir, "history.json"), "w") as f:
    json.dump(result.to_dict(), f, indent=2)

# Export CSV metrics
csv_dir = os.path.join(output_dir, "csv")
os.makedirs(csv_dir, exist_ok=True)
result.export_csv(csv_dir)
```

### 4. Extract key metrics

The final epoch snapshot contains summary metrics:

```python
history = result.to_dict()
final = history["epoch_snapshots"][-1]
welfare = final["welfare"]
toxicity = final["toxicity_rate"]
print(f"Final welfare: {welfare:.3f}")
print(f"Final toxicity: {toxicity:.3f}")
```

## Output Structure

```
<output_dir>/
├── history.json          # Full simulation history
└── csv/
    ├── epoch_metrics.csv # Per-epoch aggregate metrics
    └── agent_metrics.csv # Per-agent per-epoch metrics
```

## Common Issues

- **ImportError**: Ensure `swarm-safety` is installed with runtime deps: `pip install swarm-safety[runtime]`
- **FileNotFoundError for scenario**: Check `/root/scenarios/` for available YAMLs
- **Reproducibility**: Same seed + same scenario YAML = identical results
