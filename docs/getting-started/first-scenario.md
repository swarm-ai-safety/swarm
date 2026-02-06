# Your First Scenario

SWARM uses YAML files to define reproducible experiments. This guide shows you how to create one.

## Scenario Structure

```yaml
# scenarios/my_experiment.yaml
name: my_experiment
description: Testing adverse selection with mixed agent population

simulation:
  seed: 42
  n_epochs: 20
  steps_per_epoch: 15

agents:
  - type: honest
    count: 3
    name: "Team A"
    id_prefix: honest
  - type: opportunistic
    count: 2
    id_prefix: opp
  - type: deceptive
    count: 1
    id_prefix: dec

governance:
  transaction_tax: 0.02
  reputation_decay: 0.1
  circuit_breaker_threshold: 0.3

payoff:
  s_plus: 1.0
  s_minus: 0.5
  h: 0.3
  theta: 0.5
```

Optional: add `name` to set a human-readable display label (defaults to `agent_id`). If `count > 1`, names are suffixed for uniqueness (e.g., `Team A_1`, `Team A_2`).

## Running Your Scenario

```bash
swarm run scenarios/my_experiment.yaml
```

Or programmatically:

```python
from swarm.scenarios import ScenarioLoader
from swarm.core.orchestrator import Orchestrator

# Load scenario
scenario = ScenarioLoader.load("scenarios/my_experiment.yaml")

# Create orchestrator from scenario
orchestrator = Orchestrator.from_scenario(scenario)

# Run
metrics = orchestrator.run()
```

## Agent Types

| Type | Behavior |
|------|----------|
| `honest` | Cooperative, completes tasks diligently |
| `opportunistic` | Maximizes short-term payoff, cherry-picks |
| `deceptive` | Builds trust, then exploits |
| `adversarial` | Actively disrupts the ecosystem |

## Governance Levers

| Lever | Effect |
|-------|--------|
| `transaction_tax` | Flat tax on each interaction |
| `reputation_decay` | How quickly reputation erodes |
| `circuit_breaker_threshold` | Toxicity level that triggers agent freeze |
| `audit_probability` | Random audit frequency |
| `staking_requirement` | Minimum stake to participate |

## Parameter Sweeps

Test multiple configurations:

```yaml
# scenarios/sweep_taxes.yaml
name: tax_sweep
base_scenario: baseline.yaml

sweep:
  parameter: governance.transaction_tax
  values: [0.0, 0.01, 0.02, 0.05, 0.1]
  replications: 5
```

```bash
swarm sweep scenarios/sweep_taxes.yaml --output results/
```

## Best Practices

!!! tip "Reproducibility"
    Always set a `seed` for reproducible results.

!!! tip "Start Small"
    Begin with fewer epochs and agents, then scale up once you understand the dynamics.

!!! tip "Compare Baselines"
    Run scenarios with and without governance to measure intervention effects.

## Next Steps

- [Governance Guide](../concepts/governance.md) - Deep dive into safety mechanisms
- [Parameter Sweeps](../guides/parameter-sweeps.md) - Systematic experimentation
- [Custom Agents](../guides/custom-agents.md) - Create new agent behaviors
