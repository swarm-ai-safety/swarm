# Red Teaming

> **Implementation reference:** For adaptive adversary internals, full strategy table, governance effectiveness metrics, and evaluation results, see [docs/red-teaming.md](../red-teaming.md).

Test your governance mechanisms against adversarial agents.

## Overview

SWARM's red-teaming module provides:

- **Adversarial agents** with configurable attack strategies
- **Attack scenarios** that stress-test governance
- **Evasion metrics** to measure detection capabilities

## Quick Start

```python
from swarm.redteam import AdversarialAgent, AttackScenario
from swarm.redteam.evaluator import RedTeamEvaluator

# Create evaluator
evaluator = RedTeamEvaluator(
    base_scenario="scenarios/governed.yaml",
    attack_budget=0.3,  # 30% of agents are adversarial
)

# Run evaluation
results = evaluator.run(n_runs=10)

print(f"Attack success rate: {results.success_rate:.2%}")
print(f"Detection rate: {results.detection_rate:.2%}")
print(f"Governance evasion: {results.evasion_rate:.2%}")
```

## Attack Strategies

### Sybil Attack

Create multiple identities to manipulate reputation:

```yaml
attack:
  type: sybil
  params:
    n_identities: 5
    coordination: high
```

### Gradual Exploitation

Build trust, then exploit:

```yaml
attack:
  type: gradual
  params:
    trust_building_epochs: 10
    exploitation_intensity: 0.8
```

### Targeted Attack

Focus on specific high-value agents:

```yaml
attack:
  type: targeted
  params:
    target_selection: highest_reputation
    attack_frequency: 0.5
```

### Collusion

Coordinate with other adversarial agents:

```yaml
attack:
  type: collusion
  params:
    group_size: 3
    coordination_method: vote_manipulation
```

## Adaptive Adversaries

Agents that learn to evade governance:

```python
from swarm.agents.adaptive_adversary import AdaptiveAdversary

adversary = AdaptiveAdversary(
    agent_id="adaptive_1",
    learning_rate=0.1,
    evasion_strategies=["timing", "target_selection", "intensity_modulation"]
)
```

### Evasion Strategies

| Strategy | Description |
|----------|-------------|
| `timing` | Attack when audit probability is low |
| `target_selection` | Avoid agents with high detection capability |
| `intensity_modulation` | Keep individual attacks below threshold |
| `reputation_laundering` | Use honest interactions to offset attacks |

## Evaluation Metrics

### Attack Success Rate

```python
# Fraction of attacks that achieved their objective
success_rate = results.successful_attacks / results.total_attacks
```

### Detection Rate

```python
# Fraction of attacks detected by governance
detection_rate = results.detected_attacks / results.total_attacks
```

### Evasion Rate

```python
# Successful attacks that weren't detected
evasion_rate = results.successful_attacks - results.detected_attacks
```

### System Damage

```python
# Total harm caused by attacks
damage = results.total_externality / results.baseline_externality
```

## Red Team Scenarios

Pre-built attack scenarios:

```bash
# Run all attack scenarios
swarm redteam scenarios/governed.yaml --all

# Specific attack type
swarm redteam scenarios/governed.yaml --attack sybil
```

### Scenario Library

| Scenario | Attack | Tests |
|----------|--------|-------|
| `sybil_flood` | Sybil | Identity verification |
| `trust_exploit` | Gradual | Reputation decay |
| `coordinated_dump` | Collusion | Collusion detection |
| `adaptive_evasion` | Adaptive | Overall robustness |

## Writing Attack Scenarios

```yaml
name: custom_attack
description: Test governance against coordinated exploitation

base_scenario: governed.yaml

attack:
  type: collusion
  agent_fraction: 0.3

  params:
    group_size: 3
    target: honest_agents
    strategy: vote_manipulation

  schedule:
    warmup_epochs: 5
    attack_epochs: 10
    cooldown_epochs: 5

evaluation:
  metrics:
    - success_rate
    - detection_rate
    - welfare_impact
  success_threshold:
    detection_rate: 0.8
    welfare_impact: 0.9
```

## Best Practices

!!! warning "Don't Over-Tune"
    Governance that perfectly defeats your attacks may be overfit.

!!! tip "Test Multiple Attacks"
    No single attack tests all vulnerabilities.

!!! tip "Measure Trade-offs"
    Stronger governance has costs—track welfare alongside security.

!!! tip "Use Adaptive Adversaries"
    Static attacks underestimate real threats.

## Integration with CI

```yaml
# .github/workflows/redteam.yml
name: Red Team Tests

on: [push]

jobs:
  redteam:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run red team evaluation
        run: |
          pip install swarm-safety[dev]
          swarm redteam scenarios/governed.yaml --threshold 0.8
```
