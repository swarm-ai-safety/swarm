---
description: "AI agent governance mechanisms for multi-agent safety. Configure transaction taxes, circuit breakers, reputation decay, staking, random audits, and collusion detection in SWARM."
---

# Governance Mechanisms

SWARM provides configurable [governance levers](../getting-started/first-scenario.md) to mitigate multi-agent risks.

## Overview

[Governance mechanisms](index.md) create **incentives and constraints** that shape agent behavior at the system level. They're the primary tool for converting SWARM's metrics into actionable safety.

## Available Levers

### Transaction Tax

**Purpose:** Add friction to reduce exploitation.

```yaml
governance:
  transaction_tax: 0.02  # 2% tax on each interaction
```

**How it works:**

- Tax is deducted from both parties' payoffs
- Reduces the profit margin for low-quality interactions
- Makes exploitation less attractive

**Trade-off:** Reduces overall welfare, including for honest agents.

### Reputation Decay

**Purpose:** Make past behavior matter.

```yaml
governance:
  reputation_decay: 0.1  # 10% decay per epoch
```

**How it works:**

- Reputation contributes to payoffs
- Decay means agents must continuously behave well
- Bad actors can't coast on old reputation

**Trade-off:** Honest agents also lose reputation over time.

### Circuit Breakers

**Purpose:** Freeze toxic agents quickly.

```yaml
governance:
  circuit_breaker_threshold: 0.3  # Freeze if toxicity > 30%
  circuit_breaker_window: 10      # Over last 10 interactions
```

**How it works:**

- Monitors each agent's recent toxicity
- Agents exceeding threshold are frozen
- Can recover after cooldown period

**Trade-off:** May freeze agents incorrectly (false positives).

### Random Audits

**Purpose:** Deter hidden exploitation.

```yaml
governance:
  audit_probability: 0.05  # 5% of interactions audited
  audit_penalty: 0.5       # Penalty for failed audit
```

**How it works:**

- Random selection of interactions for review
- Failed audits result in reputation and payoff penalties
- Creates uncertainty for exploitative agents

**Trade-off:** Audit costs apply even to honest agents.

### Staking Requirements

**Purpose:** Filter undercapitalized agents.

```yaml
governance:
  staking_requirement: 10.0  # Minimum stake to participate
  stake_slash_rate: 0.1      # Fraction slashed on bad behavior
```

**How it works:**

- Agents must post collateral to participate
- Bad behavior results in stake being slashed
- Creates skin in the game

**Trade-off:** Excludes agents without capital.

### Collusion Detection

**Purpose:** Catch coordinated attacks.

```yaml
governance:
  collusion_detection: true
  collusion_threshold: 0.8   # Correlation threshold
  collusion_window: 20       # Interaction window
```

**How it works:**

- Monitors interaction patterns between agent pairs
- Detects suspiciously coordinated behavior
- Flags or penalizes colluding agents

**Trade-off:** May flag legitimate cooperation.

## Configuration

### Full Example

```yaml
governance:
  # Friction
  transaction_tax: 0.02

  # Reputation
  reputation_decay: 0.1
  initial_reputation: 1.0

  # Circuit breakers
  circuit_breaker_threshold: 0.3
  circuit_breaker_window: 10
  circuit_breaker_cooldown: 5

  # Audits
  audit_probability: 0.05
  audit_penalty: 0.5

  # Staking
  staking_requirement: 10.0
  stake_slash_rate: 0.1

  # Collusion
  collusion_detection: true
  collusion_threshold: 0.8
```

### Programmatic Configuration

```python
from swarm.governance import GovernanceConfig, GovernanceEngine

config = GovernanceConfig(
    transaction_tax=0.02,
    reputation_decay=0.1,
    circuit_breaker_threshold=0.3,
)

engine = GovernanceEngine(config)
```

## Governance Trade-offs

!!! warning "No Free Lunch"
    Every governance mechanism has costs. The goal is to find the right balance.

| Lever | Reduces | Costs |
|-------|---------|-------|
| Transaction tax | Exploitation | Welfare |
| Reputation decay | Free-riding | Honest agent burden |
| Circuit breakers | Toxic agents | False positives |
| Audits | Hidden exploitation | Audit overhead |
| Staking | Low-commitment agents | Exclusion |
| Collusion detection | Coordinated attacks | Cooperation friction |

## Measuring Effectiveness

Compare scenarios with and without governance:

```python
from swarm.scenarios import ScenarioLoader
from swarm.core.orchestrator import Orchestrator

# Run without governance
baseline = ScenarioLoader.load("scenarios/baseline.yaml")
baseline_metrics = Orchestrator.from_scenario(baseline).run()

# Run with governance
governed = ScenarioLoader.load("scenarios/governed.yaml")
governed_metrics = Orchestrator.from_scenario(governed).run()

# Compare
print(f"Baseline toxicity: {baseline_metrics[-1].toxicity_rate:.3f}")
print(f"Governed toxicity: {governed_metrics[-1].toxicity_rate:.3f}")
```

## Best Practices

1. **Start minimal** - Add governance only when metrics indicate problems
2. **Measure trade-offs** - Track welfare alongside [safety metrics](../blog/local-llama-model-size-safety.md)
3. **Tune gradually** - Small parameter changes can have large effects
4. **Combine mechanisms** - Multiple light-touch interventions often beat one heavy one

## See also

- [Parameter Sweeps](../guides/parameter-sweeps.md) — Systematically explore governance settings across parameter ranges
- [Metrics](metrics.md) — Toxicity, quality gap, and other metrics governance aims to optimize
- [Custom Governance Levers](../guides/governance-levers.md) — Build your own governance mechanisms
- [Red Teaming](../guides/red-teaming.md) — Stress-test governance with adversarial scenarios
- [Governance Mechanisms Taxonomy](../blog/governance-mechanisms-taxonomy.md) — Survey of governance approaches
