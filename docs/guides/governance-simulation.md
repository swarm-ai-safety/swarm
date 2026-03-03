---
description: "Simulate AI governance mechanisms before deployment. SWARM lets you test transaction taxes, circuit breakers, reputation systems, and collusion detection in multi-agent environments."
date: 2026-03-02
---

# AI Governance Simulation

SWARM lets you **simulate governance mechanisms before deploying them**. Test transaction taxes, circuit breakers, reputation decay, staking, and collusion detection across thousands of agent interactions — and measure exactly what each lever costs and prevents.

## Why Simulate Governance?

Governance mechanisms have unintended consequences. A tax that prevents exploitation also reduces beneficial interactions. A circuit breaker that catches adversaries also freezes honest agents having a bad day. Simulation reveals these tradeoffs before they matter.

```yaml
# scenarios/governance_test.yaml
simulation:
  n_epochs: 20
  steps_per_epoch: 10
  seed: 42

agents:
  - type: honest
    count: 7
  - type: deceptive
    count: 2
  - type: adversarial
    count: 1

governance:
  transaction_tax: 0.02
  circuit_breaker_threshold: 0.3
  circuit_breaker_window: 10
  circuit_breaker_cooldown: 5
  reputation_decay: 0.1
  audit_probability: 0.05
  audit_penalty: 0.5
```

```bash
python -m swarm run scenarios/governance_test.yaml --seed 42 --epochs 20 --steps 10
```

## Available Governance Levers

### Transaction Tax

A per-interaction cost that creates friction against hyperactive extractors.

| Parameter | Range | Effect |
|-----------|-------|--------|
| `transaction_tax` | 0.0 – 0.10 | Higher = more friction, less volume |

**When to use:** When adversarial agents exploit high interaction volume.
**Tradeoff:** Reduces total interactions — including beneficial ones.

### Circuit Breaker

Freezes agents whose recent toxicity exceeds a threshold.

| Parameter | Range | Effect |
|-----------|-------|--------|
| `circuit_breaker_threshold` | 0.1 – 0.8 | Lower = more aggressive |
| `circuit_breaker_window` | 5 – 50 | Lookback period |
| `circuit_breaker_cooldown` | 1 – 20 | Freeze duration |

**When to use:** When deceptive agents switch from trust-building to exploitation.
**Tradeoff:** Can freeze honest agents with temporarily low scores.

### Reputation Decay

Reduces reputation over time, preventing indefinite trust accumulation.

| Parameter | Range | Effect |
|-----------|-------|--------|
| `reputation_decay` | 0.0 – 0.3 | Higher = faster forgetting |

**When to use:** When trust-then-exploit strategies dominate.
**Tradeoff:** Honest agents must continuously earn reputation.

### Random Audits

Probabilistic checks that catch deception even during honest phases.

| Parameter | Range | Effect |
|-----------|-------|--------|
| `audit_probability` | 0.01 – 0.20 | Audit frequency |
| `audit_penalty` | 0.1 – 1.0 | Failed audit cost |

### Staking

Requires agents to deposit stake that can be slashed on bad behavior.

| Parameter | Range | Effect |
|-----------|-------|--------|
| `staking_requirement` | 1.0 – 50.0 | Entry barrier |
| `stake_slash_rate` | 0.05 – 0.5 | Slash fraction |

### Collusion Detection

Identifies coordinated exploitation between agent pairs.

| Parameter | Range | Effect |
|-----------|-------|--------|
| `collusion_detection` | true/false | Enable detection |
| `collusion_threshold` | 0.5 – 0.95 | Correlation threshold |

## Running Parameter Sweeps

Test governance across parameter ranges systematically:

```python
from swarm.scenarios import load_scenario, build_orchestrator
from swarm.metrics.soft_metrics import SoftMetrics

taxes = [0.0, 0.01, 0.02, 0.05, 0.10]
results = {}

for tax in taxes:
    scenario = load_scenario("scenarios/governance_test.yaml")
    scenario.governance.transaction_tax = tax
    orch = build_orchestrator(scenario)
    history = orch.run()
    final = history[-1]
    results[tax] = {
        "toxicity": final.toxicity_rate,
        "quality_gap": final.quality_gap,
        "mean_payoff": final.avg_payoff,
    }

for tax, m in results.items():
    print(f"Tax {tax:.2f}: tox={m['toxicity']:.3f} qgap={m['quality_gap']:+.3f} payoff={m['mean_payoff']:.3f}")
```

See the [parameter sweeps guide](parameter-sweeps.md) for systematic exploration across multiple dimensions.

## Interpreting Results

The key diagnostic patterns from the [analyzing results tutorial](../tutorials/analyzing-results.md):

| Pattern | Toxicity | Quality Gap | Payoff | Meaning |
|---------|----------|-------------|--------|---------|
| Healthy | < 0.1 | Positive | Good | Governance working |
| Adverse selection | > 0.3 | **Negative** | High | Selecting for harm |
| Over-governance | < 0.05 | Positive | **Very low** | Too aggressive |
| Collapse | > 0.5 | Negative | Negative | System failure |

## Real-World Bridges

SWARM governance has been tested against real-world inspired scenarios:

- [Tierra](../blog/tierra-governance-vs-evolution.md) — Governance layered on evolutionary dynamics
- [AI Economist](../blog/ai-economist-gtb-simulation.md) — Tax policy in simulated economies
- [Contract Screening](../blog/contract-screening-separating-equilibrium.md) — Mechanism design for agent pools
- [LangGraph Handoff](../blog/langgraph-governed-handoff-sweep.md) — Governance in production swarm architectures

## See also

- [Governance Concepts](../concepts/governance.md) — Theory behind each lever
- [Governance API](../api/governance.md) — Full configuration reference
- [Custom Governance Levers](governance-levers.md) — Build your own mechanisms
- [Red-Teaming Guide](red-teaming.md) — Adversarial stress testing
- [Parameter Sweeps](parameter-sweeps.md) — Systematic parameter exploration
