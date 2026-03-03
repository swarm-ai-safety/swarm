---
description: "Multi-agent AI risk assessment guide. Systematically evaluate distributional risks — information asymmetry, adverse selection, variance amplification, and governance latency — in multi-agent systems."
date: 2026-03-02
---

# Multi-Agent Risk Assessment

This guide provides a systematic approach to assessing risks in multi-agent AI systems. Rather than evaluating individual agents, focus on **population-level dynamics** that create emergent failures.

## The Four Distributional Risks

Every multi-agent system is vulnerable to four interaction-level risks. Assess each systematically:

### 1. Information Asymmetry

**Question:** Do agents have unequal access to information?

```python
from swarm.metrics.soft_metrics import SoftMetrics

metrics = SoftMetrics()
# Compare agent-specific metrics
for agent_id in agent_ids:
    agent_metrics = metrics.agent_metrics(interactions, agent_id)
    print(f"{agent_id}: avg_p={agent_metrics['mean_p']:.3f} "
          f"payoff={agent_metrics['mean_payoff']:.3f}")
```

| Risk Level | Signal | Action |
|------------|--------|--------|
| Low | Similar payoffs across agents | Monitor |
| Medium | 2x payoff gap between agents | Add transparency mechanisms |
| High | 5x+ payoff gap | Investigate information sources |

### 2. Adverse Selection

**Question:** Is the system accepting worse interactions than it rejects?

```python
quality_gap = metrics.quality_gap(interactions)
print(f"Quality gap: {quality_gap:+.3f}")
```

| Risk Level | Quality Gap | Action |
|------------|-------------|--------|
| Low | > +0.1 | System is selecting well |
| Medium | 0 to +0.1 | Near-random selection — tune thresholds |
| **High** | **< 0** | **Adverse selection — immediate intervention** |

### 3. Variance Amplification

**Question:** Do small per-interaction risks compound dangerously?

Run multiple seeds and compare outcome distributions:

```python
import numpy as np

outcomes = []
for seed in range(20):
    config = OrchestratorConfig(n_epochs=10, steps_per_epoch=10, seed=seed)
    orch = Orchestrator(config=config)
    # ... register agents ...
    metrics = orch.run()
    outcomes.append(metrics[-1].toxicity_rate)

print(f"Toxicity: mean={np.mean(outcomes):.3f} std={np.std(outcomes):.3f}")
print(f"Worst case: {np.max(outcomes):.3f}")
```

| Risk Level | Variance | Action |
|------------|----------|--------|
| Low | std < 0.05 | Stable system |
| Medium | std 0.05-0.15 | Add governance friction |
| High | std > 0.15 | System is chaotic — add circuit breakers |

### 4. Governance Latency

**Question:** Can safety mechanisms keep pace with agent adaptation?

```python
# Compare toxicity before and after circuit breaker triggers
for epoch in history:
    if epoch.circuit_breaker_triggered:
        pre_tox = history[epoch.epoch - 1].toxicity_rate
        post_tox = history[epoch.epoch + 2].toxicity_rate  # 2 epochs later
        print(f"Breaker at epoch {epoch.epoch}: "
              f"tox {pre_tox:.3f} → {post_tox:.3f}")
```

| Risk Level | Signal | Action |
|------------|--------|--------|
| Low | Governance corrects within 1-2 epochs | Adequate |
| Medium | Correction takes 3-5 epochs | Tighten thresholds |
| High | Toxicity rebounds after correction | Governance is too slow |

## Risk Assessment Workflow

### Step 1: Baseline Run

Run a scenario without governance to establish baseline risk:

```bash
python -m swarm run scenarios/baseline.yaml --seed 42 --epochs 20 --steps 10
```

### Step 2: Metric Dashboard

Check all four risk dimensions:

```python
final = history[-1]
print(f"Toxicity:     {final.toxicity_rate:.3f}")
print(f"Quality gap:  {final.quality_gap:+.3f}")
print(f"Avg payoff:   {final.avg_payoff:.3f}")
```

### Step 3: Governance Stress Test

Apply governance and verify it improves metrics without over-governing:

```python
# Check for over-governance
if final.toxicity_rate < 0.05 and final.avg_payoff < 0.1:
    print("WARNING: Over-governance — low toxicity but also low welfare")
```

### Step 4: Adversarial Validation

Run the [red-team attack library](red-teaming.md) against your governance configuration to verify robustness.

## Risk Matrix

| | Low Toxicity | High Toxicity |
|---|---|---|
| **Positive Quality Gap** | Healthy system | Governance gap — add friction |
| **Negative Quality Gap** | Over-governance — reduce restrictions | System failure — redesign |

## See also

- [Governance Simulation](governance-simulation.md) — Test governance before deployment
- [Parameter Sweeps](parameter-sweeps.md) — Systematic parameter exploration
- [Red Teaming](red-teaming.md) — Adversarial stress testing
- [Analyzing Results](../tutorials/analyzing-results.md) — Interpret metric patterns
- [Distributional Safety](../concepts/distributional-safety.md) — Theoretical foundation
