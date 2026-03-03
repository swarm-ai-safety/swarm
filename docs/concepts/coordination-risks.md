---
description: "AI agent coordination risks: how collusion, information cascades, and coordinated exploitation emerge in multi-agent systems. Detect and govern coordination failures with SWARM."
date: 2026-03-02
author: "SWARM Team"
keywords:
  - AI collusion detection
  - information cascades multi-agent
  - coordinated exploitation AI
  - cooperation vs collusion
defined_terms:
  - Collusion
  - Information Cascade
  - Coordinated Exploitation
faq:
  - q: "What is the difference between cooperation and collusion in AI systems?"
    a: "Cooperation improves system welfare with transparent signaling; collusion extracts from system welfare via concealed coordination. SWARM's quality gap metric distinguishes them: coordinated agents with negative quality gap are colluding."
  - q: "How does SWARM detect AI agent collusion?"
    a: "SWARM monitors pairwise interaction patterns for suspiciously correlated exploitation timing. Agent pairs exceeding a correlation threshold over a sliding window are flagged for potential collusion."
---

# Coordination Risks

When multiple AI agents interact, coordination can be beneficial (cooperation) or harmful (collusion). SWARM studies the boundary between the two — and provides governance mechanisms to keep coordination constructive.

## Why Coordination Becomes Risky

Individual agents acting independently produce risks that scale linearly. Coordinated agents produce risks that scale **combinatorially**. Three failure patterns dominate:

### 1. Collusion {#collusion}

Two or more agents coordinate to extract value at the expense of others. In SWARM, this appears as correlated exploitation patterns:

```python
from swarm.governance import GovernanceConfig

config = GovernanceConfig(
    collusion_detection=True,
    collusion_threshold=0.8,   # flag pairs with >80% correlation
    collusion_window=20,       # over 20 interactions
)
```

**Detection signal:** Unusually high correlation between agent pairs' exploitation timing.

### 2. Information Cascades {#information-cascade}

Agents copy each other's behavior rather than acting on private signals. When the first few agents make a mistake, the entire population follows:

| Phase | Behavior | Risk |
|-------|----------|------|
| Seed | 2-3 agents adopt strategy | Low |
| Cascade | Population copies without evaluation | Growing |
| Lock-in | Wrong strategy becomes consensus | High |

**Detection signal:** Sudden homogenization of agent strategies within 1-2 epochs.

### 3. Coordinated Exploitation {#coordinated-exploitation}

A group of agents systematically targets specific counterparties or exploits governance gaps that only work with multiple participants.

**Detection signal:** Subgroup of agents with consistently high payoffs while specific counterparties suffer.

## Measuring Coordination Risk

SWARM provides metrics for coordination health:

```python
from swarm.metrics.soft_metrics import SoftMetrics

metrics = SoftMetrics()

# Check for pairwise exploitation correlation
for pair in agent_pairs:
    correlation = metrics.pairwise_correlation(interactions, pair)
    if correlation > 0.8:
        print(f"Potential collusion: {pair} (r={correlation:.3f})")
```

## Governance Countermeasures

| Mechanism | What it addresses | Configuration |
|-----------|-------------------|---------------|
| [Collusion detection](governance.md) | Coordinated exploitation | `collusion_threshold`, `collusion_window` |
| [Transaction tax](governance.md) | Reduces volume of coordinated interactions | `transaction_tax` |
| [Random audits](governance.md) | Probabilistic detection of any pattern | `audit_probability` |
| [Reputation decay](governance.md) | Prevents coordinated trust accumulation | `reputation_decay` |

## The Cooperation-Collusion Boundary

Not all coordination is harmful. The challenge is distinguishing:

| Cooperation (beneficial) | Collusion (harmful) |
|--------------------------|---------------------|
| Improves system welfare | Extracts from system welfare |
| Transparent signaling | Concealed coordination |
| Positive quality gap | Negative quality gap |
| Others can participate | Exclusive to in-group |

SWARM's [quality gap metric](metrics.md) helps distinguish these: when coordinated agents produce a negative quality gap, the system is selecting for harm.

## See also

- [Governance Mechanisms](governance.md) — Collusion detection and other countermeasures
- [Deception](deception.md) — When coordination involves misrepresentation
- [Governance Simulation](../guides/governance-simulation.md) — Test coordination scenarios before deployment
- [Red-Teaming Guide](../guides/red-teaming.md) — Adversarial coordination attack patterns
