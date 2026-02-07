# Core Concepts

SWARM is built on several key ideas that distinguish it from traditional AI safety approaches.

## The Central Thesis

**AGI-level risks don't require AGI-level agents.** Catastrophic failures can emerge from the *interaction* of many sub-AGI agents—even when none are individually dangerous.

This happens through:

- **Information asymmetry** - Some agents know things others don't
- **Adverse selection** - Bad interactions get accepted more often than good ones
- **Variance amplification** - Small errors compound across decisions
- **Governance lag** - Safety mechanisms react too slowly

## Soft Probabilistic Labels

Instead of binary classifications (good/bad, safe/unsafe), SWARM uses **soft labels**:

$$p = P(v = +1)$$

Where $p \in [0, 1]$ represents the probability that an interaction is beneficial.

This captures:

- **Uncertainty** about outcomes
- **Gradations** of quality
- **Calibration** requirements

[Learn more about soft labels →](soft-labels.md)

## Four Key Metrics

| Metric | Formula | What It Reveals |
|--------|---------|-----------------|
| **Toxicity** | $E[1-p \mid \text{accepted}]$ | Expected harm in accepted interactions |
| **Quality Gap** | $E[p \mid \text{accepted}] - E[p \mid \text{rejected}]$ | Adverse selection (negative = bad) |
| **Conditional Loss** | $E[\pi \mid \text{accepted}] - E[\pi]$ | Selection effects on payoffs |
| **Incoherence** | $\text{Var}[\text{decision}] / E[\text{error}]$ | Variance-to-error ratio |

[Learn more about metrics →](metrics.md)

## Governance Mechanisms

SWARM provides configurable safety interventions:

- **Transaction Taxes** - Add friction to reduce exploitation
- **Reputation Decay** - Make past bad behavior costly
- **Circuit Breakers** - Freeze agents exhibiting toxic patterns
- **Random Audits** - Deter hidden exploitation
- **Staking** - Require skin in the game
- **Collusion Detection** - Identify coordinated attacks

[Learn more about governance →](governance.md)

## The Emergence Problem

Single-agent alignment asks: "How do we align one powerful agent?"

SWARM asks: "What happens when many agents—each potentially aligned—interact in ways that produce misaligned outcomes?"

This is the **emergence problem**: system-level failures that aren't predictable from individual agent properties.

[Learn more about emergence →](emergence.md)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       SWARM CORE                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Observables → ProxyComputer → v_hat → sigmoid → p          │
│                                                   ↓          │
│                                              SoftInteraction │
│                                                   ↓          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ PayoffEngine │    │  Governance  │    │   Metrics    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

<div class="grid cards" markdown>

-   :material-label-outline:{ .lg .middle } **Soft Labels**

    ---

    Understand probabilistic quality assessment

    [:octicons-arrow-right-24: Soft Labels](soft-labels.md)

-   :material-chart-line:{ .lg .middle } **Metrics**

    ---

    Learn the four key metrics and what they measure

    [:octicons-arrow-right-24: Metrics](metrics.md)

-   :material-shield-check:{ .lg .middle } **Governance**

    ---

    Explore safety mechanisms and interventions

    [:octicons-arrow-right-24: Governance](governance.md)

-   :material-waves:{ .lg .middle } **Emergence**

    ---

    Understand system-level failure modes

    [:octicons-arrow-right-24: Emergence](emergence.md)

-   :material-sync:{ .lg .middle } **Recursive Research**

    ---

    When agents study agents studying agents

    [:octicons-arrow-right-24: Recursive Research](recursive-research.md)

</div>
