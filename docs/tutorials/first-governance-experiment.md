# Your First Governance Experiment

In this tutorial you will build and run a governance experiment from scratch. By the end you will:

- Understand SWARM's agent types and how they interact
- Write a scenario YAML file
- Run the simulation and read the output
- Interpret the key governance metrics

**Time:** ~20 minutes | **Level:** Beginner

---

## Step 1: What Are We Testing?

Let's start with a concrete question:

> **Does a transaction tax reduce harmful behavior in a mixed-agent ecosystem?**

We'll compare two conditions:

| Condition | Tax |
|-----------|-----|
| Baseline  | 0%  |
| Taxed     | 5%  |

Both conditions will have the same mix of honest, opportunistic, and deceptive agents.

---

## Step 2: Create the Baseline Scenario

Create a file `scenarios/tutorial_baseline.yaml`:

```yaml
name: tutorial_baseline
description: Tutorial - no governance (baseline)

simulation:
  seed: 42
  n_epochs: 20
  steps_per_epoch: 15

agents:
  - type: honest
    count: 5
    id_prefix: h
  - type: opportunistic
    count: 3
    id_prefix: opp
  - type: deceptive
    count: 2
    id_prefix: dec

governance:
  transaction_tax: 0.0
  reputation_decay: 0.05
  circuit_breaker_threshold: 0.5
  circuit_breaker_enabled: false

payoff:
  s_plus: 2.0
  s_minus: 1.0
  h: 2.0
  theta: 0.5
```

!!! tip "Reproducibility"
    Always set a `seed`. This makes your experiment reproducible — running the same YAML with
    the same seed always produces the same results.

---

## Step 3: Create the Taxed Scenario

Create `scenarios/tutorial_taxed.yaml` — identical except for the governance block:

```yaml
name: tutorial_taxed
description: Tutorial - 5% transaction tax

simulation:
  seed: 42          # Same seed for fair comparison
  n_epochs: 20
  steps_per_epoch: 15

agents:
  - type: honest
    count: 5
    id_prefix: h
  - type: opportunistic
    count: 3
    id_prefix: opp
  - type: deceptive
    count: 2
    id_prefix: dec

governance:
  transaction_tax: 0.05   # 5% tax on each interaction
  reputation_decay: 0.05
  circuit_breaker_threshold: 0.5
  circuit_breaker_enabled: false

payoff:
  s_plus: 2.0
  s_minus: 1.0
  h: 2.0
  theta: 0.5
```

---

## Step 4: Run Both Scenarios

```bash
# Run baseline
python -m swarm run scenarios/tutorial_baseline.yaml --seed 42

# Run taxed version
python -m swarm run scenarios/tutorial_taxed.yaml --seed 42
```

Or run them programmatically:

```python
from swarm.scenarios import load_scenario, build_orchestrator

for name in ["tutorial_baseline", "tutorial_taxed"]:
    scenario = load_scenario(f"scenarios/{name}.yaml")
    orchestrator = build_orchestrator(scenario)
    epoch_metrics = orchestrator.run()[-1]

    print(f"\n=== {name} ===")
    print(f"Toxicity rate:  {epoch_metrics.toxicity_rate:.3f}")
    print(f"Quality gap:    {epoch_metrics.quality_gap:.3f}")
    print(f"Mean payoff:    {epoch_metrics.avg_payoff:.3f}")
```

---

## Step 5: Understand the Output

A typical run prints something like:

```
=== tutorial_baseline ===
Toxicity rate:  0.312
Quality gap:    -0.087
Mean payoff:    0.441

=== tutorial_taxed ===
Toxicity rate:  0.198
Quality gap:    0.043
Mean payoff:    0.389
```

### Reading the Numbers

**Toxicity rate** (`E[1-p | accepted]`) — the expected fraction of harm in accepted interactions.

- Baseline: `0.312` → about 31% of accepted interactions cause harm
- Taxed: `0.198` → tax reduced harm by ~37%

**Quality gap** (`E[p | accepted] - E[p | rejected]`) — are we selecting good interactions?

- Baseline: `-0.087` → **adverse selection**: we're accepting *worse* interactions than we reject
- Taxed: `+0.043` → tax flipped this! Now accepted interactions are better than rejected ones

**Mean payoff** — average agent payoff.

- The taxed scenario has lower mean payoff (`0.389` vs `0.441`): agents pay the tax
- But the ecosystem is healthier: less harm, better selection

!!! info "The Governance Trade-off"
    This is the core trade-off in distributional safety: governance mechanisms often reduce
    average efficiency to improve distributional safety. The question is always: *is the
    safety gain worth the efficiency cost?*

---

## Step 6: What Just Happened?

Let's trace through the simulation mechanics:

```
1. Each step: agents propose interactions
2. ProxyComputer computes p for each interaction
3. SoftPayoffEngine applies payoffs (including tax)
4. Reputation updates based on outcomes
5. Next step: agents observe reputation signals
```

The transaction tax works by making exploitative interactions less profitable. Opportunistic and deceptive agents will still try — but with lower expected payoff, they do so less aggressively.

---

## What's Next?

- **Go deeper on the math**: [Understanding Soft Labels](understanding-soft-labels.md)
- **Interpret your results**: [Analyzing Results](analyzing-results.md)
- **Vary more parameters**: [Parameter Sweeps](../guides/parameter-sweeps.md)
- **Add circuit breakers and staking**: [Custom Governance Levers](../guides/governance-levers.md)
