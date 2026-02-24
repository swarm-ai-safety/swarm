# Analyzing Results

After running a SWARM experiment, you need to interpret what the numbers mean. This tutorial covers the four key metrics, common patterns, and diagnostic strategies.

**Time:** ~20 minutes | **Level:** Intermediate

---

## The Four Key Metrics

### 1. Toxicity Rate

$$\text{Toxicity} = E[1-p \mid \text{accepted}]$$

**What it tells you:** The expected fraction of harm among accepted interactions.

| Range | Interpretation |
|-------|----------------|
| 0.0 – 0.1 | Healthy ecosystem |
| 0.1 – 0.3 | Mild concern |
| 0.3 – 0.5 | Significant degradation |
| > 0.5 | System failure |

**In code:**

```python
from swarm.metrics.soft_metrics import SoftMetrics

metrics = SoftMetrics()
toxicity = metrics.toxicity_rate(interactions)
```

---

### 2. Quality Gap

$$\text{Quality Gap} = E[p \mid \text{accepted}] - E[p \mid \text{rejected}]$$

**What it tells you:** Whether your governance mechanism is selecting for quality.

| Sign | Interpretation |
|------|----------------|
| Positive | Healthy selection — accepted > rejected |
| Zero | Random selection |
| **Negative** | **Adverse selection — accepting worse interactions** |

!!! danger "Adverse Selection"
    A **negative quality gap** is a critical failure mode. It means the system is preferentially
    admitting lower-quality interactions than it's rejecting. This is the *opposite* of what
    governance should achieve.

**In code:**

```python
quality_gap = metrics.quality_gap(interactions)
if quality_gap < 0:
    print("⚠️  Adverse selection detected!")
```

---

### 3. Conditional Loss

$$\text{Conditional Loss} = E[\pi \mid \text{accepted}] - E[\pi]$$

**What it tells you:** How much better (or worse) accepted interactions are for agent payoffs compared to the baseline.

- **Positive**: The selection mechanism adds value — accepted interactions pay better
- **Negative**: Something is selecting for low-payoff interactions

---

### 4. Mean Agent Payoff

The average payoff across all agents over the run.

Use this to understand **efficiency** — governance costs reduce mean payoff. Track whether the safety gain (lower toxicity, positive quality gap) is worth the efficiency cost.

---

## Identifying Common Patterns

### Pattern 1: Governance Working Well

```
Toxicity:     0.08  ✅ (low)
Quality gap:  +0.22 ✅ (positive, healthy selection)
Mean payoff:  0.71  (good efficiency)
```

Low toxicity + positive quality gap = governance is filtering harmful interactions.

---

### Pattern 2: Adverse Selection Failure

```
Toxicity:     0.44  ❌ (high)
Quality gap:  -0.18 ❌ (negative — adverse selection)
Mean payoff:  0.83  (high, but false)
```

High mean payoff with high toxicity and negative quality gap: the system is rewarding harmful behavior. A common cause is insufficient governance (tax too low, no circuit breaker) or deceptive agents gaming the reputation system.

---

### Pattern 3: Over-Governance Collapse

```
Toxicity:     0.05  ✅ (low)
Quality gap:  +0.31 ✅ (positive)
Mean payoff:  0.12  ❌ (very low)
```

Governance is *working* but too aggressively — it's taxing so heavily that beneficial interactions also dry up. The solution is usually to reduce tax rate or raise the audit threshold.

---

### Pattern 4: Ecosystem Collapse

```
Toxicity:     0.71  ❌
Quality gap:  -0.52 ❌
Mean payoff:  -0.08 ❌
Interactions: 3/epoch (vs 40 expected)
```

All metrics in the red, very low interaction volume. This is ecosystem collapse — agents have learned that participation is not worth the risk. Recovery requires resetting reputation or significantly changing governance parameters.

---

## Comparing Conditions

When comparing multiple scenarios, use a summary table:

```python
from swarm.scenarios import load_scenario, build_orchestrator
from swarm.metrics.soft_metrics import SoftMetrics

results = {}
for name in ["baseline", "taxed", "strict"]:
    scenario = load_scenario(f"scenarios/{name}.yaml")
    orch = build_orchestrator(scenario)
    history = orch.run()
    results[name] = history

# Print comparison
print(f"{'Scenario':12s} {'Toxicity':10s} {'Q.Gap':8s} {'Payoff':8s}")
print("-" * 42)
for name, r in results.items():
    print(f"{name:12s} {r['toxicity_rate']:.3f}      {r['quality_gap']:+.3f}    {r['mean_payoff']:.3f}")
```

---

## Time-Series Analysis

Metrics that look acceptable on average can hide transient failures. Plot metrics over epochs:

```python
import matplotlib.pyplot as plt
from swarm.analysis.export import export_to_json

# After running, the orchestrator logs epoch-level metrics
epoch_toxicity = [e.toxicity_rate for e in history]

plt.figure(figsize=(10, 4))
plt.plot(epoch_toxicity, label="Toxicity Rate")
plt.axhline(0.3, color="red", linestyle="--", label="Warning threshold")
plt.xlabel("Epoch")
plt.ylabel("Toxicity Rate")
plt.title("Toxicity Rate Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("toxicity_time_series.png")
```

Look for:

- **Rising trend**: governance is losing control over time
- **Spike then recovery**: circuit breaker triggered and worked
- **Spike and no recovery**: circuit breaker triggered but failed

---

## Per-Agent Breakdown

Aggregate metrics can mask individual agent behavior. Check per-agent stats:

```python
# After a run, `history` is a list of EpochMetrics objects.
# Each epoch contains per-agent stats that you can aggregate.
agent_stats = {}

for epoch in history:
    for agent_id, stats in epoch.agent_stats.items():
        agg = agent_stats.setdefault(
            agent_id,
            {"toxicity": [], "mean_p": [], "reputation": [], "n_interactions": 0},
        )
        agg["toxicity"].append(stats["toxicity"])
        agg["mean_p"].append(stats["mean_p"])
        agg["reputation"].append(stats["reputation"])
        agg["n_interactions"] += stats["n_interactions"]

for agent_id, agg in agent_stats.items():
    mean_toxicity = sum(agg["toxicity"]) / len(agg["toxicity"])
    if mean_toxicity > 0.4:
        print(f"High-toxicity agent: {agent_id}")
        print(f"  Interactions: {agg['n_interactions']}")
        print(f"  Mean p: {sum(agg['mean_p']) / len(agg['mean_p']):.3f}")
        print(f"  Final reputation: {agg['reputation'][-1]:.3f}")
```

Deceptive agents typically show:

- High `mean_p` early (building reputation)
- Sharp drop in `mean_p` later (exploitation phase)
- Eventually: very low reputation or circuit-breaker freeze

---

## Checking Reproducibility

Good experimental practice requires checking that your results are stable across seeds:

```python
import statistics

seeds = [42, 123, 456, 789, 1337]
toxicity_values = []

for seed in seeds:
    scenario = load_scenario("scenarios/taxed.yaml")
    scenario.simulation.seed = seed
    orch = build_orchestrator(scenario)
    result = orch.run()
    toxicity_values.append(result[-1].toxicity_rate)

print(f"Toxicity: mean={statistics.mean(toxicity_values):.3f}, "
      f"std={statistics.stdev(toxicity_values):.3f}")
```

If the standard deviation is larger than the effect size you're measuring, you need more seeds.

---

## What's Next?

- **Vary parameters systematically**: [Parameter Sweeps](../guides/parameter-sweeps.md)
- **Advanced governance levers**: [Custom Governance Levers](../guides/governance-levers.md)
- **When results generalize**: [Transferability Considerations](../guides/transferability.md)
- **Full metrics reference**: [Metrics Concept](../concepts/metrics.md)
