---
description: "Metrics for measuring multi-agent system health."
---

# Metrics API

Metrics for measuring multi-agent system health.

## SoftMetrics

Core metrics computed from [soft probabilistic labels](../concepts/soft-labels.md).

::: swarm.metrics.soft_metrics.SoftMetrics
    options:
      show_root_heading: true

### Usage

```python
from swarm.metrics.soft_metrics import SoftMetrics

metrics = SoftMetrics()

# Compute individual metrics
toxicity = metrics.toxicity_rate(interactions)
quality_gap = metrics.quality_gap(interactions)
conditional_loss = metrics.conditional_loss(interactions, payoff_engine)
```

## MetricsReporter

Dual reporting of soft and hard metrics.

```python
from swarm.metrics.reporters import MetricsReporter

reporter = MetricsReporter(threshold=0.5)

# Generate report
report = reporter.format_report(interactions, verbose=True)
print(report)

# Get structured data
data = reporter.compute_all(interactions)
print(data['soft']['toxicity_rate'])
print(data['hard']['true_positive_rate'])
```

### Report Format

```
=== SWARM Metrics Report ===
Interactions: 100 (70 accepted, 30 rejected)

Soft Metrics:
  Toxicity Rate:    0.287
  Quality Gap:      0.142
  Conditional Loss: -0.051

Hard Metrics (threshold=0.5):
  Accept Rate:      0.700
  True Positive:    0.821
  False Positive:   0.179
```

## Incoherence Metrics

Measure decision variance across replays.

```python
from swarm.metrics.incoherence import IncoherenceMetrics, DecisionRecord

incoherence = IncoherenceMetrics()

# Record decisions across replays
for replay in replays:
    record = DecisionRecord(
        decision_id=decision_id,
        replay_id=replay.id,
        decision=replay.decision,
        outcome=replay.outcome,
    )
    incoherence.record(record)

# Compute incoherence index
I = incoherence.compute_index()
print(f"Incoherence Index: {I:.3f}")
```

### Incoherence Components

| Component | Formula | Meaning |
|-----------|---------|---------|
| D | Var[decision] | Decision variance |
| E | E[error] | Expected error |
| I | D / E | Incoherence index |

## Collusion Metrics

Detect coordinated behavior.

```python
from swarm.metrics.collusion import CollusionMetrics

collusion = CollusionMetrics()

# Analyze pair-level patterns
pair_scores = collusion.pair_analysis(interactions)

# Analyze group-level patterns
group_scores = collusion.group_analysis(interactions, group_size=3)

# Get suspicious pairs
suspicious = collusion.get_suspicious_pairs(threshold=0.8)
```

## Security Metrics

Track security-related signals.

```python
from swarm.metrics.security import SecurityMetrics

security = SecurityMetrics()

# Compute security scores
attack_rate = security.attack_detection_rate(interactions)
evasion_rate = security.governance_evasion_rate(interactions, governance)
damage = security.total_externality(interactions)
```

## Capability Metrics

Track emergent capabilities.

```python
from swarm.metrics.capabilities import CapabilityMetrics

capabilities = CapabilityMetrics()

# Compute capability scores
task_completion = capabilities.task_completion_rate(interactions)
collaboration_success = capabilities.collaboration_success_rate(interactions)
composite_capability = capabilities.composite_task_capability(interactions)
```

## Custom Metrics

Create custom metrics:

```python
from swarm.metrics.base import BaseMetric

class CustomMetric(BaseMetric):
    def compute(self, interactions: list) -> float:
        # Custom computation
        accepted = [i for i in interactions if i.accepted]
        return sum(i.p for i in accepted) / len(accepted) if accepted else 0.0

# Use in reporter
reporter = MetricsReporter(
    extra_metrics={'custom': CustomMetric()}
)
```

## Aggregation

Aggregate metrics across epochs or runs.

```python
from swarm.analysis.aggregation import MetricsAggregator

aggregator = MetricsAggregator()

for epoch_metrics in all_epochs:
    aggregator.add(epoch_metrics)

summary = aggregator.summary()
print(f"Mean toxicity: {summary['toxicity_mean']:.3f}")
print(f"Std toxicity: {summary['toxicity_std']:.3f}")
```
