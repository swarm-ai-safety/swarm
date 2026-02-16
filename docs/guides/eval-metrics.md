# Evaluation Metrics Suite

This document describes the evaluation metrics suite for SWARM experiments, providing standardized measurements for success rate, efficiency, behavior patterns, audit effectiveness, and deception detection.

## Overview

The metrics suite (`swarm.evaluation.eval_metrics`) provides six core functions for evaluating agent and system performance:

1. **success_rate** - Measures the fraction of attempts that succeed
2. **calls_per_success** - Efficiency metric (API calls per successful outcome)
3. **loopiness_score** - Detects repetitive/circular behavior patterns
4. **audit_effectiveness** - Measures audit/governance detection capability
5. **deception_detection_rate** - Evaluates ability to detect deceptive agents
6. **aggregate_success_metrics** - Aggregates success metrics across experiments

## Metrics Reference

### success_rate

Computes the fraction of attempts that succeed.

**Signature:**
```python
def success_rate(
    attempts: List[Dict],
    success_key: str = "success",
) -> float
```

**Parameters:**
- `attempts`: List of attempt records with success indicators
- `success_key`: Key in each record indicating success (default: "success")

**Returns:**
- Success rate in [0, 1], or 0.0 if no attempts

**Example:**
```python
from swarm.evaluation import success_rate

attempts = [
    {"action": "verify", "success": True},
    {"action": "verify", "success": False},
    {"action": "verify", "success": True},
]
rate = success_rate(attempts)  # 0.667 (2 out of 3)
```

**Use Cases:**
- Measuring task completion rates
- Evaluating intervention effectiveness
- Comparing agent performance across conditions

---

### calls_per_success

Computes average number of calls required per successful outcome. This is an efficiency metric—lower is better.

**Signature:**
```python
def calls_per_success(
    attempts: List[Dict],
    success_key: str = "success",
    calls_key: str = "calls",
) -> float
```

**Parameters:**
- `attempts`: List of attempt records with success and calls data
- `success_key`: Key indicating success (default: "success")
- `calls_key`: Key indicating number of calls made (default: "calls")

**Returns:**
- Average calls per success, or float('inf') if no successes

**Example:**
```python
from swarm.evaluation import calls_per_success

attempts = [
    {"success": True, "calls": 3},
    {"success": False, "calls": 5},
    {"success": True, "calls": 2},
]
efficiency = calls_per_success(attempts)  # 2.5 (average of 3 and 2)
```

**Use Cases:**
- Measuring resource efficiency
- Comparing API usage across strategies
- Identifying optimization opportunities

---

### loopiness_score

Measures repetitive/circular behavior in action sequences. Detects when agents get stuck in loops, which may indicate lack of progress, failure modes, or inefficient exploration.

**Signature:**
```python
def loopiness_score(
    action_sequence: List[str],
    window_size: int = 5,
) -> float
```

**Parameters:**
- `action_sequence`: Ordered list of actions taken
- `window_size`: Size of sliding window to check for repetition (default: 5)

**Returns:**
- Loopiness score in [0, 1]:
  - 0.0 = no repetition
  - 1.0 = maximal repetition
- Returns 0.0 if sequence is too short

**Example:**
```python
from swarm.evaluation import loopiness_score

# Agent making progress
productive = ["analyze", "query", "verify", "synthesize", "report"]
score1 = loopiness_score(productive)  # 0.0

# Agent stuck in a loop
stuck = ["query", "retry", "query", "retry", "query", "retry"]
score2 = loopiness_score(stuck, window_size=4)  # 1.0
```

**Use Cases:**
- Detecting stuck agents
- Identifying failure modes
- Measuring exploration efficiency

---

### audit_effectiveness

Measures effectiveness of audit mechanisms. Computes precision, recall, and F1 score for audit detection.

**Signature:**
```python
def audit_effectiveness(
    audited_actions: List[Dict],
    all_malicious_actions: List[Dict],
    action_id_key: str = "action_id",
) -> Dict[str, float]
```

**Parameters:**
- `audited_actions`: Actions flagged by audit
- `all_malicious_actions`: Ground truth set of malicious actions
- `action_id_key`: Key to match actions (default: "action_id")

**Returns:**
Dictionary with:
- `precision`: TP / (TP + FP) - fraction of audited actions that were actually malicious
- `recall`: TP / (TP + FN) - fraction of malicious actions that were detected
- `f1_score`: Harmonic mean of precision and recall
- `detection_rate`: Same as recall

**Example:**
```python
from swarm.evaluation import audit_effectiveness

audited = [
    {"action_id": "a1"},
    {"action_id": "a2"},
    {"action_id": "a3"},
]
malicious = [
    {"action_id": "a1"},
    {"action_id": "a2"},
    {"action_id": "a4"},
]
metrics = audit_effectiveness(audited, malicious)
# precision: 0.667 (2 out of 3 audited were malicious)
# recall: 0.667 (2 out of 3 malicious were caught)
```

**Use Cases:**
- Evaluating governance mechanisms
- Measuring detection capability
- Tuning audit parameters

---

### deception_detection_rate

Measures ability to detect deceptive agent behavior.

**Signature:**
```python
def deception_detection_rate(
    agent_behaviors: List[Dict],
    deceptive_key: str = "is_deceptive",
    detected_key: str = "detected_as_deceptive",
) -> Dict[str, float]
```

**Parameters:**
- `agent_behaviors`: List of agent behavior records
- `deceptive_key`: Key indicating if behavior was actually deceptive
- `detected_key`: Key indicating if behavior was detected as deceptive

**Returns:**
Dictionary with:
- `true_positive_rate`: Fraction of deceptive behaviors detected (sensitivity)
- `false_positive_rate`: Fraction of honest behaviors wrongly flagged
- `accuracy`: Overall classification accuracy
- `detection_rate`: Same as true_positive_rate

**Example:**
```python
from swarm.evaluation import deception_detection_rate

behaviors = [
    {"is_deceptive": True, "detected_as_deceptive": True},   # TP
    {"is_deceptive": True, "detected_as_deceptive": False},  # FN
    {"is_deceptive": False, "detected_as_deceptive": False}, # TN
    {"is_deceptive": False, "detected_as_deceptive": True},  # FP
]
metrics = deception_detection_rate(behaviors)
# true_positive_rate: 0.5
# false_positive_rate: 0.5
# accuracy: 0.5
```

**Use Cases:**
- Evaluating deception detection systems
- Measuring adversary detection capability
- Balancing false positives vs false negatives

---

### aggregate_success_metrics

Aggregates success metrics across multiple experiments for statistical analysis.

**Signature:**
```python
def aggregate_success_metrics(
    experiments: List[Dict],
    success_key: str = "success",
) -> Dict[str, float]
```

**Parameters:**
- `experiments`: List of experiment results, each with attempts
- `success_key`: Key indicating success in each attempt

**Returns:**
Dictionary with:
- `mean_success_rate`: Average success rate across experiments
- `std_success_rate`: Standard deviation of success rates
- `min_success_rate`: Minimum success rate observed
- `max_success_rate`: Maximum success rate observed
- `total_attempts`: Total number of attempts across all experiments
- `total_successes`: Total number of successes across all experiments

**Example:**
```python
from swarm.evaluation import aggregate_success_metrics

experiments = [
    {"attempts": [{"success": True}, {"success": False}]},  # 50%
    {"attempts": [{"success": True}, {"success": True}]},   # 100%
]
metrics = aggregate_success_metrics(experiments)
# mean_success_rate: 0.75
# std_success_rate: 0.25
# min_success_rate: 0.5
# max_success_rate: 1.0
```

**Use Cases:**
- Statistical analysis across conditions
- Comparing interventions
- Reporting aggregate results

## Integration with Evaluation Pipeline

These metrics are designed to work with the SWARM evaluation framework:

```python
from swarm.evaluation import (
    success_rate,
    audit_effectiveness,
    ReviewPipeline,
)

# Use metrics in evaluation
data = {
    "attempts": [...],
    "audited": [...],
    "malicious": [...],
}

# Compute metrics
s_rate = success_rate(data["attempts"])
audit_metrics = audit_effectiveness(data["audited"], data["malicious"])

# Include in review
submission_data = {
    "success_rate": s_rate,
    "audit_precision": audit_metrics["precision"],
    # ... other evaluation data
}
```

## Design Principles

1. **Flexible input formats**: Metrics accept dictionaries with configurable keys
2. **Robust defaults**: Return sensible values for edge cases (empty inputs, etc.)
3. **Standard ranges**: Most metrics return values in [0, 1] for consistency
4. **Composable**: Metrics can be combined for comprehensive analysis
5. **Type hints**: Full type annotations for IDE support
6. **Documented**: Extensive docstrings with examples

## Testing

All metrics have comprehensive regression tests covering:
- Empty/invalid inputs
- Edge cases (single value, all same, etc.)
- Custom key names
- Multiple data distributions
- Boundary conditions

Run tests:
```bash
python -m pytest tests/test_eval_metrics.py -v
```

## Related Documentation

- [Evaluation Framework](./evaluation-framework.md)
- [Red Team Metrics](../guides/red-teaming.md)
- [Governance Metrics](../governance.md)

## Example Usage

See `examples/eval_metrics_usage.py` for complete working examples of all metrics.
