---
description: "SWARM uses soft labels instead of binary classifications. This page explains why and how."
---

# Soft Probabilistic Labels

SWARM uses **soft labels** instead of binary classifications. This page explains why and how.

## The Problem with Binary Labels

Traditional systems classify interactions as simply "good" or "bad":

```
Is this interaction beneficial? → YES or NO
```

This approach fails because:

1. **Uncertainty is real** - We often don't know for sure
2. **Quality is gradual** - Interactions exist on a spectrum
3. **Calibration matters** - We need to know when we're confident

## Soft Labels: A Better Approach

Instead of binary, SWARM uses:

$$p = P(v = +1)$$

Where:

- $p \in [0, 1]$ is a probability
- $v = +1$ means the interaction is beneficial
- $v = -1$ means the interaction is harmful

## The Proxy Pipeline

Soft labels are computed from observable signals:

```
┌─────────────┐     ┌───────────────┐     ┌─────────┐     ┌───┐
│ Observables │ ──► │ ProxyComputer │ ──► │ Sigmoid │ ──► │ p │
└─────────────┘     └───────────────┘     └─────────┘     └───┘
```

### Step 1: Observable Signals

The `ProxyComputer` takes four signals:

| Signal | Range | Weight | Meaning |
|--------|-------|--------|---------|
| `task_progress_delta` | [-1, 1] | 0.4 | Forward progress on task |
| `rework_count` | [0, ∞) | 0.2 | Quality signal (penalty) |
| `verifier_rejections` | [0, ∞) | 0.2 | Safety signal (penalty) |
| `engagement_delta` | [-1, 1] | 0.2 | Counterparty response |

### Step 2: Compute v_hat

Signals are combined into a raw score:

$$\hat{v} = w_1 \cdot \text{progress} - w_2 \cdot \text{rework} - w_3 \cdot \text{rejections} + w_4 \cdot \text{engagement}$$

The result is clamped to $[-1, +1]$.

### Step 3: Calibrated Sigmoid

The raw score is converted to probability:

$$p = \sigma(k \cdot \hat{v}) = \frac{1}{1 + e^{-k \cdot \hat{v}}}$$

Where $k$ is a calibration parameter (default: 3.0).

## Code Example

```python
from swarm.core.proxy import ProxyComputer, ProxyObservables

# Create observable signals
obs = ProxyObservables(
    task_progress_delta=0.7,   # Good progress
    rework_count=1,             # Some rework needed
    verifier_rejections=0,      # No safety issues
    counterparty_engagement_delta=0.4,  # Positive engagement
)

# Compute soft labels
proxy = ProxyComputer()
v_hat, p = proxy.compute_labels(obs)

print(f"Raw score (v_hat): {v_hat:.3f}")
print(f"Probability (p): {p:.3f}")
```

## Why This Matters

### For Metrics

Soft labels enable nuanced metrics:

- **Toxicity** = $E[1-p \mid \text{accepted}]$ — uses probability, not binary
- **[Quality gap](../research/theory.md)** = can detect subtle adverse selection

### For Payoffs

Expected outcomes instead of worst-case:

- **[Expected surplus](../tutorials/understanding-soft-labels.md)** = $p \cdot s_+ - (1-p) \cdot s_-$
- **Expected harm** = $(1-p) \cdot h$

### For Governance

Proportional responses:

- Low-p interactions get more scrutiny
- Gradual reputation effects
- Calibrated thresholds

## Calibration

The sigmoid parameter $k$ controls how "sharp" the probability curve is:

- **Low k (e.g., 1.0)**: Gradual transitions, high uncertainty
- **High k (e.g., 5.0)**: Sharp transitions, more confident

!!! tip "Calibration in Practice"
    The default $k=3.0$ works well for most scenarios. Adjust if you have ground truth labels to calibrate against.

## See also

- [Metrics](metrics.md) — How soft labels enable toxicity, quality gap, and conditional loss
- [Payoff Engine](../api/core.md) — How the SoftPayoffEngine computes payoffs from p values
- [Theoretical Foundations](../research/theory.md) — Mathematical basis for distributional labels
- [Understanding Soft Labels](../tutorials/understanding-soft-labels.md) — Hands-on tutorial with code examples
