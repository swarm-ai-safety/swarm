---
description: "Soft probabilistic labels for machine learning safety. Replace binary good/bad classifications with calibrated probabilities to capture uncertainty in multi-agent AI systems."
author: "SWARM Team"
keywords:
  - soft labels machine learning
  - probabilistic classification AI safety
  - calibrated sigmoid
  - proxy computer multi-agent
defined_terms:
  - Soft Label
  - p
  - v_hat
faq:
  - q: "What are soft labels in AI safety?"
    a: "Soft labels replace binary good/bad classifications with probabilities p ∈ [0,1], where p = P(v=+1) represents the probability an interaction is beneficial. They capture uncertainty and enable calibrated metrics."
  - q: "How is p computed from observable signals?"
    a: "Observable signals (task progress, rework count, verifier rejections, engagement) are combined by the ProxyComputer into a raw score v_hat ∈ [-1,+1], then mapped to p via a calibrated sigmoid: p = σ(k·v_hat)."
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

## Soft Labels: A Better Approach {#soft-label}

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

| Signal | Range | Default Weight | Meaning |
|--------|-------|----------------|---------|
| `task_progress_delta` | [-1, 1] | 0.4 | Forward progress on task |
| `rework_count` | [0, ∞) | 0.2 | Quality signal (penalty) |
| `verifier_rejections` | [0, ∞) | 0.2 | Safety signal (penalty) |
| `engagement_delta` | [-1, 1] | 0.2 | Counterparty response |

#### Weight Rationale {#weight-rationale}

The default weights ($w_1 = 0.4$, $w_2 = w_3 = w_4 = 0.2$) reflect a deliberate design choice:

- **Task progress gets double weight (0.4)** because it is the only directly measured outcome signal. The other three signals are indirect proxies—rework and rejections are penalizing signals that indicate something went wrong, while engagement is a noisy social signal. Giving progress twice the weight of any single penalty/social signal means that a genuinely productive interaction is not overwhelmed by a single noisy penalty channel.

- **Equal weight across penalty and social signals (0.2 each)** avoids privileging any one failure mode over another. In the absence of domain-specific calibration data, uniform weighting across the three non-primary channels is the least-assuming default.

- **All weights sum to 1.0** after normalization. This is enforced by `ProxyWeights.normalize()`, so custom weights that don't sum to 1.0 are automatically rescaled.

!!! warning "These weights propagate into every downstream metric"
    Because $\hat{v}$ is the input to the sigmoid that produces $p$, and $p$ feeds into toxicity, quality gap, payoffs, and governance decisions, the weight vector shapes all published metrics. When comparing results across studies, verify that the same weights were used, or document any differences.

#### Sensitivity and Configurability

The weights are fully configurable via `ProxyWeights`:

```python
from swarm.core.proxy import ProxyComputer, ProxyWeights

# Emphasize safety signals over progress
safety_focused = ProxyWeights(
    task_progress=0.2,
    rework_penalty=0.3,
    verifier_penalty=0.3,
    engagement_signal=0.2,
)

proxy = ProxyComputer(weights=safety_focused)
```

Weights can also be set in YAML scenario files under the `proxy` key:

```yaml
proxy:
  weights:
    task_progress: 0.2
    rework_penalty: 0.3
    verifier_penalty: 0.3
    engagement_signal: 0.2
```

No formal ablation study has been published for the default weight vector. If your application domain has labeled interaction data, we recommend tuning weights via cross-validation against ground-truth $v$ labels. See [Calibration](#calibration) below for guidance on the sigmoid parameter $k$.

#### Empirical Sensitivity {#weight-sensitivity}

We tested 7 weight configurations (default, uniform, progress-heavy, safety-heavy, engagement-heavy, and ±10% perturbations to the progress weight) across 5 representative scenarios (benign, mixed, toxic, deceptive, all-zero observables):

| Scenario | $p$ range across configs | Spread |
|----------|--------------------------|--------|
| Benign (high progress, no penalties) | [0.821, 0.853] | 0.033 |
| Mixed (moderate progress, some penalties) | [0.477, 0.569] | 0.092 |
| Toxic (negative progress, heavy penalties) | [0.259, 0.329] | 0.070 |
| Deceptive (high progress, high rejections) | [0.670, 0.738] | 0.068 |
| All-zero observables | [0.627, 0.769] | 0.141 |

**Key findings:**

- **Ordering is invariant.** The ranking benign > mixed > toxic is preserved across all 7 configurations. Deceptive agents always score below benign.
- **Local stability.** ±10% perturbations to the progress weight cause < 0.5 percentage-point change in $p$ for all scenarios.
- **Widest spread is on sparse signals.** The all-zero scenario (0.141 spread) is most sensitive because only the penalty channels carry information when progress and engagement are zero. Note: all-zero observables produce $p \approx 0.69$ because zero rework/rejections/misuse counts are treated as positive signals (absence of problems).
- **Aggressive reweighting shifts $p$ by 5–9pp** in the mixed and toxic scenarios but does not flip any qualitative classifications.

### Step 2: Compute v_hat {#v-hat}

Signals are combined into a raw score:

$$\hat{v} = w_1 \cdot \text{progress} - w_2 \cdot \text{rework} - w_3 \cdot \text{rejections} + w_4 \cdot \text{engagement}$$

The result is clamped to $[-1, +1]$.

### Step 3: Calibrated Sigmoid {#calibrated-sigmoid}

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

---

!!! quote "How to cite"
    SWARM Team. "Soft Labels in Multi-Agent Safety." *swarm-ai.org/concepts/soft-labels/*, 2026. Based on [arXiv:2512.16856](https://arxiv.org/abs/2512.16856).
