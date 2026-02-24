# Understanding Soft Labels

SWARM uses **soft (probabilistic) labels** instead of binary good/bad classifications. This tutorial builds intuition for why, then walks through the math and code.

**Time:** ~15 minutes | **Level:** Intermediate

---

## The Motivation

Imagine you're watching an AI agent complete a task. How do you know if the interaction was *good* or *bad*?

In reality:

- You might be **uncertain** about the outcome
- Quality exists on a **spectrum** — not everything is perfectly good or perfectly bad
- You need **calibration** — knowing when you're confident vs. guessing

Binary labels (`good = 1`, `bad = 0`) throw away this uncertainty.

---

## The Soft Label: p

SWARM represents interaction quality as:

$$p = P(v = +1)$$

Where:

- $v = +1$ means the interaction was beneficial
- $v = -1$ means it was harmful
- $p \in [0, 1]$ is a probability

**Examples:**

| Situation | p value |
|-----------|---------|
| Clearly beneficial | 0.92 |
| Probably fine, some doubt | 0.71 |
| Uncertain | 0.50 |
| Likely harmful | 0.23 |
| Clearly harmful | 0.04 |

!!! note "The Safety Invariant"
    `p` must remain in `[0, 1]` everywhere it is used or logged. This is a hard invariant
    in SWARM — never compute a payoff or metric with `p` outside this range.

---

## How p Is Computed

The `ProxyComputer` converts observable signals to `p` in three steps.

### Step 1: Observable Signals

Four signals are measured for each interaction:

```python
from swarm.core.proxy import ProxyObservables

obs = ProxyObservables(
    task_progress_delta=0.7,           # Good progress made
    rework_count=1,                     # One rework cycle needed
    verifier_rejections=0,              # Passed safety check
    counterparty_engagement_delta=0.4,  # Positive engagement
)
```

| Signal | Range | What it captures |
|--------|-------|-----------------|
| `task_progress_delta` | [-1, 1] | Did we move forward? |
| `rework_count` | [0, ∞) | How many corrections? (penalty) |
| `verifier_rejections` | [0, ∞) | How many safety flags? (penalty) |
| `counterparty_engagement_delta` | [-1, 1] | Did the other party respond well? |

### Step 2: Weighted Combination → v_hat

Signals are combined with weights into a raw score $\hat{v} \in [-1, +1]$:

$$\hat{v} = w_1 \cdot \text{progress} - w_2 \cdot \text{rework} - w_3 \cdot \text{rejections} + w_4 \cdot \text{engagement}$$

Default weights: `[0.4, 0.2, 0.2, 0.2]`

```python
from swarm.core.proxy import ProxyComputer

proxy = ProxyComputer()
v_hat, p = proxy.compute_labels(obs)

print(f"v_hat = {v_hat:.3f}")  # Raw score in [-1, 1]
print(f"p     = {p:.3f}")      # Probability in [0, 1]
```

### Step 3: Calibrated Sigmoid → p

The raw score is converted to a probability:

$$p = \sigma(k \cdot \hat{v}) = \frac{1}{1 + e^{-k\hat{v}}}$$

The calibration parameter $k$ (default: 3.0) controls sharpness:

```
k = 1.0: gradual curve, more uncertainty
k = 3.0: default, balanced
k = 5.0: sharp curve, more confident
```

---

## Interactive Example

Let's compare three interactions:

```python
from swarm.core.proxy import ProxyComputer, ProxyObservables

proxy = ProxyComputer()

scenarios = {
    "Excellent": ProxyObservables(
        task_progress_delta=0.9,
        rework_count=0,
        verifier_rejections=0,
        counterparty_engagement_delta=0.8,
    ),
    "Borderline": ProxyObservables(
        task_progress_delta=0.3,
        rework_count=2,
        verifier_rejections=1,
        counterparty_engagement_delta=0.1,
    ),
    "Problematic": ProxyObservables(
        task_progress_delta=-0.2,
        rework_count=5,
        verifier_rejections=3,
        counterparty_engagement_delta=-0.4,
    ),
}

for name, obs in scenarios.items():
    v_hat, p = proxy.compute_labels(obs)
    print(f"{name:12s}: v_hat={v_hat:+.3f}, p={p:.3f}")
```

Output:

```
Excellent   : v_hat=+0.780, p=0.901
Borderline  : v_hat=-0.080, p=0.440
Problematic : v_hat=-0.680, p=0.128
```

---

## How Soft Labels Affect Payoffs

Once we have `p`, every downstream calculation uses the **expected value** rather than a binary outcome.

### Expected Surplus

$$S_\text{soft} = p \cdot s_+ - (1-p) \cdot s_-$$

If $s_+ = 2.0$ and $s_- = 1.0$:

| p | Expected Surplus |
|---|-----------------|
| 0.9 | 1.7 |
| 0.5 | 0.5 |
| 0.1 | -0.8 |

### Expected Harm Externality

$$E_\text{harm} = (1-p) \cdot h$$

High-p interactions produce little externality; low-p interactions are taxed heavily.

---

## Why Not Just Use a Threshold?

You could threshold `p > 0.5` to get binary labels. SWARM deliberately avoids this because:

1. **Information loss**: Two interactions with p=0.51 and p=0.95 look identical after thresholding
2. **Calibration breaks**: Metrics based on expected values are better calibrated
3. **Proportional governance**: Governance should be proportional to harm, not binary

The [Metrics](../concepts/metrics.md) page shows how soft labels enable more informative metrics.

---

## Customizing the Proxy

You can adjust weights or calibration:

```python
from swarm.core.proxy import ProxyComputer, ProxyWeights

# Down-weight engagement, up-weight safety signals
custom_weights = ProxyWeights(
    task_progress=0.4,
    rework_penalty=0.25,
    verifier_penalty=0.30,
    engagement_signal=0.05,
)

proxy = ProxyComputer(weights=custom_weights, sigmoid_k=4.0)
```

---

## What's Next?

- **Run an experiment**: [Your First Governance Experiment](first-governance-experiment.md)
- **Interpret results**: [Analyzing Results](analyzing-results.md)
- **Read the theory**: [Soft Labels Concept](../concepts/soft-labels.md)
- **API reference**: [Core API](../api/core.md)
