"""Page 5: Theory — Mathematical foundations and key formulas."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="Theory", page_icon="📐", layout="wide")
st.title("Mathematical Foundations")
st.markdown(
    "This page documents the core mathematical framework behind the simulation. "
    "All formulas below are implemented in `src/core/`."
)

# ── Proxy Computation ─────────────────────────────────────────────────────────

st.header("1. Proxy Computation")
st.markdown(r"""
Observable signals from each interaction are combined into a raw proxy score:

$$\hat{v} = \sum_i w_i \cdot x_i \quad \in [-1, +1]$$

where $x_i$ are normalised observables (task progress, rework count, verifier rejections,
engagement) and $w_i$ are calibrated weights.

The raw score is then mapped to a probability through a calibrated sigmoid:

$$p = P(v = +1 \mid \hat{v}) = \frac{1}{1 + e^{-k(\hat{v} - b)}}$$

where $k$ controls steepness and $b$ is the bias. This gives us a **soft label** —
a probability that the interaction is beneficial, rather than a binary decision.
""")

# ── Payoff Structure ──────────────────────────────────────────────────────────

st.header("2. Soft Payoff Structure")
st.markdown(r"""
Given a soft label $p$, the expected surplus and externality are:

$$S_{\text{soft}} = p \cdot s^{+} - (1-p) \cdot s^{-}$$

$$E_{\text{soft}} = (1-p) \cdot h$$

where:
- $s^{+}$ = surplus when interaction is beneficial
- $s^{-}$ = cost when interaction is harmful
- $h$ = externality harm parameter

**Agent payoffs** decompose as:

$$\pi_i = \underbrace{\alpha_i \cdot S_{\text{soft}}}_{\text{surplus share}}
        + \underbrace{T_i}_{\text{transfers}}
        - \underbrace{G_i}_{\text{governance cost}}
        - \underbrace{\rho_i \cdot E_{\text{soft}}}_{\text{externality cost}}
        + \underbrace{w_{\text{rep}} \cdot \Delta R_i}_{\text{reputation}}$$

The externality internalisation parameters $\rho_a, \rho_b \in [0, 1]$ control how much
each party bears the cost of ecosystem harm.
""")

# ── Acceptance Threshold ──────────────────────────────────────────────────────

st.header("3. Acceptance Threshold")
st.markdown(r"""
An interaction is **accepted** into the ecosystem when $p \geq \theta$, where $\theta$
is the acceptance threshold (default 0.5). Rejected interactions still generate payoffs
for the participants but are excluded from the "official" ecosystem metrics.

This creates a natural tension: agents may exploit borderline interactions that
are accepted with $p$ slightly above $\theta$.
""")

# ── Safety Metrics ────────────────────────────────────────────────────────────

st.header("4. Safety Metrics")
st.markdown(r"""
**Toxicity rate** measures expected harm among accepted interactions:

$$\text{Toxicity} = \mathbb{E}[1 - p \mid \text{accepted}]
= \frac{\sum_{i \in \text{accepted}} (1 - p_i)}{|\text{accepted}|}$$

**Quality gap** measures selection effects:

$$\text{QualityGap} = \mathbb{E}[p \mid \text{accepted}] - \mathbb{E}[p \mid \text{rejected}]$$

A **negative** quality gap signals **adverse selection** — the ecosystem is preferentially
accepting lower-quality interactions.

**Total welfare** sums all agent payoffs:

$$W = \sum_{i} \pi_i$$
""")

# ── Governance Levers ─────────────────────────────────────────────────────────

st.header("5. Governance Levers")

governance_data = {
    "Lever": [
        "Transaction Tax",
        "Reputation Decay",
        "Staking",
        "Circuit Breaker",
        "Random Audit",
        "Collusion Detection",
    ],
    "Parameter": [
        "tax_rate ∈ [0, 0.5]",
        "decay_rate ∈ [0.8, 1.0]",
        "min_stake ≥ 0",
        "freeze_threshold ∈ [0.3, 1.0]",
        "audit_prob ∈ [0, 0.5]",
        "freq_threshold (z-score)",
    ],
    "Effect": [
        "Deducts fraction of surplus from each interaction, redistributed to ecosystem",
        "Reputations shrink each epoch, forcing agents to maintain good behaviour",
        "Agents must lock resources to participate, slashed on violations",
        "Agents frozen when personal toxicity exceeds threshold",
        "Random interactions are audited; violations incur multiplied penalties",
        "Detects statistically unusual interaction patterns between agent pairs",
    ],
    "Trade-off": [
        "Reduces exploitation but lowers total welfare",
        "Punishes bad actors but also erodes honest agents' standing",
        "Filters undercapitalised agents, may reduce diversity",
        "Quickly stops toxic agents but can freeze borderline honest agents",
        "Deters hidden exploitation but adds overhead",
        "Catches coordinated attacks but may flag legitimate collaboration",
    ],
}

import pandas as pd
st.dataframe(pd.DataFrame(governance_data), use_container_width=True, hide_index=True)

# ── Agent Types ───────────────────────────────────────────────────────────────

st.header("6. Agent Behavioural Types")
st.markdown(r"""
| Type | Strategy | Typical $p$ |
|------|----------|-------------|
| **Honest** | Always cooperates, high-quality interactions | $p \approx 0.8\text{–}1.0$ |
| **Opportunistic** | Cooperates when observed, exploits when possible | $p \approx 0.4\text{–}0.8$ |
| **Deceptive** | Mimics honest behaviour, gradually becomes exploitative | $p$ starts high, decays |
| **Adversarial** | Actively tries to harm ecosystem or extract value | $p \approx 0.1\text{–}0.3$ |
| **Adaptive Adversary** | Learns from detection, evolves strategies to evade governance | varies |
""")

# ── Information Flow & Boundaries ─────────────────────────────────────────────

st.header("7. Semi-Permeable Boundaries")
st.markdown(r"""
The boundary module models information flow between the sandbox and external world:

- **Inbound flows**: data entering the sandbox (queries to external services, data imports)
- **Outbound flows**: data leaving the sandbox (results, logs, potential leakage)

Boundary policies enforce:
- **Rate limiting**: max crossings per agent per time window
- **Content filtering**: blocks flows containing forbidden patterns
- **Sensitivity filtering**: blocks flows with sensitivity above threshold

The **leakage detector** scans outbound content for PII, credentials, and code patterns
using configurable regex rules, generating alerts and recommendations.
""")

# ── References ────────────────────────────────────────────────────────────────

st.header("References")
st.markdown("""
1. **Distributional safety**: Ensuring that the *distribution* of outcomes across
   all agents remains fair, not just the average.
2. **Soft labelling**: Using probabilistic labels avoids the brittleness of hard
   binary classifications and better captures uncertainty.
3. **Mechanism design**: Governance levers are inspired by economic mechanism design
   — creating incentive structures where truthful, cooperative behaviour is the
   dominant strategy.
4. **Adverse selection**: From insurance economics — when a market mechanism
   preferentially attracts the worst risks, leading to market failure.
""")
