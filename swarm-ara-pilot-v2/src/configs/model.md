# Model & Scenario Configuration

## Overview

Model parameters define the proxy computer calibration and payoff structure. All values have been empirically validated across seven scenarios (Table 2, 4) and sensitivity-tested (Table 9).

## Proxy Weights (w)

- **Symbol**: w = (w₁, w₂, w₃, w₄)
- **Default**: w = (0.40, 0.20, 0.20, 0.20)
- **Rationale**: Weighted combination of normalized observables to compute proxy score v̂:
  ```
  v̂ = w₁·task_progress + w₂·(−rework_count) + w₃·(−verifier_rejections) + w₄·engagement
  ```
  w₁ = 0.40 (double weight) because task_progress is the only direct outcome measure; rework, rejections, engagement are indirect/subjective signals that may be noisy or gamed. Downweighting indirect signals avoids information leakage from miscalibrated channels.
- **Sensitivity**: High (Table 9: uniform weights w=(0.25, 0.25, 0.25, 0.25) yield E[p|honest]=0.681; heavy-task weighting w=(0.60, 0.13, 0.13, 0.14) yields E[p|honest]=0.865; separation between honest and adversarial agent types shifts 5–60 percentage points)
- **Bounds**: All w_i ≥ 0; Σ w_i = 1.0. Acceptable ranges: w₁ ∈ [0.2, 0.8] with remaining weight distributed among {w₂, w₃, w₄}. Extreme skew w₁ → 1.0 ignores indirect signals entirely; w₁ → 0 overweights noisy signals.
- **Validation**: Table 9 ablation tests w ∈ {uniform, default, heavy-task}; downstream impact on toxicity, welfare, and agent type separation quantified
- **Source**: §4 (Proxy computer section, Eq. 1–2), Table 9 (proxy weight sensitivity analysis), H01 (heuristic)

## Sigmoid Steepness (k)

- **Symbol**: k
- **Domain**: (0, 100]
- **Default**: k = 2.0
- **Rationale**: Controls steepness of sigmoid calibration p = σ_k(v̂) = 1/(1+exp(−k·v̂)). Lower k (e.g., 0.5) makes sigmoid nearly linear, providing weak discrimination; higher k (e.g., 5.0) makes sigmoid sharp, risking overconfidence. k = 2.0 is practical midpoint:
  - σ_k(v̂=+1, k=2.0) ≈ 0.731 (high confidence beneficial)
  - σ_k(v̂=0, k=2.0) = 0.5 (maximum uncertainty)
  - σ_k(v̂=−1, k=2.0) ≈ 0.269 (low confidence, likely harmful)
  - Spread in p-space ≈ 0.46 for maximum proxy spread in v̂-space
- **Sensitivity**: Medium (no explicit ablation in paper; k = 2.0 fixed based on standard calibration heuristics; higher k increases separation, potentially over-discriminating; lower k under-discriminates)
- **Bounds**: k > 0 strictly (positive steepness); k ≤ 100 to prevent numerical overflow in exp(−k·v̂). Practical range k ∈ [1.0, 5.0]; k ∈ [1.5, 3.0] acceptable for most domains.
- **Mechanism**: Applied per-interaction after proxy score v̂ is computed; output p ∈ [0, 1] is soft label used in all downstream payoff and metrics computations
- **Validation**: Paper assumes calibration succeeds (A2); production deployment requires calibration validation on held-out data
- **Source**: §4 (Sigmoid calibration section, Eq. 3), Appendix C (calibration heuristics), H02 (heuristic)

## Scenario Payoff Structure

### Observable Quality Parameters

- **s⁺** (surplus if beneficial): Float > 0, typical range [0.5, 2.0]
- **s⁻** (loss if harmful): Float, typical range [−0.5, 0.0] (often negative or zero)
- **h** (ecosystem harm externality): Float > 0, typical range [0.1, 1.0]

Each scenario has a distinct payoff configuration to model different task domains (Table 3). Expected surplus is computed as:
```
S_soft = p·s⁺ − (1−p)·s⁻
```
Expected harm is:
```
E_soft = (1−p)·h
```

### Baseline Scenario

- **s⁺**: 1.0 (positive surplus from good interaction)
- **s⁻**: 0.0 (no loss if bad; simply no benefit)
- **h**: 0.5 (moderate ecosystem harm if interaction fails)
- **Rationale**: Neutral baseline; agents face no penalty for bad interactions, only loss of upside
- **Tax rate τ**: 0.0 (no governance)
- **Externality internalization ρ**: 0.0 (no cost for ecosystem harm)
- **Circuit breaker θ_CB**: 1.0 (permissive; only rejects p < 0.0, which never occurs)
- **Source**: Table 2 (Baseline row), Table 3 (payoff config)

### Strict Governance Scenario

- **s⁺**: 1.0
- **s⁻**: 0.0
- **h**: 0.5
- **Rationale**: Same payoff structure as Baseline, but governance applied
- **Tax rate τ**: 0.10 (10% transaction tax)
- **Externality internalization ρ**: 0.10 (10% cost for ecosystem harm)
- **Circuit breaker θ_CB**: 0.35 (strict acceptance; rejects p < 0.35)
- **Reputation decay λ**: 0.85 (moderate decay)
- **Audit probability p_audit**: 0.10
- **Expected outcome (C02)**: Welfare reduction ~40% (110 → 65); toxicity flat or slight increase (~0.30)
- **Source**: Table 2 (Strict Governance row), Table 4 (results), C02 (claim)

### Adaptive Governance Scenario

- **s⁺**: 1.0
- **s⁻**: 0.0
- **h**: 0.5
- **Rationale**: Paired governance levers to achieve Pareto improvement
- **Tax rate τ**: 0.05 (light tax)
- **Externality internalization ρ**: 0.30 (moderate externality cost)
- **Circuit breaker θ_CB**: 0.70 (permissive; high trust in agents)
- **Reputation decay λ**: 0.90 (slower decay; longer memory)
- **Audit probability p_audit**: 0.10
- **Expected outcome (C07)**: Welfare recovery toward baseline (~95–110); toxicity comparable to Strict (~0.30); demonstrates that pairing ρ with adaptive θ_CB enables welfare-toxicity Pareto improvement (hypothesis level, inferred from Table 4, 5)
- **Source**: Table 2 (Adaptive Governance row), Table 4 (results, inferred), C07 (claim)

### Additional Scenarios (Table 2)

| Scenario | s⁺ | s⁻ | h | τ | ρ | θ_CB | λ | Purpose |
|----------|----|----|---|---|---|------|---|---------|
| Threshold Dancer | 1.0 | 0.0 | 0.5 | 0.10 | 0.10 | 0.50 | 0.85 | Mid-range governance; tests circuit breaker calibration |
| Misalignment Sweeps | 1.0 | 0.0 | 0.5 | 0.10 | 0.10 | 0.35 | 0.85 | Collusion detection; coordinated agent misconduct |
| LLM Backing | 1.0 | 0.0 | 0.5 | 0.10 | 0.10 | 0.35 | 0.85 | LLM agents (Concordia); governance transfer (E04) |
| Adversarial Prompts | 1.0 | 0.0 | 0.5 | 0.10 | 0.10 | 0.35 | 0.85 | LLM robustness; adversarial instruction tuning |

- **Source**: Table 2 (scenario families), Table 3 (payoff configs)

## Distributional Payoff Decomposition

Under governance levers, total payoff is decomposed as:

```
Π_soft = S_soft 
        − τ·S_soft                      (transaction tax cost)
        − ρ·(1−p)·h                    (externality internalization cost)
        + L_rep                         (reputation incentive, accumulated over time)
        − 𝟙[p < θ_CB]·∞               (circuit breaker hard-stop)
```

where:
- **S_soft** = p·s⁺ − (1−p)·s⁻ (expected surplus under uncertainty)
- **L_rep** = f(audit_outcomes, reputation_history, λ) (reputation change from audits)
- **𝟙[p < θ_CB]** = 1 if p < θ_CB (reject), 0 otherwise (accept)

**Assumption A6**: Governance levers compose additively. Each lever contributes an independent cost term; no higher-order multiplicative interactions are assumed (though they may exist at population level; see Constraints section).

## Summary Table

| Component | Default | Domain | Sensitivity | Bounds | Source |
|-----------|---------|--------|-------------|--------|--------|
| w₁ (task_progress) | 0.40 | (0, 1] | High | [0.2, 0.8] | Table 9, H01 |
| w₂ (−rework_count) | 0.20 | [0, 1] | High | [0.1, 0.4] | Table 9, H01 |
| w₃ (−verifier_rejections) | 0.20 | [0, 1] | High | [0.1, 0.4] | Table 9, H01 |
| w₄ (engagement) | 0.20 | [0, 1] | High | [0.1, 0.4] | Table 9, H01 |
| k (sigmoid steepness) | 2.0 | (0, 100] | Medium | [1.0, 5.0] | Appendix C, H02 |
| s⁺ | 1.0 | (0, ∞) | High | [0.5, 2.0] | Table 3 |
| s⁻ | 0.0 | (−∞, ∞) | High | [−0.5, 0.0] | Table 3 |
| h | 0.5 | (0, ∞) | High | [0.1, 1.0] | Table 3 |

- **Source**: §4 (Model specification), Table 2–3 (scenario configs), Table 9 (proxy ablation), Appendix C (calibration)
