# Training & Governance Configuration

## Overview

Governance parameters control agent incentives and ecosystem safety thresholds. Each parameter has been empirically validated via ablation studies (Table 5, 6a–d) and calibrated to balance welfare and toxicity across scenarios.

## Transaction Tax Rate (τ)

- **Symbol**: τ
- **Domain**: [0.0, 1.0]
- **Default**: 0.10 (10% across most scenarios)
- **Rationale**: Reduces surplus from completed tasks, making high-risk interactions unprofitable. Incentivizes agents toward quality. Pigouvian principle: cost bearer (agent) partially internalizes externality via reduced payoff.
- **Search range**: [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
- **Sensitivity**: Medium (Table 6a: welfare monotonically declines from ~110 at τ=0.0 to ~60 at τ=0.30; toxicity flat ~0.30)
- **Bounds**: τ ≤ 0.30 practical limit; τ > 0.50 is prohibitive and collapses welfare without safety gain
- **Mechanism**: Applied to expected surplus S_soft = p·s⁺ − (1−p)·s⁻; cost_tax = τ·S_soft
- **Source**: §4 (Governance lever section), Table 6a (ablation), H07 (heuristic)

## Circuit Breaker Threshold (θ_CB)

- **Symbol**: θ_CB
- **Domain**: [0.0, 1.0]
- **Default**: 0.35 (Strict Governance), 0.70 (Adaptive Governance)
- **Rationale**: Hard acceptance boundary; if p < θ_CB, interaction is rejected regardless of payoff. Acts as a safety hard-stop, complementing soft governance levers. Enables dynamic risk tolerance: lower θ_CB is more permissive, higher θ_CB is stricter.
- **Search range**: [0.20, 0.35, 0.50, 0.65, 0.80]
- **Sensitivity**: High (Table 6b: non-monotonic toxicity curve; optimal region θ_CB ∈ [0.35, 0.50]; welfare tradeoff sharp around θ_CB=0.50)
- **Bounds**: θ_CB ∈ [0.1, 0.9] empirically; values outside this range extreme without domain justification
- **Mechanism**: Rejection criterion: if p < θ_CB, then accept = False; payoff = 0
- **Calibration**: Must be set for each domain; grid search recommended (e.g., Table 6b protocol)
- **Source**: §4 (Circuit breaker lever), Table 6b (ablation), H06 (heuristic)

## Audit Probability (p_audit)

- **Symbol**: p_audit
- **Domain**: [0.0, 0.5]
- **Default**: 0.10 (10% per interaction across scenarios)
- **Rationale**: Probability that a random interaction is audited. Audit outcome (pass/fail) directly correlates with p: if audit occurs, probability audit_passes ≈ p. Creates stochastic reputation dynamics: high-quality agents accumulate positive reputation; low-quality agents accumulate penalties. Enables reputation-based differentiation without deterministic auditing (computational savings).
- **Search range**: [0.0, 0.05, 0.10, 0.25, 0.50]
- **Sensitivity**: Low–Medium (Table 6c: weak effect; welfare ~100–120 across p_audit range; toxicity flat ~0.30)
- **Bounds**: p_audit ∈ [0.0, 0.5]; practical [0.05, 0.25]. Higher p_audit requires more auditor resources.
- **Mechanism**: Per interaction, sample u ~ Uniform[0,1]; if u < p_audit, audit_triggered = True; then sample audit_passes ~ Bernoulli(p); if audit_passes, reputation += 1.0; else reputation -= 0.5
- **Source**: §4 (Governance lever section), Table 6c (ablation)

## Reputation Decay Parameter (λ)

- **Symbol**: λ
- **Domain**: (0.0, 1.0)
- **Default**: 0.85 (Strict Governance), 0.90 (Adaptive Governance)
- **Rationale**: Exponential decay of past reputation impacts. Prevents unbounded accumulation (if λ = 1.0, reputation grows forever, enabling permanent favoritism). Enables fresh starts: agents can recover from past failures. Dynamic equation: reputation_t = reputation_{t−1}·λ + Δ_audit, where Δ_audit ∈ {−0.5, +1.0}.
- **Search range**: [0.70, 0.80, 0.90, 0.95, 1.0]
- **Sensitivity**: Medium (Table 6d: inverse relationship; welfare ~90 at λ=0.70, ~140 at λ=1.0; effect modest but directional)
- **Bounds**: λ ∈ (0, 1.0) strictly (λ=1.0 forbidden; enables permanent favoritism); practical [0.7, 0.99]. λ < 0.7 causes overly rapid forgetting.
- **Mechanism**: Decay applied per epoch or per audit event; reputation accumulates over multiple interactions with exponential weighting backward in time
- **Source**: §4 (Governance lever section), Table 6d (ablation), H05 (heuristic)

## Externality Internalization (ρ)

- **Symbol**: ρ
- **Domain**: [0.0, 1.0]
- **Default**: 0.00 (Baseline), 0.10 (Strict Governance)
- **Rationale**: Fraction of ecosystem harm h that agents bear as a cost. When ρ=0, agents pay no externality penalty (free-for-all); when ρ=1.0, agents bear full cost. Implements Pigouvian taxation principle: cost bearer internalizes externality. Key finding (C03): aggressive ρ (≥0.7) alone collapses welfare without reducing toxicity; must pair with adaptive mechanisms (θ_CB adjustment).
- **Search range**: [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
- **Sensitivity**: High (Table 5: welfare monotonically declines from 262.14 at ρ=0.0 to −67.51 at ρ=1.0; demonstrates passive mechanisms insufficient)
- **Bounds**: ρ ∈ [0.0, 1.0]; practical [0.0, 0.5] to avoid welfare collapse. ρ > 0.5 recommended only with θ_CB ≥ 0.7 adjustment.
- **Mechanism**: Cost = ρ·(1−p)·h, where h is ecosystem harm if interaction fails. Expected harm cost E_soft = (1−p)·h; internalized cost = ρ·E_soft
- **Source**: §4 (Governance lever section), Table 5 (ablation), H07 (heuristic)

## Staking (Optional, Not Ablated)

- **Symbol**: stake
- **Domain**: [0.0, ∞)
- **Default**: 0.0 (no staking in paper experiments)
- **Rationale**: Agent deposits collateral (stakes) that is forfeited if audit fails. Provides endogenous reputation accumulation: agents risk capital. Not explored in paper (Appendix A5: identified as extension).
- **Bounds**: If implemented, stake > 0; recommended [0.0, s⁺] (stake cannot exceed task surplus)
- **Mechanism**: On audit failure, stake is partially or fully forfeited (e.g., 50% penalty on failure). On audit pass, stake is returned or rewards accrue.
- **Source**: §4 (Governance lever section, mentioned), Appendix A (future work)

## Collusion Detection (Optional, Not Ablated)

- **Symbol**: collusion_sensitivity (not a parameter; operational threshold)
- **Domain**: [0.0, 1.0]
- **Default**: 0.0 (no collusion detection in baseline)
- **Rationale**: Detect coordinated low-quality behavior across agent pairs/groups. Paper includes "Misalignment Sweeps" scenario (Table 2) which tests collusion detection heuristically (agents coordinate to degrade quality). Not parameterized; detection is scenario-specific.
- **Bounds**: Collusion detection is boolean (enabled/disabled) or threshold-based (e.g., "flag if >3 agents coordinate 5+ times")
- **Mechanism**: Monitored via quality gap degradation or audit clustering (multiple agents fail audits in same epoch)
- **Source**: §6 (Scenario families), Table 2 (Misalignment Sweeps row)

## Summary Table

| Parameter | Symbol | Default | Range | Sensitivity | Optimal Region | Source |
|-----------|--------|---------|-------|-------------|----------------|--------|
| Transaction tax | τ | 0.10 | [0.0, 0.30] | Medium | [0.05, 0.15] | Table 6a |
| Circuit breaker | θ_CB | 0.35 (Strict), 0.70 (Adaptive) | [0.20, 0.80] | High | [0.35, 0.50] | Table 6b |
| Audit probability | p_audit | 0.10 | [0.05, 0.25] | Low | [0.10, 0.25] | Table 6c |
| Reputation decay | λ | 0.85 (Strict), 0.90 (Adaptive) | [0.70, 0.99] | Medium | [0.85, 0.95] | Table 6d |
| Externality internalization | ρ | 0.00 (Baseline), 0.10 (Strict) | [0.0, 0.50] | High | [0.0, 0.30] | Table 5 |
| Staking | stake | 0.0 | [0.0, s⁺] | Unknown (not ablated) | N/A | Appendix A |
| Collusion detection | — | Disabled | Boolean or threshold | Unknown (scenario-tested) | Enabled for adversarial scenarios | Table 2, Misalignment Sweeps |
