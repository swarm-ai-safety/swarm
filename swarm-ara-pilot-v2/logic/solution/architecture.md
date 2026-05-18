# System Architecture

## Component Graph

The soft-label governance system is a four-stage pipeline that transforms observable interaction data into safety-informed governance decisions and distributional metrics.

```
Observables (task_progress, rework_count, verifier_rejections, engagement)
  ↓
ProxyComputer
  ├─ Normalize observables to [-1, +1]
  ├─ Weighted combination: v̂ = w₁·o₁ + w₂·o₂ + w₃·o₃ + w₄·o₄
  └─ Output: v̂ ∈ [-1, +1]
  ↓
Sigmoid Calibration (CalibratedSigmoid)
  ├─ Apply sigmoid with steepness parameter k
  ├─ Formula: p = 1 / (1 + exp(-k·v̂))
  └─ Output: p = P(v = +1) ∈ [0, 1]
  ↓
SoftPayoffEngine
  ├─ Decompose payoff into components:
  │   ├─ S_soft = p·s⁺ - (1-p)·s⁻ (expected surplus)
  │   ├─ E_soft = (1-p)·h (expected harm externality)
  │   ├─ C_tax = τ·S_soft (transaction tax cost)
  │   ├─ C_ext = ρ·(1-p)·h (externality internalization cost)
  │   └─ L_rep = f(audit_outcome, reputation_history) (reputation loss/gain)
  ├─ Compute payoff: Π_soft = S_soft - C_tax - C_ext + L_rep
  ├─ Apply circuit breaker: if p < θ_CB, accept=False; otherwise accept=True
  ├─ Apply reputation decay: Δ_reputation *= λ (discounting)
  └─ Output: (Π_soft, accept, p, v̂)
  ↓
SoftMetrics (Population-Level Analysis)
  ├─ Toxicity: τ_rate = E[1-p | accepted]
  ├─ Quality Gap: Δ_q = E[p | accepted] - E[p | rejected] (adverse selection indicator)
  ├─ Conditional Loss: ℓ_cond = E[h | rejected] - E[h | accepted] (selection effect on externality)
  ├─ Spread: measure variance in p across accepted population
  └─ Output: (τ_rate, Δ_q, ℓ_cond, spread)
  ↓
Governance Reports + Monitoring
  ├─ Soft metrics (toxicity, quality gap) for adaptive intervention
  ├─ Hard metrics (binary pass/fail, acceptance rate) for comparison
  ├─ Agent-level audits (via audit probability p_audit)
  └─ Output: Accept/reject decision, diagnostic feedback
```

## Component Details

### 1. ProxyComputer

**Purpose**: Convert observable signals into a normalized raw proxy score.

**Inputs**:
- task_progress: Float ∈ [0, 1], fraction of task completed
- rework_count: Integer ≥ 0, number of times agent was asked to redo work
- verifier_rejections: Integer ≥ 0, number of times verification failed
- engagement: Float ∈ [0, 1], measure of agent responsiveness/effort

**Outputs**:
- v̂: Float ∈ [−1, +1], weighted combination of normalized observables

**Key Design Choices**:
- Observable signals are normalized independently to [−1, +1] using min-max scaling within each scenario
- Default weights w = (0.40, 0.20, 0.20, 0.20) prioritize task completion (w₁ = 0.4) over rework and rejections; Table 9 shows sensitivity to weight choice
- Normalization ensures v̂ remains bounded regardless of observable magnitude, enabling stable calibration
- Observable quality is assumed (A1): signals must correlate with true interaction quality for calibration to succeed

### 2. CalibratedSigmoid

**Purpose**: Map normalized proxy score to probabilistic soft label via sigmoid calibration.

**Inputs**:
- v̂: Float ∈ [−1, +1]
- k: Float > 0, steepness parameter (k = 2.0 in paper)

**Outputs**:
- p: Float ∈ [0, 1], probability that interaction is beneficial

**Key Design Choices**:
- Sigmoid with steepness k amplifies discrimination: low k (e.g., 0.5) makes sigmoid nearly linear, reducing separation; high k (e.g., 5.0) makes sigmoid sharp, risking overconfidence
- k = 2.0 is a practical midpoint balancing calibration fidelity and numerical stability
- Calibration is assumed to succeed (A2); in practice, requires validation that E[p] matches empirical frequencies

### 3. SoftPayoffEngine

**Purpose**: Compute expected payoff under distributional uncertainty and apply governance levers.

**Inputs**:
- p: Float ∈ [0, 1], soft label
- s⁺: Float, surplus if interaction is beneficial
- s⁻: Float, surplus if interaction is harmful (typically negative)
- h: Float, externality cost if interaction is harmful
- τ: Float ∈ [0, 1], transaction tax rate
- θ_CB: Float ∈ [0, 1], circuit breaker threshold
- ρ: Float ∈ [0, 1], externality internalization fraction
- λ: Float ∈ [0.7, 1.0], reputation decay factor
- p_audit: Float ∈ [0, 0.5], audit probability
- reputation_history: List of past audit outcomes

**Outputs**:
- Π_soft: Float, expected payoff
- accept: Boolean, governance decision
- audit_triggered: Boolean, whether random audit occurred
- Δ_reputation: Float, reputation change (can be negative if audit fails)

**Key Design Choices**:
- Payoff decomposition allows independent ablation of governance levers (each lever is a separate term, supporting A6)
- Circuit breaker is applied before payoff calculation: if p < θ_CB, interaction is rejected with accept=False and Π_soft=0
- Reputation accumulates over time; each audit is sampled uniformly at random with probability p_audit, and failure penalties decay with λ
- Governance levers compose additively: total payoff cost = −τ·S_soft − ρ·(1−p)·h; no higher-order terms (A6)

### 4. SoftMetrics

**Purpose**: Compute population-level distributional properties over accepted/rejected interaction sets.

**Inputs**:
- p_i: Float ∈ [0, 1] for each interaction i
- accept_i: Boolean, whether interaction i was accepted
- h_i: Float, externality for interaction i

**Outputs**:
- τ_rate: Float ∈ [0, 1], toxicity = E[1−p | accepted]
- Δ_q: Float, quality gap = E[p | accepted] − E[p | rejected]
- ℓ_cond: Float, conditional loss = E[h | rejected] − E[h | accepted]
- spread: Float, standard deviation of p among accepted interactions

**Key Design Choices**:
- Metrics are computed per-scenario and aggregated across seeds (mean ± std dev)
- Quality gap Δ_q directly measures adverse selection: negative values indicate that rejected interactions have higher p (good selection), positive values indicate that accepted interactions have higher p (benign selection)
- Conditional loss ℓ_cond measures whether rejection rule successfully filters out high-externality interactions
- Spread quantifies whether governance produces homogeneous (low spread) or diverse (high spread) accepted populations

### 5. GovernanceReporter (Dual Soft/Hard Metrics)

**Purpose**: Provide parallel soft (probabilistic) and hard (threshold-based) metric reports for direct comparison.

**Soft Metrics**:
- Toxicity, quality gap, conditional loss (from SoftMetrics)

**Hard Metrics**:
- Acceptance rate: fraction of interactions where accept=True
- Pass rate: fraction of seeds where a predefined success criterion is met
- Binary threshold results: fraction passing a hard p > 0.5 threshold (for comparison)

**Key Design Choices**:
- Dual reporting enables direct comparison of soft vs hard detection power
- Hard metrics are computed using the same acceptance decisions, so differences reveal what distributional information is lost by binary collapsing
- Enables measurement of Goodhart's Law: if hard metrics stay flat while soft metrics change, gaming is detected

## Data Flow in Experiments

**Per-interaction flow**:
1. Observables recorded (task_progress, rework_count, verifier_rejections, engagement)
2. ProxyComputer computes v̂
3. CalibratedSigmoid computes p
4. SoftPayoffEngine computes payoff, applies governance levers, makes accept/reject decision
5. Outcome feedback recorded (success/failure, actual rework counts, audit result if sampled)
6. Reputation updated for next interaction

**Per-scenario aggregation** (end of scenario run):
1. All (p, accept, h) tuples collected across all interactions and seeds
2. SoftMetrics computes τ_rate, Δ_q, ℓ_cond, spread
3. Hard metrics (acceptance rate, pass rate) computed in parallel
4. Results written to evidence tables (Table 4, 5, 6a–d)

## Extensibility Points

- **New observables**: Add to ProxyComputer weight vector w; adjust normalization range
- **New governance levers**: Add new payoff cost terms to SoftPayoffEngine; define ablation range
- **New metrics**: Add computation to SoftMetrics class; store in results table
- **Alternative sigmoid calibration**: Replace CalibratedSigmoid with other monotonic [−1,+1]→[0,1] functions; revalidate calibration

