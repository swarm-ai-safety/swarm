# Heuristics

## H01: Default Proxy Weights (Task Progress Priority)

- **Rationale**: Task completion is the only direct outcome measure; rework, rejections, and engagement are indirect signals that may be noisy or gamed. Prioritizing task_progress (w₁ = 0.4, double the other weights) centers the proxy on the primary objective while downweighting subjective or indirect signals. This avoids information leakage from miscalibrated indirect channels.
- **Sensitivity**: High. Proxy weight changes alter separation between agent types by 5–60 percentage points in expected p (Table 9: uniform weights yield E[p|honest] = 0.681, heavy-task weighting yields 0.865). Different domains may require different weight distributions.
- **Bounds**: All weights must be non-negative and normalized to sum to 1.0. Default is w = (0.40, 0.20, 0.20, 0.20); acceptable ranges are w₁ ∈ [0.2, 0.8] for task_progress with remaining weight distributed evenly among rework, verifier, and engagement signals.
- **Code ref**: [swarm/core/proxy.py, ProxyWeights class (lines 13–84), ProxyComputer.__init__ (lines 126–178), normalize() method (lines 61–83)]
- **Source**: §4 (Proxy computer section), Table 9 (proxy weight sensitivity analysis)

## H02: Sigmoid Steepness k = 2.0

- **Rationale**: Sigmoid calibration requires a steepness parameter k that determines how quickly p transitions from 0 to 1 as v̂ ranges from −1 to +1. Excessively low k (e.g., k = 0.5) makes sigmoid nearly linear, providing weak discrimination between high and low proxy scores; excessively high k (e.g., k = 5.0) makes sigmoid sharp, risking overconfidence in soft labels. k = 2.0 is a practical midpoint: sigmoid(k=2.0, v̂=1) ≈ 0.731, sigmoid(k=2.0, v̂=−1) ≈ 0.269, providing approximately 0.46 spread in p-space for maximum proxy spread in v̂-space.
- **Sensitivity**: Medium. Changing k affects toxicity estimates: higher k increases separation, potentially over-discriminating between good and bad agents; lower k under-discriminates. The paper does not ablate k; value k = 2.0 is fixed based on standard calibration heuristics.
- **Bounds**: k must satisfy k > 0 (strictly positive) and k ≤ 100 (to prevent numerical overflow in exp function). Practical range: k ∈ [1.0, 5.0]. k = 2.0 is default; k ∈ [1.5, 3.0] acceptable for most domains.
- **Code ref**: [swarm/core/proxy.py, ProxyComputer.__init__ (lines 126–150), sigmoid_k validation (lines 147–154); swarm/core/sigmoid.py, _sigmoid_fast function]
- **Source**: §4 (Sigmoid calibration section, Eq. 3), Appendix C (calibration heuristics)

## H03: Five-Seed Replication Protocol

- **Rationale**: Simulation experiments have stochastic components (random agent initialization, random interaction sampling, random audit sampling). Replication with fixed random seeds (42, 123, 456, 789, 1024) enables statistical aggregation (mean and std dev across seeds) while maintaining reproducibility. Five seeds are a practical compromise between statistical power (larger N provides tighter confidence intervals) and computational cost (five runs per scenario are feasible; thirty runs would be expensive). The choice of five seeds is acknowledged as a limitation (A5): cross-seed variation is not deeply characterized.
- **Sensitivity**: Medium. Increasing seed count from 5 to 10–30 would reduce error bars by ~25–35% (proportional to sqrt(N)), potentially changing significance of some comparisons. Decreasing to 1–3 seeds would widen error bars and obscure genuine effects.
- **Bounds**: Minimum 3 seeds (for meaningful std dev estimation); practical range 5–30 seeds. Fixed seed list is (42, 123, 456, 789, 1024) to enable reproduction. Any other fixed seed list is acceptable as long as the same list is used across all scenario runs.
- **Code ref**: [swarm/core/experiment.py, ScenarioRun class, _run_seeds method (approx. lines 300–320); swarm/core/runner.py, DEFAULT_SEEDS constant]
- **Source**: §6 (Methodology section), Appendix A (experiment protocols)

## H04: Governance Lever Additivity Assumption

- **Rationale**: Payoff decomposition assumes governance levers contribute independently to total cost: total cost = τ·S_soft + ρ·(1−p)·h + audit_penalties + reputation changes. Additivity simplifies payoff calculation and enables independent ablation of each lever. However, levers can interact at the agent-behavior level: e.g., very high τ can invert incentives when combined with moderate ρ, causing agents to seek high-risk interactions (where ρ cost is small). The paper does not fully characterize interactions beyond empirical ablation results (Table 5, 6a–d). Additivity is treated as a modeling assumption (A6), not a strict constraint.
- **Sensitivity**: High. Non-additive interactions could significantly alter Pareto frontier or optimal parameter regions. E.g., if τ and ρ interact multiplicatively (cost = τ·ρ·S_soft·h), then moderate settings of both levers could produce unexpectedly large costs. The paper empirically validates that results are qualitatively similar across ablations, suggesting additivity is reasonable but not guaranteed.
- **Bounds**: Model payoff as sum of independent lever costs: Π = S_soft − τ·S_soft − ρ·(1−p)·h + L_rep − 𝟙[p < θ_CB]·Π. Higher-order terms (e.g., τ·ρ coupling) are assumed negligible. Violation of additivity assumption would require redesign of payoff function.
- **Code ref**: [swarm/core/payoff.py, SoftPayoffEngine.compute_payoff method (approx. lines 200–280); SoftPayoffEngine._apply_tax, _apply_externality_internalization, _apply_audit, _apply_reputation methods (separate cost functions, each independent)]
- **Source**: §4 (Payoff decomposition, Eq. 8–11), §6 (Ablation methodology), Appendix B (payoff derivation)

## H05: Reputation Decay Parameter λ

- **Rationale**: Agent reputation accumulates over time and is used to modulate incentives (higher reputation → higher audit trust → lower future audit cost). To prevent permanent favoritism and enable recovery from failures, reputation must decay over time. Decay factor λ ∈ [0.7, 1.0] controls the rate: lower λ (e.g., 0.7) causes rapid forgetting; higher λ (e.g., 0.95) causes slow decay. λ = 0.85 is the default (Table 2, Strict Governance scenario), balancing fresh starts with long-term incentives. Formal reputation model: reputation_t = reputation_{t-1}·λ + Δ_audit, where Δ_audit ∈ {−0.5, +1.0} depending on audit outcome.
- **Sensitivity**: Medium. Lower λ favors adaptive behavior (agents can recover quickly); higher λ favors consistency (past performance matters more). Table 6d ablation shows inverse relationship: higher λ increases welfare (less penalty for past failures), but effect is modest (welfare varies from ~90 to ~140 across λ ∈ [0.7, 1.0]).
- **Bounds**: λ must satisfy λ ∈ (0, 1.0); practically λ ∈ [0.7, 0.99]. λ = 1.0 (no decay) is forbidden because it enables unbounded reputation accumulation and permanent favoritism. λ < 0.7 causes overly rapid forgetting and weakens long-term incentives.
- **Code ref**: [swarm/core/payoff.py, SoftPayoffEngine class, _update_reputation method (approx. lines 350–380); swarm/core/agent.py, Agent.reputation_history, Agent.update_reputation_from_audit]
- **Source**: §4 (Governance lever section), Table 6d (reputation decay ablation)

## H06: Circuit Breaker Threshold Calibration

- **Rationale**: Circuit breaker threshold θ_CB is a hard acceptance boundary: interactions with p < θ_CB are rejected, regardless of payoff. The threshold must be calibrated to balance rejection of low-quality interactions (low p, high risk) against false rejection of acceptable interactions. Optimal region is θ_CB ≈ 0.35–0.50 (Table 6b): below this range (e.g., θ_CB = 0.20), toxicity increases to 0.335 (vs baseline 0.300) and welfare drops to 38.21 (vs baseline 108.50), indicating overactive rejection; above this range (e.g., θ_CB = 0.65), toxicity plateaus at 0.326 and welfare increases to ~147, indicating permissive acceptance with modest safety degradation. Practitioners should empirically tune θ_CB for their domain using ablation similar to Table 6b.
- **Sensitivity**: High. Non-monotonic toxicity curve and welfare trade-offs mean small changes in θ_CB (±0.05) can significantly alter outcomes. Table 6b shows that θ_CB = 0.35 and θ_CB = 0.50 have similar welfare (~109 vs ~144) but different toxicity profiles (~0.300 vs ~0.333).
- **Bounds**: θ_CB must satisfy θ_CB ∈ [0.0, 1.0] (within p range). Empirically, θ_CB ∈ [0.2, 0.8] is reasonable; values outside [0.1, 0.9] are extreme and should be avoided without specific domain justification. Default is θ_CB = 0.35 (Strict Governance scenario).
- **Code ref**: [swarm/core/payoff.py, SoftPayoffEngine._apply_circuit_breaker method (approx. lines 250–270); acceptance logic in compute_payoff]
- **Source**: §6 (Ablation results), Table 6b (circuit breaker threshold ablation with θ_CB ∈ [0.2, 0.8]), Abstract (lines 37–39: calibration guidance)

## H07: Externality Internalization Scaling

- **Rationale**: Externality internalization parameter ρ controls how much ecosystem harm cost h an agent bears. Formally, cost = ρ·(1−p)·h. When ρ = 0, agents bear no externality cost (free-for-all); when ρ = 1.0, agents bear full cost. Intermediate values (ρ ∈ [0.1, 0.5]) create partial internalization. The paper finds (Table 5) that aggressive internalization (ρ ≥ 0.7) collapses welfare without reducing toxicity, demonstrating that passive cost-imposing mechanisms alone are insufficient. Pairing ρ with adaptive acceptance mechanisms (C07) enables Pareto improvement: Adaptive Governance scenario uses ρ = 0.3 (implicit, inferred from Table 4), achieving welfare recovery.
- **Sensitivity**: High. Welfare exhibits strong monotonic decline as ρ increases (262.14 → −67.51 from ρ = 0.0 to ρ = 1.0, Table 5). This is one of the paper's key negative findings: single-lever governance fails.
- **Bounds**: ρ ∈ [0.0, 1.0]. Values ρ > 0.5 risk welfare collapse without safety gain; recommended range ρ ∈ [0.0, 0.5]. Pair ρ with adaptive acceptance threshold (θ_CB) for Pareto improvement.
- **Code ref**: [swarm/core/payoff.py, SoftPayoffEngine._apply_externality_internalization method (approx. lines 290–310)]
- **Source**: §4 (Governance lever section), §6 (Ablation results), Table 5 (externality internalization ablation)

