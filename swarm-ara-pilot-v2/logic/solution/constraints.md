# Constraints and Limitations

## Boundary Conditions

### Soft Label Validity

**Condition:** p ∈ [0, 1] must be maintained everywhere soft labels are surfaced or logged.

- ProxyComputer output v̂ ∈ [−1, +1] is guaranteed by normalization
- Sigmoid function is strictly bounded: σ_k(v̂) ∈ (0, 1) for all v̂ ∈ ℝ
- Numerical edge cases: when v̂ approaches ±∞, sigmoid approaches 0 or 1; but bounded input ensures this does not occur in practice
- **Risk**: Catastrophic overflow in exp(−k · v̂) if k is extremely large (k > 10) and |v̂| is uncontrolled. Mitigation: k = 2.0 is well-behaved; numerical libraries (NumPy, PyTorch) have built-in overflow guards

### Reputation Stability

**Condition:** reputation_t must not diverge over long runs.

- Reputation decay λ ∈ [0.7, 1.0] ensures exponential discounting of past events
- Audit outcomes are bounded: Δ_audit ∈ [−0.5, +1.0] per audit
- With λ = 0.8 and audit interval ~1/p_audit: reputation decays by ~20% per epoch, enabling fresh starts after failures
- **Risk**: If λ = 1.0 (no decay), reputation accumulates unboundedly; this enables permanent favoritism. Mitigation: enforce λ < 1.0

### Governance Lever Interactions

**Condition:** Governance levers compose additively (A6); no significant higher-order interactions.

- Tax and externality internalization both reduce payoff but act independently: total cost = τ·S_soft + ρ·(1−p)·h
- Circuit breaker is a binary hard constraint, applied after payoff calculation
- Audit probability is sampled independently; outcomes are independent across interactions
- **Known limitation**: Levers do interact at the population level. E.g., high τ can invert incentives when combined with high ρ, potentially causing agents to seek high-risk interactions (where ρ cost is minimal due to low p). The paper does not fully characterize lever interactions beyond empirical ablation (Table 5, 6a–d)

### Observable Signal Quality

**Assumption A1:** Observable signals are meaningful proxies for underlying interaction quality.

- task_progress: assumes agents cannot cheat progress reports (e.g., via false timestamps). Mitigation: verification feedback is independent, raising red flags if reported progress is not substantiated
- rework_count: assumes rework genuinely indicates quality issues, not agent sloppiness. In multi-step tasks, rework is an objective signal; in one-shot tasks, rework correlation with quality is weaker
- verifier_rejections: assumes verification is not biased or gamed. In practice, verifiers may have preferences unrelated to true quality
- engagement: assumes engagement correlates with quality. High engagement can also indicate desperate agents trying to cover poor quality
- **Mitigation**: Default weighting (w = [0.4, 0.2, 0.2, 0.2]) prioritizes task_progress, downweighting subjective signals. Table 9 ablation shows proxy robustness across weight configurations

### Calibration Assumption (A2)

**Assumption:** A calibrated sigmoid (with k = 2.0) maps proxy scores to well-calibrated probabilities.

- Calibration requires that agents with p ≈ 0.6 actually succeed ~60% of the time
- In practice, calibration is scenario-dependent: honest agents may have higher base success rates than adversarial agents, violating uniform calibration
- Miscalibration examples:
  - If honest agents are systematically easier (higher s⁺, lower h), they will accumulate higher p on average, but the sigmoid steepness k may not accurately reflect their true success rate
  - If verifier rejections are noisy or inconsistent, the proxy signal becomes unreliable
- **Mitigation**: Paper does not describe explicit calibration validation; this is a known limitation. In production, calibration should be verified on held-out validation set
- **Impact on claims:** C01 depends critically on A2; if calibration fails, soft labels lose information value and collapse toward binary (violating C01)

## Known Limitations

### Limited Scenario Coverage (A4)

The paper tests 7 scenario configurations (Baseline, Strict, Adaptive, Threshold Dancer, Misalignment Sweeps, LLM Backing, Adversarial Prompts). Each scenario is relatively simple:
- Agents are scripted or LLM-backed; behavior is not evolved or learned in real-time
- Environment is stateless or memoryless (no long-term dependencies beyond reputation)
- No coalition formation or collusion (except for "Misalignment Sweeps" scenario which is designed to test collusion detection)
- No heterogeneous agent objectives beyond honest vs adversarial dichotomy

**Impact:** Results may not generalize to:
- Complex hierarchical environments (multi-level approval chains, nested governance)
- Environments with true learning/adaptation (agents that improve quality over time or learn to evade governance)
- Highly asymmetric agent types (spectrum from pure honest to pure adversarial, with many intermediate types)
- Non-stationary environments (where proxy signal distributions shift over time)

### Five-Seed Replication (A5)

The paper uses five fixed seeds (42, 123, 456, 789, 1024) to estimate mean and std dev across runs. This is a small sample; typical ML studies use ≥10–30 seeds.

**Impact:**
- Confidence intervals are wide (see Table 4: error bars of ±12–13 on welfare); some comparisons may not be significant
- Edge cases (e.g., one seed fails to accept any interaction) are not well-characterized
- Cross-seed variation is acknowledged but not deeply investigated

### No Formal Optimality Guarantees

The soft-label framework does not prove that any particular governance configuration is optimal. The paper is empirical:
- Optimal θ_CB is found by grid search (Table 6b), not by principled optimization
- Pareto frontier (C06, C07) is estimated from discrete ablation points, not computed analytically
- No formal welfare-toxicity tradeoff bound

**Impact:** Practitioners must empirically tune their own governance parameters; the paper provides guidance (e.g., θ_CB ≈ 0.35–0.50) but cannot guarantee generalization to new domains.

### Externality Modeling

The externality cost h is assumed to be proportional to (1−p), i.e., the harm to the ecosystem is fully captured by interaction quality. This is a simplification:
- In reality, externalities are heterogeneous: some low-quality interactions cause large harm (e.g., privacy violations), while others cause small harm (e.g., wasted time)
- Temporal externalities (long-term reputation loss, cascading failures) are not modeled; reputation decay λ is a crude proxy
- No modeling of systemic risk or cascading failures (e.g., if many agents fail simultaneously)

**Impact:** C03 (aggressive externality internalization collapses welfare) is robust, but the paper's recommendation to pair ρ with adaptive mechanisms (C07) is heuristic, not theoretically justified.

### LLM Agent Scope (E04)

LLM backing experiments (C05) are limited to:
- Small models (Llama 3.1 8B, Claude Haiku, GPT-4o Mini) and one large model (Claude 3.5 Sonnet)
- Governance applied to behavioral outputs only; no fine-tuning or RL on governance feedback
- No investigation of how different instruction tuning or RLHF training affects governance robustness

**Impact:** Unclear whether governance mechanisms transfer to:
- Frontier-scale models (e.g., Claude 3.5 Sonnet operating at scale, GPT-4 variants)
- Models trained with specific safety objectives (e.g., Constitutional AI)
- Models fine-tuned on adversarial tasks

## Assumptions (Fully Listed)

### A1: Observable Signals Are Meaningful Proxies

Observable signals (task_progress, rework_count, verifier_rejections, engagement) correlate with true interaction quality and are not easily manipulated by strategic agents.

- **Justification:** In multi-agent settings with verification, these observables are objective and third-party validated
- **Violation:** If agents learn to fake observables (e.g., bribe verifiers, game engagement metrics), proxy signals become unreliable

### A2: Calibrated Sigmoid Succeeds

A sigmoid with k = 2.0 can calibrate proxy scores to well-calibrated probabilities across diverse agent types and scenarios.

- **Justification:** Sigmoid is a simple, well-studied calibration function; k = 2.0 is a practical midpoint
- **Violation:** If proxy signal distributions are multi-modal (e.g., honest agents have p ~ 0.9, adversarial agents have p ~ 0.1, with no overlap), a single sigmoid cannot calibrate both simultaneously

### A3: Agents Respond to Payoff Incentives

Agents are rational and risk-aware; they maximize expected payoff subject to governance constraints.

- **Justification:** Mechanism design theory assumes rational agents
- **Violation:** If agents have heterogeneous risk preferences (some risk-loving, some risk-averse) or non-monetary objectives (prestige, spite), payoff structures may not drive desired behavior

### A4: Scenarios and Agent Types Are Representative

The 7 scenarios and agent types tested (honest, adversarial, self-optimizing, LLM-backed) are sufficient to generalize insights about governance lever effects.

- **Justification:** Scenarios cover a range from ungoverned baseline to strict and adaptive governance; agent types span scripted and LLM-based
- **Violation:** Unknown unknowns: scenarios not tested may exhibit different tradeoffs. E.g., in a scenario with high-value interactions (s⁺ >> 1.0), even aggressive governance may be acceptable

### A5: Five-Seed Replication Provides Sufficient Signal

Five fixed seeds provide adequate sample size to estimate mean and std dev for claims comparison.

- **Justification:** Statistical power is adequate for detecting large effects (e.g., welfare reduction of 40%)
- **Violation:** Small effects (e.g., 5% welfare improvement) may not be significant at N=5 seeds

### A6: Governance Levers Compose Additively

Payoff cost contributions from tax, externality internalization, and audit penalties combine additively with no significant higher-order interactions.

- **Justification:** Each lever applies a separate cost term to payoff; no multiplicative or conditional dependencies assumed
- **Violation:** If tax rate and externality internalization are correlated in agent response (e.g., both cause agents to seek different interaction types), composition may be non-additive

## Recommendations for Extension

1. **Calibration validation:** Develop explicit calibration testing (e.g., expected calibration error metric)
2. **Scenario expansion:** Test governance mechanisms on more complex environments (hierarchical approval, learning agents, coalition formation)
3. **Larger replication studies:** Increase seed count to 10–30 for tighter confidence intervals
4. **Formal optimization:** Apply Bayesian optimization or reinforcement learning to find welfare-toxicity Pareto frontier analytically
5. **Heterogeneous externalities:** Model externality costs as distributions rather than scalar multiples of (1−p)
6. **LLM fine-tuning:** Investigate whether governance feedback can be used to fine-tune LLM agents for better alignment

