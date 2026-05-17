# Concepts

## Soft Probabilistic Label

- **Notation**: p, where p = P(v = +1) ∈ [0, 1]
- **Definition**: A continuous probability estimate that an interaction is beneficial (v = +1 rather than v = −1), computed by applying a calibrated sigmoid function to a proxy score v̂ ∈ [−1, +1]. Formally, p = σ_k(v̂) = 1 / (1 + exp(−k · v̂)), where k is a steepness parameter (k = 2.0 in the paper).
- **Boundary conditions**: p ∈ [0, 1] always. When p = 0.5, the interaction is neutral (maximum uncertainty). When p → 0, the interaction is highly likely harmful; when p → 1, highly likely beneficial. Calibration requires downstream observables (task completion, rework, verification rejections) to be informative about true interaction quality.
- **Related concepts**: Proxy score (v̂), calibrated sigmoid, uncertainty preservation

## Proxy Score

- **Notation**: v̂ ∈ [−1, +1]
- **Definition**: A raw score computed by weighted combination of four observable signals: task_progress, rework_count, verifier_rejections, and engagement. Formally, v̂ = w₁ · norm(task_progress) + w₂ · norm(rework_count) + w₃ · norm(verifier_rejections) + w₄ · norm(engagement), where w = (w₁, w₂, w₃, w₄) are positive weights summing to unity, and norm() maps each observable to [−1, +1].
- **Boundary conditions**: Proxy scores are well-defined only when observable signals are available and meaningful. Normalization may fail if observables have degenerate distributions (e.g., all agents have identical task_progress). The choice of weights w significantly affects separation between honest and adversarial agents (see Table 9).
- **Related concepts**: Soft probabilistic label (p), observable signal, calibration, weight sensitivity

## Toxicity Rate

- **Notation**: τ or E[1 − p | accepted]
- **Definition**: The expected value of (1 − p) over the set of interactions accepted by governance mechanisms. Equivalently, the expected harmfulness of an accepted interaction population. For a set of accepted interactions A, τ = (1/|A|) ∑_{i ∈ A} (1 − p_i).
- **Boundary conditions**: τ ∈ [0, 1]. τ = 0 indicates a perfectly safe population (all p_i = 1); τ = 1 indicates maximum harm (all p_i = 0). In practice, observed values range from 0.3 to 0.35 across baseline and ablated scenarios in Table 4.
- **Related concepts**: Soft probabilistic label (p), distributional metrics, acceptance criterion, quality gap

## Quality Gap (Adverse Selection Metric)

- **Notation**: Δ_q or E[p | accepted] − E[p | rejected]
- **Definition**: The difference in expected soft-label values between accepted and rejected interactions. Positive Δ_q indicates benign selection (accepted interactions have higher p than rejected ones); negative Δ_q indicates adverse selection (high-risk interactions preferentially pass governance). Formally, Δ_q = (1/|A|) ∑_{i ∈ A} p_i − (1/|R|) ∑_{i ∈ R} p_i, where A is the set of accepted interactions and R is the set of rejected interactions.
- **Boundary conditions**: Δ_q is undefined when either |A| = 0 or |R| = 0 (no accepted or no rejected interactions). The sign and magnitude of Δ_q depend critically on governance calibration; overly permissive thresholds push Δ_q toward zero (no selection effect), while overly restrictive thresholds may invert it if rejection rules are misaligned.
- **Related concepts**: Soft probabilistic label (p), adverse selection, toxicity rate, governance calibration

## Governance Lever

- **Notation**: Generic notation (τ for tax, θ_CB for circuit breaker threshold, λ for reputation decay, ρ for externality internalization, p_audit for audit probability)
- **Definition**: A parameterized intervention that affects agent incentives through payoff modification or acceptance criteria. Examples: (1) transaction tax τ ∈ [0.0, 0.3] reduces per-interaction surplus; (2) circuit breaker θ_CB ∈ [0.2, 0.8] rejects interactions where estimated p < θ_CB; (3) reputation decay λ ∈ [0.7, 1.0] discounts past history; (4) externality internalization ρ ∈ [0.0, 1.0] assigns ecosystem harm cost h to agents proportionally; (5) audit probability p_audit ∈ [0.0, 0.5] imposes random verification with cost penalties on failure.
- **Boundary conditions**: Governance levers compose additively in payoff effects (assumption A6). Individual levers have optimal operating regions (e.g., θ_CB ≈ 0.35–0.50, Table 6b); overly extreme settings cause welfare collapse. Levers interact with agent risk-awareness (A3) and proxy calibration (A2).
- **Related concepts**: Payoff decomposition, soft payoff engine, tradeoff analysis, calibration sensitivity

## Distributional Safety

- **Notation**: No single notation; encompasses soft metrics (τ, Δ_q) and ensemble properties
- **Definition**: Safety evaluated on population-level distributional properties rather than per-interaction thresholds. Formally, distributional safety is achieved when: (1) soft metrics (toxicity, quality gap, conditional loss) remain within acceptable ranges across agent subpopulations; (2) governance mechanisms detect gaming behavior (proxy degradation, cost-cutting) that binary thresholds miss; (3) adaptive acceptance mechanisms maintain Pareto frontiers between welfare and toxicity rather than imposing asymmetric harm.
- **Boundary conditions**: Distributional safety is meaningful only when proxy calibration is reliable (A2) and governance levers compose additively (A6). It presupposes heterogeneous agents (mix of honest, adversarial, self-optimizing types) and sufficient population size to estimate distributional moments. Not applicable to single-agent or fully scripted scenarios.
- **Related concepts**: Soft probabilistic labels, toxicity, quality gap, governance mechanisms, adverse selection

## Payoff Decomposition

- **Notation**: Π_soft = S_soft − τ · S_soft − ρ · (1 − p) · h + α · (Δ_reputation)
- **Definition**: Expected payoff decomposed into four components: (1) S_soft = p · s⁺ − (1 − p) · s⁻ (expected surplus); (2) −τ · S_soft (transaction tax cost); (3) −ρ · (1 − p) · h (externality internalization cost, scaled by harm risk); (4) α · (Δ_reputation) (reputation incentive, where Δ_reputation depends on audit outcomes and past performance). The paper in §4 derives payoff equations (4)–(7) as combinations of these components.
- **Boundary conditions**: Payoff decomposition assumes agents maximize expected payoff. It is valid only when p estimates are well-calibrated and agent risk-awareness (A3) holds. Components may interact (e.g., high ρ can invert incentives), necessitating empirical ablation to measure welfare-toxicity tradeoffs.
- **Related concepts**: Soft payoff engine, governance levers, expected value, mechanism design

## Calibration

- **Notation**: k (sigmoid steepness) or calibration quality measured by expected p alignment with observed outcomes
- **Definition**: The property that estimated soft labels p match empirical frequencies in observed outcomes. A sigmoid is well-calibrated when, among interactions assigned p ≈ 0.6, approximately 60% are beneficial (v = +1) and 40% are harmful (v = −1). Calibration requires choosing sigmoid steepness k and proxy weights w such that soft-label distributions align with ground truth frequencies. Table 9 demonstrates weight sensitivity: uniform weights (w = [0.25, 0.25, 0.25, 0.25]) yield E[p | honest] = 0.681, while heavy-task weighting (w = [0.80, 0.05, 0.05, 0.10]) yields E[p | honest] = 0.865, a significant shift.
- **Boundary conditions**: Perfect calibration is unattainable; some miscalibration is inevitable due to proxy imperfection. The paper assumes calibrated sigmoid (A2) as foundational; violation of this assumption invalidates soft-label safety metrics. Calibration must be verified across heterogeneous agent types and scenarios.
- **Related concepts**: Soft probabilistic label, proxy score, sigmoid function, weight sensitivity

