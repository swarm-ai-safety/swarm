# Algorithm

## Mathematical Formulation

### Proxy Score Computation (§4, Eq. 1–2)

Observable signals are normalized and weighted:

```
o_j = normalize(observable_j) ∈ [-1, +1]

v̂ = Σ w_j · o_j  where w ∈ ℝ⁴₊, Σ w_j = 1.0
```

Default weights: w = (0.40, 0.20, 0.20, 0.20).

Observables:
- o₁: task_progress (normalized ∈ [0,1] → [-1, +1])
- o₂: −rework_count (normalized, negative because rework indicates quality issues)
- o₃: −verifier_rejections (normalized, negative because rejections indicate failure)
- o₄: engagement (normalized ∈ [0,1] → [-1, +1])

### Sigmoid Calibration (§4, Eq. 3)

Soft label is computed via logistic sigmoid with steepness parameter k:

```
p = σ_k(v̂) = 1 / (1 + exp(-k · v̂))

where k = 2.0 (steepness; controls sensitivity)
```

- When v̂ = 0: p = 0.5 (maximum uncertainty)
- When v̂ = +1: p ≈ 0.731 (high confidence beneficial)
- When v̂ = −1: p ≈ 0.269 (low confidence, likely harmful)

### Soft Payoff Decomposition (§4, Eq. 4–7)

Expected surplus under uncertainty:
```
S_soft = p · s⁺ − (1 − p) · s⁻
```

where s⁺ is surplus if interaction is beneficial, s⁻ is loss if harmful (typically negative or zero).

Expected externality cost:
```
E_soft = (1 − p) · h
```

where h is ecosystem harm cost incurred if interaction is harmful.

### Payoff Under Governance Levers (§4, Eq. 8–11)

Total expected payoff with all governance levers:

```
Π_soft = S_soft 
       − τ · S_soft                    (transaction tax)
       − ρ · (1 − p) · h              (externality internalization)
       + L_rep                         (reputation incentive, accumulated over time)
       − 𝟙[p < θ_CB] · (Π_soft + ∞)  (circuit breaker: accept iff p ≥ θ_CB)
```

where:
- τ ∈ [0.0, 0.3]: transaction tax rate
- ρ ∈ [0.0, 1.0]: externality internalization fraction
- θ_CB ∈ [0.2, 0.8]: circuit breaker threshold (soft acceptance criterion)
- L_rep: reputation change from audit outcomes (can be negative)

Reputation dynamics:
```
reputation_t = reputation_{t-1} · λ + Δ_audit

where λ ∈ [0.7, 1.0] is reputation decay (lower λ = faster forgetting),
and Δ_audit = +1 if audit passes, −penalty if audit fails (sampled with prob p_audit).
```

### Distributional Metrics (§5, Eq. 12–15)

Over a set of accepted interactions A and rejected interactions R:

**Toxicity rate:**
```
τ_rate = E[1 − p | accepted] = (1/|A|) Σ_{i ∈ A} (1 − p_i)
```

**Quality gap (adverse selection metric):**
```
Δ_q = E[p | accepted] − E[p | rejected] 
    = (1/|A|) Σ_{i ∈ A} p_i − (1/|R|) Σ_{i ∈ R} p_i
```

Interpretation: Δ_q > 0 indicates benign selection (accepted interactions have higher p); Δ_q < 0 indicates adverse selection.

**Conditional loss (externality selection effect):**
```
ℓ_cond = E[h | rejected] − E[h | accepted]
       = (1/|R|) Σ_{i ∈ R} h_i − (1/|A|) Σ_{i ∈ A} h_i
```

**Spread (diversity of accepted population):**
```
spread = std[p_i : i ∈ A]
```

## Pseudocode

```python
def soft_label_governance_step(
    observable: Dict[str, float],    # {task_progress, rework_count, verifier_rejections, engagement}
    params: Dict[str, float],         # {s_plus, s_minus, h, tau, theta_CB, rho, lambda, p_audit, k}
    weights: np.ndarray,              # w = (w1, w2, w3, w4)
    reputation_history: List[float],
    random_state: np.random.Generator
) -> Dict[str, Any]:
    """
    Single interaction step: compute soft label, apply governance, update reputation.
    
    Returns:
        payoff: float, expected payoff after governance
        accept: bool, accept/reject decision
        soft_label: float, p ∈ [0, 1]
        reputation_delta: float, change in agent reputation
        audit_triggered: bool, whether audit was sampled
    """
    
    # Step 1: Normalize observables
    normalized = normalize_observables(observable)  # each ∈ [-1, +1]
    
    # Step 2: Compute proxy score
    v_hat = np.dot(weights, [
        normalized['task_progress'],
        -normalized['rework_count'],
        -normalized['verifier_rejections'],
        normalized['engagement']
    ])
    
    # Step 3: Apply sigmoid calibration
    k = params['k']  # default 2.0
    p = 1.0 / (1.0 + np.exp(-k * v_hat))  # ∈ [0, 1]
    
    # Step 4: Compute soft payoff
    s_plus = params['s_plus']
    s_minus = params['s_minus']
    h = params['h']
    S_soft = p * s_plus - (1 - p) * s_minus
    E_soft = (1 - p) * h
    
    # Step 5: Apply governance levers
    tau = params['tau']
    rho = params['rho']
    tax_cost = tau * S_soft
    externality_cost = rho * (1 - p) * h
    
    # Step 6: Reputation incentive
    L_rep = sum(reputation_history) * params['lambda']  # decay
    audit_prob = params['p_audit']
    audit_triggered = random_state.random() < audit_prob
    if audit_triggered:
        audit_passes = random_state.random() < p  # audit outcome correlates with p
        L_rep += 1.0 if audit_passes else -0.5
    
    # Step 7: Compute net payoff
    payoff_net = S_soft - tax_cost - externality_cost + L_rep
    
    # Step 8: Apply circuit breaker
    theta_CB = params['theta_CB']
    accept = (p >= theta_CB) and (payoff_net >= 0)  # only accept if above threshold AND positive payoff
    
    if not accept:
        payoff_net = 0.0
    
    return {
        'payoff': payoff_net,
        'accept': accept,
        'soft_label': p,
        'proxy_score': v_hat,
        'reputation_delta': L_rep - sum(reputation_history),
        'audit_triggered': audit_triggered,
        'externality_cost': externality_cost
    }

def scenario_run(
    agent_population: List[Agent],
    scenario_params: ScenarioParams,  # governance + payoff settings
    n_epochs: int = 10,
    n_steps_per_epoch: int = 100,
    seed: int = 42
) -> Tuple[List[Dict], MetricsResult]:
    """
    Full scenario execution: run population through multiple epochs, collect metrics.
    
    Returns:
        history: list of per-interaction results
        metrics: aggregated toxicity, quality_gap, welfare, etc.
    """
    
    random_state = np.random.Generator(np.random.PCG64(seed))
    history = []
    
    for epoch in range(n_epochs):
        for step in range(n_steps_per_epoch):
            # Sample agent and action
            agent = random_state.choice(agent_population)
            observable = agent.step(scenario_params)  # generate observables
            
            # Apply governance
            result = soft_label_governance_step(
                observable,
                scenario_params.governance,
                scenario_params.weights,
                agent.reputation_history,
                random_state
            )
            
            # Update agent state
            agent.reputation_history.append(result['reputation_delta'])
            agent.payoff += result['payoff']
            
            # Log interaction
            history.append({
                'agent_id': agent.id,
                'epoch': epoch,
                'soft_label': result['soft_label'],
                'accept': result['accept'],
                'payoff': result['payoff'],
                'externality': result['externality_cost']
            })
    
    # Compute distributional metrics
    accepted = [h for h in history if h['accept']]
    rejected = [h for h in history if not h['accept']]
    
    if len(accepted) > 0:
        toxicity = np.mean([1 - h['soft_label'] for h in accepted])
        quality_gap = (
            np.mean([h['soft_label'] for h in accepted]) -
            np.mean([h['soft_label'] for h in rejected]) if len(rejected) > 0 else 0.0
        )
    else:
        toxicity = np.nan
        quality_gap = np.nan
    
    total_welfare = np.sum([h['payoff'] for h in history])
    
    return history, MetricsResult(
        toxicity=toxicity,
        quality_gap=quality_gap,
        welfare=total_welfare,
        interaction_count=len(history),
        pass_rate=compute_pass_rate(history, scenario_params)
    )

def compute_pass_rate(history: List[Dict], params: ScenarioParams) -> float:
    """
    Determines whether run meets predefined success criterion.
    Criterion: toxicity ≤ 0.30 and welfare ≥ 100.0 (example thresholds).
    Returns: fraction (0.0 or 1.0 for single-seed, aggregated across seeds).
    """
    toxicity = np.mean([1 - h['soft_label'] for h in history if h['accept']])
    welfare = np.sum([h['payoff'] for h in history])
    
    return 1.0 if (toxicity <= 0.30 and welfare >= 100.0) else 0.0
```

## Complexity Analysis

**Per-interaction complexity:**
- Observable normalization: O(1) (4 observables)
- Proxy score computation: O(1) (dot product of 4-element vectors)
- Sigmoid: O(1) (scalar exponential)
- Payoff computation: O(1) (arithmetic operations)
- Reputation update: O(H) where H is length of reputation history (typically H ≈ 10–100)
- **Total**: O(H) per interaction

**Per-epoch complexity:**
- Run S steps: O(S · H)
- Aggregate metrics: O(S) (compute mean over S interactions)
- **Total**: O(S · H) per epoch

**Full scenario (N epochs, S steps/epoch, A agents):**
- O(N · S · H) + O(N · S) = O(N · S · H) overall

**Practical scale** (from paper):
- N = 10 epochs, S ≥ 100 steps, A ≥ 50 agents, H ≈ 10
- Per scenario: ~10,000 interactions, O(100,000) arithmetic operations
- 7 scenarios × 5 seeds = 35 scenario runs, ~3.5M total operations (negligible on CPU)

## Convergence Properties

The soft-label governance system does not optimize a global objective; instead, it:
1. **Equilibrates** reputation and payoff incentives: agents with high p accumulate reputation, enabling future acceptance; this incentivizes quality
2. **Stabilizes** accept/reject decisions: once θ_CB is set, boundary of accepted interactions is fixed in p-space
3. **Does not guarantee Pareto improvement**: aggressive externality internalization (ρ → 1.0) can drive welfare negative while toxicity remains invariant (C03), demonstrating that passive mechanisms alone are insufficient

The paper finds empirically that Adaptive Governance (with paired ρ ≈ 0.3 and θ_CB ≈ 0.70) recovers welfare while maintaining safety (C07, hypothesis level).

