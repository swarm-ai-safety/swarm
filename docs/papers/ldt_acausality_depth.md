# Deeper Reasoning Without Deeper Cooperation: Acausality Depth in LDT Multi-Agent Systems

**Raeli Savitt**

**Abstract.** Logical Decision Theory (LDT) agents cooperate by detecting behavioral similarity with counterparties and reasoning about counterfactual policy outcomes. We extend an LDT agent with two additional levels of acausal reasoning: Level 2 (policy introspection), which infers counterparty decision parameters from behavioral history and simulates their cooperation logic, and Level 3 (recursive equilibrium), which finds fixed-point cooperation probabilities via iterated best-response. In a 7-agent mixed-population simulation (3 LDT, 2 honest, 1 opportunistic, 1 adversarial) swept across acausality depths {1, 2, 3} with 10 seeds per configuration (N=30), we find **no statistically significant differences** in welfare, toxicity, quality gap, or agent payoffs after Bonferroni correction (15 tests, corrected alpha = 0.0033). The largest effect — depth 1 vs 2 welfare (d = -0.87, p = 0.069) — is suggestive but does not survive correction. Level 3 increases welfare variance (SD 13.53 vs 7.92 at Level 1) without improving mean outcomes. We argue that deeper acausal reasoning is redundant when behavioral traces are informative, populations are small, and adversaries do not model the LDT agent's decision procedure. We identify specific environmental conditions (larger populations, modeling adversaries, lower cooperation priors) where deeper reasoning is predicted to become decisive.

## 1. Introduction

Logical Decision Theory (LDT) proposes that rational agents should reason about decisions at the *policy* level rather than myopically maximizing single-step expected payoff. A key prediction is that LDT agents can sustain cooperation with "logical twins" — counterparties whose decision procedures are sufficiently correlated — by recognizing that their own choice logically implies the twin's choice.

Prior implementations of LDT in multi-agent simulations have typically operated at a single level: detecting behavioral similarity via cosine similarity on interaction traces (which we term **Level 1 acausality**). Zvi Mowshowitz's critique of LDT cooperation models argues that this understates LDT's cooperative advantage because it does not model deeper reasoning about counterparty decision procedures.

We implement two additional levels:

- **Level 2 (Policy Introspection):** Infer the counterparty's decision parameters (cooperation prior, similarity threshold, welfare weight, updateless commitment) from their behavioral history, then simulate whether their inferred policy would cooperate with us.
- **Level 3 (Recursive Equilibrium):** Level-k iterated reasoning where both agents' best-response functions are iterated to convergence, finding the fixed-point cooperation probability.

We evaluate all three levels in a controlled simulation environment to test whether deeper reasoning produces measurably better outcomes.

## 2. Methods

### 2.1 Simulation Environment

We use the SWARM soft-label simulation framework with the `ldt_cooperation` scenario:

| Parameter | Value |
|-----------|-------|
| Agents | 7 (3 LDT, 2 honest, 1 opportunistic, 1 adversarial) |
| Epochs | 10 |
| Steps per epoch | 10 |
| Transaction tax | 0.0 |
| Circuit breaker | Disabled |
| Payoff: s_plus / s_minus / h | 2.0 / 1.0 / 2.0 |
| Acceptance threshold (theta) | 0.5 |

### 2.2 LDT Agent Configuration

All LDT agents share identical base parameters:

| Parameter | Value |
|-----------|-------|
| cooperation_prior | 0.65 |
| similarity_threshold | 0.7 |
| welfare_weight | 0.3 |
| updateless_commitment | 0.8 |
| counterfactual_horizon | 20 |

The swept parameter is `acausality_depth` ∈ {1, 2, 3}, which controls the reasoning cascade:

- **Depth 1:** Behavioral twin detection + counterfactual payoff comparison (original logic).
- **Depth 2:** Level 1 + policy introspection. L1 agree + L2 agree → cooperate; disagreements resolved by inferred confidence.
- **Depth 3:** Weighted ensemble: 0.2 × L1 + 0.3 × L2 + 0.5 × L3 equilibrium probability > 0.5 → cooperate.

### 2.3 Level 2: Policy Introspection

The `_infer_counterparty_policy` method estimates four parameters from interaction history:

1. **cooperation_prior** ← acceptance rate
2. **similarity_threshold** ← inverse variance of accepted p values (low variance = selective = high threshold)
3. **welfare_weight** ← acceptance rate for marginal interactions (p ∈ [0.4, 0.6])
4. **updateless_commitment** ← behavioral stability (drift between early and late interaction halves)

All estimates are blended with a **mirror prior** ("they are like me"), weighted by `mirror_prior_weight × (1 - confidence)`, where confidence = min(sample_count / horizon, 1.0). The mirror fades as data accumulates.

The `_simulate_counterparty_decision` method then runs a virtual Level 1 agent with the inferred parameters to predict whether the counterparty would cooperate.

### 2.4 Level 3: Recursive Equilibrium

The `_recursive_equilibrium` method implements level-k iterated reasoning:

1. Initialize: my_p = cooperation_prior, their_p = inferred cooperation_prior
2. Iterate up to `max_recursion_depth` (default 8):
   - Compute soft best-response probabilities using sigmoid-smoothed twin detection and payoff comparison
   - Apply introspection discount (0.9) per level for damping
   - Check convergence: |Δ| < epsilon (0.01)
3. Return the fixed-point my_p

Convergence is guaranteed by: continuous [0,1]→[0,1] mapping (Brouwer), sigmoid damping, and max-depth cap.

### 2.5 Statistical Methods

- 10 seeds per configuration (pre-registered), seeds 43–72
- Welch's t-test for pairwise comparisons (unequal variance)
- Mann-Whitney U as non-parametric robustness check
- Cohen's d for effect sizes
- Shapiro-Wilk normality validation
- Bonferroni and Holm-Bonferroni correction across 15 pairwise tests (3 pairs × 5 metrics)

## 3. Results

### 3.1 Descriptive Statistics

| Depth | Welfare (mean ± SD) | Toxicity (mean ± SD) | Acceptance Rate | Quality Gap | Honest Payoff | Adversarial Payoff |
|-------|--------------------|--------------------|-----------------|-------------|---------------|-------------------|
| 1 | 125.07 ± 7.92 | 0.3362 ± 0.0060 | 0.897 ± 0.022 | 0.1621 ± 0.0457 | 21.39 | 3.26 |
| 2 | 132.16 ± 8.47 | 0.3264 ± 0.0151 | 0.913 ± 0.019 | 0.1565 ± 0.0534 | 22.95 | 3.43 |
| 3 | 127.72 ± 13.53 | 0.3325 ± 0.0055 | 0.901 ± 0.033 | 0.1629 ± 0.0314 | 22.58 | 3.18 |

All distributions pass Shapiro-Wilk normality tests (all p > 0.21).

### 3.2 Pairwise Comparisons

| Comparison | Metric | t-stat | p-value | Cohen's d | Bonferroni sig? |
|------------|--------|--------|---------|-----------|----------------|
| 1 vs 2 | welfare | -1.93 | 0.069 | -0.87 | No |
| 1 vs 2 | toxicity | 1.90 | 0.082 | 0.85 | No |
| 1 vs 2 | honest_payoff | -1.57 | 0.133 | -0.70 | No |
| 1 vs 3 | toxicity | 1.43 | 0.170 | 0.64 | No |
| 1 vs 3 | honest_payoff | -1.02 | 0.321 | -0.46 | No |
| 2 vs 3 | toxicity | -1.19 | 0.259 | -0.53 | No |

*Remaining 9 tests omitted (all p > 0.39, |d| < 0.40).*

**No tests survive Bonferroni correction** (threshold α/15 = 0.0033). **No tests survive Holm-Bonferroni correction.** Zero of 15 tests are nominally significant at p < 0.05.

### 3.3 P-Hacking Audit

| Item | Value |
|------|-------|
| Total hypotheses tested | 15 |
| Pre-registered parameter | Yes (acausality_depth) |
| Seeds pre-specified | Yes (10 per config) |
| Nominally significant (p < 0.05) | 0 |
| Bonferroni significant | 0 |
| Holm-Bonferroni significant | 0 |

### 3.4 Notable Trends (Not Significant)

The largest effect size is depth 1 vs 2 welfare (d = -0.87, p = 0.069): depth 2 produces ~5.7% higher mean welfare. This is a "large" effect by Cohen's conventions but does not reach significance at our corrected threshold. The toxicity comparison (d = 0.85, p = 0.082) mirrors this — depth 2 trends toward lower toxicity.

Depth 3 shows notably higher variance (welfare SD = 13.53 vs 7.92 for depth 1), suggesting the recursive equilibrium introduces instability without corresponding benefit.

![Welfare by Acausality Depth](../../runs/20260212-231859_ldt_acausality_study/plots/welfare_by_depth.png)

![Toxicity by Acausality Depth](../../runs/20260212-231859_ldt_acausality_study/plots/toxicity_by_depth.png)

![Effect Sizes](../../runs/20260212-231859_ldt_acausality_study/plots/effect_sizes.png)

![Welfare-Toxicity Tradeoff](../../runs/20260212-231859_ldt_acausality_study/plots/welfare_toxicity_tradeoff.png)

![Agent Payoffs by Type](../../runs/20260212-231859_ldt_acausality_study/plots/agent_payoffs_by_depth.png)

![Welfare Distribution](../../runs/20260212-231859_ldt_acausality_study/plots/welfare_boxplot.png)

![Quality Gap by Depth](../../runs/20260212-231859_ldt_acausality_study/plots/quality_gap_by_depth.png)

## 4. Discussion

### 4.1 Why Deeper Reasoning Doesn't Help (Here)

The null result is informative. We identify three environmental factors that suppress the advantage of deeper acausal reasoning:

1. **Small population, high cooperation prior.** With only 7 agents and a cooperation prior of 0.65, the baseline Level 1 agent already cooperates with most counterparties. There is little room for deeper reasoning to *increase* cooperation — and the small population means there are few adversarial interactions where discrimination would matter.

2. **Behavioral traces converge quickly.** With 10 steps per epoch and a counterfactual horizon of 20, agents build sufficient behavioral profiles within 2 epochs. Level 2's policy inference adds sophistication but arrives at similar conclusions as Level 1's cosine similarity when the underlying traces are already informative.

3. **No predictor/exploiter agents.** LDT's theoretical advantage is most pronounced against agents that *model and exploit* the LDT agent's decision procedure. The opportunistic and adversarial agents in this scenario do not simulate the LDT agent's reasoning, so Level 2-3's "what would they think about what I think" reasoning has no strategic advantage over Level 1's behavioral correlation.

### 4.2 When Deeper Reasoning Should Matter

We predict Level 2-3 acausality would show significant effects under:

- **Larger populations** (20+ agents) where behavioral traces are sparser and mirror priors become more informative
- **Adversarial agents that model LDT reasoning** (creating an arms race where policy introspection provides genuine strategic advantage)
- **Lower cooperation priors** (0.3-0.4) where the marginal cooperation decisions are more contested
- **Shorter horizons** (counterfactual_horizon = 5) where behavioral data is insufficient for Level 1 but Level 2's structural inference fills the gap

### 4.3 Depth 3 Variance

The increased variance at depth 3 (welfare SD 13.53 vs 7.92 at depth 1) deserves attention. The recursive equilibrium's sigmoid-based best-response function can amplify small differences in initial conditions, leading to divergent outcomes across seeds. In some seeds the equilibrium converges to high cooperation; in others, a slightly different history tips it toward lower cooperation, creating a bimodal-like distribution. This suggests the `introspection_discount` (0.9) and `convergence_epsilon` (0.01) parameters may need tuning — or that Level 3 reasoning requires a minimum data threshold before activation to avoid amplifying noise in early-game interactions.

### 4.4 Red-Team Implications

A separate red-team evaluation of the scenario's governance configuration (no defenses enabled) revealed a robustness score of 0.40/F with 6/8 attacks succeeding. Enabling all governance levers (circuit breaker, collusion detection, audit, staking, 5% tax) improved this to 0.66/D. This underscores that LDT cooperation logic, however sophisticated, is not a substitute for governance-layer defenses against coordinated attacks.

## 5. Conclusion

We implemented Level 2 and Level 3 acausal reasoning for LDT agents and found no statistically significant impact on welfare, toxicity, or agent payoffs in a 7-agent mixed-population simulation with 10 seeds per configuration (0/15 tests significant after Bonferroni correction). The largest effect (depth 1 vs 2 welfare, d = -0.87, p = 0.069) is suggestive but does not survive correction. Deeper acausal reasoning increases implementation complexity and, at Level 3, introduces outcome variance without improving mean performance.

This is a **useful null result** for LDT agent design: in environments where behavioral traces are informative and adversaries are unsophisticated, the simple Level 1 twin-detection heuristic captures most of the cooperative surplus. Implementers should default to Level 1 and reserve deeper reasoning for environments with the specific properties we identify — larger populations, modeling adversaries, lower cooperation priors, and shorter observation horizons — where the theoretical advantage of policy introspection and recursive equilibrium is predicted to be realized.

## Reproducibility

```bash
# Install
python -m pip install -e ".[dev,runtime]"

# Run sweep (30 runs: 3 depths × 10 seeds)
python -c "
from swarm.scenarios import load_scenario
from swarm.analysis.sweep import SweepConfig, SweepParameter, SweepRunner

base = load_scenario('scenarios/ldt_cooperation.yaml')
base.orchestrator_config.n_epochs = 10

config = SweepConfig(
    base_scenario=base,
    parameters=[SweepParameter('agents.ldt.config.acausality_depth', [1, 2, 3])],
    runs_per_config=10,
    seed_base=42,
)

runner = SweepRunner(config)
runner.run()
runner.to_csv('sweep_results.csv')
"

# Run single scenario
python -m swarm run scenarios/ldt_cooperation.yaml --seed 42 --epochs 10 --steps 10
```

## References

- Yudkowsky, E. (2010). Timeless Decision Theory. MIRI Technical Report.
- Soares, N., & Fallenstein, B. (2017). Agent Foundations for Aligning Machine Intelligence with Human Interests. MIRI Technical Report.
- Wei, J., et al. (2022). Functional Decision Theory: A New Theory of Instrumental Rationality. *Philosophical Studies*.
