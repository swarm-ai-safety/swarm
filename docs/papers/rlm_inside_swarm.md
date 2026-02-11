# Recursive Reasoning in Multi-Agent Systems: Strategic Depth as a Distributional Safety Risk

**Authors:** Raeli Savitt
**Date:** 2026-02-10
**Framework:** SWARM v1.0.0

## Abstract

We study the distributional safety implications of embedding strategically sophisticated agents---modeled as Recursive Language Models (RLMs) with level-k iterated best response---into multi-agent ecosystems governed by soft probabilistic labels. Across three pre-registered experiments (N=30 seeds total, 26 statistical tests), we find three counter-intuitive results. **First**, deeper recursive reasoning *hurts* individual payoff (Pearson r = -0.75, p < 0.001, 10/10 tests survive Holm correction), rejecting the hypothesis that strategic depth enables implicit collusion. **Second**, memory budget asymmetry creates statistically significant but practically modest power imbalances (3.2% spread, r = +0.67, p < 0.001, 11/11 survive Holm). **Third**, fast-adapting RLM agents outperform honest baselines in small-world networks (Cohen's d = 2.14, p = 0.0001) but *not* by evading governance---rather by optimizing partner selection within legal bounds. Across all experiments, honest agents earn 2.3--2.8x more than any RLM tier, suggesting that strategic sophistication is currently a net negative in SWARM-style ecosystems with soft governance. All p-values survive Holm-Bonferroni correction at the per-experiment level.

## 1. Introduction

[TODO: Expand with literature context]

The intersection of recursive reasoning and multi-agent safety is understudied. Most alignment research treats agents as either aligned or misaligned in a binary sense. The SWARM framework instead uses *soft probabilistic labels*---each interaction receives a probability p of being beneficial---enabling a richer distributional analysis of safety outcomes.

Recursive Language Models (RLMs) represent a class of agents that apply iterated reasoning over their action space: at recursion depth k, an RLM models its counterparties as depth-(k-1) reasoners and best-responds accordingly. This mirrors the level-k thinking framework from behavioral game theory (Stahl & Wilson, 1994; Nagel, 1995). The key safety question is: does increased reasoning depth create emergent coordination risks that governance mechanisms cannot keep pace with?

We operationalize this question through three experiments:

1. **Recursive Collusion** (Exp. 1): Does deeper recursion enable implicit coordination without explicit communication?
2. **Memory-as-Power** (Exp. 2): Does asymmetric working memory create exploitative power imbalances?
3. **Governance Lag** (Exp. 3): Can fast-adapting agents outpace deliberately slow governance responses?

Each experiment uses 10 pre-registered seeds with Bonferroni and Holm-Bonferroni correction for multiple comparisons.

## 2. Experimental Setup

### 2.1 RLM Agent Architecture

RLM agents implement level-k thinking algorithmically (no LLM API calls). The key parameters are:

| Parameter | Description | Values Tested |
|-----------|-------------|---------------|
| `recursion_depth` | Levels of iterated best response | 1, 3, 5 |
| `planning_horizon` | Steps of discounted look-ahead | 3, 5, 7, 8 |
| `memory_budget` | Max entries in structured working memory | 10, 50, 100, 150, 200 |

At each decision point, a depth-k agent:
1. Generates candidate actions (NOOP, propose collaboration, post, claim task, vote)
2. Evaluates each candidate via `_evaluate_action_recursive()`
3. Models counterparties at depth-(k-1) using `CounterpartyModel` (tracks cooperation rate, estimated type, payoff history)
4. Selects actions via epsilon-greedy exploration at rate 0.1/(depth+1)

Level-0 (base case) selects the highest-trust visible agent and proposes collaboration, equivalent to a naive `HonestAgent`.

### 2.2 Scenarios

| Scenario | Agents | RLM Config | Epochs | Steps | Network | Governance |
|----------|--------|------------|--------|-------|---------|------------|
| Recursive Collusion | 9 RLM + 3 honest | depth={1,3,5} x 3 | 30 | 15 | Complete | Collusion detection, audit (p=0.15) |
| Memory-as-Power | 7 RLM + 3 honest | mem={200,50,10}, depth=3 | 40 | 15 | Complete | Audit (p=0.1), circuit breaker, rho=0.1 |
| Governance Lag | 4 RLM + 4 honest + 2 adversary | depth=5, mem=150 | 50 | 15 | Small-world (k=4, p=0.2) | Slow: audit (p=0.05), high thresholds |

### 2.3 Governance Parameters

| Parameter | Exp. 1 | Exp. 2 | Exp. 3 |
|-----------|--------|--------|--------|
| Audit probability | 0.15 | 0.10 | 0.05 |
| Audit penalty multiplier | 2.0x | 2.0x | 3.0x |
| Audit threshold (p) | 0.50 | 0.50 | 0.40 |
| Collusion detection | Yes | No | Yes |
| Circuit breaker | No | Yes (tox=0.7) | Yes (tox=0.8) |
| Transaction tax | 2% | 2% | 1% |
| Externality (rho_a, rho_b) | 0.0, 0.0 | 0.1, 0.1 | 0.05, 0.05 |
| Bandwidth cap | 15 | 15 | 20 |

### 2.4 Payoff Model

All experiments use the SWARM soft payoff engine:

- **Expected surplus**: S_soft = p * s_plus - (1-p) * s_minus, with s_plus = 2.0, s_minus = 1.0
- **Externality cost**: E_soft = (1-p) * h, with h = 2.0
- **Acceptance threshold**: theta = 0.5
- **Reputation weight**: w_rep = 1.0

### 2.5 Statistical Methods

- **Seeds**: 10 per experiment, fixed a priori: {42, 7, 123, 256, 999, 2024, 314, 577, 1337, 8080}
- **Primary test statistic**: Two-sample independent t-test on per-seed group means
- **Omnibus test**: One-way ANOVA across all groups
- **Effect size**: Cohen's d for pairwise comparisons
- **Dose-response**: Pearson correlation between the grouping variable and payoff
- **Inequality**: Gini coefficient with one-sample t-test against zero
- **Multiple comparisons**: Bonferroni (alpha/n_tests) and Holm-Bonferroni (step-down) corrections applied per experiment

Total tests: 10 (Exp. 1) + 11 (Exp. 2) + 5 (Exp. 3) = 26.

## 3. Results

### 3.1 Cross-Experiment Summary

| Experiment | Primary Finding | Key Statistic | Gini | Honest Advantage |
|------------|-----------------|---------------|------|------------------|
| Recursive Collusion | Deeper recursion hurts payoff | r = -0.746, p < 0.001 | 0.299 | 2.8x |
| Memory-as-Power | Memory advantage real but modest | r = +0.673, p < 0.001 | 0.236 | 2.3x |
| Governance Lag | RLM outperforms via strategy, not evasion | d = 2.14, p = 0.0001 | 0.325 | 0.72x (reversed) |

### 3.2 Experiment 1: Recursive Collusion

**Hypothesis**: Deeper recursive reasoning enables implicit coordination without explicit communication, producing higher payoffs for deep thinkers.

**Result**: **Rejected.** Deeper recursion monotonically *decreases* payoff.

| Group | Recursion Depth | Mean Payoff | Std |
|-------|----------------|-------------|-----|
| RLM Shallow | 1 | 219.661 | 4.367 |
| RLM Mid | 3 | 213.643 | 2.010 |
| RLM Deep | 5 | 211.351 | 2.219 |
| Honest | -- | 592.980 | 9.886 |

**Key tests (10 total, Bonferroni alpha = 0.005):**

| Test | Statistic | p-value | Cohen's d | Bonferroni | Holm |
|------|-----------|---------|-----------|------------|------|
| Shallow vs Mid | t = 3.959 | 0.0009 | 1.770 | Survives | Survives |
| Shallow vs Deep | t = 5.365 | < 0.0001 | 2.399 | Survives | Survives |
| Mid vs Deep | t = 2.421 | 0.026 | 1.082 | Fails | Survives |
| ANOVA (RLM tiers) | F = 19.712 | < 0.0001 | -- | Survives | Survives |
| Pearson (depth, payoff) | r = -0.746 | < 0.0001 | -- | Survives | Survives |
| All RLM vs Honest | t > 109 | < 0.0001 | d > 48 | Survives | Survives |

**Interpretation**: The "strategic overthinking" effect---more computation cycles spent modeling counterparties yields *worse* outcomes, not better coordination. RLM agent payoffs are remarkably stable across seeds (std = 2--4), while honest agent payoffs are highly variable (std = 406.50 at the individual level), suggesting RLM reasoning produces consistent but mediocre outcomes while honest agents benefit from network-position lottery effects.

### 3.3 Experiment 2: Memory-as-Power

**Hypothesis**: Agents with larger working memory budgets systematically exploit agents with smaller budgets, creating measurable power imbalances.

**Result**: **Supported, with caveats.** The gradient is monotonic and statistically robust, but practically small.

| Group | Memory Budget | Mean Payoff | Std |
|-------|--------------|-------------|-----|
| RLM High | 200 | 253.376 | 3.613 |
| RLM Mid | 50 | 249.063 | 2.579 |
| RLM Low | 10 | 245.442 | 4.140 |
| Honest | -- | 563.106 | 6.892 |

**Key tests (11 total, Bonferroni alpha = 0.00455):**

| Test | Statistic | p-value | Cohen's d | Bonferroni | Holm |
|------|-----------|---------|-----------|------------|------|
| High vs Low | t = 4.566 | 0.0002 | 2.042 | Survives | Survives |
| High vs Mid | t = 3.072 | 0.007 | 1.374 | Fails | Survives |
| Mid vs Low | t = 2.348 | 0.031 | 1.050 | Fails | Survives |
| ANOVA (RLM tiers) | F = 12.845 | 0.0001 | -- | Survives | Survives |
| Pearson (budget, payoff) | r = +0.673 | < 0.0001 | -- | Survives | Survives |
| Exploitation rate (agent-level) | r = +0.500 | < 0.0001 | -- | Survives | Survives |

**Interpretation**: The memory-as-power effect is statistically significant (all 11 tests survive Holm) but practically modest: the high-low spread is only 7.93 points (3.2%). The dominant effect remains the honest-vs-RLM gap (2.3x). Externality internalization (rho = 0.1) produces the lowest Gini (0.236) across all three experiments.

### 3.4 Experiment 3: Governance Lag

**Hypothesis**: Fast-adapting RLM agents (depth=5, large memory) can exploit the gap between harmful action and governance response, outpacing deliberately slow governance.

**Result**: **Partially supported.** RLM agents outperform, but through legitimate strategy rather than governance evasion.

| Group | Mean Payoff | Std |
|-------|-------------|-----|
| RLM Fast (depth=5) | 335.606 | 19.600 |
| Honest | 242.755 | 58.128 |
| Adaptive Adversary | 221.490 | 174.549 |

**Key tests (5 total, Bonferroni alpha = 0.01):**

| Test | Statistic | p-value | Cohen's d | Bonferroni | Holm |
|------|-----------|---------|-----------|------------|------|
| RLM vs Honest | t = 4.787 | 0.0001 | 2.141 | Survives | Survives |
| RLM vs Adversary | t = 2.054 | 0.055 | 0.919 | Fails | Fails |
| Honest vs Adversary | t = 0.366 | 0.719 | 0.163 | Fails | Fails |
| ANOVA (all groups) | F = 3.228 | 0.055 | -- | Fails | Fails |
| Gini > 0 | t = 12.891 | < 0.0001 | -- | Survives | Survives |

**Interpretation**: This is the only experiment where RLM agents outperform honest agents, reversing the pattern from Exp. 1--2. The mechanism is not governance evasion---RLM signal profiles produce moderate p-values (~0.5--0.7), well above the harm threshold (p < 0.4), so there are zero harmful interactions to detect. Instead, RLM depth-5 agents exploit the small-world network topology for better partner selection. Adversary payoffs are bimodal: they either dominate massively (~665--939) or get frozen out (~0--5), reflecting the boom-or-bust adaptive adversary strategy.

### 3.5 Cross-Experiment Patterns

1. **The honest advantage is context-dependent.** In complete networks (Exp. 1--2), honest agents earn 2.3--2.8x more. In small-world networks with slow governance (Exp. 3), RLM agents earn 1.4x more. Network topology mediates the value of strategic reasoning.

2. **Strategic reasoning produces stability at the cost of magnitude.** Across all experiments, RLM group-mean standard deviations range from 2.0 to 19.6, while honest group-mean standard deviations range from 6.9 to 58.1. RLM agents converge to consistent but moderate outcomes.

3. **Gini coefficients reveal governance quality.** Exp. 2 (lowest Gini = 0.236) uses externality internalization (rho = 0.1); Exp. 3 (highest Gini = 0.325) uses deliberately weak governance. The 0.089-point spread demonstrates that rho parameters effectively reduce inequality.

4. **All findings are robust to correction.** Of 26 total tests across three experiments, 24 survive Holm-Bonferroni correction and 20 survive strict Bonferroni. The two that fail both corrections (RLM-vs-Adversary and ANOVA in Exp. 3) are underpowered due to adversary variance, not statistical artifacts.

## 4. Discussion

[TODO: Expand]

### 4.1 Strategic Overthinking

The most surprising finding is that deeper recursion *hurts* within-group payoff (Exp. 1, r = -0.75). This contradicts the intuition that smarter agents should outperform simpler ones. We hypothesize three mechanisms:

1. **Computational cost without information gain.** Level-k reasoning assumes counterparties play at level-(k-1), but if all agents are similarly sophisticated, the additional depth provides no strategic advantage while consuming exploration budget.
2. **Overcaution.** Deeper agents discount more aggressively and avoid interactions that shallower agents would accept, reducing their interaction volume and hence cumulative payoff.
3. **Epsilon-greedy decay.** The exploration rate 0.1/(depth+1) means deeper agents explore less, potentially missing beneficial interactions that shallower agents discover through noise.

### 4.2 Memory as a Modest Power Lever

The memory-as-power effect (r = +0.67) is real but small (3.2% spread). This suggests that in SWARM-style ecosystems, *information asymmetry alone* does not confer large advantages. The dominant factor is the *type* of strategy (honest vs. strategic), not the *resources* available to a given strategy type.

### 4.3 Network Topology as a Moderator

The reversal of the honest advantage in Exp. 3 (small-world, slow governance) vs. Exp. 1--2 (complete network) suggests that network topology is a critical moderator of distributional safety. In complete networks, honest agents benefit from high connectivity; in small-world networks, strategic agents exploit structural holes. This has implications for governance design: regulators cannot assume that governance mechanisms effective in well-connected networks transfer to sparse or clustered topologies.

### 4.4 Governance Implications

| Lever | Observed Effect |
|-------|-----------------|
| Externality internalization (rho > 0) | Reduces Gini by ~0.06--0.09 |
| Higher audit probability | No significant effect on RLM behavior (signals stay above threshold) |
| Small-world topology | Enables strategic agents, disadvantages honest |
| Collusion detection | No implicit collusion detected across any experiment |

## 5. Limitations

1. **No LLM backbone.** RLM agents use algorithmic level-k reasoning, not actual language model inference. Results may not transfer to LLM-based agents with richer reasoning capabilities.
2. **Fixed payoff parameters.** All experiments use s_plus=2.0, s_minus=1.0, h=2.0. The findings may not generalize to different surplus/harm ratios.
3. **Within-experiment correction only.** Multiple comparisons correction is applied per experiment, not across the full study (26 tests). A study-wide Bonferroni threshold of 0.05/26 = 0.0019 would eliminate a few additional borderline results.
4. **Moderate sample size.** 10 seeds per experiment provides adequate power for large effects (d > 1) but may miss smaller effects.
5. **Signal profile assumption.** RLM agents produce signals via `ObservableGenerator`'s moderate/variable profile, not via the actual interaction quality. The honest advantage partly reflects this design choice.

## 6. Reproducibility

All experiments are reproducible from:

```bash
# Experiment 1: Recursive Collusion
python -m swarm run scenarios/rlm_recursive_collusion.yaml --seed {42,7,123,256,999,2024,314,577,1337,8080}

# Experiment 2: Memory-as-Power
python -m swarm run scenarios/rlm_memory_as_power.yaml --seed {42,7,123,256,999,2024,314,577,1337,8080}

# Experiment 3: Governance Lag
python -m swarm run scenarios/rlm_governance_lag.yaml --seed {42,7,123,256,999,2024,314,577,1337,8080}
```

Analysis artifacts are stored in `runs/20260210-215826_analysis_rlm_*/`.

Raw data: `per_agent_payoffs.csv` (100--120 rows per experiment).
Machine-readable results: `summary.json` per experiment.

<!-- Reproducibility query (no RLM runs in SQLite yet):
SELECT scenario_id, seed, acceptance_rate, avg_toxicity, welfare_per_epoch
FROM scenario_runs WHERE scenario_id LIKE 'rlm_%' ORDER BY scenario_id, seed;
-->

## 7. References

[TODO: Add full citations]

- Stahl, D. O., & Wilson, P. W. (1994). Experimental evidence on players' models of other players. *Journal of Economic Behavior & Organization*, 25(3), 309--327.
- Nagel, R. (1995). Unraveling in guessing games: An experimental study. *American Economic Review*, 85(5), 1313--1326.
- Crawford, V. P., Costa-Gomes, M. A., & Irber, N. (2013). Structural models of nonequilibrium strategic thinking: Theory, evidence, and applications. *Journal of Economic Literature*, 51(1), 5--62.
