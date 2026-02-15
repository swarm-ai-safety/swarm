---
title: "Governance Effects on Skill-Evolving Agents: A Parameter Sweep Study"
author: Raeli Savitt
date: 2026-02-14
scenario: scenarios/skillrl.yaml
run_dir: runs/20260214-142400_skillrl_study
seeds: 30
---

# Governance Effects on Skill-Evolving Agents: A Parameter Sweep Study

**Raeli Savitt**

*February 14, 2026*

## Abstract

We study how governance parameters — transaction tax rates and circuit breaker mechanisms — affect welfare, toxicity, and agent payoffs in a multi-agent simulation featuring SkillRL agents that learn reusable interaction strategies through recursive skill evolution. In an initial sweep of 80 runs (4 tax rates x 2 circuit breaker settings x 10 seeds), we find that transaction taxes have a large, statistically robust negative effect on aggregate welfare (Cohen's d up to 3.58), while the default circuit breaker (toxicity threshold 0.7) never triggers. A refined follow-up sweep of 420 runs (7 tax rates x 2 circuit breaker thresholds x 30 seeds) confirms the welfare-tax relationship is linear (R²=0.63, slope=-444 welfare/unit tax) and reveals that lowering the circuit breaker threshold to 0.5 successfully freezes agents (d=3.53) and suppresses adversarial payoffs by 42% (d=-1.29), but still has no effect on aggregate toxicity or welfare. These results demonstrate a fundamental welfare-toxicity decoupling: flat transaction taxes and agent-level circuit breakers cannot reduce harmful interaction rates because toxicity is determined by acceptance decisions made before governance costs are levied.

## 1. Introduction

The SkillRL framework (Xia et al., 2026) introduces recursive skill-augmented reinforcement learning where agents extract, refine, and promote reusable strategies from interaction outcomes. In multi-agent governance settings, the interaction between learned skills and governance mechanisms is not well understood. Do transaction taxes discourage harmful behavior, or do they simply reduce overall economic activity? Do circuit breakers provide additional safety when agents already learn from experience?

We address these questions through a systematic parameter sweep over the SkillRL scenario, combining 3 SkillRL agents with 2 honest, 1 opportunistic, and 1 adversarial agent under varying governance regimes.

## 2. Method

### 2.1 Scenario Configuration

| Parameter | Value |
|---|---|
| SkillRL agents | 3 (acceptance_threshold=0.4, skill_exploration_rate=0.15) |
| Honest agents | 2 |
| Opportunistic agents | 1 |
| Adversarial agents | 1 |
| Epochs | 20 |
| Steps per epoch | 10 |
| Payoff: s_plus / s_minus / h | 2.0 / 1.0 / 2.0 |
| Reputation decay | 0.95 |

### 2.2 Swept Parameters

| Parameter | Values |
|---|---|
| Transaction tax rate | 0.00, 0.05, 0.10, 0.15 |
| Circuit breaker enabled | False, True |

Total configurations: 8. Seeds per configuration: 10. Total runs: **80**.

### 2.3 Statistical Methods

- Welch's t-test (unequal variances) for all pairwise comparisons
- Mann-Whitney U as non-parametric robustness check
- Cohen's d for effect sizes
- Shapiro-Wilk normality validation per group
- Bonferroni and Holm-Bonferroni correction for multiple comparisons (28 tests total)
- Pre-registered seed range: 42-122

## 3. Results

### 3.1 Welfare

Transaction tax rate has a strong, monotonically negative effect on welfare:

| Tax Rate | CB Off (mean +/- SD) | CB On (mean +/- SD) |
|---|---|---|
| 0.00 | 411.35 +/- 15.44 | 417.18 +/- 17.42 |
| 0.05 | 393.73 +/- 14.52 | 398.96 +/- 21.50 |
| 0.10 | 364.95 +/- 12.10 | 371.32 +/- 13.44 |
| 0.15 | 351.63 +/- 11.32 | 340.31 +/- 24.69 |

All pairwise tax comparisons on welfare survive Bonferroni correction (alpha=0.05). The largest effect is tax 0% vs 15%: **d = 3.58, p < 0.0001**. Even the smallest gap (tax 10% vs 15%) is significant: **d = 1.25, p = 0.0004**.

Circuit breaker has no significant effect on welfare (d = -0.05, p = 0.83).

![Welfare vs Tax Rate](../runs/20260214-142400_skillrl_study/plots/welfare_vs_tax.png)

### 3.2 Toxicity

Toxicity rates are remarkably stable across all conditions (range: 0.271-0.276):

| Tax Rate | Mean Toxicity (pooled) |
|---|---|
| 0.00 | 0.2710 |
| 0.05 | 0.2736 |
| 0.10 | 0.2729 |
| 0.15 | 0.2744 |

No toxicity comparison survives Bonferroni correction. The only raw-significant result (tax 0% vs 15%, p=0.02, d=-0.76) does not survive correction.

![Toxicity vs Tax Rate](../runs/20260214-142400_skillrl_study/plots/toxicity_vs_tax.png)

### 3.3 Quality Gap and Agent Payoffs

Quality gap (positive = good selection) is stable across conditions (~0.29), indicating that the acceptance/rejection mechanism consistently selects higher-quality interactions regardless of governance.

Honest agent payoffs decline with tax rate (from ~76 at 0% to ~65 at 15%), while adversarial payoffs remain uniformly low (~2.0-2.5), confirming that SkillRL agents effectively marginalize adversarial participants.

![Agent Payoff by Type](../runs/20260214-142400_skillrl_study/plots/agent_payoff_by_type.png)

### 3.4 Welfare-Toxicity Tradeoff

![Welfare-Toxicity Tradeoff](../runs/20260214-142400_skillrl_study/plots/welfare_toxicity_tradeoff.png)

The scatter reveals a tight vertical band: welfare varies substantially (290-460) while toxicity barely moves (0.271-0.276). This demonstrates a fundamental **welfare-toxicity decoupling** — taxes reduce total surplus without proportionally affecting the rate of harmful interactions.

### 3.5 P-Hacking Audit

| Category | Count |
|---|---|
| Total hypothesis tests | 28 |
| Significant at raw p<0.05 | 14 |
| Surviving Bonferroni (alpha=0.05) | 10 |
| Surviving Holm-Bonferroni | 10 |

All 10 surviving tests concern welfare differences across tax rates. Zero toxicity or circuit breaker comparisons survive correction.

### 3.6 Normality

Shapiro-Wilk confirms normality (p>0.05) for all groups except toxicity_rate at tax=0.15 (W=0.889, p=0.025), justifying the parametric tests with Mann-Whitney U robustness checks confirming identical conclusions.

## 4. Refined Sweep: Council Recommendations

Following a multi-LLM council review (mechanism designer, statistician, red-teamer), we ran a refined sweep addressing three recommendations: (1) intermediate tax rates for curve characterization, (2) lower circuit breaker threshold to actually trigger freezing, and (3) increased seeds for statistical power.

### 4.1 Method

| Parameter | Values |
|---|---|
| Transaction tax rate | 0.000, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150 |
| CB freeze threshold | 0.5 (active), 0.7 (inactive) |

All runs had circuit breaker enabled. Total: 14 configurations x 30 seeds = **420 runs**.

### 4.2 Circuit Breaker Threshold Effect

Lowering the CB threshold from 0.7 to 0.5 has a dramatic effect on agent freezing but no effect on aggregate outcomes:

| Metric | CB=0.5 | CB=0.7 | Cohen's d | p (Bonferroni) |
|---|---|---|---|---|
| Frozen agents | 0.86 | 0.00 | 3.53 | < 0.0001 |
| Adversarial payoff | 1.27 | 2.19 | -1.29 | < 0.0001 |
| Welfare | 385.2 | 383.3 | 0.07 | 1.00 (NS) |
| Toxicity | 0.2721 | 0.2721 | -0.00 | 1.00 (NS) |
| Honest payoff | 71.3 | 71.2 | 0.02 | 1.00 (NS) |

The CB threshold=0.5 successfully freezes ~0.86 agents on average and suppresses adversarial payoffs by 42%, but this has **zero detectable effect on welfare or toxicity**. The frozen adversarial agents were already earning so little (~2.2) that removing them doesn't change system-level metrics.

![Frozen Agents by CB Threshold](../runs/20260214-142400_skillrl_study/plots/refined_frozen_agents.png)

![Adversarial Payoff Suppression](../runs/20260214-142400_skillrl_study/plots/refined_adversarial_payoff.png)

### 4.3 Welfare-Tax Relationship

Linear regression on 420 runs confirms a strong linear relationship:

- **Slope**: -444.3 welfare per unit tax (i.e., -4.4 welfare per 1% tax)
- **R²**: 0.634
- **p**: 3.2 x 10^-93

All 21 pairwise tax comparisons are significant at raw p<0.05; 21 survive Bonferroni correction. Even adjacent tax rates (e.g., 0% vs 2.5%: d=0.60, p=0.001) produce detectable welfare differences with n=60.

![Refined Welfare vs Tax Rate](../runs/20260214-142400_skillrl_study/plots/refined_welfare_vs_tax.png)

### 4.4 Toxicity Remains Invariant

Toxicity regression: slope=0.0075, R²=0.008, p=0.07. Effectively zero. Neither tax rate nor circuit breaker threshold produces any detectable change in toxicity.

![Refined Toxicity](../runs/20260214-142400_skillrl_study/plots/refined_toxicity_vs_tax.png)

### 4.5 Summary Panel

![Summary Panel](../runs/20260214-142400_skillrl_study/plots/refined_summary_panel.png)

## 5. Discussion

### 5.1 Why Taxes Cannot Reduce Toxicity

Code analysis reveals the mechanism: transaction taxes are applied as a governance cost (`c_a`, `c_b`) in the payoff equation *after* the acceptance decision has been made. Toxicity is measured as `E[1-p | accepted]` — the average harm probability among accepted interactions. Since agents decide to accept or reject based on expected payoff *before* tax is levied, the tax cannot influence which interactions enter the accepted pool. The tax is a pure post-hoc friction cost.

### 5.2 Why Circuit Breakers Don't Affect Toxicity Either

Even when the circuit breaker successfully freezes adversarial agents (CB=0.5), toxicity doesn't change because:
1. Adversarial agents were already earning near-zero payoff (~2.2), meaning they were rarely selected as interaction partners
2. Toxicity is an average over all accepted interactions, and the adversarial agent's contribution to the pool is negligible
3. The remaining toxicity (~0.27) comes from the inherent noise in honest and opportunistic agents' interactions, not from adversarial agents

### 5.3 Implications for Governance Design

These results suggest that effective toxicity reduction requires mechanisms that intervene *before* the acceptance decision:
- **Pre-acceptance screening**: governance costs incorporated into the expected payoff calculation
- **Reputation-weighted matching**: reducing the probability of pairing with low-reputation agents
- **Progressive taxation**: higher rates for agents with low average p, creating incentive to improve quality

Flat transaction taxes and post-hoc circuit breakers are structurally incapable of reducing toxicity in this framework.

### 5.4 Limitations

- Fixed agent composition (3 SkillRL, 2 honest, 1 opportunistic, 1 adversarial)
- Toxicity floor (~0.27) may reflect proxy scoring noise rather than true harmful interactions
- Circuit breaker only tested at two thresholds; intermediate values may reveal nonlinear effects

## 6. Reproducibility

```bash
# Reproduce initial sweep (80 runs)
python examples/parameter_sweep.py \
  --scenario scenarios/skillrl.yaml \
  --output runs/skillrl_sweep.csv \
  --seed 42 --epochs 20 --runs_per_config 10

# Reproduce refined sweep (420 runs)
python runs/20260214-142400_skillrl_study/refined_sweep.py

# Reproduce analysis
python runs/20260214-142400_skillrl_study/analyze_sweep.py
python runs/20260214-142400_skillrl_study/analyze_refined.py

# Reproduce plots
python runs/20260214-142400_skillrl_study/generate_plots.py
python runs/20260214-142400_skillrl_study/generate_refined_plots.py
```

## References

- Xia et al. (2026). SkillRL: Recursive Skill-Augmented Reinforcement Learning.
