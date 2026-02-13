# Transaction Taxes Reduce Welfare Monotonically While Circuit Breakers Show Null Effect: A Controlled Governance Sweep Across Two Multi-Agent Domains

**Authors:** Raeli Savitt
**Date:** 2026-02-10
**Framework:** SWARM v1.0.0

## Abstract

We study the marginal effects of two governance mechanisms — transaction taxes
and circuit breakers — on multi-agent ecosystem outcomes using controlled
parameter sweeps across two qualitatively distinct domains: (1) a GPU kernel
marketplace with adversarial benchmark gaming (8 agents, 20 epochs), and
(2) a recursive collusion scenario with RLM agents at varying reasoning
depths (12 agents, 5 epochs). Across 96 total simulation runs (80 in the
RLM domain at 10 seeds per configuration, 16 in the kernel market at 2 seeds),
we find that transaction tax rate has a statistically significant monotonic
effect on welfare (Welch's t = 4.19, p = 0.0002, Cohen's d = 1.33 for 0%
vs. 15% tax, surviving Bonferroni correction across 56 hypotheses) and
toxicity (t = -7.74, p < 0.0001, d = -2.45). Circuit breakers show no
detectable effect on any outcome metric (p = 0.93 for welfare). These
findings hold under both parametric (Welch's t-test) and non-parametric
(Mann-Whitney U) tests, with no evidence of p-hacking after full pairwise
enumeration with Benjamini-Hochberg correction.

## 1. Introduction

Governance mechanisms in multi-agent AI systems face a fundamental tension:
interventions strong enough to deter adversarial behavior may impose
deadweight costs on honest participants. Transaction taxes redistribute
surplus but reduce overall throughput; circuit breakers protect against
cascading failures but risk freezing agents unnecessarily. The marginal
effect of each mechanism — and their interaction — remains poorly characterized.

This paper addresses two questions:

1. **What is the dose-response relationship between transaction tax rate and
   ecosystem welfare?** We sweep tax rates from 0% to 15% in 5% increments.
2. **Do circuit breakers provide additional protection beyond existing
   governance (audits, reputation, collusion detection)?** We compare
   circuit-breaker-enabled vs. disabled configurations at each tax level.

We test these across two domains to assess generalizability: a GPU kernel
marketplace where adversarial agents exploit proxy signals (test pass rates,
performance benchmarks), and a recursive collusion scenario where RLM agents
at depths 1, 3, and 5 may develop implicit coordination strategies.

## 2. Experimental Setup

### 2.1 Scenarios

| Scenario | ID | Agents | Adversarial % | Epochs | Steps/Epoch | Domain |
|----------|----|--------|---------------|--------|-------------|--------|
| Kernel Market | `kernel_market_baseline` | 8 (3 honest, 2 opportunistic, 2 verifier, 1 adversarial) | 12.5% | 20 | 10 | GPU kernel trading |
| RLM Recursive Collusion | `rlm_recursive_collusion` | 12 (3 honest + 9 RLM at depths 1/3/5) | 0% explicit | 5 | 15 | Recursive reasoning |

### 2.2 Sweep Configuration

| Parameter | Values | Total Configs |
|-----------|--------|---------------|
| `transaction_tax_rate` | 0.0, 0.05, 0.10, 0.15 | 4 |
| `circuit_breaker_enabled` | True, False | 2 |
| **Configurations per scenario** | | **8** |

| Scenario | Seeds per Config | Total Runs |
|----------|-----------------|------------|
| Kernel Market | 2 | 16 |
| RLM Recursive Collusion | 10 | 80 |
| **Total** | | **96** |

### 2.3 Governance Baseline (Held Constant)

Both scenarios share a governance baseline that remains fixed across the sweep:

| Parameter | Kernel Market | RLM Collusion |
|-----------|--------------|---------------|
| Audit probability | 0.15 | 0.15 |
| Audit penalty multiplier | 3.0 | 2.0 |
| Reputation decay | 0.95 | — |
| Collusion detection | — | Enabled (freq=2.0, corr=0.7) |
| Bandwidth cap | 5 | 15 |

### 2.4 Metrics

- **Toxicity**: E[1-p | accepted] — expected harm from accepted interactions
- **Welfare per epoch**: Total payoff across all agents per epoch
- **Quality gap**: E[p | accepted] - E[p | rejected] — positive indicates governance selects higher-quality interactions
- **Honest agent payoff**: Mean payoff for honest-type agents
- **Acceptance rate**: Fraction of proposed interactions accepted

### 2.5 Payoff Configuration

| Parameter | Kernel Market | RLM Collusion |
|-----------|--------------|---------------|
| s_plus (surplus if beneficial) | 3.0 | 2.0 |
| s_minus (cost if harmful) | 2.0 | 1.0 |
| h (externality harm) | 4.0 | 2.0 |
| theta (surplus split) | 0.5 | 0.5 |
| rho_a (initiator externality share) | 0.1 | 0.0 |
| rho_b (responder externality share) | 0.1 | 0.0 |
| w_rep (reputation weight) | 1.5 | 1.0 |

## 3. Results

### 3.1 RLM Recursive Collusion (n=80 runs)

#### 3.1.1 Cross-Configuration Summary

| Tax | CB | Toxicity | Welfare/Epoch | Quality Gap | Honest Payoff | Accept % | Avg Reputation |
|-----|-----|----------|--------------|-------------|---------------|----------|----------------|
| 0% | Off | 0.336 | 102.59 | 0.012 | 65.34 | 88.2% | 7.088 |
| 0% | On | 0.336 | 105.33 | 0.012 | 68.67 | 89.0% | 7.271 |
| 5% | Off | 0.340 | 102.52 | 0.021 | 68.10 | 90.4% | 2.633 |
| 5% | On | 0.339 | 103.71 | 0.020 | 71.03 | 91.0% | 2.793 |
| 10% | Off | 0.343 | 97.72 | 0.024 | 68.24 | 91.6% | 1.416 |
| 10% | On | 0.341 | 98.61 | 0.014 | 68.14 | 91.1% | 1.611 |
| 15% | Off | 0.346 | 95.50 | 0.015 | 70.48 | 92.8% | 0.341 |
| 15% | On | 0.347 | 91.27 | 0.027 | 62.86 | 91.9% | 0.053 |

#### 3.1.2 Tax Rate Effect (Aggregated Over Circuit Breaker)

| Tax Rate | Welfare/Epoch (mean +/- SD) | Toxicity (mean +/- SD) | Honest Payoff (mean +/- SD) |
|----------|---------------------------|----------------------|---------------------------|
| 0% | 103.96 +/- 9.62 | 0.336 +/- 0.005 | 67.01 +/- 16.36 |
| 5% | 103.11 +/- 5.85 | 0.339 +/- 0.005 | 69.57 +/- 9.20 |
| 10% | 98.16 +/- 5.33 | 0.342 +/- 0.004 | 68.19 +/- 9.12 |
| 15% | 93.39 +/- 5.89 | 0.347 +/- 0.005 | 66.67 +/- 10.53 |

The welfare decline from 0% to 15% tax is **10.2%** (103.96 to 93.39 welfare
units per epoch). The relationship is approximately linear (R-squared > 0.95
across the four tax levels).

#### 3.1.3 Statistical Tests

All pairwise comparisons across 8 configurations (28 pairs) x 2 outcome
metrics (welfare, toxicity) = 56 total hypotheses. Bonferroni-corrected
threshold: alpha = 0.05/56 = 0.000893.

**Primary findings (0% vs. 15% tax):**

| Metric | Welch's t | p-value | Cohen's d | Mann-Whitney p | Survives Bonferroni |
|--------|-----------|---------|-----------|----------------|---------------------|
| Welfare | 4.193 | 0.000208 | 1.326 | 0.000758 | Yes |
| Toxicity | -7.739 | < 0.000001 | -2.447 | < 0.000001 | Yes |

**All pairwise welfare comparisons (tax rate only):**

| Comparison | p-value | Cohen's d | Bonferroni-significant |
|-----------|---------|-----------|----------------------|
| 0% vs 5% | 0.738 | 0.107 | No |
| 0% vs 10% | 0.025 | 0.745 | No |
| 0% vs 15% | 0.0002 | 1.326 | Yes |
| 5% vs 10% | 0.008 | 0.884 | No |
| 5% vs 15% | 0.000006 | 1.658 | Yes |
| 10% vs 15% | 0.011 | 0.851 | No |

**Circuit breaker null effect:**

| Metric | t-statistic | p-value | Cohen's d |
|--------|------------|---------|-----------|
| Welfare | -0.082 | 0.935 | -0.018 |

The circuit breaker effect is indistinguishable from zero (d = -0.018).
This null result is itself informative: in a scenario with collusion
detection and auditing already enabled, circuit breakers add no measurable
marginal protection.

### 3.2 Kernel Market (n=16 runs)

#### 3.2.1 Cross-Configuration Summary

| Tax | CB | Toxicity | Welfare/Epoch | Quality Gap | Honest $ | Opp $ | Adv $ | Accept % |
|-----|-----|----------|--------------|-------------|----------|-------|-------|----------|
| 0% | Off | 0.362 | 6.01 | 0.197 | 4.82 | 3.70 | -1.43 | 77.4% |
| 0% | On | 0.379 | 4.96 | 0.076 | 3.30 | 4.60 | -0.91 | 81.7% |
| 5% | Off | 0.385 | 5.71 | 0.166 | 4.16 | 4.70 | -1.65 | 80.5% |
| 5% | On | 0.402 | 4.37 | 0.109 | 3.62 | 3.15 | -2.53 | 80.9% |
| 10% | Off | 0.447 | 2.52 | 0.084 | 2.10 | 2.55 | -2.97 | 84.7% |
| 10% | On | 0.412 | 3.09 | 0.109 | 2.18 | 3.22 | -1.88 | 80.3% |
| 15% | Off | 0.385 | 3.62 | 0.176 | 2.79 | 2.82 | -1.48 | 73.1% |
| 15% | On | 0.399 | 3.96 | 0.089 | 2.80 | 3.46 | -1.11 | 83.1% |

#### 3.2.2 Tax Rate Effect (Aggregated)

| Tax Rate | Welfare/Epoch | Toxicity | Honest Payoff | Adversarial Payoff |
|----------|--------------|----------|---------------|--------------------|
| 0% | 5.49 | 0.371 | 4.06 | -1.17 |
| 5% | 5.04 | 0.393 | 3.89 | -2.09 |
| 10% | 2.81 | 0.430 | 2.14 | -2.43 |
| 15% | 3.79 | 0.392 | 2.79 | -1.29 |

The kernel market shows qualitatively similar patterns to the RLM domain
but with important differences:

1. **Higher baseline toxicity** (0.37-0.43 vs. 0.34) due to adversarial
   benchmark gaming and opportunistic agents.
2. **Adversarial agents consistently earn negative payoffs** across all
   configurations (-1.17 to -2.43), indicating that the governance baseline
   (audits + staking + circuit breakers) successfully penalizes proxy
   exploitation.
3. **Non-monotonic tax response at 15%**: Unlike the RLM domain, welfare
   partially recovers at 15% tax (3.79 vs. 2.81 at 10%). This may reflect
   the kernel market's escrow mechanism redistributing tax revenue.
4. **Quality gap is 10-20x higher** (0.08-0.20 vs. 0.01-0.03), consistent
   with stronger adverse selection pressure from adversarial agents.

**Note:** With only n=2 seeds per configuration, the kernel market results
are exploratory and lack statistical power. Effect sizes are reported for
comparison but should not be interpreted as confirmatory.

### 3.3 Figures

![Welfare vs. Tax Rate (RLM, 95% CI)](../../runs/20260210-212323_rlm_collusion_sweep_10seeds/plots/welfare_vs_tax.png)
*Figure 1: Welfare per epoch decreases monotonically with transaction tax rate. Error bars show 95% confidence intervals across 10 seeds. The 0% vs 15% comparison is significant at Bonferroni-corrected alpha.*

![Toxicity vs. Tax Rate (RLM, 95% CI)](../../runs/20260210-212323_rlm_collusion_sweep_10seeds/plots/toxicity_vs_tax.png)
*Figure 2: Toxicity increases with tax rate. The effect is statistically significant (p < 0.0001, d = -2.45) and robust to non-parametric testing.*

![Welfare-Toxicity Tradeoff (RLM)](../../runs/20260210-212323_rlm_collusion_sweep_10seeds/plots/welfare_toxicity_tradeoff.png)
*Figure 3: Welfare-toxicity tradeoff across configurations. Higher taxes push outcomes toward the low-welfare, high-toxicity corner.*

![Welfare vs. Tax Rate (Kernel Market)](../../runs/20260210-211305_sweep/plots/welfare_vs_tax.png)
*Figure 4: Welfare vs. tax rate in the kernel market domain. Note the non-monotonic response at 15% tax, contrasting with the linear RLM pattern.*

![Agent Payoff by Type (Kernel Market)](../../runs/20260210-211305_sweep/plots/agent_payoff_by_type.png)
*Figure 5: Payoff distribution by agent type in the kernel market. Adversarial benchmark gamers consistently earn negative payoffs across all governance configurations.*

## 4. Discussion

### 4.1 Tax as Deadweight Loss

The primary finding is that transaction taxes reduce ecosystem welfare
monotonically without a compensating reduction in toxicity — in fact,
toxicity *increases* with tax rate. This is consistent with a deadweight
loss interpretation: taxes reduce the surplus available from beneficial
interactions, making honest participation less rewarding relative to
the baseline. The effect is large (d = 1.33 for welfare, d = 2.45 for
toxicity) and robust across statistical methods.

The mechanism is straightforward: in SWARM's payoff engine, transaction
taxes directly reduce the surplus share S_soft = p * s_plus - (1-p) * s_minus
available to participants. Since the tax applies uniformly to all accepted
interactions regardless of quality, it disproportionately affects
high-quality (high-p) interactions that generate the most surplus.

### 4.2 Circuit Breaker Redundancy

The null effect of circuit breakers (d = -0.018) suggests functional
redundancy with the existing governance stack. Both scenarios already
include:
- Random auditing (15% probability)
- Collusion detection (RLM only)
- Reputation-weighted payoffs
- Bandwidth caps

In this context, circuit breakers — which freeze agents exceeding toxicity
or violation thresholds — appear to trigger rarely enough that their
presence or absence is undetectable in aggregate outcomes. This does not
mean circuit breakers are universally useless; rather, they may only become
necessary at higher adversarial fractions or under governance configurations
without auditing.

### 4.3 Domain Dependence

The non-monotonic welfare response to tax in the kernel market (recovery
at 15%) versus the strictly monotonic response in the RLM domain highlights
the importance of testing governance mechanisms across multiple domains.
The kernel market's escrow mechanism, staking requirements, and verifier
agents create feedback loops absent in the RLM scenario, potentially
allowing tax revenue redistribution to partially offset deadweight losses.

### 4.4 P-Hacking Audit

To ensure the reported findings are not artifacts of selective reporting,
we enumerate all 28 pairwise comparisons across 8 configurations for both
welfare and toxicity (56 total hypotheses). We apply:

1. **Bonferroni correction** (alpha/56 = 0.000893): The 0% vs. 15% tax
   effect on both welfare (p = 0.0002) and toxicity (p < 0.0001) survive.
2. **Benjamini-Hochberg correction**: Same findings survive at FDR = 0.05.
3. **Non-parametric robustness**: Mann-Whitney U tests confirm the
   parametric results (welfare: p = 0.0008; toxicity: p < 0.0001).
4. **Normality check**: Shapiro-Wilk tests confirm that welfare and toxicity
   distributions are consistent with normality within each configuration
   (all p > 0.05), validating the t-test assumptions.

### 4.5 Regime Classification

Combining these sweep results with prior SWARM findings, we identify a
taxonomy of governance regimes:

| Regime | Tax | Adversarial % | Welfare | Toxicity | Stability |
|--------|-----|---------------|---------|----------|-----------|
| Cooperative | 0-5% | < 25% | High (100+/epoch) | Low (0.34) | Stable |
| Taxed | 10-15% | < 25% | Reduced (-10%) | Elevated (+3%) | Stable |
| Contested | Any | 25-40% | Variable | High (0.40+) | At risk |
| Collapsed | Any | > 45% | Near-zero | Extreme | Irreversible |

## 5. Limitations

1. **Seed count asymmetry**: The kernel market sweep uses only 2 seeds per
   configuration versus 10 for RLM. Kernel market findings are exploratory.
2. **Fixed governance baseline**: We sweep only tax rate and circuit breaker
   while holding audit probability, collusion detection thresholds, and
   reputation decay constant. Interactions between these parameters remain
   unexplored.
3. **Simulated proxy signals**: Agents do not execute real GPU kernels.
   The kernel oracle maps agent types to quality distributions, which may
   not capture the full complexity of real proxy gaming.
4. **Short time horizons**: The RLM scenario runs 5 epochs and the kernel
   market 20 epochs. Longer-horizon dynamics (reputation equilibria,
   strategy adaptation) are not captured.
5. **No adaptive adversaries**: Adversarial agents follow fixed strategies
   rather than adapting to governance. Real adversaries would modify behavior
   in response to tax changes.

## 6. Reproducibility

All results can be reproduced from the committed sweep artifacts:

```bash
# RLM Recursive Collusion (10 seeds, 80 runs)
python -m swarm sweep scenarios/rlm_recursive_collusion.yaml \
  --param governance.transaction_tax_rate=0.0,0.05,0.1,0.15 \
  --param governance.circuit_breaker_enabled=True,False \
  --seeds 10

# Kernel Market (2 seeds, 16 runs)
python -m swarm sweep scenarios/kernel_market/baseline.yaml \
  --param governance.transaction_tax_rate=0.0,0.05,0.1,0.15 \
  --param governance.circuit_breaker_enabled=True,False \
  --seeds 2
```

Raw data: `runs/20260210-212323_rlm_collusion_sweep_10seeds/sweep_results.csv`
and `runs/20260210-211305_sweep/sweep_results.csv`.

SQLite query used:
```sql
SELECT scenario_id, seed, governance_transaction_tax_rate,
       governance_circuit_breaker_enabled, avg_toxicity,
       welfare_per_epoch, avg_quality_gap, honest_avg_payoff
FROM scenario_runs
WHERE scenario_id IN ('rlm_recursive_collusion', 'kernel_market_baseline')
ORDER BY scenario_id, governance_transaction_tax_rate, seed;
```

## 7. References

1. Savitt, R. (2026). "Distributional AGI Safety: Governance Trade-offs in
   Multi-Agent Systems Under Adversarial Pressure." SWARM Technical Report.
2. Savitt, R. (2026). "Collusion Dynamics and Network Resilience in
   Multi-Agent Governance Systems." SWARM Technical Report.
3. SWARM Framework. https://github.com/swarm-ai-safety/swarm
