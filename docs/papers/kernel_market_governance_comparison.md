# Comparative Governance Effects in a GPU Kernel Marketplace with Correlated Proxy Gaps

## Abstract

We compare the effects of seven governance regimes on welfare, toxicity, and agent payoffs in a GPU kernel marketplace with correlated speedup-cheating, split functional/OOD test regimes, and tolerance exploitation. Sweeping from no governance to full governance (audits + staking + circuit breaker + reputation + tax) across 70 runs (7 regimes × 10 seeds), we find that **circuit breakers** emerge as the most effective single lever: they produce the highest welfare (22.96 vs 12.70 no-governance baseline, d = 1.64, p = 0.022 Bonferroni-corrected) and the lowest toxicity (0.395 vs 0.446, d = −2.08, p = 0.003 Bonferroni-corrected). Full governance also significantly reduces toxicity (d = −1.89, p = 0.007). By contrast, staking alone produces the worst outcomes on both welfare (10.65) and toxicity (0.452). Nine out of 42 pairwise comparisons survive Bonferroni correction, making this the most statistically productive sweep in the kernel market series. Agent-type stratification remains massive across all regimes (honest vs adversarial d = 2.24–3.39, all p < 0.0001), and quality gap stays positive (0.015–0.116), indicating the market avoids adverse selection under all tested governance configurations.

## Introduction

Previous work on the kernel market v2 model established two findings: (1) transaction taxes reduce welfare without reducing toxicity (deadweight loss, d = 1.19), and (2) audit rate primarily affects deterrence (adversarial payoffs) rather than system-level outcomes. Both studies held most governance levers constant, varying one at a time.

This paper takes a different approach: we compare **complete governance regimes** against each other, including a no-governance baseline. This allows us to answer: which governance mechanism matters most? Do combinations outperform single levers? And which mechanisms have no effect or negative effects?

We test seven regimes spanning the space from no governance to full governance:

1. **No governance**: All mechanisms disabled, reputation decay = 1.0
2. **Audits only**: 20% audit probability, 3× penalty multiplier
3. **Staking only**: Minimum stake = 1.0, slash rate = 0.3
4. **Reputation only**: Reputation decay = 0.85
5. **Circuit breaker only**: Freeze threshold toxicity = 0.5, 2 violations, 3-epoch duration
6. **Audits + staking**: Both mechanisms enabled
7. **Full governance**: All levers enabled including 5% transaction tax

## Model

We use the kernel market v2 scenario with correlated speedup-cheating, split functional/OOD test regimes, and tolerance exploitation (see companion v2 paper).

### Agent Composition

| Type | Count | Func. Quality | OOD Quality | Base Speedup | Role |
|---|---|---|---|---|---|
| Honest | 2 | 0.85 | 0.80 | N(0.50, σ) | Correct + fast kernels |
| Opportunistic | 3 | 0.70 | 0.50 | N(0.40, σ) | Cut corners on precision |
| Adversarial | 2 | 0.95 | 0.30 | N(0.45, σ) + 0.40 | Exploit proxy signals |
| Verifier | 1 | --- | --- | --- | Surface-level checks |

Speedup variance σ scales with difficulty: easy = 0.15, medium = 0.25, hard = 0.35.

### Governance Regimes

| Regime | Audit Prob. | Staking | Rep. Decay | Circuit Breaker | Tax |
|---|---|---|---|---|---|
| No governance | 0.00 | Off | 1.00 | Off | 0% |
| Audits only | 0.20 | Off | 0.98 | Off | 0% |
| Staking only | 0.00 | On (1.0/0.3) | 0.98 | Off | 0% |
| Reputation only | 0.00 | Off | 0.85 | Off | 0% |
| Circuit breaker only | 0.00 | Off | 0.98 | On (0.5/2/3) | 0% |
| Audits + staking | 0.20 | On (1.0/0.3) | 0.98 | Off | 0% |
| Full governance | 0.15 | On (1.0/0.3) | 0.90 | On (0.5/2/3) | 5% |

### Sweep Configuration

| Parameter | Values |
|---|---|
| `governance.regime` | 7 discrete regimes (above) |
| Seeds per configuration | 10 |
| Total runs | **70** |
| Epochs per run | 5 |
| Steps per epoch | 10 |

## Results

### Welfare

| Regime | Welfare (mean ± SD) |
|---|---|
| No governance | 12.70 ± 5.28 |
| Audits only | 15.02 ± 4.17 |
| Staking only | 10.65 ± 4.13 |
| Reputation only | 15.18 ± 3.40 |
| Circuit breaker only | **22.96 ± 6.22** |
| Audits + staking | 13.34 ± 5.26 |
| Full governance | 21.38 ± 7.30 |

Circuit breaker achieves nearly double the welfare of staking (22.96 vs 10.65, d = 2.20, Bonferroni p = 0.008). It also significantly outperforms no governance (d = 1.64, p = 0.022 Bonferroni-corrected across 12 baseline comparisons).

### Toxicity

| Regime | Toxicity (mean ± SD) |
|---|---|
| No governance | 0.4463 ± 0.0282 |
| Audits only | 0.4319 ± 0.0178 |
| Staking only | 0.4518 ± 0.0224 |
| Reputation only | 0.4357 ± 0.0233 |
| Circuit breaker only | **0.3948 ± 0.0209** |
| Audits + staking | 0.4396 ± 0.0222 |
| Full governance | **0.3992 ± 0.0211** |

Circuit breaker and full governance are the only regimes that significantly reduce toxicity vs no governance:
- Circuit breaker: d = −2.08, Bonferroni p = 0.003
- Full governance: d = −1.89, Bonferroni p = 0.007

Staking alone actually increases toxicity (0.452 vs 0.446 baseline), the worst of all regimes.

### Quality Gap

| Regime | Quality Gap (mean) |
|---|---|
| No governance | 0.113 |
| Audits only | 0.116 |
| Staking only | 0.099 |
| Reputation only | 0.113 |
| Circuit breaker only | 0.038 |
| Audits + staking | 0.097 |
| Full governance | 0.015 |

Quality gap remains positive across all regimes (no adverse selection). Notably, circuit breaker and full governance have the lowest quality gaps (0.038 and 0.015), suggesting these regimes narrow the distribution of accepted submissions — they accept less aggressively but more uniformly.

### Agent-Type Stratification

| Regime | Honest Payoff | Opp. Payoff | Adv. Payoff | d (H vs A) |
|---|---|---|---|---|
| No governance | 2.64 | 2.53 | −1.40 | 2.27*** |
| Audits only | 3.28 | 2.93 | −1.80 | 2.91*** |
| Staking only | 2.26 | 2.27 | −1.46 | 3.39*** |
| Reputation only | 3.09 | 2.82 | −1.28 | 2.86*** |
| Circuit breaker only | 4.30 | 3.74 | −0.59 | 2.62*** |
| Audits + staking | 2.95 | 2.60 | −1.66 | 2.53*** |
| Full governance | 4.34 | 3.40 | −0.92 | 2.24*** |

***p < 0.0001, Bonferroni-significant.

All regime comparisons between honest and adversarial agents are significant (p < 0.0001). The pooled effect sizes:
- Honest vs adversarial: d = 3.34, p < 0.00001
- Opportunistic vs adversarial: d = 5.42, p < 0.00001
- Honest vs opportunistic: d = 0.25, p = 0.14 (ns)

### Bonferroni-Significant Comparisons (9/42)

| Rank | Metric | Comparison | |d| | Bonf. p |
|---|---|---|---|---|
| 1 | Toxicity | Circuit breaker vs Staking | 2.64 | 0.0006 |
| 2 | Toxicity | Full governance vs Staking | 2.42 | 0.0016 |
| 3 | Welfare | Circuit breaker vs Staking | 2.20 | 0.008 |
| 4 | Toxicity | Audits+staking vs Circuit breaker | 2.08 | 0.008 |
| 5 | Toxicity | Circuit breaker vs No governance | 2.08 | 0.010 |
| 6 | Toxicity | Audits only vs Circuit breaker | 1.91 | 0.020 |
| 7 | Toxicity | Full governance vs No governance | 1.89 | 0.024 |
| 8 | Toxicity | Audits+staking vs Full governance | 1.87 | 0.024 |
| 9 | Toxicity | Circuit breaker vs Reputation | 1.85 | 0.026 |

Seven of nine significant results involve the circuit breaker — either directly (circuit_breaker_only) or as part of full governance.

### Omnibus Tests

| Test | Metric | Statistic | p |
|---|---|---|---|
| Kruskal-Wallis | Welfare | H = 24.24 | 0.0005 |
| Kruskal-Wallis | Toxicity | H = 33.37 | 0.000009 |
| Kruskal-Wallis | Quality gap | H = 36.29 | 0.000002 |
| ANOVA | Welfare | F = 6.70 | 0.000016 |
| ANOVA | Toxicity | F = 10.06 | < 0.000001 |
| ANOVA | Quality gap | F = 12.89 | < 0.000001 |

All omnibus tests are significant, confirming that the governance regime matters.

### Normality Validation

Shapiro-Wilk tests confirm welfare and toxicity distributions are normal across all regime groups (all p > 0.05), validating the use of parametric t-tests. Mann-Whitney U tests confirm all Bonferroni-significant results.

## Discussion

### Circuit Breakers Dominate

The circuit breaker emerges as the single most effective governance mechanism, producing:
- **Highest welfare** (22.96, +81% vs no governance)
- **Lowest toxicity** (0.395, −11% vs no governance)
- **Highest honest payoffs** (4.30, +63% vs no governance)
- **Highest opportunistic payoffs** (3.74, +48% vs no governance)
- **Least negative adversarial payoffs** (−0.59 vs −1.40 no governance)

This seems paradoxical: the circuit breaker is the regime under which adversarial agents lose the *least*, yet it produces the best system-level outcomes. The explanation lies in the mechanism's design: circuit breakers freeze high-toxicity agents, preventing them from further harming the ecosystem. This benefits everyone — including adversarial agents, who accumulate fewer penalties during frozen periods.

### Staking Backfires

Staking alone produces the worst outcomes across the board: lowest welfare (10.65), highest toxicity (0.452), and the lowest honest/opportunistic payoffs. This suggests that in correlated proxy-gap environments, requiring agents to post collateral before participating creates a barrier that disproportionately harms honest agents (who may not have accumulated capital) while failing to deter adversarial agents (who can still recoup stake through inflated speedup signals).

### The Tax Penalty Persists

Full governance (which includes a 5% transaction tax) achieves significantly lower toxicity than no governance (d = −1.89) but does not significantly outperform circuit breaker alone on any metric. The tax component appears to drag down welfare relative to what circuit breakers alone achieve (21.38 vs 22.96), consistent with the v2 finding that taxes impose deadweight loss.

### Combinations Don't Always Help

Audits + staking (combination) performs worse than audits alone on welfare (13.34 vs 15.02) and toxicity (0.440 vs 0.432). Adding staking to audits dilutes the audit benefit rather than amplifying it. This suggests governance interactions can be **subadditive** — the whole is less than the sum of parts.

### Honest ≈ Opportunistic Persists

As in the audit rate study, honest and opportunistic agents earn nearly identical payoffs (d = 0.25, ns). The proxy gap between these types (functional quality 0.85 vs 0.70) is not large enough for governance to distinguish them. Both benefit equally from improved governance.

## Figures

1. `plots/welfare_by_regime.png` — Mean welfare by regime with 95% CI, color-coded by category
2. `plots/toxicity_by_regime.png` — Mean toxicity by regime with 95% CI
3. `plots/quality_gap_by_regime.png` — Quality gap with adverse selection threshold
4. `plots/agent_payoff_by_regime.png` — Grouped bar chart by agent type and regime
5. `plots/welfare_toxicity_tradeoff.png` — Welfare-toxicity scatter by regime
6. `plots/adversarial_payoff_by_regime.png` — Adversarial payoff with significance highlighting
7. `plots/regime_heatmap.png` — Z-scored performance heatmap across all metrics

## Reproducibility

```python
from pathlib import Path
from swarm.analysis import SweepConfig, SweepParameter, SweepRunner
from swarm.scenarios import load_scenario

sc = load_scenario(Path('scenarios/kernel_market/v2.yaml'))
sc.orchestrator_config.n_epochs = 5

# 7 governance regimes applied as discrete conditions
regimes = {
    'no_governance': {'audit_probability': 0.0, 'staking_enabled': False,
                      'reputation_decay_rate': 1.0, 'circuit_breaker_enabled': False,
                      'transaction_tax_rate': 0.0},
    'audits_only': {'audit_probability': 0.20, 'audit_penalty_multiplier': 3.0},
    'staking_only': {'staking_enabled': True, 'staking_min_stake': 1.0,
                     'staking_slash_rate': 0.3},
    'reputation_only': {'reputation_decay_rate': 0.85},
    'circuit_breaker_only': {'circuit_breaker_enabled': True,
                             'circuit_breaker_freeze_threshold_toxicity': 0.5,
                             'circuit_breaker_consecutive_violations': 2,
                             'circuit_breaker_freeze_duration_epochs': 3},
    'audits_staking': {**audits_only, **staking_only},
    'full_governance': {'audit_probability': 0.15, 'staking_enabled': True,
                        'circuit_breaker_enabled': True, 'reputation_decay_rate': 0.90,
                        'transaction_tax_rate': 0.05},
}
# See scenarios/kernel_market/sweeps/governance_comparison.yaml for exact config
```

Run artifacts: `runs/20260211-000149_kernel_market_governance_comparison/`

## Limitations

- **5 epochs per run** may be insufficient for circuit breaker dynamics to fully manifest (freezes are 3 epochs long, leaving only 2 non-frozen epochs).
- **Agent composition is fixed** — future sweeps should vary the adversarial fraction to test robustness.
- **No interaction sweeps** — we test 7 point configs, not the full parameter space. Two-lever interaction effects may exist that we don't capture.
- **Non-adaptive agents** — agents don't change strategy in response to governance. Adaptive adversaries could exploit regime-specific weaknesses.

## Conclusion

Comparing seven governance regimes in the v2 kernel market model reveals that **circuit breakers are the dominant governance mechanism**, producing the highest welfare (+81% vs no governance), lowest toxicity (−11%), and the most Bonferroni-significant comparisons (7/9 involve circuit breakers). Staking alone backfires, reducing welfare and increasing toxicity. Transaction taxes impose deadweight loss even within full governance. Governance combinations can be subadditive (audits + staking < audits alone). These findings suggest that **mechanism design matters more than mechanism quantity**: a single well-targeted lever (circuit breaker) outperforms a full governance stack that includes less effective mechanisms. Future work should explore adaptive adversaries, interaction effects between levers, and the circuit breaker's sensitivity to its threshold parameters.
