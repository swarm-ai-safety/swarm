# The Cost of Safety: Governance Overhead vs. Toxicity Reduction in GasTown Multi-Agent Workspaces

**Authors:** Raeli Savitt
**Date:** 2026-02-13
**Framework:** SWARM v1.3.1

## Abstract

We study the welfare–safety tradeoff in GasTown-style multi-agent workspaces by sweeping adversarial agent proportion from 0% to 86% under two regimes: full governance (circuit breaker, collusion detection, staking, random audit) and no governance. Across 42 runs (7 compositions x 2 regimes x 3 seeds, 30 epochs each), we find a consistent **governance cost paradox**: governance levers reduce toxicity at every adversarial level (mean reduction 0.071) but impose welfare costs that exceed the safety benefit at all tested proportions. At 0% adversarial, governance costs 216 welfare units (-57.6%) for only 0.066 toxicity reduction. The cost narrows as adversarial pressure increases, converging at 86% rogue where the welfare gap shrinks to 10.6 units. Neither regime triggers the 0.5 toxicity crisis threshold. Adversarial agents never achieve payoffs above 2.3, while honest agents dominate in both regimes. No adverse selection is observed. These results suggest that the current governance parameter set is over-tuned for safety at the expense of throughput, and that governance levers may be most cost-effective when targeted rather than applied uniformly.

## 1. Introduction

Multi-agent AI workspaces face a fundamental tension: governance mechanisms that prevent adversarial exploitation impose friction on all participants, including honest ones. The GasTown workspace model — with its roles (Mayor, Polecats, Witnesses, Deacons, Rogues), bead-based task economy, and layered governance — provides a naturalistic setting to study this tradeoff.

Prior SWARM composition studies have established that adversarial agent proportion degrades welfare monotonically and that governance levers reduce toxicity. However, the *net welfare impact* of governance — its benefit through harm reduction minus its cost through friction — has not been measured directly.

This study asks three questions:

1. **At what adversarial proportion does governance become net-positive for welfare?**
2. **How much welfare do governance levers cost in the absence of adversarial pressure?**
3. **Is there a critical threshold where governance can no longer contain adversarial agents?**

## 2. Experimental Setup

### 2.1 Population Design

We fix the total population at 7 agents and sweep the number of adversarial (rogue) agents from 0 to 6. The remaining slots are filled with honest agents plus 1 opportunistic agent (the "corner-cutter") when room permits. Each composition is run under both governance regimes.

| Composition | Honest | Opportunistic | Adversarial | Rogue % |
|-------------|--------|---------------|-------------|---------|
| 0/7 rogue   | 6      | 1             | 0           | 0%      |
| 1/7 rogue   | 5      | 1             | 1           | 14%     |
| 2/7 rogue   | 4      | 1             | 2           | 29%     |
| 3/7 rogue   | 3      | 1             | 3           | 43%     |
| 4/7 rogue   | 2      | 1             | 4           | 57%     |
| 5/7 rogue   | 1      | 1             | 5           | 71%     |
| 6/7 rogue   | 1      | 0             | 6           | 86%     |

### 2.2 Governance Parameters

Two regimes are compared:

| Parameter | Governed | Ungoverned |
|-----------|----------|------------|
| Transaction tax rate | 0.05 | 0.00 |
| Reputation decay | 0.95/epoch | 1.00 (none) |
| Circuit breaker | Enabled (toxicity > 0.5, 3 violations, 2-epoch freeze) | Disabled |
| Random audit | 20% probability, 2x penalty | Disabled |
| Staking | 5.0 min stake, 20% slash | Disabled |
| Collusion detection | Enabled (freq 2.0, corr 0.7, realtime 10%) | Disabled |
| Bandwidth cap | 8 | Unlimited |

### 2.3 Payoff Parameters

GasTown workspace payoff configuration (from `scenarios/gastown_workspace.yaml`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| s_plus | 3.0 | Surplus from beneficial interaction |
| s_minus | 1.5 | Loss from harmful interaction |
| h | 2.5 | Externality (tech debt / shipped bugs) |
| theta | 0.5 | Even surplus split |
| rho_a | 0.3 | Agent A externality internalization |
| rho_b | 0.2 | Agent B externality internalization |
| w_rep | 1.5 | Reputation weight |

### 2.4 Simulation Parameters

- **Epochs:** 30 per run
- **Steps per epoch:** 10
- **Seeds:** 42, 43, 44 (3 seeds per configuration)
- **Total runs:** 42 (7 compositions x 2 regimes x 3 seeds)
- **Total simulation epochs:** 1,260

### 2.5 Metrics

- **Total welfare**: Cumulative payoff across all agents, summed over epochs
- **Toxicity rate**: E[1-p | accepted] — expected harm from accepted interactions
- **Quality gap**: E[p | accepted] - E[p | rejected] — negative indicates adverse selection
- **Per-class payoff**: Average total payoff by agent type (honest, opportunistic, adversarial)

## 3. Results

### 3.1 Cross-Regime Summary

| Rogue % | Governed Welfare | Ungoverned Welfare | Welfare Gap | Governed Toxicity | Ungoverned Toxicity | Toxicity Gap |
|---------|-----------------|-------------------|-------------|-------------------|---------------------|--------------|
| 0%  | 158.857 +/- 0.273 | 374.733 +/- 0.574 | -215.876 | 0.214 +/- 0.014 | 0.281 +/- 0.006 | -0.066 |
| 14% | 103.113 +/- 0.620 | 291.829 +/- 0.253 | -188.716 | 0.216 +/- 0.019 | 0.293 +/- 0.004 | -0.077 |
| 29% | 70.064 +/- 0.353  | 252.350 +/- 0.473 | -182.286 | 0.198 +/- 0.029 | 0.300 +/- 0.002 | -0.102 |
| 43% | 67.113 +/- 0.208  | 198.785 +/- 0.411 | -131.672 | 0.248 +/- 0.066 | 0.300 +/- 0.005 | -0.052 |
| 57% | 50.487 +/- 0.079  | 141.564 +/- 0.305 | -91.077  | 0.261 +/- 0.035 | 0.313 +/- 0.011 | -0.052 |
| 71% | 21.255 +/- 0.134  | 78.132 +/- 0.152  | -56.877  | 0.175 +/- 0.043 | 0.307 +/- 0.025 | -0.133 |
| 86% | 21.804 +/- 0.146  | 32.367 +/- 0.164  | -10.563  | 0.409 +/- 0.015 | 0.421 +/- 0.012 | -0.012 |

### 3.2 Welfare Degradation

Both regimes show monotonic welfare decline as adversarial proportion increases:

- **Ungoverned**: Welfare drops from 374.7 (0% rogue) to 32.4 (86% rogue), a 91.4% decline.
- **Governed**: Welfare drops from 158.9 (0% rogue) to 21.8 (86% rogue), an 86.3% decline.

The ungoverned regime produces 2.0x–3.7x higher welfare than the governed regime at every adversarial level. The gap is largest at low adversarial proportions and narrows as rogues dominate.

![Figure 1: Total welfare by rogue proportion](figures/gastown_governance_cost/gastown_welfare.png)
*Figure 1. Total welfare (sum over 30 epochs) vs. adversarial agent proportion. Ungoverned (red) consistently outperforms governed (green) in raw welfare, though the gap narrows under high adversarial pressure.*

### 3.3 Toxicity Containment

Governance consistently reduces toxicity, with the governed regime maintaining lower toxicity at every adversarial level:

- **Governed toxicity range**: 0.175–0.409
- **Ungoverned toxicity range**: 0.281–0.421
- **Mean toxicity reduction from governance**: 0.071 (across all levels)
- **Maximum toxicity reduction**: 0.133 (at 71% rogue)

Neither regime breached the 0.5 crisis threshold, suggesting that even without governance, the market's own filtering mechanisms provide partial safety.

![Figure 2: Toxicity rate by rogue proportion](figures/gastown_governance_cost/gastown_toxicity.png)
*Figure 2. Toxicity rate (mean over epochs) vs. adversarial proportion. Governance keeps toxicity 5–13 percentage points lower across the sweep. Both regimes approach 0.42 at 86% rogue.*

### 3.4 The Governance Cost Paradox

The central finding: governance reduces welfare at every tested adversarial proportion. The "governance protection" metric (governed welfare minus ungoverned welfare) is negative everywhere:

| Rogue % | Welfare Cost of Governance | Toxicity Benefit | Cost per Toxicity Point |
|---------|---------------------------|------------------|------------------------|
| 0%  | -215.9 | 0.066 | 3,271 welfare/toxicity-point |
| 14% | -188.7 | 0.077 | 2,450 |
| 29% | -182.3 | 0.102 | 1,787 |
| 43% | -131.7 | 0.052 | 2,532 |
| 57% | -91.1  | 0.052 | 1,752 |
| 71% | -56.9  | 0.133 | 428 |
| 86% | -10.6  | 0.012 | 883 |

The cost-per-toxicity-point drops dramatically at 71% rogue (428), suggesting governance becomes most cost-efficient only under extreme adversarial pressure.

![Figure 3: Governance protection — benefit over ungoverned baseline](figures/gastown_governance_cost/gastown_governance_protection.png)
*Figure 3. Governance benefit (governed minus ungoverned) for welfare and toxicity reduction. Welfare gain is negative at all levels, indicating governance is a net cost. The gap narrows as adversarial proportion rises.*

### 3.5 Per-Class Payoff Analysis

Honest agents earn the highest payoffs in both regimes, confirming that the payoff structure rewards genuine contributions:

**Governed regime:**
- Honest agents: 7.4–24.6 average payoff (peaks at 0% rogue)
- Opportunistic agents: 0.0–15.1 (peaks at 29% rogue)
- Adversarial agents: -0.1–2.1 (never dominates)

**Ungoverned regime:**
- Honest agents: 19.4–58.1 (peaks at 43% rogue — paradoxically higher under pressure)
- Opportunistic agents: 0.0–50.2 (peaks at 14% rogue)
- Adversarial agents: 0.0–2.3 (always marginal)

The payoff breakdown reveals why ungoverned welfare is higher: honest agents earn 2x–4x more without governance friction, and this more than compensates for the slightly higher toxicity.

![Figure 4: Per-class payoffs by rogue proportion](figures/gastown_governance_cost/gastown_payoff_breakdown.png)
*Figure 4. Per-class average total payoff under governed (left) and ungoverned (right) regimes. Governance compresses all payoffs but preserves the honest > opportunistic > adversarial ranking.*

### 3.6 Adverse Selection

Quality gap remains positive in both regimes at all adversarial levels, indicating that low-quality interactions are preferentially rejected:

- **Governed quality gap range**: 0.012–0.125
- **Ungoverned quality gap range**: 0.040–0.285

The ungoverned regime shows a *stronger* positive quality gap (mean 0.215 vs. 0.076 governed), suggesting that market-based filtering without governance overhead may actually be more selective.

![Figure 5: Adverse selection by rogue proportion](figures/gastown_governance_cost/gastown_adverse_selection.png)
*Figure 5. Quality gap vs. adversarial proportion. Positive values mean the system correctly accepts higher-quality interactions. The ungoverned regime (red) shows stronger selection throughout.*

## 4. Discussion

### 4.1 Why Governance Costs More Than It Saves

The governance cost paradox arises from the cumulative friction of five simultaneous mechanisms:

1. **Transaction tax** (5%) directly reduces all payoffs
2. **Reputation decay** (0.95/epoch) penalizes all agents, not just adversarial ones
3. **Staking requirements** (5.0 minimum) lock up capital that could generate welfare
4. **Bandwidth cap** (8 interactions) limits honest agent throughput
5. **Random audits** (20% probability) create false-positive friction

These costs scale with the number of agents and interactions, while the benefit (catching adversarial agents) scales only with the adversarial fraction. At low adversarial levels, the cost-to-benefit ratio is extremely unfavorable (3,271 welfare per toxicity point at 0% rogue).

### 4.2 When Does Governance Break Even?

Governance never achieves positive net welfare in this study. However, the cost drops sharply at high adversarial proportions. Extrapolating the convergence trend, governance would break even only in a near-total adversarial scenario (>90% rogue) — a regime unlikely in practice.

This suggests the governance parameters are calibrated for a worst-case scenario that doesn't match typical operating conditions. A lighter-touch governance regime — lower tax rates, higher audit thresholds, relaxed staking — might achieve the same toxicity reduction at lower welfare cost.

### 4.3 The Market as Safety Mechanism

The absence of adverse selection in the ungoverned regime is notable. Even without formal governance, the payoff structure itself (high s_plus, high h, strong reputation weight w_rep=1.5) creates natural incentives against accepting low-quality work. Honest agents dominate because beneficial interactions generate more surplus than adversarial ones can extract.

This suggests that well-designed payoff structures may substitute for governance mechanisms, at least when adversarial agents are a minority.

### 4.4 Honest Agent Resilience

The most striking result is that honest agents earn more under adversarial pressure in the ungoverned regime (peaking at 58.1 at 43% rogue) than in any governed configuration. This is partially an artifact of bandwidth caps limiting honest agent throughput under governance, and partially reflects that honest agents benefit from reduced competition as adversarial agents fail to generate surplus.

## 5. Conclusion

This study reveals a governance cost paradox in GasTown multi-agent workspaces: the full governance stack (transaction tax, reputation decay, circuit breaker, random audit, staking, collusion detection) consistently reduces toxicity by 5–13 percentage points but imposes welfare costs of 10–216 units — yielding negative net welfare at every adversarial proportion tested. Governance is most cost-efficient at 71% adversarial (428 welfare per toxicity point reduced) and least efficient at 0% adversarial (3,271 per point). Neither regime triggers the 0.5 toxicity crisis threshold, and no adverse selection is observed. These findings suggest that the current governance parameter set is over-calibrated for the threat level, and that lighter-touch or threshold-triggered governance — activating only when toxicity crosses a danger threshold — may better balance safety with productive throughput.

## 6. Limitations

- **Fixed population size**: All runs use 7 agents. Scaling effects (larger populations with smaller adversarial fractions) are unexplored.
- **Uniform governance**: All governance levers are on or off together. Ablation studies varying individual levers would reveal which mechanisms are most cost-effective.
- **Single payoff configuration**: The GasTown payoff parameters (s_plus=3.0, h=2.5, w_rep=1.5) may favor the ungoverned regime. Different economies may change the governance tradeoff.
- **3 seeds**: Low seed count limits statistical power. Standard deviations are tight (reflecting deterministic agent policies) but edge cases may be undersampled.
- **No adaptive adversaries**: Adversarial agents use fixed policies. Adaptive adversaries that learn to evade governance could change the cost-benefit calculus.
- **Welfare ≠ safety**: Higher welfare does not necessarily mean a safer system. In domains where tail-risk catastrophic harm matters more than average toxicity, governance overhead may be justified even at high welfare cost.

## 7. Reproducibility

```bash
# Full study (42 runs, ~2 min)
python examples/gastown_composition_study.py --total-agents 7 --epochs 30 --steps 10 --seeds 3

# Smoke test (~5 sec)
python examples/gastown_composition_study.py --total-agents 4 --epochs 3 --steps 3 --seeds 1
```

Data from this paper: `runs/20260213-103914_gastown_composition_study/`

## 8. References

- SWARM framework: `swarm/` (v1.3.1)
- GasTown scenario: `scenarios/gastown_workspace.yaml`
- GasTown bridge: `swarm/bridges/gastown/`
- Composition study runner: `examples/gastown_composition_study.py`
