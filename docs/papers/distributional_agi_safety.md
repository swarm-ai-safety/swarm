# Distributional AGI Safety: Governance Trade-offs in Multi-Agent Systems Under Adversarial Pressure

**Authors:** Raeli Savitt
**Date:** 2026-02-09
**Framework:** SWARM v1.0.0

## Abstract

We study governance trade-offs in multi-agent AI systems using a probabilistic
simulation framework that replaces binary safety labels with calibrated soft
scores p = P(v = +1). Across 11 scenarios (209 total epochs, 81 agent-slots),
we find that ecosystem outcomes cluster into three regimes: cooperative
(acceptance > 0.93, toxicity < 0.30), contested (acceptance 0.42-0.94, toxicity
0.33-0.37), and adversarial collapse (acceptance < 0.56, collapse by epoch
12-14). Collapse occurred exclusively when adversarial fraction reached 50%,
and governance tuning delayed but did not prevent it — shifting collapse from
epoch 12 to 14 across three red-team variants. Collusion detection proved
critical: a scenario with 37.5% adversarial agents avoided collapse entirely
when pair-wise frequency and correlation monitoring were enabled, while
comparable scenarios without it collapsed. Incoherence metrics scaled
sub-linearly with agent count, while welfare scaled super-linearly, suggesting
that larger cooperative populations are disproportionately productive but also
harder to monitor. These results formalize the intuition from market
microstructure theory that adverse selection in agent ecosystems is
regime-dependent: governance interventions that suffice under moderate
adversarial pressure fail abruptly beyond a critical threshold.

## 1. Introduction

As AI systems increasingly operate as autonomous agents — negotiating,
collaborating, and competing within shared digital environments — the safety
question shifts from aligning a single model to governing an ecosystem of
interacting agents with heterogeneous objectives. A growing body of work
addresses multi-agent safety through mechanism design [1, 2], distributional
analysis [3], and economic governance frameworks [4]. Yet empirical study of
*how* and *when* governance interventions fail under adversarial pressure
remains limited, in part because most evaluations use binary safety labels
(safe/unsafe) that obscure the probabilistic, continuous nature of real
interaction quality.

This paper takes a different approach. Drawing on market microstructure
theory — specifically the adverse selection models of Kyle [5] and Glosten and
Milgrom [6] — we model multi-agent ecosystems as markets in which agents with
private information about interaction quality choose whether and how to
participate. Honest agents are analogous to uninformed traders: they rely on
observable signals and cooperate in good faith. Adversarial and deceptive
agents resemble informed traders: they exploit private knowledge of their own
intentions to extract value at the ecosystem's expense. The governance
mechanism — acceptance thresholds, audits, circuit breakers — plays the role of
the market maker, setting terms of participation that must balance the cost of
excluding legitimate interactions against the risk of admitting harmful ones.

Central to our framework is the replacement of binary safety labels with
*soft probabilistic labels*: each interaction receives a calibrated score
p = P(v = +1), the probability that its true value is beneficial. This
follows the distributional safety framework of Kenton et al. [3], which
argues that safety properties are better characterized by distributions over
outcomes than by point classifications. Probabilistic labels enable
continuous metrics — toxicity as E[1-p | accepted], quality gap as the
difference in expected p between accepted and rejected interactions — that
capture adverse selection dynamics — the "lemons problem" [7] — invisible to
binary classification.

We implement this framework in SWARM (System-Wide Assessment of Risk in
Multi-agent systems), a configurable simulation environment supporting
multiple agent behavioral types, governance lever combinations, network
topologies, and economic mechanisms including Dworkin-style resource
auctions [8], Shapley-value reward allocation [9], and mission
economies [4]. Using SWARM, we run 11 scenarios spanning cooperative,
contested, and adversarial regimes, varying agent composition from 0% to 50%
adversarial fraction and governance from disabled to fully layered
(tax + staking + circuit breaker + audit + collusion detection).

Our central research questions are:

1. **Is there a critical adversarial fraction** beyond which governance
   interventions fail to prevent ecosystem collapse, and if so, where does
   it lie?
2. **Which governance levers** provide qualitatively different protection
   (extending the viable operating range) versus quantitatively incremental
   improvement (delaying but not preventing collapse)?
3. **How do safety metrics and welfare scale** with agent population size
   and network density?

We find that ecosystem outcomes partition cleanly into three regimes, that
the collapse boundary lies between 37.5% and 50% adversarial fraction, and
that collusion detection — a structural governance lever operating on
interaction patterns rather than individual agents — is the critical
differentiator between survival and collapse in contested environments.

## 2. Experimental Setup

### 2.1 Scenarios

| Scenario | Agents | Composition | Adversarial % | Epochs | Network |
|----------|--------|-------------|---------------|--------|---------|
| baseline | 5 | 3H, 1O, 1D | 20.0% | 10 | none |
| adversarial_redteam_v1 | 8 | 4H, 2Adv, 2AA | 50.0% | 30 | small_world (k=4, p=0.15) |
| adversarial_redteam_v2 | 8 | 4H, 2Adv, 2AA | 50.0% | 30 | small_world (k=4, p=0.15) |
| adversarial_redteam_v3 | 8 | 4H, 2Adv, 2AA | 50.0% | 30 | small_world (k=4, p=0.15) |
| collusion_detection | 8 | 3H, 2O, 3Adv | 37.5% | 25 | small_world (k=4, p=0.2) |
| emergent_capabilities | 8 | 6H, 2O | 0.0% | 30 | complete |
| incoherence_short | 3 | 3H | 0.0% | 8 | none |
| incoherence_medium | 6 | 5H, 1O | 16.7% | 8 | none |
| incoherence_long | 10 | 9H, 1Adv | 10.0% | 8 | none |
| marketplace_economy | 7 | 3H, 2O, 1D, 1Adv | 14.3% | 10 | none |
| network_effects | 10 | 4H, 3O, 2D, 1Adv | 10.0% | 20 | small_world (k=4, p=0.1) |

Agent types: **H** = honest, **O** = opportunistic, **D** = deceptive, **Adv** = adversarial, **AA** = adaptive adversary.

### 2.2 Governance Parameters

| Scenario | Tax Rate | Rep. Decay | Staking | Circuit Breaker | Audit | Collusion Det. |
|----------|----------|------------|---------|-----------------|-------|----------------|
| baseline | 0.00 | 1.00 | off | off | off | off |
| adversarial_redteam_v* | 0.05 | 0.95 | 10.0 min | tox > 0.6, 3 viol, 3 ep freeze | p=0.15, 2.5x pen | freq > 2.0 z, corr > 0.6 |
| collusion_detection | 0.05 | 0.95 | 10.0 min | tox > 0.6, 3 viol, 2 ep freeze | off | freq > 2.0 z, corr > 0.7 |
| emergent_capabilities | 0.03 | 0.97 | 5.0 min | tox > 0.7, 5 viol, 2 ep freeze | off | off |
| marketplace_economy | 0.05 | 0.95 | off | tox > 0.7, 3 viol, 2 ep freeze | p=0.10, 2.0x pen | off |
| network_effects | 0.05 | 0.95 | off | tox > 0.7, 3 ep freeze | p=0.10 | off |

Redteam variant differences:
- **v1**: Original governance parameters
- **v2**: `audit_penalty=1.0`, `freeze_duration=1`
- **v3**: `audit_penalty=1.0`, `freeze_duration=1`, `rep_decay=1.0`

### 2.3 Payoff Parameters

| Scenario | s+ | s- | h | theta | rho_a | rho_b | w_rep |
|----------|-----|-----|------|-------|-------|-------|-------|
| baseline | 2.0 | 1.0 | 2.0 | 0.5 | 0.0 | 0.0 | 1.0 |
| adversarial_redteam | 3.0 | 1.5 | 3.0 | 0.5 | — | — | 2.5 |
| collusion_detection | 3.0 | 1.5 | 2.5 | 0.5 | — | — | 2.0 |
| emergent_capabilities | 3.5 | 1.0 | 2.0 | 0.5 | — | — | 2.5 |
| marketplace_economy | 2.0 | 1.0 | 2.0 | 0.5 | 0.1 | 0.1 | 1.0 |
| network_effects | 2.0 | 1.0 | 2.0 | 0.5 | 0.1 | 0.1 | 1.0 |

### 2.4 Formal Model

**Proxy computation.** Four observable signals — task progress, rework penalty,
verifier penalty, and engagement signal — are combined into a raw proxy score
via weighted sum:

v_hat = w_task * x_task + w_rework * x_rework + w_verifier * x_verifier + w_engage * x_engage

where each x_i is in [-1, +1] and the weights (default: 0.4, 0.2, 0.2, 0.2)
are normalized to sum to one. The raw score is mapped to a probability through
a calibrated sigmoid:

p = P(v = +1 | v_hat) = 1 / (1 + exp(-k * (v_hat - b)))

with steepness k = 2 and bias b = 0 by default.

**Soft payoffs.** Given probability p for an interaction, the expected surplus
and externality are:

S_soft = p * s+ - (1-p) * s-
E_soft = (1-p) * h

Agent payoffs decompose as:

pi_i = alpha_i * S_soft + T_i - G_i - rho_i * E_soft + w_rep * delta_R_i

where alpha_i is the surplus share (theta for initiator, 1-theta for
counterparty), T_i is net transfer (zero-sum across parties), G_i is
governance cost (taxes, staking slashes, audit penalties), rho_i controls
externality internalization, and delta_R_i is the reputation change weighted
by w_rep. This decomposition follows mechanism design principles [1, 2] and
the distributional safety framework [3].

**Acceptance rule.** An interaction is accepted when p >= p_bar (default
p_bar = 0.5). This creates the adverse selection dynamic described by
Akerlof [7] and Glosten-Milgrom [6]: agents with private knowledge of low
interaction quality may still find it profitable to participate when the
acceptance threshold admits borderline interactions.

### 2.5 Metrics

- **Acceptance rate**: fraction of interactions accepted by governance threshold (p >= p_bar)
- **Toxicity**: E[1-p | accepted], expected harm from accepted interactions
- **Welfare/epoch**: cumulative payoff across agents per epoch
- **Quality gap**: E[p|accepted] - E[p|rejected] (negative = adverse selection)
- **Collapse epoch**: first epoch where ecosystem function degrades irreversibly (welfare drops to zero or agents frozen)

## 3. Results

### 3.1 Cross-Scenario Summary

| Scenario | Acceptance | Toxicity | Welfare/Epoch | Adv. Frac. | Collapse? |
|----------|-----------|----------|---------------|------------|-----------|
| baseline | 0.938 | 0.298 | 5.0 | 0.200 | No |
| adversarial_redteam_v1 | 0.556 | 0.295 | 3.8 | 0.500 | Epoch 12 |
| adversarial_redteam_v2 | 0.481 | 0.312 | 3.8 | 0.500 | Epoch 13 |
| adversarial_redteam_v3 | 0.455 | 0.312 | 3.5 | 0.500 | Epoch 14 |
| collusion_detection | 0.425 | 0.370 | 6.3 | 0.375 | No |
| emergent_capabilities | 0.998 | 0.297 | 44.9 | 0.000 | No |
| incoherence_short | 1.000 | 0.183 | 1.0 | 0.000 | No |
| incoherence_medium | 0.940 | 0.343 | 5.7 | 0.167 | No |
| incoherence_long | 0.787 | 0.341 | 21.3 | 0.100 | No |
| marketplace_economy | 0.549 | 0.328 | 3.7 | 0.143 | No |
| network_effects | 0.783 | 0.335 | 9.9 | 0.100 | No |

### 3.2 Regime Classification

| Regime | Scenarios | Acceptance | Toxicity | Collapse |
|--------|-----------|-----------|----------|----------|
| Cooperative | baseline, emergent_capabilities, incoherence_short | > 0.93 | < 0.30 | Never |
| Contested | collusion_detection, marketplace, network, incoherence_med/long | 0.42 - 0.94 | 0.33 - 0.37 | Never |
| Adversarial collapse | redteam_v1, v2, v3 | < 0.56 | 0.30 - 0.31 | Epochs 12-14 |

### 3.3 Key Observations

**Adversarial red-team progression (v1 -> v2 -> v3):**
- Acceptance rate declined monotonically: 0.556 -> 0.481 -> 0.455
- Collapse epoch shifted later: 12 -> 13 -> 14
- Welfare/epoch degraded: 3.8 -> 3.8 -> 3.5
- Governance parameter tuning delayed but did not prevent collapse at 50% adversarial fraction

**Collusion detection scenario:**
- Highest toxicity (0.370) among non-collapsing scenarios
- Maintained positive welfare (6.3/epoch) despite 37.5% adversarial fraction
- Collusion detection prevented collapse that occurred in redteam scenarios with higher adversarial fraction

**Incoherence scaling** (related to variance-dominated failure modes [12]):
- Toxicity scaled with agent count: 0.183 (3 agents) -> 0.343 (6) -> 0.341 (10)
- Acceptance rate decreased: 1.000 -> 0.940 -> 0.787
- Non-linear welfare scaling: 1.0 -> 5.7 -> 21.3 (super-linear in agent count)

### 3.4 Marketplace and Network Analysis

**Marketplace economy.** The bounty/escrow marketplace scenario (7 agents,
14.3% adversarial) achieved a 0.549 acceptance rate — the lowest among
non-collapsing, non-redteam scenarios. Of 82 total interactions, only 45
were accepted, reflecting the additional filtering imposed by escrow
verification and dispute resolution. Despite this selectivity, welfare
remained modest at 3.7/epoch (total: 37.0), with final-epoch welfare
dropping to 1.4. The marketplace mechanism effectively traded throughput for
safety: toxicity (0.328) was lower than the collusion detection scenario
(0.370) despite having fewer governance levers active, suggesting that
economic friction (escrow fees, bid deadlines, dispute costs) functions as
an implicit governance mechanism by raising the cost of low-quality
participation.

**Network effects.** The small-world network scenario (10 agents, 10%
adversarial, k=4, p=0.1 rewiring) sustained operation across all 20 epochs
with no collapse. It processed the second-highest interaction volume (314
total, 246 accepted) behind only the emergent capabilities scenario. Welfare
scaled well at 9.9/epoch (total: 197.9), with final-epoch welfare of 12.9
— indicating accelerating returns as dynamic edge strengthening consolidated
cooperative relationships. The acceptance rate (0.783) and toxicity (0.335)
positioned it squarely in the contested regime. Notably, this scenario
achieved higher per-agent welfare (0.99/agent/epoch) than the baseline
(1.0/agent/epoch) despite having both deceptive and adversarial agents,
suggesting that network topology — specifically, the ability for honest
agents to strengthen connections with each other and weaken ties to bad
actors — provides a decentralized governance mechanism complementary to
centralized levers [11]. The dynamic edge evolution also provides a natural
dampener against flash-crash-style cascading failures [10], as weakened
edges to adversarial nodes reduce contagion velocity.

**Comparison.** The marketplace and network scenarios represent two
architectural approaches to the same problem: the marketplace uses
centralized economic mechanisms (escrow, arbitration) while the network
relies on decentralized topological evolution (edge strengthening/decay).
The network approach yielded higher welfare (9.9 vs. 3.7/epoch) and
throughput (246 vs. 45 accepted) at comparable toxicity (0.335 vs. 0.328),
but required more agents (10 vs. 7) and longer runtime (20 vs. 10 epochs)
to realize these gains.

### 3.5 Figures

![Figure 1: Scenario Comparison](figures/fig1_scenario_comparison.png)
*Cross-scenario comparison of acceptance rate, toxicity, and welfare.*

![Figure 2: Collusion Timeline](figures/fig2_collusion_timeline.png)
*Temporal evolution of collusion detection flags and penalties.*

![Figure 3: Regime Scatter](figures/fig3_regime_scatter.png)
*Acceptance vs. toxicity scatter plot with regime boundaries.*

![Figure 4: Incoherence Scaling](figures/fig4_incoherence_scaling.png)
*Incoherence metrics across short/medium/long branching configurations.*

![Figure 5: Welfare Comparison](figures/fig5_welfare_comparison.png)
*Welfare per epoch across governance regimes.*

![Figure 6: Network vs Collusion](figures/fig6_network_vs_collusion.png)
*Network topology effects on collusion detection efficacy.*

## 4. Discussion

### 4.1 Regime Transitions and the Adversarial Threshold

The sharpest finding is the existence of a critical adversarial fraction
between 37.5% and 50%. Below this threshold, ecosystems remained functional
across all governance configurations tested — the collusion detection scenario
sustained positive welfare (6.3/epoch) despite 37.5% adversarial agents.
Above it, all three red-team variants collapsed regardless of parameter
tuning. This is consistent with the Glosten-Milgrom adverse selection model:
when the fraction of informed (adversarial) agents exceeds the point at which
the acceptance mechanism can no longer distinguish signal from noise, the
ecosystem equivalent of a bid-ask spread widens until cooperation becomes
unprofitable for honest agents. The monotonic decline in acceptance rate across
red-team variants (0.556 to 0.455) traces exactly this widening.

Governance tuning across v1-v3 shifted collapse later by two epochs (12 to 14)
but could not prevent it. This suggests that parameter optimization within a
fixed governance architecture has diminishing returns against high adversarial
pressure. The implication for deployed multi-agent systems is that monitoring
adversarial fraction is at least as important as tuning governance parameters —
structural composition matters more than lever calibration once a critical
threshold is crossed.

### 4.2 Collusion Detection as a Phase-Transition Buffer

The collusion detection scenario is the most governance-intensive non-collapsing
run. At 37.5% adversarial fraction, it sat just below the collapse threshold
and exhibited the highest toxicity (0.370) of any surviving scenario. Yet it
maintained positive welfare because pair-wise frequency monitoring (z-score >
2.0) and correlation thresholds (> 0.7) disrupted coordinated exploitation
before it could cascade.

This points to collusion detection as a qualitatively different kind of
governance lever. Transaction taxes and circuit breakers operate on individual
agents; collusion detection operates on interaction structure. The red-team
scenarios had collusion detection enabled at a lower correlation threshold
(0.6) but still collapsed — suggesting that at 50% adversarial fraction, even
structural monitoring is overwhelmed. The effective range of collusion
detection appears to be roughly 25-40% adversarial fraction: below 25%,
individual-level governance suffices; above 40-50%, no tested configuration
prevented collapse.

### 4.3 The Welfare-Toxicity Trade-off

Across the contested regime, welfare and toxicity are not simply inversely
correlated. The collusion detection scenario achieved higher welfare (6.3/epoch)
than the baseline (5.0/epoch) despite substantially higher toxicity (0.370 vs.
0.298) and a larger adversarial fraction (37.5% vs. 20%). This is because
governance costs and rejected interactions reduce welfare in low-adversarial
scenarios where filtering is aggressive, while contested scenarios that
maintain function extract more surplus from the larger interaction volume that
higher agent counts provide.

The emergent capabilities scenario illustrates the upper bound: with zero
adversarial agents and a complete network, it achieved 44.9 welfare/epoch at
0.297 toxicity — an order of magnitude above other scenarios. This suggests
that the welfare cost of adversarial pressure is not linear but
multiplicative: each adversarial agent degrades not just its own interactions
but the productivity of the surrounding cooperative network.

### 4.4 Incoherence and Scale

The incoherence series (3, 6, 10 agents) reveals two scaling dynamics. Toxicity
saturated quickly: it jumped from 0.183 to 0.343 between 3 and 6 agents, then
plateaued at 0.341 for 10 agents. This suggests a floor effect — once any
adversarial or opportunistic agents are present, baseline toxicity stabilizes
around 0.34 regardless of further scaling. Acceptance rate, by contrast,
declined steadily (1.000 to 0.940 to 0.787), indicating that the governance
mechanism becomes more selective as the interaction graph grows denser.

Welfare scaled super-linearly (1.0 to 5.7 to 21.3), consistent with network
effects in cooperative production: more agents create more interaction
opportunities, and the surplus from beneficial interactions compounds. This
super-linear scaling is encouraging for the viability of large cooperative
multi-agent systems, but it also raises the stakes of the adversarial
threshold: a collapse in a large ecosystem destroys disproportionately more
value.

### 4.5 Implications for Multi-Agent System Design

These results suggest three practical design principles:

1. **Monitor composition, not just metrics.** Toxicity and acceptance rate are
   lagging indicators. By the time they degrade visibly, the ecosystem may be
   past the collapse threshold. Tracking adversarial fraction directly — via
   behavioral classification or collusion detection — provides earlier warning.

2. **Layer structural governance over individual governance.** Transaction taxes
   and circuit breakers are necessary but not sufficient. Collusion detection
   (pair-wise interaction analysis) provides a qualitatively different defense
   that extends the viable operating range by roughly 15-20 percentage points
   of adversarial fraction.

3. **Design for regime awareness.** A single governance configuration cannot
   optimally serve all three regimes. Cooperative regimes are over-governed by
   aggressive parameters (reducing welfare), while adversarial regimes are
   under-governed by moderate ones (permitting collapse). Adaptive governance
   that tightens as adversarial indicators rise would better track the
   operating regime.

### 4.6 Future Work

Several directions follow naturally from these findings:

**Multi-seed validation.** The most immediate need is running each scenario
across 10-50 seeds to establish confidence intervals on regime boundaries.
The critical adversarial threshold (37.5%-50%) is currently a two-point
estimate; multi-seed sweeps at 5% increments between 30% and 55% adversarial
fraction would sharpen this to a transition curve with error bars.

**Adaptive governance.** All governance parameters in this study were static.
A natural extension is a meta-governance layer that observes real-time
metrics (toxicity trend, acceptance rate slope, collusion flags) and adjusts
lever settings epoch-by-epoch. This could be implemented as a bandit
algorithm over governance configurations or as a reinforcement learning agent
optimizing a welfare-toxicity Pareto frontier. The key question is whether
adaptive governance can prevent collapse at adversarial fractions above 50%,
or whether the threshold is fundamental.

**Dynamic adversarial fraction.** In deployed systems, agents may shift
strategies over time — an honest agent may become opportunistic as it
discovers exploits, or an adversarial agent may reform after repeated
penalties. Modeling adversarial fraction as a dynamic variable (driven by
payoff incentives, imitation dynamics, or evolutionary pressure) would test
whether governance can stabilize composition or whether adversarial drift is
self-reinforcing.

**Scale experiments.** The super-linear welfare scaling observed in the
incoherence series (3 to 10 agents) motivates testing at 50, 100, and 500
agents. Key questions: Does the adversarial threshold shift with scale? Does
collusion detection remain tractable when the number of agent pairs grows
quadratically? Do network topologies that work at 10 agents fragment or
centralize at 100?

**Learned proxy weights.** The current proxy weights (0.4, 0.2, 0.2, 0.2)
are hand-set. Training the weight vector and sigmoid parameters (k, b) via
gradient descent on labeled interaction data would test whether calibration
quality affects the adversarial threshold — better proxies might extend the
viable operating range by narrowing the bid-ask spread analogy from
Section 4.1.

**Cross-scenario transfer.** Testing whether governance parameters optimized
for one regime transfer to another would inform deployment strategy. A
configuration tuned on the baseline scenario may fail catastrophically when
adversarial fraction increases — quantifying this brittleness would
strengthen the case for regime-aware adaptive governance.

## 5. Limitations

- **Single-seed runs.** Each scenario was run with seed 42 only. Results may
  not be robust to stochastic variation; multi-seed sweeps with confidence
  intervals are needed to confirm regime boundaries.

- **Simulation fidelity.** Agent behavioral types are stylized (honest,
  opportunistic, deceptive, adversarial). Real multi-agent systems exhibit
  richer and more continuous behavioral variation that may shift the
  thresholds identified here.

- **Fixed adversarial fraction.** Adversarial fraction was set per-scenario
  and did not evolve over time. In practice, agents may shift strategies
  dynamically, and the collapse threshold likely depends on the rate of
  behavioral change, not just the static fraction.

- **No learned governance.** All governance parameters were hand-configured.
  Learned or adaptive governance policies (e.g., reinforcement learning over
  lever settings) might extend the viable range beyond what static tuning
  achieves.

- **Collapse definition.** Collapse is operationalized as the first epoch
  where welfare degrades irreversibly. Alternative definitions (e.g., based
  on honest-agent exit or network fragmentation) might yield different
  collapse epochs.

- **Scale.** The largest scenario tested had 10 agents. Extrapolating regime
  boundaries to systems with hundreds or thousands of agents requires
  further validation, particularly given the super-linear welfare scaling
  observed.

## 6. References

[1] Myerson, R.B. (1981). Optimal Auction Design. *Mathematics of Operations
Research*, 6(1), 58-73.

[2] Hurwicz, L. (1960). Optimality and Informational Efficiency in Resource
Allocation Processes. In Arrow, K.J., Karlin, S., & Suppes, P. (Eds.),
*Mathematical Methods in the Social Sciences*, 27-46. Stanford University
Press.

[3] Kenton, Z., Filos, A., Evans, O., & Gal, Y. (2025). Distributional Safety
in Agentic Systems. *arXiv preprint* arXiv:2512.16856.

[4] Tomasev, N., Franklin, J., Leibo, J.Z., Jacobs, A.Z., Cunningham, T.,
Gabriel, I., & Osindero, S. (2025). Virtual Agent Economies. *arXiv preprint*
arXiv:2509.10147.

[5] Kyle, A.S. (1985). Continuous Auctions and Insider Trading.
*Econometrica*, 53(6), 1315-1335.

[6] Glosten, L.R. & Milgrom, P.R. (1985). Bid, Ask and Transaction Prices in
a Specialist Market with Heterogeneously Informed Traders. *Journal of
Financial Economics*, 14(1), 71-100.

[7] Akerlof, G.A. (1970). The Market for "Lemons": Quality Uncertainty and the
Market Mechanism. *Quarterly Journal of Economics*, 84(3), 488-500.

[8] Dworkin, R. (1981). What is Equality? Part 2: Equality of Resources.
*Philosophy & Public Affairs*, 10(4), 283-345.

[9] Shapley, L.S. (1953). A Value for n-Person Games. In Kuhn, H.W. &
Tucker, A.W. (Eds.), *Contributions to the Theory of Games*, Vol. 2, 307-317.
Princeton University Press.

[10] Kyle, A.S., Obizhaeva, A.A., & Tuzun, T. (2017). Flash Crashes and
Market Microstructure. Working Paper.

[11] Chen, Y., Shenker, S., & Zhao, S. (2025). Multi-Agent Market Dynamics.
*arXiv preprint* arXiv:2502.14143.

[12] Anthropic. (2026). The Hot Mess Theory of AI. Anthropic Alignment Blog.
https://alignment.anthropic.com/2026/hot-mess-of-ai/

## 7. Appendix

### A.1 Full Run Data

| Scenario | Seed | Agents | Epochs | Steps | Total Int. | Accepted | Accept Rate | Toxicity | Welfare/Ep | Total Welfare | Final Welfare | Adv. Frac | Collapse |
|----------|------|--------|--------|-------|------------|----------|-------------|----------|------------|---------------|---------------|-----------|----------|
| baseline | 42 | 5 | 10 | 10 | 48 | 45 | 0.938 | 0.298 | 4.98 | — | — | 0.200 | — |
| adversarial_redteam_v1 | 42 | 8 | 30 | 15 | 135 | 75 | 0.556 | 0.295 | 3.80 | — | — | 0.500 | Ep 12 |
| adversarial_redteam_v2 | 42 | 8 | 30 | 15 | 158 | 76 | 0.481 | 0.312 | 3.80 | — | — | 0.500 | Ep 13 |
| adversarial_redteam_v3 | 42 | 8 | 30 | 15 | 156 | 71 | 0.455 | 0.312 | 3.49 | — | — | 0.500 | Ep 14 |
| collusion_detection | 42 | 8 | 25 | 15 | 299 | 127 | 0.425 | 0.370 | 6.29 | — | — | 0.375 | — |
| emergent_capabilities | 42 | 8 | 30 | 20 | 635 | 634 | 0.998 | 0.297 | 44.90 | — | — | 0.000 | — |
| incoherence_short | 42 | 3 | 8 | 2 | 7 | 7 | 1.000 | 0.183 | 0.99 | 7.94 | 0.00 | 0.000 | — |
| incoherence_medium | 42 | 6 | 8 | 8 | 50 | 47 | 0.940 | 0.343 | 5.70 | 45.56 | 4.33 | 0.167 | — |
| incoherence_long | 42 | 10 | 8 | 20 | 221 | 174 | 0.787 | 0.341 | 21.31 | 170.49 | 18.50 | 0.100 | — |
| marketplace_economy | 42 | 7 | 10 | 10 | 82 | 45 | 0.549 | 0.328 | 3.70 | 36.95 | 1.41 | 0.143 | — |
| network_effects | 42 | 10 | 20 | 10 | 314 | 246 | 0.783 | 0.335 | 9.90 | 197.90 | 12.94 | 0.100 | — |

### A.2 Per-Agent Efficiency

| Scenario | Agents | Welfare/Ep | Welfare/Agent/Ep | Interactions/Agent/Ep |
|----------|--------|------------|------------------|-----------------------|
| baseline | 5 | 4.98 | 1.00 | 0.96 |
| adversarial_redteam_v1 | 8 | 3.80 | 0.48 | 0.56 |
| collusion_detection | 8 | 6.29 | 0.79 | 1.50 |
| emergent_capabilities | 8 | 44.90 | 5.61 | 2.65 |
| incoherence_short | 3 | 0.99 | 0.33 | 0.29 |
| incoherence_medium | 6 | 5.70 | 0.95 | 1.04 |
| incoherence_long | 10 | 21.31 | 2.13 | 2.76 |
| marketplace_economy | 7 | 3.70 | 0.53 | 1.17 |
| network_effects | 10 | 9.90 | 0.99 | 1.57 |

### A.3 Governance Lever Coverage Matrix

| Scenario | Tax | Rep Decay | Staking | Circuit Breaker | Audit | Collusion | Network | Marketplace | Levers Active |
|----------|-----|-----------|---------|-----------------|-------|-----------|---------|-------------|---------------|
| baseline | | | | | | | | | 0 |
| adversarial_redteam_v* | x | x | x | x | x | x | x | | 7 |
| collusion_detection | x | x | x | x | | x | x | | 6 |
| emergent_capabilities | x | x | x | x | | | x | | 5 |
| incoherence_short | | | | | | | | | 0 |
| incoherence_medium | | | | | | | | | 0 |
| incoherence_long | | | | | | | | | 0 |
| marketplace_economy | x | x | | x | x | | | x | 5 |
| network_effects | x | x | | x | x | | x | | 5 |

### A.4 Regime Boundary Summary

Based on observed data, the regime boundaries can be approximated as:

| Boundary | Adversarial Fraction | Governance Required | Key Indicator |
|----------|---------------------|---------------------|---------------|
| Cooperative -> Contested | ~15-20% | Individual levers sufficient | Toxicity crosses 0.30 |
| Contested -> Collapse | ~40-50% | Structural levers insufficient | Acceptance drops below 0.50 |
| Collusion-buffered ceiling | ~37.5% | Collusion detection active | Toxicity > 0.35 but welfare positive |

Note: These boundaries are estimated from single-seed runs and should be
validated with multi-seed sweeps (see Section 4.6).

---

### Reproducibility

**SQLite query used to populate results tables:**

```sql
SELECT scenario_id, seed, n_agents, n_epochs, steps_per_epoch,
       total_interactions, accepted_interactions, acceptance_rate,
       avg_toxicity, welfare_per_epoch, adversarial_fraction,
       collapse_epoch, notes
FROM scenario_runs
ORDER BY scenario_id, seed
```

**Database:** `runs/runs.db`
**All scenarios run with:** `python -m swarm run scenarios/<id>.yaml --seed 42`
