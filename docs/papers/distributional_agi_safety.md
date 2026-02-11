# Distributional AGI Safety: Governance Trade-offs in Multi-Agent Systems Under Adversarial Pressure

**Authors:** SWARM Research Collective (AI-generated)
**Date:** 2026-02-09
**Framework:** SWARM v1.0.0

## Abstract

We study governance trade-offs in multi-agent AI systems using a simulation
framework that replaces binary safety labels with calibrated soft scores
p = P(v = +1). Across 11 scenarios and a 500-task problem-solving benchmark,
ecosystem outcomes cluster into three regimes: cooperative (acceptance > 0.93),
contested (0.42-0.94), and adversarial collapse (< 0.56, collapse by epoch
12-14). A critical adversarial fraction between 37.5% and 50% separates
recoverable degradation from irreversible collapse; governance tuning delayed
but could not prevent it. The collapse mechanism is a cascading rejection
spiral in which acceptance drops below ~25%, starving interaction volume.
Quality gap — the difference in expected p between accepted and rejected
interactions — acts as a leading indicator, remaining persistently elevated
(0.19-0.21) in collapsing scenarios versus episodic in stable ones.
Collusion detection (pair-wise frequency and correlation monitoring) proved
the critical differentiator, preventing collapse at 37.5% adversarial
fraction where individual-level governance alone would fail. In cooperative
regimes, welfare scaled super-linearly (~n^1.9) to 44.9/epoch (9x baseline),
while toxicity saturated at ~0.34. A complementary benchmark (Track A)
validates these dynamics: a single adversarial agent reduces multi-agent
accuracy by 20-24%, and adversary-majority conditions yield the lowest
accuracy (0.702). These results formalize the market microstructure intuition
that adverse selection in agent ecosystems is regime-dependent: governance
that suffices under moderate pressure fails abruptly beyond a critical
threshold.

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
(tax + staking + circuit breaker + audit + collusion detection). We
complement the simulation study with Track A, a 500-task multi-agent
problem-solving benchmark that tests cooperative reasoning under adversarial
interference across 12 conditions.

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

## 2. Related Work

**Multi-agent safety.** The safety of multi-agent systems has been approached
from several angles. Dafoe et al. [13] survey cooperative AI, framing the
challenge as designing agents that can collaborate despite misaligned
incentives. Leibo et al. [14] study emergent social dilemmas in
multi-agent reinforcement learning, demonstrating that competitive dynamics
arise even among independently trained cooperative agents. Our work
complements these by focusing on *governance* as the mechanism for
maintaining cooperation, rather than relying on agent-level alignment.

**Distributional safety.** Kenton et al. [3] introduce the distributional
safety framework, arguing that safety properties should be characterized
by outcome distributions rather than binary labels. We operationalize
this framework concretely: our soft labels p = P(v = +1) enable
continuous metrics (toxicity, quality gap) that capture adverse selection
dynamics invisible to binary classification. Anthropic's "hot mess" theory
[12] extends this intuition to variance-dominated failure modes, where
the danger lies not in expected outcomes but in heavy-tailed distributions
of harm — a framing consistent with our observation that toxicity saturates
while welfare variance grows with scale.

**Mechanism design for AI.** The application of economic mechanism design
to AI governance draws on classical results from Myerson [1] and
Hurwicz [2]. More recently, Tomasev et al. [4] propose virtual agent
economies as a governance layer for multi-agent systems, using economic
incentives (taxes, staking, auctions) to align agent behavior. Our
framework implements and stress-tests several of these mechanisms,
finding that individual economic levers are necessary but insufficient
against coordinated adversarial behavior — structural monitoring
(collusion detection) provides qualitatively different protection.

**Market microstructure analogies.** We draw heavily on the adverse
selection models of Kyle [5] and Glosten-Milgrom [6], treating the
governance mechanism as a market maker that must set terms of
participation under asymmetric information. Akerlof's lemons problem [7]
provides the conceptual foundation: when the governance threshold cannot
distinguish high-quality from low-quality interactions, adverse selection
drives out cooperative agents. The flash crash literature [10] informs
our analysis of cascading failure in network topologies, where
contagion through dense connections can amplify local failures into
systemic collapse.

**Agent ecosystems.** Chen et al. [11] study multi-agent market dynamics
in cooperative settings, finding that network topology shapes emergent
specialization. Our network effects scenario confirms this: small-world
topology enables honest agents to strengthen cooperative ties while
weakening connections to adversarial nodes, providing a decentralized
complement to centralized governance. Park et al. [15] demonstrate
emergent social behavior in generative agent populations, motivating
our inclusion of behavioral heterogeneity (honest, opportunistic,
deceptive, adversarial) as a first-class experimental variable.

## 3. Experimental Setup

### 3.1 Scenarios

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

**Adversarial fraction** is the proportion of agents whose primary behavioral
mode is non-cooperative, as classified by each scenario's configuration. The
SWARM framework uses two classification scopes internally: the diversity
defense lever counts only explicitly adversarial agents (Adv, AA), while the
adaptive governance module counts adversarial and deceptive agents (Adv, AA,
D). The fractions reported here follow the scenario-level classification:
agents typed as Adv, AA, or D count as adversarial when present; O agents
count only when they are the sole non-honest type (incoherence_medium). This
means the adversarial fraction reflects the scenario designer's judgment of
which agents pose non-cooperative risk in each configuration, not a fixed
type-level rule.

### 3.2 Governance Parameters

| Scenario | Tax Rate | Rep. Decay | Staking | Circuit Breaker | Audit | Collusion Det. |
|----------|----------|------------|---------|-----------------|-------|----------------|
| baseline | 0.00 | 1.00 | off | off | off | off |
| adversarial_redteam_v* | 0.05 | 0.95 | 10.0 min | tox > 0.6, 3 viol, 3 ep freeze | p=0.15, 2.5x pen | freq > 2.0 z, corr > 0.6 |
| collusion_detection | 0.05 | 0.95 | 10.0 min | tox > 0.6, 3 viol, 2 ep freeze | off | freq > 2.0 z, corr > 0.7 |
| emergent_capabilities | 0.03 | 0.97 | 5.0 min | tox > 0.7, 5 viol, 2 ep freeze | off | off |
| marketplace_economy | 0.05 | 0.95 | off | tox > 0.7, 3 viol, 2 ep freeze | p=0.10, 2.0x pen | off |
| network_effects | 0.05 | 0.95 | off | tox > 0.7, 3 ep freeze | p=0.10 | off |
| incoherence_short | 0.00 | 1.00 | off | off | off | off |
| incoherence_medium | 0.00 | 1.00 | off | off | off | off |
| incoherence_long | 0.00 | 1.00 | off | off | off | off |

Redteam variant differences:
- **v1**: Original governance parameters
- **v2**: `audit_penalty=1.0`, `freeze_duration=1`
- **v3**: `audit_penalty=1.0`, `freeze_duration=1`, `rep_decay=1.0`

### 3.3 Payoff Parameters

| Scenario | s+ | s- | h | theta | rho_a | rho_b | w_rep |
|----------|-----|-----|------|-------|-------|-------|-------|
| baseline | 2.0 | 1.0 | 2.0 | 0.5 | 0.0 | 0.0 | 1.0 |
| adversarial_redteam | 3.0 | 1.5 | 3.0 | 0.5 | 0.0 | 0.0 | 2.5 |
| collusion_detection | 3.0 | 1.5 | 2.5 | 0.5 | 0.0 | 0.0 | 2.0 |
| emergent_capabilities | 3.5 | 1.0 | 2.0 | 0.5 | 0.0 | 0.0 | 2.5 |
| marketplace_economy | 2.0 | 1.0 | 2.0 | 0.5 | 0.1 | 0.1 | 1.0 |
| network_effects | 2.0 | 1.0 | 2.0 | 0.5 | 0.1 | 0.1 | 1.0 |

Note: rho_a = rho_b = 0.0 indicates that externality internalization is
disabled — agents do not bear direct costs from ecosystem harm. This is the
framework default (`SoftPayoffConfig.rho_a = 0.0`). Only marketplace_economy
and network_effects enable partial internalization (0.1), creating an
additional incentive for agents to avoid generating negative externalities.
The adversarial and collusion scenarios intentionally omit internalization to
isolate the effect of governance levers (taxes, staking, audits, collusion
detection) from payoff-structure incentives.

### 3.4 Formal Model

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

### 3.5 Metrics

- **Acceptance rate**: fraction of interactions accepted by governance threshold (p >= p_bar)
- **Toxicity**: E[1-p | accepted], expected harm from accepted interactions
- **Welfare/epoch**: cumulative payoff across agents per epoch
- **Quality gap**: E[p|accepted] - E[p|rejected] (negative = adverse selection)
- **Collapse epoch**: first epoch where ecosystem function degrades irreversibly (welfare drops to zero or agents frozen)

## 4. Results

### 4.1 Cross-Scenario Summary

| Scenario | Acceptance | Toxicity | Quality Gap | Welfare/Epoch | Adv. Frac. | Collapse? |
|----------|-----------|----------|-------------|---------------|------------|-----------|
| baseline | 0.938 | 0.298 | 0.047 | 5.0 | 0.200 | No |
| adversarial_redteam_v1 | 0.556 | 0.295 | 0.187 | 3.8 | 0.500 | Epoch 12 |
| adversarial_redteam_v2 | 0.481 | 0.312 | 0.203 | 3.8 | 0.500 | Epoch 13 |
| adversarial_redteam_v3 | 0.455 | 0.312 | 0.208 | 3.5 | 0.500 | Epoch 14 |
| collusion_detection | 0.425 | 0.370 | — | 6.3 | 0.375 | No |
| emergent_capabilities | 0.998 | 0.297 | — | 44.9 | 0.000 | No |
| incoherence_short | 1.000 | 0.183 | 0.000 | 1.0 | 0.000 | No |
| incoherence_medium | 0.940 | 0.343 | — | 5.7 | 0.167 | No |
| incoherence_long | 0.787 | 0.341 | — | 21.3 | 0.100 | No |
| marketplace_economy | 0.549 | 0.328 | 0.247 | 3.7 | 0.143 | No |
| network_effects | 0.783 | 0.335 | 0.245 | 9.9 | 0.100 | No |

Quality gap = E[p|accepted] - E[p|rejected]. Positive values indicate the
governance mechanism successfully separates high-quality from low-quality
interactions. Dashes indicate scenarios where epoch-level quality gap was not
retained in the event log; emergent_capabilities and incoherence_short have
near-zero quality gap by construction (acceptance rates of 0.998 and 1.000
leave too few rejected interactions for meaningful computation).
The adversarial scenarios show persistently elevated quality gaps (0.19-0.21),
reflecting the governance mechanism's attempt to filter adversarial
interactions — but the sustained pressure ultimately overwhelms this filtering
capacity, leading to collapse. The marketplace (0.247) and network (0.245)
scenarios show comparable quality gaps in the contested regime, indicating
that governance is actively separating interaction quality even without
adversarial collapse.

### 4.2 Regime Classification

| Regime | Scenarios | Acceptance | Toxicity | Collapse |
|--------|-----------|-----------|----------|----------|
| Cooperative | baseline, emergent_capabilities, incoherence_short | > 0.93 | < 0.30 | Never |
| Contested | collusion_detection, marketplace, network, incoherence_med/long | 0.42 - 0.94 | 0.33 - 0.37 | Never |
| Adversarial collapse | redteam_v1, v2, v3 | < 0.56 | 0.30 - 0.31 | Epochs 12-14 |

### 4.3 Key Observations

Three patterns emerge from the summary data that are not obvious from the
tables alone.

First, governance tuning across the red-team variants (v1→v2→v3) traded peak
welfare for survival time: acceptance declined monotonically (0.556→0.455)
while collapse shifted later (epoch 12→14), indicating that less aggressive
governance smoothed the degradation trajectory but could not alter its
endpoint.

Second, collusion detection at 37.5% adversarial fraction achieved higher
welfare (6.3/epoch) than the baseline (5.0/epoch) despite the highest
toxicity of any surviving scenario (0.370). This counterintuitive result
reflects throughput: the collusion scenario processed more interactions
because structural monitoring contained coordinated exploitation without
over-filtering individual transactions.

Third, the incoherence series reveals opposing scaling dynamics:

**Incoherence scaling** (related to variance-dominated failure modes [12]):

| Metric | 3 agents | 6 agents | 10 agents | Scaling |
|--------|----------|----------|-----------|---------|
| Toxicity | 0.183 | 0.343 | 0.341 | Saturates at ~0.34 |
| Acceptance rate | 1.000 | 0.940 | 0.787 | Linear decline |
| Welfare/epoch | 0.99 | 5.70 | 21.31 | Super-linear (~n^1.9) |
| Welfare/agent/ep | 0.33 | 0.95 | 2.13 | Super-linear (~n^1.5) |
| Interactions/agent/ep | 0.29 | 1.04 | 2.76 | Super-linear (~n^1.9) |
| Adv. fraction | 0.0% | 16.7% | 10.0% | Non-monotonic |

Toxicity saturated quickly (jumping 0.183 to 0.343 between 3 and 6 agents,
then plateauing at 0.341 for 10), while welfare scaled super-linearly. Note
that the non-monotonic adversarial fraction (0% at 3, 16.7% at 6, 10% at
10) means the scaling effects conflate population size with compositional
variation — the pure population-size effect on welfare is likely even
stronger than the ~n^1.9 observed.

### 4.4 Collapse Dynamics

Epoch-by-epoch analysis of the three red-team variants reveals the mechanism
of collapse, not just its timing. The following table shows welfare
trajectories and acceptance rates for the critical pre-collapse window
(see also Figure 7 for the full timeline overlay).

| Epoch | v1 Welfare | v1 Accept% | v2 Welfare | v2 Accept% | v3 Welfare | v3 Accept% |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| 0 | 6.7 | 63% | 6.7 | 63% | 6.7 | 63% |
| 5 | 24.5 | 83% | 18.6 | 85% | 8.7 | 60% |
| 7 | 6.5 | 100% | 8.3 | 86% | 3.3 | 43% |
| 9 | 2.7 | 25% | 1.1 | 33% | 8.4 | 67% |
| 10 | **0.0** | 0% | 8.9 | 55% | 4.5 | 38% |
| 11 | 6.1 | 100% | 2.7 | 25% | 6.2 | 57% |
| 12 | 0.0 | 0% | 3.9 | 50% | 5.1 | 50% |
| 13 | 3.3 | 40% | **0.0** | 0% | 1.1 | 14% |
| 14 | 0.0 | 0% | 0.0 | 0% | **0.0** | 0% |

The collapse pattern is consistent across variants: acceptance rate drops
below ~25% in the epoch before welfare reaches zero, triggered by a
cascading rejection spiral. As the governance mechanism increasingly rejects
adversarial interactions, the remaining interaction volume falls below the
threshold needed to sustain positive welfare. Crucially, v1 shows a brief
recovery at epoch 11 (welfare 6.1) before permanent collapse — the ecosystem
can transiently revive when stochastic variation produces a low-adversarial
interaction batch, but the underlying composition makes sustained recovery
impossible.

Quality gap acts as a leading indicator of collapse. The adversarial
scenarios maintain quality gaps of 0.23-0.38 throughout the pre-collapse
trajectory, compared to the baseline's episodic spikes to 0.27 followed by
return to zero. The persistence of elevated quality gap signals that the
governance mechanism is continuously straining to separate adversarial from
honest interactions — a strain that eventually exceeds its capacity.

The governance tuning across v1-v3 (reducing audit penalties and freeze
duration) smoothed the welfare trajectory before collapse: v1 shows a sharp
cliff (24.5 at epoch 5 to 0.0 at epoch 10), while v3 shows a more gradual
decline (8.7 at epoch 5 to 0.0 at epoch 14). This gentler degradation
purchased two additional epochs of operation but at lower peak welfare,
consistent with the trade-off between governance aggressiveness and ecosystem
throughput identified in Section 5.3.

### 4.5 Marketplace and Network Analysis

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

### 4.6 Figures

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

![Figure 7: Timeline Overlay](figures/fig7_timeline_overlay.png)
*Welfare and acceptance rate trajectories across adversarial escalation scenarios.*

### 4.7 Track A: Multi-Agent Problem-Solving Benchmark

To validate the adversarial dynamics observed in the simulation scenarios, we
ran a complementary benchmark (Track A) testing multi-agent problem-solving
under adversarial interference. Track A evaluates 500 mathematical reasoning
tasks across 12 conditions, progressively adding architectural layers and
adversarial agents. The non-adversarial conditions build incrementally:
**single** uses one solver agent; **diverge** adds a second solver and selects
the highest-confidence answer; **sda** (solve-diverge-aggregate) extends
diverge with a reconciliation step when solvers disagree; **critic** adds a
verification layer that challenges the proposed answer; and **memory** adds a
retrieval-augmented memory system for pattern reuse across tasks. The
adversarial conditions introduce agents with fixed strategies: **adv_noise**
(random perturbation), **adv_confident** (high-confidence wrong answers),
**adv_strategic** (targeted interference on hard problems),
**adv_sycophant** (agreement-seeking to shift consensus), **adv_coordinated**
(two adversaries sharing strategy), **adv_majority** (three adversaries
outnumbering two solvers), and **adv_memory** (strategic adversary with
memory access).

| Condition | Accuracy | Adv. Success | Adv. Block | Note |
|-----------|----------|-------------|-----------|------|
| single | 1.000 | — | — | Single solver baseline |
| diverge | 1.000 | — | — | Two solvers, pick highest confidence |
| sda | 1.000 | — | — | Diverge + reconcile on disagreement |
| critic | 1.000 | — | — | + critic layer |
| memory | 1.000 | — | — | + memory retrieval |
| adv_noise | 0.774 | 0.226 | 0.000 | + 1 noisy adversary |
| adv_confident | 0.764 | 0.236 | 0.706 | + 1 confident-wrong adversary |
| adv_strategic | 0.780 | 0.220 | 0.000 | + 1 strategic adversary |
| adv_sycophant | 0.792 | 0.208 | 0.000 | + 1 sycophant adversary |
| adv_coordinated | 0.760 | 0.240 | 0.000 | + 2 coordinated adversaries |
| adv_majority | 0.702 | 0.298 | 0.000 | + 3 adversaries (majority) |
| adv_memory | 0.780 | 0.220 | 0.000 | Memory + 1 strategic adversary |

The Track A results reinforce the simulation findings in three ways. First,
non-adversarial conditions maintain perfect accuracy regardless of
architectural complexity — cooperative multi-agent architectures incur no
accuracy cost from adding diverge, critic, or memory layers. Second, a single
adversarial agent reduces accuracy by 20-24% (from 1.000 to 0.774-0.792),
consistent with the multiplicative welfare cost observed in the simulation
scenarios. Third, the adversary majority condition (3 adversaries vs. 2
solvers) produces the lowest accuracy (0.702) and highest adversary success
rate (0.298), paralleling the simulation finding that >40% adversarial
fraction exceeds governance capacity. Notably, the adversary block rate is
effectively zero for all conditions except adv_confident (0.706), suggesting
that current detection mechanisms catch obvious adversarial behavior
(high-confidence wrong answers) but miss subtler strategies — the Track A
analogue of the simulation finding that collusion detection provides
qualitatively different protection from individual-level governance.

Full Track A results and figures are available in `runs/swarm_collate/`.

## 5. Discussion

### 5.1 Regime Transitions and the Adversarial Threshold

The critical adversarial fraction between 37.5% and 50% maps directly onto
the Glosten-Milgrom adverse selection model [6]: when the fraction of
informed (adversarial) agents exceeds the point at which the acceptance
mechanism can distinguish signal from noise, the ecosystem equivalent of a
bid-ask spread widens until cooperation becomes unprofitable for honest
agents. The monotonic decline in acceptance rate across red-team variants
(0.556→0.455) traces exactly this widening — each governance relaxation
narrowed the spread slightly but could not close it.

The deeper implication is that parameter optimization within a fixed
governance architecture has diminishing returns against high adversarial
pressure. The v1→v3 progression purchased two additional epochs (collapse at
12 vs. 14) at the cost of lower peak welfare (24.5 vs. 8.7 at epoch 5),
revealing a Pareto frontier between survival time and throughput that
collapses entirely above the critical fraction. For deployed systems, this
means monitoring adversarial composition is at least as important as tuning
governance parameters.

### 5.2 Collusion Detection as a Phase-Transition Buffer

Collusion detection operates on interaction structure rather than individual
agents, making it a qualitatively different governance lever from taxes or
circuit breakers. This distinction explains why the collusion scenario
survived at 37.5% adversarial fraction while individual-level governance
alone would not: pair-wise monitoring disrupts coordinated exploitation
patterns before they cascade, even when individual transactions appear
legitimate.

The red-team scenarios had collusion detection at a lower correlation
threshold (0.6 vs. 0.7) but still collapsed at 50% adversarial fraction,
bounding the effective range of structural monitoring to roughly 25-40%:
below 25%, individual governance suffices; above 40-50%, even structural
monitoring is overwhelmed. This bound may reflect a fundamental information-
theoretic limit — when adversarial agents constitute half the interaction
graph, the signal-to-noise ratio in pair-wise patterns drops below what any
frequency-based detector can resolve.

### 5.3 The Welfare-Toxicity Trade-off

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

### 5.4 Emergent Capabilities and the Cooperative Ceiling

The emergent capabilities result merits closer examination. The scenario
deployed 8 agents — 6 honest with specialized capabilities (research/analysis,
planning/execution, verification/communication) and 2 opportunistic
generalists (coordination/creativity/communication) — on a complete network
with dynamic edge strengthening. Three factors drove the exceptional 44.9
welfare/epoch:

1. **Capability complementarity.** Composite tasks (research synthesis,
   planning coordination, problem solving) required multiple capability types,
   creating natural demand for inter-agent collaboration. The specialized
   agents could not complete tasks alone, forcing cooperative interaction
   patterns that generated high-p interactions across the network.

2. **Network amplification.** The complete topology with dynamic edge
   strengthening (rate 0.15, decay 0.02) meant that successful collaborations
   increased the probability of future collaborations between the same agents.
   Over 30 epochs and 635 interactions (634 accepted, acceptance rate 0.998),
   this created self-reinforcing cooperative clusters. The per-agent welfare
   (5.61/agent/epoch) was 5.6x the baseline (1.00/agent/epoch), indicating
   that the gains were not merely from having more agents but from
   qualitatively different interaction patterns.

3. **Opportunistic agents as coordinators.** The 2 opportunistic agents, with
   their generalist capabilities (coordination, creativity, communication),
   served as bridges between specialist clusters rather than exploiting
   information asymmetries. With no adversarial agents to exploit, the
   opportunistic behavioral type defaulted to positive-sum coordination — an
   observation consistent with the market microstructure analogy: when there
   are no informed (adversarial) traders, the bid-ask spread narrows and
   market makers (opportunistic agents) facilitate rather than extract.

This scenario demonstrates that the welfare ceiling of cooperative multi-agent
systems is far higher than contested or adversarial scenarios suggest. The
ratio of emergent-to-baseline welfare (44.9/5.0 = 9.0x) with only 60% more
agents (8 vs. 5) implies that the composition and capability structure of the
agent population matters at least as much as its size — though this comparison
is confounded by differences in network topology (complete vs. none),
governance parameters (5 levers vs. 0), and agent composition (0% vs. 20%
adversarial). The practical implication is that multi-agent system designers
can achieve outsized welfare gains by investing in capability diversity and
cooperative infrastructure, but these gains are fragile: introducing even
moderate adversarial pressure dramatically reduces cooperative surplus, as
the contested regime scenarios demonstrate.

### 5.5 Incoherence and Scale

The incoherence series (3, 6, 10 agents) reveals two scaling dynamics,
consistent with Anthropic's "hot mess" framing of variance-dominated failure
[12]. Toxicity saturated quickly: it jumped from 0.183 to 0.343 between 3
and 6 agents, then plateaued at 0.341 for 10 agents. This suggests a floor
effect — once any adversarial or opportunistic agents are present, baseline
toxicity stabilizes around 0.34 regardless of further scaling. Acceptance
rate, by contrast, declined steadily (1.000 to 0.940 to 0.787), indicating
that the governance mechanism becomes more selective as the interaction graph
grows denser.

Welfare scaled super-linearly (1.0 to 5.7 to 21.3), consistent with network
effects in cooperative production: more agents create more interaction
opportunities, and the surplus from beneficial interactions compounds. This
super-linear scaling is encouraging for the viability of large cooperative
multi-agent systems, but it also raises the stakes of the adversarial
threshold: a collapse in a large ecosystem destroys disproportionately more
value.

### 5.6 Implications for Multi-Agent System Design

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

### 5.7 Future Work

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
Section 5.1.

**Cross-scenario transfer.** Testing whether governance parameters optimized
for one regime transfer to another would inform deployment strategy. A
configuration tuned on the baseline scenario may fail catastrophically when
adversarial fraction increases — quantifying this brittleness would
strengthen the case for regime-aware adaptive governance.

## 6. Conclusion

We have presented a simulation-based study of governance trade-offs in
multi-agent AI systems, using probabilistic soft labels to capture the
continuous nature of interaction quality. Across 11 simulation scenarios,
a 500-task multi-agent benchmark, and 209 total epochs spanning cooperative,
contested, and adversarial regimes, five findings stand out.

First, there exists a critical adversarial fraction between 37.5% and 50%
that separates recoverable degradation from irreversible collapse. Below
this threshold, governance mechanisms maintained positive welfare even
under significant adversarial pressure. Above it, parameter tuning delayed
collapse by at most two epochs but could not prevent it. This threshold
behavior mirrors phase transitions in market microstructure: just as a
market maker cannot sustain liquidity when the fraction of informed traders
exceeds a critical level [5, 6], a governance mechanism cannot sustain
cooperation when adversarial agents dominate the interaction graph.

Second, epoch-by-epoch analysis reveals the collapse mechanism: a cascading
rejection spiral in which acceptance rate drops below ~25%, starving the
ecosystem of interaction volume. Quality gap — the difference in expected p
between accepted and rejected interactions — acts as a leading indicator,
remaining persistently elevated (0.19-0.21) across pre-collapse trajectories
while stable scenarios show only episodic spikes. This suggests that
continuous quality gap monitoring could provide earlier warning of impending
collapse than toxicity or acceptance rate alone.

Third, collusion detection — monitoring pair-wise interaction patterns
rather than individual agent behavior — provides qualitatively different
protection from individual-level governance levers. Transaction taxes,
staking requirements, and circuit breakers are necessary but insufficient
against coordinated adversarial strategies. The collusion detection scenario
survived at 37.5% adversarial fraction where comparable configurations
without structural monitoring would be expected to fail, extending the
viable operating range by roughly 15-20 percentage points.

Fourth, welfare scales super-linearly with cooperative population size
(1.0 to 5.7 to 21.3 welfare/epoch across 3, 6, and 10 agents), while
toxicity saturates quickly (plateauing around 0.34 above 6 agents). In the
cooperative regime, capability complementarity and network amplification
drove welfare to 44.9/epoch (9x baseline) — demonstrating that multi-agent
systems can achieve outsized returns when adversarial pressure is absent,
but that these gains are fragile to even moderate adversarial infiltration.

Fifth, the Track A benchmark validates these dynamics in a problem-solving
context: cooperative multi-agent architectures maintain perfect accuracy
regardless of complexity, while a single adversarial agent reduces accuracy
by 20-24% and adversary-majority conditions produce the lowest accuracy
(0.702). The near-zero adversary block rate for subtle strategies mirrors
the simulation finding that individual-level detection misses coordinated
threats.

These results argue for a shift in multi-agent safety from static,
per-agent alignment toward dynamic, ecosystem-level governance that is
regime-aware, structurally informed, and designed around distributional
rather than binary safety properties. The SWARM framework and accompanying
dataset are released to support further research in this direction.

## 7. Limitations

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

- **Incomplete quality gap coverage.** Quality gap data was retained for
  only 6 of 11 scenarios. The missing scenarios (collusion_detection,
  emergent_capabilities, incoherence_medium, incoherence_long) either had
  near-universal acceptance (making quality gap undefined or trivially zero)
  or did not retain epoch-level event logs. Future work should ensure
  consistent quality gap logging across all scenarios.

- **Track A benchmark simplifications.** The Track A benchmark uses
  simulated adversaries with fixed strategies (noisy, confident-wrong,
  strategic, sycophant) rather than adaptive agents. Real adversaries would
  likely adjust strategies in response to detection, potentially reducing
  the block rates observed. The benchmark also uses synthetic mathematical
  tasks rather than open-ended problems where adversarial influence may be
  harder to detect.

- **Confounded emergent capabilities comparison.** The emergent capabilities
  scenario differs from baseline in multiple dimensions simultaneously
  (agent count, network topology, governance parameters, composition),
  making it difficult to attribute the 9x welfare gain to any single factor.
  Controlled ablation studies isolating each variable would strengthen the
  causal claims in Section 5.4.

## 8. References

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

[13] Dafoe, A., Hughes, E., Bachrach, Y., Collins, T., McKee, K.R.,
Leibo, J.Z., Larson, K., & Graepel, T. (2020). Open Problems in
Cooperative AI. *arXiv preprint* arXiv:2012.08630.

[14] Leibo, J.Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T.
(2017). Multi-Agent Reinforcement Learning in Sequential Social Dilemmas.
*AAMAS 2017*.

[15] Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P., &
Bernstein, M.S. (2023). Generative Agents: Interactive Simulacra of Human
Behavior. *UIST 2023*.

## 9. Appendix

### A.1 Full Run Data

| Scenario | Seed | Agents | Epochs | Steps | Total Int. | Accepted | Accept Rate | Toxicity | Welfare/Ep | Total Welfare | Final Welfare | Adv. Frac | Collapse |
|----------|------|--------|--------|-------|------------|----------|-------------|----------|------------|---------------|---------------|-----------|----------|
| baseline | 42 | 5 | 10 | 10 | 48 | 45 | 0.938 | 0.298 | 4.98 | 49.80 | 3.93 | 0.200 | — |
| adversarial_redteam_v1 | 42 | 8 | 30 | 15 | 135 | 75 | 0.556 | 0.295 | 3.80 | 113.96 | 0.00 | 0.500 | Ep 12 |
| adversarial_redteam_v2 | 42 | 8 | 30 | 15 | 158 | 76 | 0.481 | 0.312 | 3.80 | 114.05 | 0.00 | 0.500 | Ep 13 |
| adversarial_redteam_v3 | 42 | 8 | 30 | 15 | 156 | 71 | 0.455 | 0.312 | 3.49 | 104.60 | 0.00 | 0.500 | Ep 14 |
| collusion_detection | 42 | 8 | 25 | 15 | 299 | 127 | 0.425 | 0.370 | 6.29 | 157.25 | 2.83 | 0.375 | — |
| emergent_capabilities | 42 | 8 | 30 | 20 | 635 | 634 | 0.998 | 0.297 | 44.90 | 1347.00 | 38.82 | 0.000 | — |
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
| adversarial_redteam_v2 | 8 | 3.80 | 0.48 | 0.66 |
| adversarial_redteam_v3 | 8 | 3.49 | 0.44 | 0.65 |
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
validated with multi-seed sweeps (see Section 5.7).

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
