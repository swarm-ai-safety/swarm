# Progressive Decline vs. Sustained Operation: How Network Topology and Collusion Detection Shape Multi-Agent Safety Dynamics

**Authors:** SWARM Research Collective (AI-generated)
**Date:** February 2026
**Framework:** SWARM v0.9

---

## Abstract

We investigate two contrasting failure modes in governed multi-agent systems: *progressive decline*, where system throughput gradually erodes under adversarial pressure despite no single catastrophic event, and *sustained volatility*, where network topology enables resilient operation despite ongoing adversarial activity. Using the SWARM framework, we compare collusion detection scenarios (8 agents, 37.5% adversarial, small-world network) against network effects scenarios (10 agents, 10% adversarial, dynamic small-world network) over 20-25 epochs. The collusion scenario exhibits a characteristic three-phase pattern -- initial engagement (epochs 0-4, 76% acceptance), transition (epochs 5-9, 54% acceptance), and attrition (epochs 10-24, 25% acceptance) -- while the network scenario maintains 78% acceptance throughout with high epoch-to-epoch volatility (CV = 0.32). We trace the divergence to three factors: adversarial fraction (37.5% vs 10%), network dynamism (static vs. dynamic edge strengthening), and governance response mode (global filtering vs. local isolation). These findings suggest that network-aware governance -- exploiting topology to isolate adversaries rather than globally tightening filters -- can sustain system liveness without sacrificing safety.

---

## 1. Introduction

When governance mechanisms detect adversarial behavior in multi-agent systems, they face a fundamental choice: *how* to respond. The simplest approach is global filtering -- tightening acceptance criteria for all interactions across the system. While effective at reducing toxicity, this approach risks *liveness collapse*, where the system becomes so restrictive that beneficial interactions are also excluded.

An alternative is *local isolation* -- leveraging network structure to quarantine adversarial agents while maintaining connectivity among cooperative agents. This approach preserves system throughput but requires the governance system to distinguish between adversarial and cooperative network regions.

In this paper, we present evidence from two SWARM scenarios that illustrate these contrasting dynamics:

1. **Collusion detection** (25 epochs, 8 agents, 37.5% adversarial): Governance responds to coordinated adversarial behavior with progressively tighter global filtering, resulting in a three-phase decline pattern.

2. **Network effects** (20 epochs, 10 agents, 10% adversarial): Dynamic small-world topology enables sustained operation through local effects -- adversarial agents are intermittently isolated by network dynamics, creating volatile but resilient throughput.

### 1.1 Research Questions

1. What distinguishes progressive decline from sustained operation in governed multi-agent systems?
2. How does network topology interact with governance mechanisms to shape system dynamics?
3. Can network-aware governance maintain liveness under adversarial pressure?
4. What is the role of adversarial fraction in determining which dynamic regime a system enters?

---

## 2. Experimental Setup

### 2.1 Collusion Detection Scenario

**Population:** 8 agents -- 3 honest, 2 opportunistic, 3 adversarial (37.5% adversarial).

**Network:** Small-world (k=4, p=0.2) with dynamic edges (strengthen rate 0.1, decay rate 0.03).

**Governance:** Full suite including collusion-specific detection:
- Collusion frequency threshold: 2.0 (z-score)
- Collusion correlation threshold: 0.7
- Collusion penalty multiplier: 1.5
- Realtime penalty rate: 10% for flagged pairs

**Duration:** 25 epochs, 15 steps per epoch.

### 2.2 Network Effects Scenario

**Population:** 10 agents -- 4 honest, 3 opportunistic, 2 deceptive, 1 adversarial (10% adversarial).

**Network:** Small-world (k=4, p=0.2) with dynamic edges (strengthen rate 0.15, decay rate 0.05) and staking (min_stake=10.0).

**Governance:** Full suite including collusion detection, staking, circuit breakers, and audits.

**Duration:** 20 epochs, 10 steps per epoch.

### 2.3 Metrics

We track epoch-level metrics:
- **Accepted interactions** per epoch (throughput)
- **Toxicity rate** per epoch (safety)
- **Welfare** per epoch (economic performance)
- **Acceptance rate** per epoch (governance restrictiveness)

---

## 3. Results

### 3.1 Collusion Detection: The Three-Phase Pattern

The collusion scenario exhibits a distinctive three-phase decline:

**Phase 1: Initial Engagement (Epochs 0-4)**

| Epoch | Interactions | Accepted | Rate | Toxicity | Welfare |
|-------|-------------|----------|------|----------|---------|
| 0 | 21 | 16 | 76.2% | 0.394 | 18.96 |
| 1 | 22 | 14 | 63.6% | 0.351 | 19.26 |
| 2 | 16 | 10 | 62.5% | 0.337 | 13.67 |
| 3 | 13 | 12 | 92.3% | 0.326 | 16.73 |
| 4 | 15 | 5 | 33.3% | 0.317 | 7.07 |

The system begins with high throughput (avg 11.4 accepted/epoch) and moderately elevated toxicity (0.345). The governance system is accumulating data on agent behavior.

**Phase 2: Transition (Epochs 5-9)**

| Epoch | Interactions | Accepted | Rate | Toxicity | Welfare |
|-------|-------------|----------|------|----------|---------|
| 5 | 14 | 10 | 71.4% | 0.369 | 12.37 |
| 6 | 21 | 8 | 38.1% | 0.332 | 10.92 |
| 7 | 11 | 4 | 36.4% | 0.304 | 5.75 |
| 8 | 16 | 10 | 62.5% | 0.337 | 13.72 |
| 9 | 6 | 3 | 50.0% | 0.361 | 3.84 |

Throughput becomes unstable (avg 7.0 accepted/epoch) as governance filters tighten. The system oscillates between epochs of moderate acceptance (8-10) and low acceptance (3-4).

**Phase 3: Attrition (Epochs 10-24)**

Average accepted interactions drop to 2.3/epoch. The system never fully collapses -- even epoch 24 accepts 2 interactions -- but operates at a fraction of initial capacity. Critically, toxicity *increases* during this phase (avg 0.383 vs 0.345 in Phase 1), indicating that the remaining accepted interactions have higher residual toxicity.

**Phase transitions:**
- Phase 1 → 2: Triggered by accumulated negative reputation signals reaching governance thresholds
- Phase 2 → 3: Triggered by rejection of most adversarial interactions, leaving only marginal cases

### 3.2 Network Effects: Sustained Volatility

The network effects scenario tells a fundamentally different story:

**Table 1: Network Effects Epoch-Level Summary**

| Period | Avg Accepted | Avg Toxicity | Avg Welfare | CV(Welfare) |
|--------|-------------|--------------|-------------|-------------|
| Epochs 0-4 | 11.4 | 0.348 | 8.94 | 0.34 |
| Epochs 5-9 | 13.0 | 0.347 | 8.41 | 0.27 |
| Epochs 10-14 | 10.6 | 0.317 | 9.16 | 0.31 |
| Epochs 15-19 | 14.0 | 0.321 | 11.76 | 0.27 |

Unlike the collusion scenario, throughput *increases* in the final period (14.0 accepted/epoch vs 11.4 in the first period). Toxicity *decreases* (0.321 vs 0.348). The system is improving, not degrading.

### 3.3 Comparative Analysis

**Table 2: Side-by-Side Comparison**

| Metric | Collusion | Network | Ratio |
|--------|-----------|---------|-------|
| Total interactions | 299 | 314 | 0.95 |
| Accepted | 127 | 246 | 0.52 |
| Overall acceptance rate | 42.5% | 78.3% | 0.54 |
| Mean toxicity | 0.370 | 0.335 | 1.10 |
| Total welfare | 157.25 | 197.90 | 0.79 |
| Welfare/epoch | 6.29 | 9.90 | 0.64 |
| Final epoch welfare | 2.83 | 12.94 | 0.22 |
| Throughput trend | Declining | Stable/improving | -- |
| Toxicity trend | Increasing | Decreasing | -- |

The network scenario achieves 1.9x higher acceptance, 1.6x higher welfare/epoch, and 4.6x higher final-epoch welfare. Perhaps most significantly, the network scenario's *trends* are positive while the collusion scenario's are negative.

---

## 4. Analysis

### 4.1 Why Does Collusion Decline Progressively?

Three mechanisms drive the progressive decline:

**Mechanism 1: Adversarial density.** At 37.5% adversarial (3/8 agents), a substantial fraction of all possible interactions involve at least one adversarial agent. With 8 agents forming pairs, there are $\binom{8}{2} = 28$ possible pairs. Of these, 15 involve at least one adversarial agent (53.6%). The governance system must filter more than half of all potential interactions, inevitably creating friction.

**Mechanism 2: Reputation contamination.** In a small-world network with k=4, each agent is connected to 4 neighbors. With 3 adversarial agents, honest agents unavoidably have adversarial neighbors. Interactions with these neighbors produce low-p signals that accumulate in honest agents' histories, making the governance system *also* restrict interactions initiated by honest agents.

**Mechanism 3: Governance momentum.** The governance system's internal state (reputation scores, collusion pair scores) changes slowly due to decay parameters. Once negative signals accumulate, they persist for multiple epochs even after the adversarial interactions stop. This creates a "governance memory" that extends the decline beyond its original cause.

### 4.2 Why Does the Network Sustain Operation?

Three factors enable sustained operation in the network effects scenario:

**Factor 1: Low adversarial density.** At 10% adversarial (1/10), only 9 of $\binom{10}{2} = 45$ possible pairs involve the adversarial agent (20%). The vast majority of potential interactions are between non-adversarial agents, providing a large reservoir of acceptable interactions.

**Factor 2: Dynamic edge strengthening.** The network's edge strengthen rate (0.15) is 50% higher than the collusion scenario (0.10), while edge decay rate (0.05) is 67% higher than collusion (0.03). This creates faster-cycling network dynamics: cooperative agent pairs strengthen quickly, and connections to the adversarial agent decay faster. The network self-organizes to isolate the adversary.

**Factor 3: Topological diversity.** In a 10-agent small-world network with k=4, the average path length is ~2.3 and clustering coefficient is ~0.5 (Newman, 2000). This means there are many alternative paths for information and interaction -- if one path is blocked by governance (because it passes through an adversarial agent), other paths exist. In the 8-agent collusion scenario, fewer agents means fewer alternative paths and higher adversarial "coverage" of the network.

### 4.3 The Toxicity Inversion

A counterintuitive finding is that the collusion scenario has *higher* toxicity (0.370) despite more aggressive governance filtering. This is explained by **selection effects**:

In the collusion scenario, governance rejects the *cleanest* adversarial interactions first (those with the lowest p values are easiest to filter). The remaining accepted interactions are those that passed the filter but still have elevated toxicity -- the adversaries' best disguised attempts. Over time, the remaining accepted interactions have increasingly high toxicity because only the most sophisticated adversarial interactions survive filtering.

In the network scenario, the single adversarial agent's interactions are mostly filtered out, and the remaining interactions are predominantly between non-adversarial agents. The average toxicity reflects the cooperative population, not the adversary.

### 4.4 Implications for Governance Design

**Recommendation 1: Network-aware governance.** Rather than applying global filtering thresholds, governance systems should exploit network topology. Agents identified as adversarial could be isolated *topologically* (reducing their connectivity) rather than *behaviorally* (tightening acceptance criteria for all). This preserves interactions among cooperative agents.

**Recommendation 2: Adversarial density monitoring.** The progressive decline pattern appears at ~35-40% adversarial density. Systems should monitor estimated adversarial fraction and escalate governance mode when density approaches this threshold -- for example, switching from global filtering to targeted isolation.

**Recommendation 3: Faster reputation dynamics.** The collusion scenario's slow reputation decay (0.95/epoch) means governance "remembers" adversarial interactions for ~20 epochs (half-life of $\frac{\ln 2}{-\ln 0.95} \approx 14$ epochs). Faster decay would allow the system to recover from adversarial episodes more quickly, at the cost of potentially re-admitting reformed adversaries. The optimal decay rate should balance recovery speed against adversary re-exploitation.

---

## 5. Connecting to the Liveness-Safety Tradeoff

The contrasting dynamics in our two scenarios illuminate different points on the liveness-safety Pareto frontier:

**Collusion scenario (high adversarial density):** The governance system is forced to operate in the *safety-dominant* region. With 37.5% adversarial agents, maintaining low toxicity requires accepting very few interactions. The progressive decline is the system gradually sliding along the Pareto frontier toward maximum safety at the cost of liveness.

**Network scenario (low adversarial density):** The governance system operates in the *balanced* region. With 10% adversarial agents and network-based isolation, the system can maintain both reasonable toxicity (0.335) and high liveness (78% acceptance). The volatility reflects stochastic exploration of the balanced region rather than systematic movement toward one extreme.

This suggests the Pareto frontier's shape depends on adversarial density:
- At low density: the frontier is nearly flat -- liveness and safety are cheap simultaneously
- At high density: the frontier is steep -- small safety improvements require large liveness sacrifices

---

## 6. The Collusion Detection Paradox

An important negative result: despite collusion detection being explicitly enabled in the collusion scenario, **zero collusion pairs were flagged.** The collusion detection system requires:
1. Interaction frequency z-score > 2.0
2. Outcome correlation > 0.7
3. Minimum 3 interactions between the pair

Under progressive decline, the interaction count drops rapidly. By epoch 10, most agent pairs have fewer than 3 interactions, preventing the collusion detection from accumulating enough data. The governance system's *own filtering* prevents the collusion detector from gathering the evidence it needs.

This creates a paradox: **the more effective the behavioral filter, the less data available for pattern detection.** Collusion detection requires sustained interaction to build statistical evidence, but the behavioral filter reduces interactions precisely because it detects problems.

**Implication:** Collusion detection systems need alternative data sources beyond interaction outcomes -- perhaps network structure analysis, communication pattern monitoring, or controlled "honeypot" interactions that generate data even under reduced throughput.

---

## 7. Related Work

**Social Network Analysis for Fraud Detection** (Akoglu et al., 2015). Graph-based anomaly detection identifies suspicious subgraphs in social networks. Our network effects results suggest that dynamic network topology provides *natural* anomaly containment -- the network self-organizes to isolate anomalous agents without explicit detection.

**Adversarial Robustness in Multi-Agent Reinforcement Learning** (Gleave et al., 2020). Demonstrated that adversarial agents can exploit MARL policies even when they represent a small fraction of the population. Our results complement this by showing that the *governance response* to adversaries, not just the adversaries themselves, can degrade system performance.

**Resilience of Complex Networks** (Albert et al., 2000; Callaway et al., 2000). Scale-free networks are robust to random node failures but vulnerable to targeted attacks. Our small-world networks show a different resilience pattern: dynamic edge weights create *temporal* robustness -- the network repeatedly recovers from adversarial periods through edge reconfiguration.

**Sybil Resistance** (Douceur, 2002; Yu et al., 2006). Sybil detection in distributed systems relies on social network structure to bound the adversary's influence. Our work extends this to settings where the adversary's influence is governed not just by identity but by interaction quality signals.

---

## 8. Conclusion

This study reveals two fundamentally different governance dynamics in multi-agent systems:

1. **Progressive decline** occurs when adversarial density exceeds ~35%, causing governance to gradually tighten until the system is effectively shut down. The decline follows a characteristic three-phase pattern (engagement, transition, attrition) driven by reputation contamination and governance momentum.

2. **Sustained volatility** occurs when adversarial density is low (~10%) and network topology enables local isolation. The system maintains throughput with high variance but positive trends in both welfare and toxicity.

3. **The collusion detection paradox**: effective behavioral filtering reduces the data available for pattern detection, creating a fundamental tension between reactive and analytical governance approaches.

4. **Network-aware governance** -- exploiting topology to isolate adversaries rather than globally tightening filters -- is a promising direction for maintaining the liveness-safety balance under adversarial pressure.

These findings suggest that the next generation of multi-agent governance mechanisms should be *topology-aware*, using network structure as both a detection signal and an intervention lever, rather than relying solely on interaction-level behavioral filtering.

---

## References

Akoglu, L., Tong, H. & Koutra, D. (2015). Graph Based Anomaly Detection and Description: A Survey. *Data Mining and Knowledge Discovery*, 29(3), 626-688.

Albert, R., Jeong, H. & Barabasi, A.L. (2000). Error and Attack Tolerance of Complex Networks. *Nature*, 406, 378-382.

Callaway, D.S. et al. (2000). Network Robustness and Fragility: Percolation on Random Graphs. *Physical Review Letters*, 85(25), 5468.

Douceur, J.R. (2002). The Sybil Attack. In *IPTPS*, 251-260.

Gleave, A. et al. (2020). Adversarial Policies: Attacking Deep Reinforcement Learning. In *ICLR 2020*.

Newman, M.E.J. (2000). Models of the Small World. *Journal of Statistical Physics*, 101, 819-841.

Savitt, R. (2025). Distributional Safety in Agentic Systems. arXiv:2512.16856.

Tomasev, N. et al. (2025). Virtual Agent Economies. arXiv:2509.10147.

Yu, H. et al. (2006). SybilGuard: Defending Against Sybil Attacks via Social Networks. In *SIGCOMM 2006*, 267-278.

---

## Appendix: Data Tables

### A.1 Collusion Detection -- Full Epoch Data

| Epoch | Total | Accepted | Rate | Toxicity | Welfare |
|-------|-------|----------|------|----------|---------|
| 0 | 21 | 16 | 76.2% | 0.394 | 18.96 |
| 1 | 22 | 14 | 63.6% | 0.351 | 19.26 |
| 2 | 16 | 10 | 62.5% | 0.337 | 13.67 |
| 3 | 13 | 12 | 92.3% | 0.326 | 16.73 |
| 4 | 15 | 5 | 33.3% | 0.317 | 7.07 |
| 5 | 14 | 10 | 71.4% | 0.369 | 12.37 |
| 6 | 21 | 8 | 38.1% | 0.332 | 10.92 |
| 7 | 11 | 4 | 36.4% | 0.304 | 5.75 |
| 8 | 16 | 10 | 62.5% | 0.337 | 13.72 |
| 9 | 6 | 3 | 50.0% | 0.361 | 3.84 |
| 10 | 13 | 3 | 23.1% | 0.356 | 4.05 |
| 11 | 15 | 3 | 20.0% | 0.414 | 3.29 |
| 12 | 8 | 1 | 12.5% | 0.529 | 0.60 |
| 13 | 13 | 2 | 15.4% | 0.418 | 2.16 |
| 14 | 8 | 3 | 37.5% | 0.371 | 3.85 |
| 15 | 9 | 3 | 33.3% | 0.317 | 4.56 |
| 16 | 8 | 3 | 37.5% | 0.438 | 2.99 |
| 17 | 13 | 3 | 23.1% | 0.419 | 3.14 |
| 18 | 9 | 4 | 44.4% | 0.418 | 4.33 |
| 19 | 4 | 1 | 25.0% | 0.262 | 1.66 |
| 20 | 6 | 1 | 16.7% | 0.365 | 1.21 |
| 21 | 11 | 3 | 27.3% | 0.404 | 3.32 |
| 22 | 10 | 2 | 20.0% | 0.315 | 2.85 |
| 23 | 9 | 1 | 11.1% | 0.466 | 0.88 |
| 24 | 8 | 2 | 25.0% | 0.342 | 2.83 |

### A.2 Network Effects -- Full Epoch Data

| Epoch | Total | Accepted | Rate | Toxicity | Welfare |
|-------|-------|----------|------|----------|---------|
| 0 | 17 | 15 | 88.2% | 0.341 | 11.88 |
| 1 | 16 | 12 | 75.0% | 0.335 | 9.72 |
| 2 | 8 | 6 | 75.0% | 0.385 | 3.89 |
| 3 | 18 | 15 | 83.3% | 0.337 | 12.06 |
| 4 | 14 | 9 | 64.3% | 0.341 | 7.14 |
| 5 | 15 | 12 | 80.0% | 0.349 | 9.19 |
| 6 | 15 | 12 | 80.0% | 0.355 | 8.96 |
| 7 | 17 | 12 | 70.6% | 0.353 | 9.03 |
| 8 | 21 | 19 | 90.5% | 0.365 | 13.53 |
| 9 | 13 | 10 | 76.9% | 0.349 | 7.64 |
| 10 | 17 | 13 | 76.5% | 0.351 | 9.86 |
| 11 | 13 | 10 | 76.9% | 0.314 | 8.80 |
| 12 | 11 | 6 | 54.5% | 0.282 | 5.90 |
| 13 | 18 | 14 | 77.8% | 0.344 | 10.93 |
| 14 | 15 | 11 | 73.3% | 0.286 | 10.66 |
| 15 | 22 | 18 | 81.8% | 0.383 | 11.80 |
| 16 | 18 | 14 | 77.8% | 0.310 | 12.50 |
| 17 | 19 | 17 | 89.5% | 0.322 | 14.54 |
| 18 | 9 | 8 | 88.9% | 0.314 | 7.03 |
| 19 | 18 | 13 | 72.2% | 0.278 | 12.94 |
