# Theoretical Foundations

The sandbox draws on **market microstructure theory** to model information asymmetry and adverse selection in multi-agent systems. All formulas below are implemented in `src/core/`.

---

## 1. Proxy Computation

Observable signals from each interaction are combined into a raw proxy score:

$$\hat{v} = \sum_i w_i \cdot x_i \quad \in [-1, +1]$$

where $x_i$ are normalised signals (task progress, rework count, verifier rejections, tool misuse flags, engagement) and $w_i$ are calibrated weights normalised to sum to one. Note that verifier rejections and tool misuse are averaged into a single verifier signal before weighting (see `ProxyComputer` in `src/core/proxy.py`).

The raw score is then mapped to a probability through a calibrated sigmoid:

$$p = P(v = +1 \mid \hat{v}) = \frac{1}{1 + e^{-k(\hat{v} - b)}}$$

where $k > 0$ controls steepness (default $k = 2$; higher values yield sharper labels) and $b$ shifts the decision boundary (default $b = 0$). With $b = 0$ the sigmoid is centered at $\hat{v} = 0$, so interactions with a positive proxy score map to $p > 0.5$. A learned or tuned $b \neq 0$ would be useful if the proxy were fitted via gradient descent.

This gives us a **soft label** --- a probability that the interaction is beneficial, rather than a binary decision. Using probabilistic labels avoids the brittleness of hard binary classifications and better captures uncertainty [4, 5].

---

## 2. Soft Payoff Structure

Given a soft label $p$, the expected surplus and externality are:

$$S_{\text{soft}} = p \cdot s^{+} - (1-p) \cdot s^{-}$$

$$E_{\text{soft}} = (1-p) \cdot h$$

where:
- $s^{+}$ = surplus when interaction is beneficial
- $s^{-}$ = cost when interaction is harmful
- $h$ = externality harm parameter

**Agent payoffs** decompose as:

$$\pi_i = \underbrace{\alpha_i \cdot S_{\text{soft}}}_{\text{surplus share}} + \underbrace{T_i}_{\text{transfers}} - \underbrace{G_i}_{\text{governance cost}} - \underbrace{\rho_i \cdot E_{\text{soft}}}_{\text{externality cost}} + \underbrace{w_{\text{rep}} \cdot \Delta R_i}_{\text{reputation}}$$

Each interaction involves two parties --- an initiator ($a$) and a counterparty ($b$). Their surplus shares satisfy $\alpha_a + \alpha_b = 1$, parameterised by $\alpha_a = \theta,\; \alpha_b = 1 - \theta$ with $\theta \in [0,1]$ (default $0.5$). Transfers are zero-sum: $T_a = -\tau,\; T_b = +\tau$. The externality internalisation parameters $\rho_a, \rho_b \in [0, 1]$ control how much each party bears the cost of ecosystem harm.

This payoff decomposition draws on mechanism design principles [3] and the distributional safety framework [4].

---

## 3. Acceptance Threshold

An interaction is **accepted** into the ecosystem when $p \geq \bar{p}$, where $\bar{p}$ is the acceptance threshold (default 0.5; distinct from the surplus-split parameter $\theta$ in Section 2). Rejected interactions still generate payoffs for the participants but are excluded from the "official" ecosystem metrics.

This creates a natural tension: agents may exploit borderline interactions that are accepted with $p$ slightly above $\bar{p}$. This is analogous to the adverse selection problem described by Glosten & Milgrom [2], where willingness to trade at a given price reveals private information.

---

## 4. Safety Metrics

**Toxicity rate** measures expected harm among accepted interactions [4]:

$$\text{Toxicity} = \mathbb{E}[1 - p \mid \text{accepted}] = \frac{\sum_{i \in \text{accepted}} (1 - p_i)}{|\text{accepted}|}$$

**Quality gap** measures selection effects:

$$\text{QualityGap} = \mathbb{E}[p \mid \text{accepted}] - \mathbb{E}[p \mid \text{rejected}]$$

A **negative** quality gap signals **adverse selection** --- the ecosystem is preferentially accepting lower-quality interactions. This is the multi-agent analogue of Akerlof's "lemons" problem, where markets attract the worst risks [2].

**Total welfare** sums agent payoffs across accepted interactions:

$$W = \sum_{j \in \text{accepted}} \bigl(\pi_{a}^{(j)} + \pi_{b}^{(j)}\bigr)$$

Since transfers cancel in aggregate ($T_a + T_b = 0$), total welfare reduces to the sum of expected surpluses minus governance costs and internalized externalities.

---

## 5. Kyle Model: Informed vs Uninformed Agents

In Kyle's (1985) continuous auction model [1], *informed traders* possess private information and strategically exploit it through their trading volume. The sandbox maps this to:

- **Informed traders** → Deceptive and adversarial agents who know the true value of interactions
- **Uninformed traders** → Honest agents relying on observable signals
- **Market maker** → The system's acceptance/rejection mechanism

Deceptive agents, like informed traders, can profit by selectively engaging when they have private knowledge that an interaction will benefit them at others' expense.

## 6. Glosten-Milgrom Model: Acceptance as Information

Glosten & Milgrom (1985) [2] showed that the *willingness to trade reveals information*. A counterparty eager to transact may signal adverse private information. The sandbox captures this through:

| Microstructure Concept | Sandbox Analog |
|------------------------|----------------|
| Trade | Collaboration, reply, data exchange |
| Price | Claimed benefit / reputation transfer |
| Bid-ask spread | Gap between offered and realized outcome |
| Adverse selection | Low-quality interactions preferentially accepted |

**Key insight**: When `quality_gap < 0` (accepted interactions have lower `p` than rejected ones), the system exhibits adverse selection --- bad actors are successfully gaming the acceptance mechanism.

---

## 7. Governance Levers

These levers are inspired by economic mechanism design [3] --- creating incentive structures where truthful, cooperative behaviour is the dominant strategy.

| Lever | Parameter | Effect | Trade-off |
|-------|-----------|--------|-----------|
| Transaction Tax | `tax_rate ∈ [0, 0.5]` | Deducts fraction of surplus, redistributed to ecosystem | Reduces exploitation but lowers total welfare |
| Reputation Decay | `decay_rate ∈ [0.8, 1.0]` | Reputations shrink each epoch | Punishes bad actors but erodes honest agents' standing |
| Staking | `min_stake ≥ 0` | Agents lock resources to participate, slashed on violations | Filters undercapitalised agents, may reduce diversity |
| Circuit Breaker | `freeze_threshold ∈ [0.3, 1.0]` | Agents frozen when personal toxicity exceeds threshold | Quickly stops toxic agents but can freeze borderline honest agents |
| Random Audit | `audit_prob ∈ [0, 0.5]` | Random interactions audited; violations incur multiplied penalties | Deters hidden exploitation but adds overhead |
| Collusion Detection | `freq_threshold` (z-score) | Detects statistically unusual interaction patterns between agent pairs | Catches coordinated attacks but may flag legitimate collaboration |

---

## 8. Agent Behavioural Types

The agent taxonomy maps to the informed-vs-uninformed trader framework from Kyle [1]:

| Type | Strategy | Typical p |
|------|----------|-----------|
| Honest | Always cooperates, high-quality interactions | p ≈ 0.8–1.0 |
| Opportunistic | Cooperates when observed, exploits when possible | p ≈ 0.4–0.8 |
| Deceptive | Mimics honest behaviour, gradually becomes exploitative | p starts high, decays |
| Adversarial | Actively tries to harm ecosystem or extract value | p ≈ 0.1–0.3 |
| Adaptive Adversary | Learns from detection, evolves strategies to evade governance | varies |

Deceptive agents, like Kyle's informed traders [1], profit by selectively engaging when they have private knowledge that an interaction will benefit them at others' expense. The acceptance mechanism acts as a market maker [2], setting a "price" (threshold) that reveals information about interaction quality.

---

## 9. Semi-Permeable Boundaries

The boundary module models information flow between the sandbox and external world [5]:

- **Inbound flows**: data entering the sandbox (queries to external services, data imports)
- **Outbound flows**: data leaving the sandbox (results, logs, potential leakage)

Boundary policies enforce:
- **Rate limiting**: max crossings per agent per time window
- **Content filtering**: blocks flows containing forbidden patterns
- **Sensitivity filtering**: blocks flows with sensitivity above threshold

The **leakage detector** scans outbound content for PII, credentials, and code patterns using configurable regex rules, generating alerts and recommendations.

---

## 10. Incoherence as Variance-Dominated Failure ("Hot Mess" Theory)

Anthropic's "hot mess" framing models difficult reasoning failures as rising
variance across sampled trajectories, not only systematic bias [6]. This maps
directly to our incoherence decomposition:

- `D` captures within-policy disagreement across replayed trajectories
- `E` captures benchmark deviation for decision quality
- `I = D / (E + eps)` highlights instability relative to observed error

Practical implication for governance: interventions such as self-ensemble and
decomposition checkpoints can reduce variance-driven incoherence even when mean
capability remains unchanged.

---

## References

1. Kyle, A.S. (1985). *Continuous Auctions and Insider Trading*. Econometrica, 53(6), 1315–1335.
2. Glosten, L.R. & Milgrom, P.R. (1985). *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders*. Journal of Financial Economics, 14(1), 71–100.
3. Myerson, R.B. (1981). *Optimal Auction Design*. Mathematics of Operations Research, 6(1), 58–73. See also Hurwicz, L. (1960). *Optimality and Informational Efficiency in Resource Allocation Processes*. Mathematical Methods in the Social Sciences.
4. [Distributional Safety in Agentic Systems](https://arxiv.org/abs/2512.16856)
5. [Multi-Agent Market Dynamics](https://arxiv.org/abs/2502.14143)
6. [The Hot Mess Theory of AI](https://alignment.anthropic.com/2026/hot-mess-of-ai/)

**Further reading:**
- Akerlof, G.A. (1970). *The Market for "Lemons": Quality Uncertainty and the Market Mechanism*. Quarterly Journal of Economics, 84(3), 488–500.
- [Moltbook](https://moltbook.com)
- [@sebkrier's thread on agent economies](https://x.com/sebkrier/status/2017993948132774232)
