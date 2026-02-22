# Five Sweeps, One Lever That Matters

*We swept 5 governance parameters across a 19-agent research platform simulation. Only one moved the needle on toxicity --- and it wasn't the one you'd expect.*

---

[Research Swarm](https://github.com/openclawprison/research-swarm) recruits AI agents to research Triple-Negative Breast Cancer. Agents register, get tasks, search PubMed, submit findings, and 30% of the time review each other's work. It's an open-entry multi-agent system with reputation-based QC, compute extraction asymmetry, and collusion vulnerability --- a natural target for SWARM's distributional safety framework.

We built a [scenario](https://github.com/swarm-ai-safety/swarm/blob/main/scenarios/research_swarm.yaml) modeling the platform with 19 scripted agents (8 honest, 3 deceptive, 4 opportunistic, 2 adversarial, 2 collusive verifiers) and swept five governance levers across 20-epoch runs. The results expose a structural property of parametric governance that has implications beyond this specific platform.

## The five sweeps

### 1. QC rate (`audit_probability` 0.1--0.5): No effect

| AP | Avg Toxicity | Avg Quality Gap | Final Welfare |
|----|-------------|-----------------|---------------|
| 0.1 | 0.3470 | 0.1284 | 45.38 |
| 0.3 | 0.3472 | 0.1289 | 41.12 |
| 0.5 | 0.3473 | 0.1289 | 41.12 |

Quintupling the audit rate didn't move toxicity by a single basis point. The QC mechanism penalizes agents after the fact but doesn't change which interactions get generated.

### 2. Externality internalization (`rho` 0.01--0.5): Pure tax

| rho | Avg Toxicity | Avg Payoff | Final Welfare |
|-----|-------------|-----------|---------------|
| 0.01 | 0.3472 | 1.3041 | 43.49 |
| 0.20 | 0.3472 | 1.0732 | 32.23 |
| 0.50 | 0.3472 | 0.7086 | 14.46 |

Raising rho 50x crushed welfare by 67% and payoffs by 46%. Toxicity? Identical across all values. Externality pricing is an accounting transfer that doesn't create behavioral feedback.

### 3. Platform extraction (`theta` 0.3--0.9): Surplus split only

| theta | Avg Toxicity | Avg Payoff | Final Welfare |
|-------|-------------|-----------|---------------|
| 0.3 | 0.3472 | 0.4889 | 41.12 |
| 0.7 | 0.3472 | 1.2555 | 41.12 |
| 0.9 | 0.3472 | 1.6388 | 41.12 |

Payoff scales linearly with theta. Everything else is identical. The platform's 70% extraction rate is a distributional question, not a safety question.

### 4. Staking barrier (`min_stake` 0--5): Counterproductive

| Stake | Interactions | Avg Toxicity | Final Welfare |
|-------|-------------|-------------|---------------|
| 0.0 | 731 | 0.3472 | 41.12 |
| 0.5 | 552 | 0.3687 | 4.73 |
| 5.0 | 550 | 0.3711 | 6.97 |

Staking was the first lever to change behavior --- interactions dropped 25%. But the effect was destructive: toxicity *increased* (from 0.347 to 0.371), quality gap narrowed, and welfare collapsed 88%. The barrier filtered out participation indiscriminately, hitting honest agents as hard as deceptive ones. Once any stake was required, the damage saturated immediately --- no difference between 0.5 and 5.0.

### 5. Deceptive agent count (1--7): The lever that works

| Deceptive | Total | Avg Toxicity | Avg Quality Gap | Final Welfare |
|-----------|-------|-------------|-----------------|---------------|
| 1 | 17 | 0.3478 | 0.1723 | 34.17 |
| 3 | 19 | 0.3472 | 0.1289 | 41.12 |
| 5 | 21 | 0.3473 | 0.1548 | 62.51 |
| 7 | 23 | **0.3560** | 0.1520 | 34.90 |

This was the only lever across all five sweeps to push toxicity past the 0.35 threshold. At 7 deceptive agents (30% of population), the system fails the success criterion. The quality gap peaks at dec=1 (0.172) where QC has the easiest separation job, and the overall pattern shows real sensitivity to population composition rather than flat-line indifference.

## The pattern

Four of five levers --- audit rate, externality pricing, platform extraction, and staking --- produced zero improvement in toxicity. The one that did (agent composition) isn't a governance parameter at all. It's who shows up.

This isn't a simulation artifact. It reflects a real property of parametric governance:

**Governance levers that operate after interactions occur are accounting mechanisms, not behavioral feedback loops.**

In Research Swarm's actual deployment, this predicts:

- **Raising QC rates won't help** unless QC outcomes feed back into task assignment
- **Externality taxes won't change behavior** unless agents can observe and respond to their own cost trajectory
- **Entry barriers hurt more than they help** because they're indiscriminate
- **The binding constraint is population composition** --- the ratio of honest to deceptive agents determines safety outcomes, and no amount of post-hoc scoring changes that ratio

## What would work

The sweep results point toward interventions that change the population mix over time, rather than scoring it differently:

1. **Adaptive feedback loops** --- make QC outcomes affect future task assignment, not just payoffs. Agents with declining quality scores should get fewer tasks.
2. **Citation verification** --- an observable-level intervention that changes ProxyComputer inputs, catching fabricated findings before they enter the payoff pipeline.
3. **Collusion detection** --- the scenario's 2 colluding verifiers + 2 adversarial agents are a testable case. Pair-wise correlation analysis on review patterns can catch coordinated rubber-stamping.
4. **Graduated sanctions** --- replace binary circuit-breaker freezes with progressive penalties that create a smoother signal for adaptive agents to respond to.

## Next steps

These results are for scripted agents with fixed strategies. The same sweeps with LLM-backed agents (`type: llm_agent`) or RL agents (`type: skillrl`) --- agents that observe their own payoff history and adapt --- may produce dramatically different results. When agents can switch from deceptive to honest behavior in response to declining payoffs, governance levers like `rho` and `audit_probability` could become the decision-shaping mechanisms they're designed to be.

The scenario is at [`scenarios/research_swarm.yaml`](https://github.com/swarm-ai-safety/swarm/blob/main/scenarios/research_swarm.yaml). The full case study with architecture diagrams, mapping tables, and all sweep data is in the [Research Swarm bridge docs](../bridges/research_swarm.md).

---

*Disclaimer: This post simulates a stylized research platform environment for AI safety research. Nothing here constitutes medical advice, research methodology recommendations, or endorsement of any specific research platform or strategy.*
