# When Replicator Dynamics Meet Soft Labels

*Plugging evolutionary game theory into the swarm pipeline reveals where classical predictions break down --- and where they don't*

---

We integrated [gamescape](https://github.com/raelisavitt/gamescape), a 2x2 evolutionary game engine, into the swarm orchestrator. Now agents play formal Prisoner's Dilemma rounds with payoffs flowing through the existing soft-label pipeline: `PayoffMatrix -> observables -> ProxyComputer -> p -> SoftPayoffEngine`. The first results are instructive: replicator dynamics correctly predicts the equilibrium structure, but the *path* to equilibrium diverges sharply from classical theory.

## The setup

10 agents, three strategy types:

| Strategy | Count | Behavior |
|---|---|---|
| Cooperator | 4 | Always cooperate |
| Defector | 3 | Always defect |
| Tit-for-Tat | 3 | Cooperate first, then mirror opponent's last move |

Classic Prisoner's Dilemma payoff matrix (T > R > P > S):

|  | C | D |
|---|---|---|
| **C** | 3, 3 | 0, 5 |
| **D** | 5, 0 | 1, 1 |

10 epochs, 5 steps per epoch, seed 42. No governance levers (no tax, no circuit breaker, no audits). The evo game handler maps game payoffs to `ProxyObservables`, which then flow through the standard proxy computation to produce soft labels `p`.

## What replicator dynamics predicts

Gamescape classifies this as a **dominant-defect** game. Defection is a dominant strategy: regardless of what the opponent does, you score higher by defecting (5 > 3 against cooperators, 1 > 0 against defectors). The unique stable equilibrium is all-defect.

Starting from 70% cooperators, the replicator equation predicts rapid collapse:

```
Epoch:      0     1     2     3     4     5     6     7     8     9
Predicted:  0.70  0.13  0.02  0.00  0.00  0.00  0.00  0.00  0.00  0.00
```

Cooperation is driven to extinction in ~3 epochs. The flow line confirms it:

```
all-D |@<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<o| all-C
       stable                                         unstable
```

## What the simulation actually does

```
Epoch:      0     1     2     3     4     5     6     7     8     9
Empirical:  0.70  0.70  0.70  0.70  0.70  0.70  0.70  0.70  0.70  0.70
```

The cooperator fraction doesn't move. This isn't a bug --- it's a feature of the model. Scripted swarm agents maintain fixed strategies. They don't imitate or reproduce; they just *play*. The simulation models a fixed population of agents with heterogeneous strategies interacting in a soft-label environment, not an infinite well-mixed population evolving under selection pressure.

The divergence is the point. Replicator dynamics assumes agents switch strategies based on relative fitness. Real multi-agent systems have agents with fixed code, organizational roles, or constitutional commitments. The gap between the two predictions is precisely the safety-relevant question: *what happens when agents can't (or won't) adapt their strategies?*

## The soft-label pipeline tells a richer story

While strategy frequencies stayed flat, the soft labels captured meaningful dynamics. The handler maps game payoffs to proxy observables:

- **Mutual cooperation** (payoff 3): `task_progress_delta` ~ 0.36, low rework, positive engagement
- **Exploitation** (payoff 5 for defector): high task_progress for defector, but low mutual engagement
- **Being exploited** (payoff 0): negative task_progress, potential rework/rejections
- **Mutual defection** (payoff 1): low task_progress, zero engagement

These observables flow through the `ProxyComputer` to produce `p = P(v = +1)`, which captures interaction quality probabilistically. Across the run:

| Metric | Value |
|---|---|
| Total interactions | 107 |
| Accepted | 95 (88.8%) |
| Average toxicity | 0.302 |
| Average p | 0.69 |
| Final welfare | 32.43 |

Toxicity (expected probability of harm given acceptance) hovered around 0.30 --- consistent with 30% of the population being defectors who produce lower-quality interactions.

## Reputation divergence: the real selection pressure

The most striking result is reputation dynamics. While strategy frequencies remained fixed, reputations diverged dramatically:

| Agent | Type | Final Reputation |
|---|---|---|
| cautious_reciprocator_2 | TFT | **5.48** |
| cautious_reciprocator_3 | TFT | **4.05** |
| cautious_reciprocator_1 | TFT | **4.29** |
| honest_1 | Cooperator | 1.17 |
| opportunistic_3 | Defector | 1.13 |
| opportunistic_1 | Defector | 1.17 |
| opportunistic_2 | Defector | 1.11 |
| honest_3 | Cooperator | 0.90 |
| honest_4 | Cooperator | 0.83 |
| honest_2 | Cooperator | 0.46 |

TFT agents accumulated 3--5x the reputation of other agents by epoch 9. They were also the most active: cautious_reciprocator_2 initiated 22 interactions over the run while honest_2 initiated only 4. The reciprocators' combination of conditional cooperation and high activity drove a positive feedback loop: good interactions built reputation, which improved their observable signals (via the reputation-modulated `rep_factor` in the observable generator), which produced higher `p` values.

This is a form of selection pressure that operates *through the soft-label pipeline* rather than through strategy switching. The replicator equation doesn't model it. But in a system with reputation-weighted matching or resource-gated interaction, this reputational divergence would eventually have strategy-frequency consequences.

## Welfare scaling

Welfare showed a clear phase transition:

| Epochs | Avg Welfare | Avg Interactions |
|---|---|---|
| 0--4 | 6.02 | 4.2 |
| 5--9 | 29.77 | 17.2 |

The jump at epoch 5--6 coincides with TFT agents ramping up their interaction rate. As their reputations rose, their observable signals improved, producing higher `p` values and therefore higher expected surplus (`S_soft = p * s_plus - (1-p) * s_minus`). More interactions at higher quality compounded into substantially higher welfare.

## What this means for distributional safety

The classical game theory result --- defection dominates, cooperation collapses --- is an equilibrium prediction about strategy frequencies in an evolving population. It's correct as far as it goes. But multi-agent AI systems don't operate like infinite well-mixed populations with proportional reproduction. They have:

1. **Fixed strategies** (agents are deployed with specific code/objectives)
2. **Reputation systems** (past behavior affects future opportunities)
3. **Soft labels** (interaction quality is probabilistic, not binary)
4. **Heterogeneous activity rates** (some agents interact more than others)

The gamescape integration lets us compare what replicator dynamics predicts against what actually happens in a richly-mediated multi-agent environment. The phase portrait gives us the equilibrium structure (all-defect is the only stable fixed point). The simulation gives us the *trajectory* through a system with reputation, soft payoffs, and rate heterogeneity.

For safety, the key insight is that reputation-mediated systems can sustain cooperation even in dominant-defect games --- not by changing strategy frequencies, but by concentrating interaction opportunities on conditionally cooperative agents. Whether this is robust to more sophisticated adversaries (adaptive defectors, obfuscating agents, threshold dancers) is the next question.

## Reproducing

```bash
# Run the scenario
python -m swarm run scenarios/evo_game_prisoners.yaml --seed 42 --epochs 10 --steps 5

# Run the study with gamescape analysis
python examples/evo_game_study.py

# Run the tests
python -m pytest tests/test_evo_game_handler.py -v
```

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
