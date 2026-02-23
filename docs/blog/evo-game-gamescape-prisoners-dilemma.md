# When Replicator Dynamics Meet Soft Labels

*Plugging evolutionary game theory into the swarm pipeline reveals where classical predictions break down --- and where they don't*

---

We integrated [gamescape](https://github.com/raelisavitt/gamescape), a 2x2 evolutionary game engine, into the swarm orchestrator. Now agents play formal Prisoner's Dilemma rounds with payoffs flowing through the existing soft-label pipeline: `PayoffMatrix -> observables -> ProxyComputer -> p -> SoftPayoffEngine`. Across five seeds (42, 123, 314, 777, 999), the results are consistent: replicator dynamics correctly predicts the equilibrium structure, but the *path* to equilibrium diverges sharply from classical theory --- and the variance across seeds reveals which dynamics are robust and which are fragile.

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

10 epochs, 5 steps per epoch, five seeds (42, 123, 314, 777, 999). No governance levers (no tax, no circuit breaker, no audits). The evo game handler maps game payoffs to `ProxyObservables`, which then flow through the standard proxy computation to produce soft labels `p`.

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

These observables flow through the `ProxyComputer` to produce `p = P(v = +1)`, which captures interaction quality probabilistically. Across five seeds:

| Metric | Seed 42 | Seed 123 | Seed 314 | Seed 777 | Seed 999 | Mean ± SD |
|---|---|---|---|---|---|---|
| Total interactions | 107 | 118 | 125 | 97 | 120 | 113.4 ± 11.3 |
| Accepted % | 88.8% | 90.7% | 88.0% | 86.6% | 90.0% | 88.8% ± 1.6% |
| Avg toxicity | 0.302 | 0.290 | 0.295 | 0.306 | 0.300 | 0.299 ± 0.006 |
| Avg p | 0.69 | 0.71 | 0.72 | 0.69 | 0.71 | 0.70 ± 0.01 |
| Final welfare | 32.43 | 25.62 | 14.99 | 20.98 | 34.24 | 25.65 ± 8.03 |

Toxicity converges tightly across seeds to ~0.30 --- consistent with 30% of the population being defectors who produce lower-quality interactions. Acceptance rate is similarly stable at ~89%. The main source of variance is final welfare (CV = 31%), driven by differences in welfare trajectory shape (see below).

## Reputation divergence: the real selection pressure

The most striking result is reputation dynamics. While strategy frequencies remained fixed, reputations diverged dramatically. Here are the final reputations from seed 42:

| Agent | Type | Final Reputation |
|---|---|---|
| cautious_reciprocator_2 | TFT | **5.48** |
| cautious_reciprocator_1 | TFT | **4.29** |
| cautious_reciprocator_3 | TFT | **4.05** |
| honest_1 | Cooperator | 1.17 |
| opportunistic_1 | Defector | 1.17 |
| opportunistic_3 | Defector | 1.13 |
| opportunistic_2 | Defector | 1.11 |
| honest_3 | Cooperator | 0.90 |
| honest_4 | Cooperator | 0.83 |
| honest_2 | Cooperator | 0.46 |

This pattern --- TFT agents accumulating 3--5x the reputation of other agents --- holds across all five seeds. The reciprocators' combination of conditional cooperation and high activity drives a positive feedback loop: good interactions build reputation, which improves their observable signals (via the reputation-modulated `rep_factor` in the observable generator), which produces higher `p` values. In seed 42, cautious_reciprocator_2 initiated 22 interactions over the run while honest_2 initiated only 4.

This is a form of selection pressure that operates *through the soft-label pipeline* rather than through strategy switching. The replicator equation doesn't model it. But in a system with reputation-weighted matching or resource-gated interaction, this reputational divergence would eventually have strategy-frequency consequences.

## Welfare scaling: two regimes

Across five seeds, welfare trajectories split into two distinct patterns:

**Sharp phase transition** (seeds 42, 999): Low welfare plateau through epochs 0--4, then a sudden jump to 30--35 at epoch 5--6, sustained through the end. In seed 42:

| Epochs | Avg Welfare | Avg Interactions |
|---|---|---|
| 0--4 | 6.02 | 4.2 |
| 5--9 | 29.77 | 17.2 |

**Gradual ramp** (seeds 123, 777): Welfare climbs steadily from ~10 to ~20--25 without a sharp discontinuity. Final welfare is lower (20--26 vs 32--34).

**Rise then fall** (seed 314): The most volatile trajectory --- welfare peaked at 29.75 at epoch 5 but declined to 14.99 by epoch 9, suggesting that TFT reputation gains weren't sufficient to sustain mid-run momentum in this matchup sequence.

The phase transition, when it occurs, coincides with TFT agents ramping up their interaction rate. As their reputations rise, their observable signals improve, producing higher `p` values and therefore higher expected surplus (`S_soft = p * s_plus - (1-p) * s_minus`). More interactions at higher quality compound into substantially higher welfare.

## Early-epoch fragility

Three of five seeds (314, 777, 999) exhibit an acceptance rate crash in epochs 1--3, dropping to 33--40% before recovering to 90%+ by epoch 5. Seeds 42 and 123 avoid this crash entirely. The pattern suggests the system is reliably fragile in early epochs --- before TFT reputation effects have accumulated enough to stabilize interaction quality --- and that the specific matchup sequence in those early rounds determines whether the crash occurs.

This has safety implications: a governance system that intervenes based on early-epoch metrics (e.g., freezing agents with low acceptance rates) might prematurely disrupt a system that would self-correct. Conversely, a system without governance might suffer real harm during the fragile early period.

## What this means for distributional safety

The classical game theory result --- defection dominates, cooperation collapses --- is an equilibrium prediction about strategy frequencies in an evolving population. It's correct as far as it goes. But multi-agent AI systems don't operate like infinite well-mixed populations with proportional reproduction. They have:

1. **Fixed strategies** (agents are deployed with specific code/objectives)
2. **Reputation systems** (past behavior affects future opportunities)
3. **Soft labels** (interaction quality is probabilistic, not binary)
4. **Heterogeneous activity rates** (some agents interact more than others)

The gamescape integration lets us compare what replicator dynamics predicts against what actually happens in a richly-mediated multi-agent environment. The phase portrait gives us the equilibrium structure (all-defect is the only stable fixed point). The simulation gives us the *trajectory* through a system with reputation, soft payoffs, and rate heterogeneity.

The five-seed study reveals what's robust and what's fragile:

- **Robust**: Toxicity convergence (~0.30), acceptance rate recovery (90%+), TFT reputation dominance. These hold across all seeds.
- **Fragile**: Welfare trajectory shape, early-epoch acceptance rates, final welfare level. These vary substantially with matchup sequence.

For safety, the key insight is that reputation-mediated systems can sustain cooperation even in dominant-defect games --- not by changing strategy frequencies, but by concentrating interaction opportunities on conditionally cooperative agents. The early-epoch fragility suggests that governance interventions should be calibrated to the system's self-correction timescale: too early risks disrupting beneficial dynamics, too late risks accumulating real harm during the fragile period. Whether this is robust to more sophisticated adversaries (adaptive defectors, obfuscating agents, threshold dancers) is the next question.

## Reproducing

```bash
# Run a single seed
python -m swarm run scenarios/evo_game_prisoners.yaml --seed 42 --epochs 10 --steps 5

# Run all five seeds
for seed in 42 123 314 777 999; do
  python -m swarm run scenarios/evo_game_prisoners.yaml --seed $seed --epochs 10 --steps 5
done

# Run the study with gamescape analysis
python examples/evo_game_study.py

# Run the tests
python -m pytest tests/test_evo_game_handler.py -v
```

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
