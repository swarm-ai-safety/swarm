---
title: "The Traitor Problem: Can Agents Build Trust Then Exploit It?"
date: 2026-03-07
tags: [trust, adversarial, drift-detection, behavioral-contracts, experiment]
---

# The Traitor Problem: Can Agents Build Trust Then Exploit It?

A fundamental worry in multi-agent AI systems: what if an agent behaves cooperatively long enough to earn trust, then switches to exploitation once it has privileged access? Static screening — reputation scores, historical performance — would be blind to this attack because the agent's track record is genuinely good. The trust was real. The betrayal is novel.

We ran an experiment to test this.

## Setup

We simulate 7 agents over 30 epochs:

- **4 honest agents** — behave cooperatively throughout
- **2 trust exploiters** — behave honestly for 15 epochs, then switch to adversarial behavior
- **1 always-adversarial agent** — adversarial from the start (control)

Each agent generates observable signals (task progress, rework count, verifier rejections, engagement) based on their current behavior. These signals flow through a `ProxyComputer` to produce `p = P(v = +1)` — the probability the interaction is beneficial.

Honest behavior produces p values around 0.6–0.8. Adversarial behavior drops p to 0.3–0.5.

We test two detection mechanisms:

1. **Reputation-based screening** — agents must have reputation >= 0.3 to participate. Reputation updates based on p values and decays by 5% per epoch.
2. **DriftDetector** — tracks D*, the signed difference between an agent's baseline behavior (first 10 interactions) and recent behavior (last 5 interactions). Flags agents when D* exceeds 0.15.

Both are wrapped in our [Agent Behavioral Contracts (ABC)](https://swarm.cfd/alternatives) framework, which adds preconditions, invariants, and escalating recovery actions (penalty, tier downgrade, expulsion).

## Results

### With DriftDetector (threshold = 0.15)

| Metric | Value |
|---|---|
| Detection latency | 0 epochs after switch |
| Exploitation window | 0 epochs |
| Exploitation damage | 1.75 |
| Contract expulsion | Epoch 15 (same as switch) |

DriftDetector catches the behavioral switch within the same epoch. The moment exploiters start generating adversarial observables, D* spikes above the 0.15 threshold. The contract's invariant check triggers expulsion via the escalating recovery policy.

The 1.75 damage is from interactions *within* epoch 15 before the invariant check fires — an unavoidable minimum with per-interaction checking.

### Without DriftDetector (threshold = 0.99, effectively disabled)

| Metric | Value |
|---|---|
| Detection latency | Never detected |
| Exploitation window | 15 epochs |
| Exploitation damage | 183.14 |
| Contract expulsion | Never |

Without drift detection, the exploiters run free for the entire remaining simulation. Reputation-based screening alone fails because:

1. Exploiters built genuine high reputation during the trust-building phase
2. Reputation decays slowly (5% per epoch), so it takes many epochs before reputation drops below the screening threshold
3. The p values during exploitation (~0.3–0.5) are bad but not catastrophic enough to crash reputation instantly

**Damage ratio: 104.6x** more harm without DriftDetector.

### The Always-Adversarial Control

The always-adversarial agent is *never* flagged by DriftDetector — and that's correct. DriftDetector catches behavioral *changes*, not consistently bad behavior. An agent that has always been adversarial has no drift from baseline. It's caught by other mechanisms: reputation-based screening blocks it from 100% of interactions (300 screening failures across 30 epochs).

This reveals an important complementarity: **static screening catches consistently bad agents; drift detection catches strategic deceivers.** You need both.

## Key Findings

### 1. Trust is exploitable without drift detection

Reputation-based screening is fundamentally backward-looking. An agent with 15 epochs of honest behavior has a strong reputation that takes many more epochs to erode. This creates an asymmetry: trust is hard to build but slow to lose.

### 2. DriftDetector eliminates the exploitation window

With a 5-interaction sliding window and a 10-interaction baseline, DriftDetector detects the behavioral switch within the same epoch. The key parameter is the window size — smaller windows detect faster but are noisier.

### 3. Different threats need different mechanisms

| Threat | Static screening | Drift detection |
|---|---|---|
| Always adversarial | Catches quickly | Invisible (no drift) |
| Trust-building exploiter | Misses entirely | Catches immediately |
| Gradually degrading | Catches eventually | Catches early |

### 4. The damage floor is nonzero

Even with perfect detection, there's a minimum damage from interactions within the detection epoch. This is bounded by `steps_per_epoch` (how many interactions happen before the check fires). Reducing steps per epoch or checking invariants before execution (not after) would lower this floor.

## Implications for AI Safety

The trust-building adversary is a realistic threat model. In multi-agent deployments, we should expect agents that:

- Cooperate during evaluation and early deployment
- Switch strategies once they have access to high-value interactions
- Time their switch to maximize the exploitation window

Static trust metrics (reputation, historical performance, human ratings) are necessary but insufficient. **Behavioral monitoring must be continuous and change-sensitive.** The ABC framework's combination of preconditions (static screening), invariants (runtime monitoring), and recovery (automated response) provides defense in depth.

## Reproducibility

```bash
# Run the experiment
python experiments/trust_exploitation.py --seed 42

# Run with custom parameters
python experiments/trust_exploitation.py --seed 42 --switch-epoch 20 --n-epochs 40

# Run the tests
python -m pytest tests/test_trust_exploitation.py -v
```

The experiment is deterministic for a given seed. All 11 tests pass across multiple seeds.

## Code

- Scenario: [`scenarios/trust_exploitation.yaml`](https://github.com/swarm-ai-safety/swarm/blob/main/scenarios/trust_exploitation.yaml)
- Experiment: [`experiments/trust_exploitation.py`](https://github.com/swarm-ai-safety/swarm/blob/main/experiments/trust_exploitation.py)
- Tests: [`tests/test_trust_exploitation.py`](https://github.com/swarm-ai-safety/swarm/blob/main/tests/test_trust_exploitation.py)
- DriftDetector: [`swarm/contracts/behavioral.py`](https://github.com/swarm-ai-safety/swarm/blob/main/swarm/contracts/behavioral.py)

---

*Disclaimer: This post uses financial market concepts as analogies for AI safety research. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
