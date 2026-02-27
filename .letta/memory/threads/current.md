---
description: "Active research thread — current hypothesis and next steps"
updated: 2026-02-26
---

# Active Thread

## Current hypothesis

Adaptive behavioral responses (learning from rejection) are necessary and sufficient to make externality internalization (rho) welfare-positive at moderate levels. The governance sweet spot generalizes across game structures when agents can adapt.

## What we're testing

Extended the Mesa bridge governance study with a third regime: adaptive_learning, where agents improve task_progress in response to rejection (diminishing returns, archetype-specific learning rates). 165 runs total (11 rho values × 3 regimes × 5 seeds) on a 30-agent heterogeneous population.

## Last session summary

- Created `examples/mesa_adaptive_agents_study.py` — extends the governance sweep with learning agents
- Found: learning agents recover +137% welfare at rho=1.0 (807 vs 340), Pareto-dominate adaptive at every rho
- Selfish agents learn most (task_progress 0.26→0.69); exploitative agents barely improve (0.14→0.20)
- Governance safe zone widens from [0.3, 0.7] to [0.0, 0.8] with learning agents
- Effect sizes are *** (p<0.001) for welfare at all rho values
- Generated 8 plots; pushed artifacts to swarm-artifacts
- Hypothesis partially confirmed: adaptive response overcomes welfare collapse, but the "sweet spot" concept may be obsolete — with learning, rho is beneficial at all levels

## Next experiment

1. Try Stag Hunt and Hawk-Dove payoff matrices via Mesa bridge to test governance sweet spot generalization across game structures
2. Connect Mesa bridge to a real Mesa model (Schelling segregation or Sugarscape) for non-synthetic validation
3. Test adversarial learners — agents that learn to game the threshold rather than genuinely improve

## Blockers

None currently.
