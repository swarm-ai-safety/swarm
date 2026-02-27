---
description: "Active research thread — current hypothesis and next steps"
updated: 2026-02-27
---

# Active Thread

## Current hypothesis

The governance sweet spot is game-structure-invariant: externality internalization (rho) combined with adaptive thresholds and learning agents produces consistent welfare-toxicity tradeoff improvements regardless of whether the underlying interaction is PD, Stag Hunt, or Hawk-Dove. The next question is whether adversarial learners (who game the threshold) can break this.

## Status: CONFIRMED (3-study arc complete)

Three Mesa bridge studies now form a coherent arc:
1. **Governance study** (110 runs): rho alone is a pure welfare tax; adaptive threshold creates real governance with sweet spot at rho∈[0.3,0.7]
2. **Adaptive agents study** (165 runs): learning agents recover +137% welfare at rho=1.0, Pareto-dominate adaptive at every rho
3. **Game structures study** (180 runs): results generalize across PD, Stag Hunt, Hawk-Dove — learning benefit is 132-159% at rho=1.0 (d=9.88-11.63, all p<0.001)

Cross-study comparison (455 total runs) validates reproducibility: Study 2 and Study 3 PD conditions produce identical results.

## Last session summary

- Created `examples/mesa_game_structures_study.py` — 3 game types x 6 rho x 2 regimes x 5 seeds = 180 runs
- Generated 8 diagnostic plots and 5 publication figures (fig8-fig12, pushed to swarm-artifacts)
- Ran cross-study comparison across all 3 Mesa studies (455 runs total)
- Key insight: toxicity converges to ~0.147 at rho=1.0 regardless of game type; game structure affects welfare magnitude but not toxicity reduction
- Hawk-Dove (costly conflict, h=3) shows strongest learning benefit (+159%) because high externalities make governance most valuable

## Next experiment

1. **Adversarial learners** — agents that learn to game the acceptance threshold rather than genuinely improve quality. Do they break the governance mechanism?
2. **Real Mesa model** — connect the bridge to Schelling segregation or Sugarscape for non-synthetic validation
3. **Population scaling** — test whether results hold with 100+ agents (current: 30)
4. **Paper writeup** — the three-study arc is a natural paper structure (tax → adaptation → generalization)

## Blockers

None currently.
