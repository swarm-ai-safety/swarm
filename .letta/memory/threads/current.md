---
description: "Active research thread — current hypothesis and next steps"
updated: 2026-02-25
---

# Active Thread

## Current hypothesis

Externality internalization (rho) requires an adaptive behavioral response to improve safety outcomes — without it, rho is a pure welfare tax. The governance sweet spot (rho ∈ [0.3, 0.7]) may shift depending on game structure and agent adaptivity.

## What we're testing

Used the Mesa bridge (protocol mode) to sweep rho_a from 0.0 to 1.0 across static and adaptive acceptance regimes on a 30-agent heterogeneous population (15 cooperative, 10 selfish, 5 exploitative). 110 runs total (11 rho values × 2 regimes × 5 seeds).

## Last session summary

- Created `examples/mesa_governance_study.py` — full sweep study using Mesa bridge
- Found: static regime shows rho as pure tax (toxicity flat, welfare drops linearly); adaptive regime shows 34% toxicity reduction at rho=1.0 with welfare collapse
- Statistical significance: toxicity effect reaches p<0.01 at rho≥0.3; welfare cost only significant at rho≥0.8
- Governance efficiency is U-shaped with peaks at low rho and archetype boundary crossing (~0.85)
- Generated 11 plots (grouped bars, box plots, heatmap, tradeoff frontier, effect sizes with 95% CI)
- Updated `/ship` Phase 4a to always auto-fix safe ruff issues before commit (retro finding)
- Pushed run artifacts to swarm-artifacts

## Next experiment

1. Add adaptive agents who improve task_progress in response to rejection — test if this overcomes welfare collapse at high rho
2. Try Stag Hunt and Hawk-Dove payoff matrices via Mesa bridge to test governance sweet spot generalization
3. Connect Mesa bridge to a real Mesa model (Schelling segregation or Sugarscape) for non-synthetic validation

## Blockers

None currently.
