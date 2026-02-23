---
description: "Active research thread — current hypothesis and next steps"
updated: 2026-02-22
---

# Active Thread

## Current hypothesis

Reputation-mediated systems can sustain cooperation in dominant-defect games without governance — but are fragile to early-epoch matchup sequences and may not resist adaptive adversaries.

## What we're testing

Integrated gamescape evolutionary game engine into the swarm orchestrator. 10 agents (4 cooperators, 3 defectors, 3 TFT) play iterated Prisoner's Dilemma with payoffs flowing through the soft-label pipeline. Ran 5 seeds (42, 123, 314, 777, 999).

## Last session summary

- Built `EvolutionaryGameHandler` integrating gamescape's `PayoffMatrix` into the swarm orchestrator
- Ran 5 seeds of the evo game prisoners scenario
- Found: toxicity (~0.30) and acceptance rate (~89%) are robust across seeds; welfare trajectory shape is fragile (CV=31%)
- TFT agents accumulate 3-5x reputation — selection pressure through soft-label pipeline
- 3/5 seeds show early-epoch acceptance crashes (33-40%) that self-correct by epoch 5
- Updated blog post with 5-seed comparison table and new sections (welfare regimes, early-epoch fragility)

## Next experiment

1. Add governance levers (tax, circuit breaker, audits) to `evo_game_prisoners.yaml` and compare welfare/toxicity against the ungoverned baseline
2. Introduce adaptive adversaries (threshold dancers) to test whether reputation-mediated cooperation is robust
3. Try different payoff matrices (Stag Hunt, Hawk-Dove) to see if the welfare trajectory patterns generalize

## Blockers

None currently.
