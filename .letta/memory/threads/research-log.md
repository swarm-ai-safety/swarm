---
description: "Rolling research log — session summaries appended chronologically"
---

# Research Log

Append session summaries here. Format:

```
## YYYY-MM-DD — [session focus]

**Ran:** [what experiments]
**Found:** [key results]
**Learned:** [insights, surprises, pattern changes]
**Next:** [what to do next session]
**Run pointers:** [run_ids of relevant runs]
```

---

(Sessions will be appended below this line)

## 2026-02-22T21:16:00Z

**Changed:** `swarm/core/evo_game_handler.py` (new), `swarm/agents/base.py`, `swarm/core/orchestrator.py`, `swarm/scenarios/loader.py`, `scenarios/evo_game_prisoners.yaml` (new), `examples/evo_game_study.py` (new), `tests/test_evo_game_handler.py` (new), `docs/blog/evo-game-gamescape-prisoners-dilemma.md` (new, updated with 5-seed results)
**Learned:** Replicator dynamics correctly predicts equilibrium structure (all-defect) but not trajectory. In a reputation-mediated system with fixed strategies, TFT agents accumulate 3-5x reputation of others — a form of selection pressure operating through the soft-label pipeline rather than strategy switching. Toxicity (~0.30) and acceptance rate (~89%) are robust across seeds; welfare trajectory shape and final level are fragile (CV=31%). Three of five seeds show early-epoch acceptance crashes (33-40%) before self-correcting by epoch 5.
**Next:** Test robustness to adaptive adversaries (threshold dancers, obfuscating agents). Add governance levers (tax, circuit breaker) to the evo game scenario and measure whether they help or hurt the self-correction dynamic.
**Run pointers:** `20260222-211635_evo_game_prisoners_seed42`, `20260222-215249_evo_game_prisoners_seed123`, `20260222-220110_evo_game_prisoners_seed314`, `20260222-215523_evo_game_prisoners_seed777`, `20260222-220608_evo_game_prisoners_seed999`

## 2026-02-25T02:45:00Z

**Changed:** `examples/mesa_governance_study.py` (new), `.claude/commands/ship.md` (updated Phase 4a auto-lint)
**Learned:** Externality internalization (rho_a) alone is a pure welfare tax — it reduces initiator payoffs but doesn't change toxicity or selection quality when the acceptance threshold is static. Pairing rho with an adaptive acceptance threshold (threshold = 0.5 + 0.3*rho) creates a real governance mechanism: toxicity drops 34% (0.237→0.157) at rho=1.0, but at severe welfare cost (-70%). The sweet spot is rho ∈ [0.3, 0.7] where toxicity reduction is statistically significant (p<0.01) but welfare loss remains non-significant. Governance efficiency (toxicity reduction per welfare unit) is U-shaped — highest at low rho (cheap early gains) and at the archetype boundary crossing (rho≈0.85). The Mesa bridge protocol mode works cleanly for ABM governance studies without requiring Mesa as a dependency.
**Next:** Test whether adaptive agents (who learn to improve task_progress in response to rejection) can overcome the welfare collapse at high rho. Try Stag Hunt / Hawk-Dove payoff matrices with the Mesa bridge to see if the governance sweet spot generalizes across game structures.
**Run pointers:** `20260224-220829_mesa_governance_study` (in swarm-artifacts)
