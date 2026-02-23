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
