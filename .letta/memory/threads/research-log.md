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

## 2026-02-26T05:30:00Z

**Changed:** `examples/mesa_adaptive_agents_study.py` (new)
**Learned:** Learning agents that improve task_progress in response to rejection recover the welfare collapse at high rho. At rho=1.0: adaptive+learning welfare=807 vs adaptive-only=340 (+137% recovery), while toxicity also improves slightly (0.147 vs 0.157). The learning regime Pareto-dominates adaptive at every rho value. Selfish agents learn most aggressively (task_progress: 0.26→0.69), exploitative agents barely improve (0.14→0.20) due to lower learning rate and higher rework. The governance safe zone widens from rho∈[0.3,0.7] to roughly [0.0,0.8] with learning agents. Welfare recovery exceeds 100% at low-to-mid rho — learning agents actually *exceed* the static baseline because higher acceptance rates from improved quality generate more surplus. Effect sizes are *** (p<0.001) for welfare at all rho values. Generated 8 plots including Pareto frontier showing learning pushes the tradeoff curve outward.
**Next:** Try Stag Hunt and Hawk-Dove payoff matrices via Mesa bridge to test if the governance sweet spot generalizes across game structures. Connect Mesa bridge to a real Mesa model (Schelling segregation or Sugarscape) for non-synthetic validation.
**Run pointers:** `20260226-201109_mesa_adaptive_agents_study` (in swarm-artifacts)

## 2026-02-27T00:15:00Z

**Changed:** `examples/mesa_game_structures_study.py` (new), publication figures fig8-fig12 (in swarm-artifacts), cross-study comparison (local)
**Learned:** The governance sweet spot generalizes across game structures. Tested 3 game types (Prisoner's Dilemma, Stag Hunt, Hawk-Dove) with distinct payoff matrices and break-even probabilities (0.33/0.11/0.57). Key findings: (1) Learning agents Pareto-dominate adaptive-only across ALL game types. (2) At rho=1.0, learning recovers +132-159% welfare (d=9.88-11.63, all p<0.001). (3) Hawk-Dove shows strongest % recovery (+159%) because high externalities (h=3) make the cost of admitting bad actors most severe. (4) Toxicity converges to ~0.147 at rho=1.0 regardless of game type — game structure affects welfare magnitude but not toxicity reduction. (5) Normalized welfare decay curves nearly overlap across games, confirming game-invariant governance dynamics. (6) Cross-study comparison (455 total runs) confirms Study 2 and Study 3 PD results are identical, validating reproducibility. (7) Governance tax with learning (32-41%) is roughly half of without learning (66-77%) across all games.
**Next:** (1) Test adversarial learners that game the threshold rather than genuinely improve. (2) Connect Mesa bridge to a real Mesa model for non-synthetic validation. (3) Population scaling — test if results hold with 100+ agents. (4) Consider writing up the three-study arc as a paper.
**Run pointers:** `20260226-211430_mesa_game_structures_study` (in swarm-artifacts), `cross_study_mesa_comparison` (local)
