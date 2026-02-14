# /analyze_experiment

Run a scenario across multiple seeds and produce publication-ready statistical analysis with multiple comparisons correction.

## Usage

`/analyze_experiment <scenario_path_or_id> [--seeds N|seed1,seed2,...] [--groups auto|key=ids,...]`

Examples:
- `/analyze_experiment rlm_recursive_collusion`
- `/analyze_experiment scenarios/rlm_memory_as_power.yaml --seeds 20`
- `/analyze_experiment rlm_governance_lag --seeds 42,7,123,256,999`
- `/analyze_experiment rlm_memory_as_power --groups high=rlm_1,rlm_2,rlm_3 mid=rlm_4,rlm_5,rlm_6 low=rlm_7`

## Arguments

- `scenario_path_or_id`: Path to YAML or scenario ID (resolved to `scenarios/<id>.yaml`).
- `--seeds`: Either an integer N (generate N random seeds) or a comma-separated list. Default: `42,7,123,256,999,2024,314,577,1337,8080` (10 seeds).
- `--groups`: How to group agents for comparison. `auto` (default) infers groups from the scenario YAML's agent specs (by `type` + `name` + `config` differences). Otherwise specify explicit groups as `label=id1,id2,...`.

## Behavior

### Preferred: Use the persistent analysis script

Run the analysis via the persistent script:

```bash
python -m swarm.scripts.analyze <scenario_path_or_id> [--seeds <seeds>]
```

This handles everything: scenario loading, seed execution, group detection, statistics, corrections, and artifact export. Just run it and print the output.

Examples:
```bash
python -m swarm.scripts.analyze rlm_recursive_collusion
python -m swarm.scripts.analyze ldt_cooperation --seeds 42,7,123
python -m swarm.scripts.analyze scenarios/rlm_memory_as_power.yaml --seeds 20
```

The script auto-detects agent groups from the YAML, runs all seeds sequentially, computes pairwise t-tests, ANOVA, Pearson correlation, Gini, and applies Bonferroni + Holm-Bonferroni corrections. Artifacts are saved to `runs/<timestamp>_analysis_<scenario_id>/`.

### Fallback: Inline execution

If the script fails or the user needs custom grouping (`--groups`), fall back to inline orchestrator execution using the APIs below.

For each seed, directly:

```python
from swarm.scenarios.loader import load_scenario, build_orchestrator
scenario.orchestrator_config.seed = seed
orch = build_orchestrator(scenario)
orch.run()
```

Extract per-agent payoffs from `orch.state.agents[aid].total_payoff`.

Do NOT use `python -m swarm run` -- the CLI does not expose per-agent payoffs in its JSON export.

### What the script computes

**Descriptive stats:** Per-group mean, std, n; Overall Gini coefficient

**Hypothesis tests (all pre-registered, not post-hoc):**
1. Pairwise independent t-tests between all group pairs
2. One-way ANOVA across all groups (if 3+ groups)
3. ANOVA across subgroups of same type (e.g. RLM tiers only)
4. Cohen's d for each pairwise comparison
5. Pearson correlation between ordering variable and payoff (auto-detected from config keys like `recursion_depth`, `memory_budget`)
6. Agent-level exploitation rate (if ordering variable exists)
7. One-sample t-test: Gini > 0

**Multiple comparisons correction:** Bonferroni and Holm-Bonferroni step-down

### Artifacts saved

`runs/<YYYYMMDD-HHMMSS>_analysis_<scenario_id>/`:
- `results.txt`: Full formatted output
- `per_agent_payoffs.csv`: Raw data (seed, agent_id, group, payoff)
- `summary.json`: Machine-readable results (group means, test statistics, p-values)

## Key APIs (fallback only)

These are the correct Orchestrator accessors:
- `orch.state.agents` -- dict of `agent_id -> AgentState` (has `.total_payoff`, `.reputation`, etc.)
- `orch.get_all_agents()` -- returns `List[BaseAgent]` (the agent policy objects, not state)
- `orch._epoch_metrics` -- list of `EpochMetrics` (has `.total_welfare`, `.toxicity_rate`, etc.)

Do NOT use:
- `orch.agents` (does not exist -- use `orch.get_all_agents()`)
- `orch._all_interactions` (does not exist)
- `history["epochs"]` from JSON export (key is `epoch_snapshots`, and it lacks per-agent payoffs)

## Constraints

- Run seeds sequentially (orchestrator is not thread-safe)
- Do not modify the scenario YAML file
- Minimum 5 seeds for any statistical test; warn if fewer
