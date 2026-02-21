# /analyze_experiment

Run statistical analysis on SWARM experiment data. Two modes: run a scenario across multiple seeds, or analyze an existing sweep CSV. Consolidates the former `/stats` command (now `/analyze_experiment --csv`).

## Usage

```
/analyze_experiment <scenario_path_or_id> [--seeds N|seed1,seed2,...] [--groups auto|key=ids,...]
/analyze_experiment --csv <sweep_csv> [--output <dir>]
```

Examples:
- `/analyze_experiment rlm_recursive_collusion`
- `/analyze_experiment scenarios/rlm_memory_as_power.yaml --seeds 20`
- `/analyze_experiment rlm_governance_lag --seeds 42,7,123,256,999`
- `/analyze_experiment rlm_memory_as_power --groups high=rlm_1,rlm_2,rlm_3 mid=rlm_4,rlm_5,rlm_6 low=rlm_7`
- `/analyze_experiment --csv runs/20260210-223119_kernel_market_v2/sweep_results.csv`
- `/analyze_experiment --csv runs/latest/sweep_results.csv --output runs/latest/`

## Argument parsing

Parse `$ARGUMENTS` to extract:
- `--csv <path>`: CSV mode — run statistics on an existing sweep CSV (see CSV Mode below)
- `--seeds`: Either an integer N (generate N random seeds) or a comma-separated list. Default: `42,7,123,256,999,2024,314,577,1337,8080` (10 seeds).
- `--groups`: How to group agents for comparison. `auto` (default) infers from scenario YAML. Otherwise `label=id1,id2,...`.
- `--output <dir>`: Output directory (CSV mode only). Default: same directory as the CSV.

If `--csv` is present → CSV mode. Otherwise → Scenario mode.

---

## Scenario Mode (default)

### Preferred: Use the persistent analysis script

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

### Key APIs (fallback only)

These are the correct Orchestrator accessors:
- `orch.state.agents` -- dict of `agent_id -> AgentState` (has `.total_payoff`, `.reputation`, etc.)
- `orch.get_all_agents()` -- returns `List[BaseAgent]` (the agent policy objects, not state)
- `orch._epoch_metrics` -- list of `EpochMetrics` (has `.total_welfare`, `.toxicity_rate`, etc.)

Do NOT use:
- `orch.agents` (does not exist -- use `orch.get_all_agents()`)
- `orch._all_interactions` (does not exist)
- `history["epochs"]` from JSON export (key is `epoch_snapshots`, and it lacks per-agent payoffs)

---

## CSV Mode (`--csv`)

Run the full statistical analysis battery on an existing sweep CSV.

### Step 1: Load and normalize columns

Read the CSV with pandas. Normalize known column aliases:

```
avg_toxicity       → toxicity_rate
total_welfare      → welfare
honest_avg_payoff  → honest_payoff
opportunistic_avg_payoff → opportunistic_payoff
adversarial_avg_payoff   → adversarial_payoff
deceptive_avg_payoff     → deceptive_payoff
avg_quality_gap    → quality_gap
```

Keep originals if canonical names already exist. Print detected sweep parameters (columns that are not metrics).

### Step 2: Identify swept parameters

Any column whose name contains a `.` (e.g. `governance.transaction_tax_rate`) is a sweep parameter. All other numeric columns are metrics.

### Step 3: Pairwise comparisons

For each swept parameter with discrete values:
- For each pair of values and each metric in `[welfare, toxicity_rate, honest_payoff, opportunistic_payoff, adversarial_payoff, quality_gap]` (using whichever exist):
  1. **Welch's t-test** (two-sided, unequal variance)
  2. **Mann-Whitney U** (non-parametric robustness check)
  3. **Cohen's d** effect size (pooled SD)
  4. Record group means, SDs, and sample sizes

### Step 4: Multiple comparisons correction

Count total hypotheses tested. Apply:
1. **Bonferroni**: reject if `p < 0.05 / n_tests`
2. **Benjamini-Hochberg**: rank p-values, reject if `p_i <= (i / n_tests) * 0.05`

Flag each result with `bonferroni_sig` and `bh_sig` booleans.

### Step 5: Normality checks

Run **Shapiro-Wilk** on the primary metric (welfare) for each group of the first swept parameter. Report W statistic and p-value. Flag groups as NORMAL (p > 0.05) or NON-NORMAL.

### Step 6: Agent-type stratification (if applicable)

If columns for 2+ agent types exist (e.g. `honest_payoff`, `adversarial_payoff`):
- Run **paired t-test** for all agent-type pairs across all runs
- Compute Cohen's d
- Apply Bonferroni correction (over agent-type pairs only)

### Step 7: Output

Print a formatted report:

```
Statistical Analysis: <csv_filename>
============================================================
Swept parameters: governance.transaction_tax_rate (4 values), governance.circuit_breaker_enabled (2 values)
Total runs: 80
Total hypotheses: 42

=== Significant Results (Bonferroni) ===
  welfare: 0.0 vs 0.15 — p=0.0006, d=1.19

=== Agent-Type Stratification ===
  honest=2.21, opp=2.34, adv=-1.65
  honest vs adversarial: d=3.45***

=== Normality (Shapiro-Wilk) ===
  All groups normal (all p > 0.05)
```

Save `summary.json` with structure:

```json
{
  "csv": "<path>",
  "total_runs": 80,
  "total_hypotheses": 42,
  "n_bonferroni_significant": 1,
  "n_bh_significant": 1,
  "swept_parameters": {},
  "results": [],
  "agent_stratification": [],
  "normality": []
}
```

### Key APIs (CSV mode)

```python
from scipy import stats
import numpy as np
import pandas as pd

# Welch's t-test
stats.ttest_ind(g1, g2, equal_var=False)

# Mann-Whitney U
stats.mannwhitneyu(g1, g2, alternative='two-sided')

# Cohen's d
pooled_sd = np.sqrt((np.std(g1, ddof=1)**2 + np.std(g2, ddof=1)**2) / 2)
d = (np.mean(g1) - np.mean(g2)) / pooled_sd

# Shapiro-Wilk
stats.shapiro(values)

# Paired t-test (for agent stratification)
stats.ttest_rel(g1, g2)
```

---

## Statistical rigor requirements (both modes)

- Always report effect sizes alongside p-values
- Always apply multiple comparisons correction
- Always report total number of hypotheses tested
- Always validate normality assumption before interpreting t-tests
- Use Welch's t-test (not Student's) — do not assume equal variance
- Report both parametric (t-test) and non-parametric (Mann-Whitney) results

## Constraints

- Run seeds sequentially (orchestrator is not thread-safe) — Scenario mode
- Do not modify the scenario YAML file — Scenario mode
- Never modify the input CSV — CSV mode
- Minimum 5 seeds/observations per group for parametric tests; warn if fewer
- Column normalization must be idempotent — CSV mode
- Print progress as analysis proceeds

## Migration from old commands

| Old command | Equivalent |
|---|---|
| `/stats runs/latest/sweep_results.csv` | `/analyze_experiment --csv runs/latest/sweep_results.csv` |
| `/stats runs/sweep.csv --output runs/` | `/analyze_experiment --csv runs/sweep.csv --output runs/` |
