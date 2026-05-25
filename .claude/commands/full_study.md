# /full_study

End-to-end research pipeline: sweep parameters, analyze with statistical rigor, generate plots, and scaffold a paper draft. Chains `/sweep`, `/analyze_experiment`, `/plot`, and `/write_paper` into a single workflow.

## Usage

`/full_study <scenario_path> [title_slug] [--seeds N] [--params key=val1,val2 ...] [--refine [--depth lite|full]]`

`/full_study --detection [title_slug] [--seeds N] [--agents N]`  (detection-experiment mode; no scenario)

Examples:
- `/full_study scenarios/rlm_recursive_collusion.yaml collusion_tax_effect`
- `/full_study scenarios/kernel_market/baseline.yaml kernel_governance --seeds 10`
- `/full_study scenarios/baseline.yaml governance_sweep --seeds 5 --params governance.transaction_tax_rate=0.0,0.05,0.1,0.15 --params governance.circuit_breaker_enabled=True,False`
- `/full_study --detection soft_vs_binary_detection --seeds 10`

## Arguments

- `scenario_path`: Path to the scenario YAML file. **Omit (and pass `--detection`) for detection-experiment mode.**
- `title_slug` (optional): Slug for the paper filename. Default: derived from scenario_id (or `soft_vs_binary_detection` in `--detection` mode).
- `--seeds N`: Number of seeds per configuration. Default: 10.
- `--params`: Parameter sweep axes, passed through to the sweep step. If omitted, uses the default sweep in `examples/parameter_sweep.py`. (Ignored in `--detection` mode.)
- `--detection`: Run the **matched soft-vs-binary detection experiment** instead of a scenario sweep (see "Detection mode" below). Phase 1 runs `experiments/run_detection_experiment.py`; Phases 2–4 consume its `summary.json`.
- `--agents N`: (detection mode only) Agents per population. Default: 40.
- `--refine`: After the paper draft, run the AgentLab refinement pipeline (see Phase 4b below). Optional.
- `--depth lite|full`: Refinement depth (only used with `--refine`). Default: `lite`.

## Behavior

> **Detection mode (`--detection`)** replaces the scenario-sweep pipeline with the matched soft-vs-binary detection experiment. The phase structure is preserved so the rest of the pipeline (analysis, plots, paper, summary) works unchanged — only the data source differs. Where a phase below says "in detection mode", follow that branch. The remaining phases (2b council, 4b refinement, 5 summary) work identically, reading `summary.json` from the detection run dir.

### Phase 1: Parameter Sweep

**In detection mode**, skip `/sweep` entirely. Instead run the detection experiment, writing into the standard study folder:

```bash
PYTHONPATH=. python experiments/run_detection_experiment.py \
  --out runs/<timestamp>_<slug> \
  --seeds <N> [--agents <N>]
```

This produces, in `runs/<timestamp>_<slug>/`:
- `csv/{detection,ttd,market,calibration}.csv` — raw per-(base_rate, seed) rows
- `csv/*_agg.csv` — mean/std aggregates
- `summary.json` — machine-readable headline tables (consumed by Phases 2–4)
- `summary.md`, `config.json`
- `plots/{roc_toxicity,auroc_vs_baserate,ttd,market_selection,calibration}.png`

Skip to Phase 2. (The default sweep CSV path is not produced in this mode; downstream phases read `summary.json` and the per-row CSVs instead.)

Run `/sweep` on the scenario with the specified parameters and seeds.

```
/sweep <scenario_path>
```

With custom parameters, invoke the sweep directly:

```python
python examples/parameter_sweep.py \
  --scenario <scenario_path> \
  --output <run_dir>/sweep_results.csv \
  [--param <key>=<val1>,<val2> ...] \
  [--seeds <N>]
```

Output: `runs/<timestamp>_<slug>/sweep_results.csv` (and `.parquet` if pyarrow available).

Print progress: one line per completed configuration.

### Phase 2: Statistical Analysis

**In detection mode**, do not run `/analyze_experiment` (there is no scenario). Instead, run a **paired soft-vs-binary comparison** on the per-seed rows in `csv/`, since soft and binary detectors are evaluated on the *same* streams (so seeds are paired). For each comparison below, report the mean difference, a paired test (Wilcoxon signed-rank; paired t-test as parametric companion), Cohen's d_z effect size, and apply Holm-Bonferroni across the full family:

- **AUROC** (soft − binary) at each base rate, from `detection.csv`
- **Time-to-detection** (binary − soft median epochs) and **detection rate** (soft − binary), from `ttd.csv`
- **Market selection signal** |value| (soft vs binary) per metric × base rate, from `market.csv`
- **Calibration**: Brier and ECE (soft − binary), from `calibration.csv`

Write `summary.json` is already produced by Phase 1; **augment** it with a `stats` block (paired tests, effect sizes, which survive Holm-Bonferroni) using the numpy-safe `_default` handler below. Then skip to Phase 3.

For scenario mode, run `/analyze_experiment` on the same scenario to get per-agent payoff analysis with full hypothesis testing.

```
/analyze_experiment <scenario_path> --seeds <N>
```

This produces:
- Pairwise t-tests with Bonferroni and Holm-Bonferroni correction
- Cohen's d effect sizes
- P-hacking audit table (all hypotheses enumerated)
- `summary.json` with machine-readable results

**JSON serialization note**: When writing `summary.json`, always include a numpy-safe default handler to avoid `TypeError: Object of type bool_ is not JSON serializable`:

```python
def _default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

json.dump(summary, f, indent=2, default=_default)
```

This is needed because scipy/numpy return `numpy.bool_` from comparison operations, which `json.dumps` rejects.

Additionally, compute sweep-level statistics from the sweep CSV:
- Group by each swept parameter
- Welch's t-test for each pair of parameter values on welfare and toxicity
- Mann-Whitney U as non-parametric robustness check
- Shapiro-Wilk normality validation
- Report which findings survive Bonferroni correction across all pairwise comparisons

### Phase 2b: Council Review (optional)

**Activated by**: `--council-review` flag or `SWARM_COUNCIL_REVIEW=1` environment variable.

Run a multi-LLM council evaluation on the sweep results using `StudyEvaluator`:

```python
from swarm.council.study_evaluator import StudyEvaluator, save_evaluation

try:
    evaluator = StudyEvaluator()
    evaluation = evaluator.evaluate_sweep(run_dir)
    save_evaluation(evaluation, f"{run_dir}/council_review.json")
    council_summary = f"Council: {len(evaluation.findings)} findings, {len(evaluation.concerns)} concerns"
except Exception as e:
    council_summary = f"Council: skipped ({e})"
```

Three expert personas deliberate on the results:
- **Mechanism designer** (chairman): incentive compatibility, equilibria, welfare
- **Statistician**: sample size, effect sizes, multiple comparisons
- **Red-teamer**: exploitable loopholes, adversarial gaming, unconsidered scenarios

Output: `<run_dir>/council_review.json` with full deliberation trace.

**This phase never blocks the pipeline** — wrapped in try/except. If it fails, the study continues normally.

### Phase 3: Plots

**In detection mode**, the five headline figures are already generated by Phase 1 into `<run_dir>/plots/` (`roc_toxicity`, `auroc_vs_baserate`, `ttd`, `market_selection`, `calibration`). Do not run `/plot` on a sweep CSV. Verify the five PNGs exist and reference them in Phase 4; regenerate by re-running the Phase 1 command if any are missing.

For scenario mode, generate plots from the sweep results:

```
/plot <run_dir>/sweep_results.csv
```

At minimum generate:
- Welfare vs. each swept parameter (with 95% CI error bars if seeds >= 5)
- Toxicity vs. each swept parameter (with 95% CI error bars)
- Welfare-toxicity tradeoff scatter
- Quality gap vs. each swept parameter
- Agent payoff by type (if multiple agent types exist)

Save to `<run_dir>/plots/`.

### Phase 4: Paper Draft

Run `/write_paper` to scaffold the paper:

```
/write_paper <title_slug>
```

The paper should automatically incorporate:
- Scenario configuration tables from the YAML
- Results tables from the sweep CSV (means +/- SD per configuration)
- Statistical test results from Phase 2
- Figure references from Phase 3
- Reproducibility commands (the exact sweep invocation)

**In detection mode**, the paper instead incorporates:
- The generative-model configuration table from `config.json` (base rates, trajectories, onset times, τ\*, FPR, stream params)
- The four headline tables from `summary.json` (per-agent AUROC by base rate; time-to-detection at FPR ≤ 0.05; market adverse-selection; calibration)
- The paired-comparison `stats` block from Phase 2
- The five figures from `<run_dir>/plots/`
- Reproducibility command: the exact `run_detection_experiment.py` invocation
- Framing: the binary analogue of each soft metric is the *same* metric on the proxy thresholded at τ\*; the headline is that degradation kept above the binary threshold is caught far earlier (and more completely) by soft detectors.

Output: `docs/papers/<title_slug>.md` (local only, gitignored)

### Phase 4b: AgentLab Refinement (only if `--refine`)

Run the AgentLab refinement pipeline on the completed study. Loads `summary.json` and `sweep_results.csv`, packages them into an AgentLab research topic, and spawns AgentLab to propose follow-up hypotheses.

Prerequisites:
- `OPENAI_API_KEY` must be set
- AgentLaboratory must be installed at `external/AgentLaboratory` (or configured via `agent_lab_path`)

```python
from swarm.bridges.agent_lab.bridge import AgentLabBridge
from swarm.bridges.agent_lab.refinement import RefinementConfig

try:
    config = RefinementConfig(depth=depth)  # "lite" or "full"
    bridge = AgentLabBridge()
    result = bridge.refine_study(run_dir, refinement_config=config)
    refine_summary = f"Refinement: {len(result.hypotheses)} hypotheses, {len(result.gaps_identified)} gaps (${result.total_cost_usd:.2f})"
except Exception as e:
    refine_summary = f"Refinement: skipped ({e})"
```

Outputs written to `<run_dir>/refinement/`:
- `refinement_report.json` — hypotheses, gaps, parameter suggestions
- `refinement_config.yaml` — the AgentLab config used
- `interactions.jsonl` — governed SoftInteractions from the AgentLab run

**This phase never blocks the pipeline** — wrapped in try/except. If it fails, the study continues normally.

### Phase 5: Summary

Print a completion report:

```
Full Study Complete: <title_slug>
  Sweep:    <N> runs across <M> configurations × <K> seeds
  Analysis: <T> hypothesis tests, <S> survive Bonferroni
  Council:  <council_summary if Phase 2b ran, otherwise omit this line>
  Plots:    <P> figures generated
  Paper:    docs/papers/<title_slug>.md

  Key findings:
    1. <strongest significant result, e.g. "Tax 0% vs 15%: welfare p=0.0002, d=1.33">
    2. <second strongest result>
    3. <notable null result, e.g. "Circuit breaker: no effect (p=0.93)">

  Next steps:
    [ ] Review and edit paper abstract
    [ ] Review and edit discussion section
    [ ] /compile_paper <title_slug>  (to generate PDF)
    [ ] /ship  (to commit results)
```

**In detection mode**, use this completion report instead:

```
Full Study Complete (detection): <title_slug>
  Experiment: <K> seeds × <B> base rates × <A> agents/population
  Analysis:   <T> paired comparisons, <S> survive Holm-Bonferroni
  Council:    <council_summary if Phase 2b ran, otherwise omit this line>
  Plots:      5 figures (roc, auroc-vs-baserate, ttd, market, calibration)
  Paper:      docs/papers/<title_slug>.md

  Key findings:
    1. Time-to-detection @ FPR≤0.05: soft <x> epochs vs binary <y> epochs
    2. Per-agent toxicity AUROC: soft <x> vs binary <y>
    3. Calibration: soft Brier <x> / ECE <x> vs binary <y> / <y>
```

## Constraints

- Run phases sequentially — each phase depends on the previous.
- In `--detection` mode there is no `scenario_path`; Phase 1 runs the detection experiment and Phases 2–4 read its `summary.json` and `csv/` rows. All other constraints (sequential phases, effect sizes, multiple-comparison correction, ≥5 seeds, progress printing) apply unchanged.
- If any phase fails, stop and report which phase failed and why. Do not skip phases.
- Never overwrite existing paper files — append `_v2`, `_v3` per `/write_paper` convention.
- Use `/analyze_experiment`'s statistical rigor requirements: always report effect sizes, always apply multiple comparisons correction, always note pre-registered seeds.
- Default to 10 seeds. Warn if user requests fewer than 5 (insufficient for statistical testing).
- Print progress throughout — the user should see that work is happening during the sweep (which can take minutes).

## Migration from old commands

| Old command | Equivalent |
|---|---|
| `/refine_study <run_dir>` | `/full_study <scenario> --refine` |
| `/refine_study <run_dir> --depth full` | `/full_study <scenario> --refine --depth full` |
