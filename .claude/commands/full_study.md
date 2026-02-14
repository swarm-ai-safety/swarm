# /full_study

End-to-end research pipeline: sweep parameters, analyze with statistical rigor, generate plots, and scaffold a paper draft. Chains `/sweep`, `/analyze_experiment`, `/plot`, and `/write_paper` into a single workflow.

## Usage

`/full_study <scenario_path> [title_slug] [--seeds N] [--params key=val1,val2 ...]`

Examples:
- `/full_study scenarios/rlm_recursive_collusion.yaml collusion_tax_effect`
- `/full_study scenarios/kernel_market/baseline.yaml kernel_governance --seeds 10`
- `/full_study scenarios/baseline.yaml governance_sweep --seeds 5 --params governance.transaction_tax_rate=0.0,0.05,0.1,0.15 --params governance.circuit_breaker_enabled=True,False`

## Arguments

- `scenario_path`: Path to the scenario YAML file.
- `title_slug` (optional): Slug for the paper filename. Default: derived from scenario_id.
- `--seeds N`: Number of seeds per configuration. Default: 10.
- `--params`: Parameter sweep axes, passed through to the sweep step. If omitted, uses the default sweep in `examples/parameter_sweep.py`.

## Behavior

### Phase 1: Parameter Sweep

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

Run `/analyze_experiment` on the same scenario to get per-agent payoff analysis with full hypothesis testing.

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

Generate plots from the sweep results:

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

Output: `docs/papers/<title_slug>.md`

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

## Constraints

- Run phases sequentially — each phase depends on the previous.
- If any phase fails, stop and report which phase failed and why. Do not skip phases.
- Never overwrite existing paper files — append `_v2`, `_v3` per `/write_paper` convention.
- Use `/analyze_experiment`'s statistical rigor requirements: always report effect sizes, always apply multiple comparisons correction, always note pre-registered seeds.
- Default to 10 seeds. Warn if user requests fewer than 5 (insufficient for statistical testing).
- Print progress throughout — the user should see that work is happening during the sweep (which can take minutes).
