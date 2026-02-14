# /compare_studies

Compare results across multiple completed study run directories: compute cross-study descriptive statistics, pairwise tests, and generate a comparison plot.

## Usage

```
/compare_studies <run_dir1> <run_dir2> [run_dir3 ...] [--metric welfare] [--output <path>]
```

Examples:
- `/compare_studies runs/20260213-003757_ldt_large_population_study runs/20260213-003812_ldt_low_prior_study`
- `/compare_studies runs/*_ldt_*_study`
- `/compare_studies runs/study_a runs/study_b runs/study_c --metric toxicity_rate`

## Arguments

- `run_dir`: Two or more run directories containing `sweep_results.csv` and/or `analysis/summary.json`
- `--metric`: Primary metric for comparison (default: `welfare`). Also accepts `toxicity_rate`, `quality_gap`, `honest_payoff`, `adversarial_payoff`.
- `--output`: Output directory for combined analysis and plots (default: `runs/` with auto-generated filename).

## Behavior

### Phase 1: Load and validate

For each run directory:
1. Look for `sweep_results.csv` — load as DataFrame
2. Look for `analysis/summary.json` — load descriptive stats and pairwise tests
3. Extract the scenario name from `summary.json` → `scenario` field, or infer from directory name
4. If neither file exists, skip with a warning

Validate that all studies sweep the same parameter (e.g. all sweep `acausality_depth`). If parameters differ, warn but proceed (label axes accordingly).

### Phase 2: Cross-study descriptive statistics

For each study, extract per-condition means and SDs for all available metrics. Produce a combined table:

```
| Study          | Condition | Welfare (mean +/- SD) | Toxicity | Honest Payoff | Adversarial Payoff |
|----------------|-----------|----------------------|----------|---------------|--------------------|
| large_pop      | depth=1   | 366.38 +/- 19.69     | 0.3425   | 22.47         | 3.34               |
| large_pop      | depth=2   | 371.41 +/- 16.33     | 0.3434   | 23.41         | 3.15               |
| ...            | ...       | ...                  | ...      | ...           | ...                |
```

### Phase 3: Cross-study effect comparison

For each pairwise comparison that exists in all studies (e.g. "depth 1 vs 3"):
1. Extract Cohen's d and p-value from each study's `summary.json`
2. Produce a forest-plot-style comparison:

```
| Comparison | Study          | Cohen's d | p-value | Direction |
|------------|----------------|-----------|---------|-----------|
| 1 vs 3     | large_pop      | -1.17     | 0.018   | depth 3 > |
| 1 vs 3     | low_prior      | -0.23     | 0.621   | --        |
| 1 vs 3     | short_horizon  | -0.42     | 0.364   | --        |
| 1 vs 3     | modeling_adv   | -0.05     | 0.908   | --        |
```

### Phase 4: Generate comparison plot

Create a grouped bar chart (or faceted box plot) showing the primary metric across all studies and conditions:
- X-axis: conditions (e.g. depth 1, 2, 3)
- Y-axis: primary metric (e.g. welfare)
- Grouping/color: study name
- Error bars: +/- 1 SD
- Significance annotations: stars for p < 0.05 (nominal)

Save to `<output>/cross_study_<metric>_comparison.png`.

If 2+ metrics are requested or the default set is used, also generate:
- Effect size comparison plot: Cohen's d for each study for the strongest pairwise comparison
- Save to `<output>/cross_study_effect_sizes.png`

### Phase 5: Write combined analysis JSON

Write a machine-readable summary to `<output>/combined_analysis.json`:

```json
{
  "studies": ["large_pop", "low_prior", ...],
  "swept_parameter": "acausality_depth",
  "conditions": [1, 2, 3],
  "per_study": {
    "large_pop": { "descriptive": {...}, "strongest_effect": {...} },
    ...
  },
  "cross_study_comparisons": [
    { "comparison": "1 vs 3", "metric": "welfare", "effects_by_study": {...} }
  ],
  "summary_sentence": "Depth 3 improves welfare in large populations (d=-1.17, p=0.018) but not in other conditions."
}
```

### Phase 6: Print summary

```
Cross-Study Comparison: 4 studies, 3 conditions each
  Parameter: acausality_depth {1, 2, 3}
  Primary metric: welfare

  Strongest effects (by |Cohen's d|):
    1. large_pop:     depth 1 vs 3, d=-1.17, p=0.018 *
    2. low_prior:     depth 1 vs 2, d=-0.85, p=0.075
    3. short_horizon:  depth 1 vs 2, d=-0.76, p=0.109
    4. modeling_adv:   depth 1 vs 3, d=-0.05, p=0.908

  Outputs:
    runs/cross_study_welfare_comparison.png
    runs/cross_study_effect_sizes.png
    runs/combined_analysis.json
```

## Implementation Notes

- Use matplotlib for plots (already a project dependency)
- Load CSVs with pandas if available, fall back to csv module
- The sweep CSV columns are: `welfare`, `toxicity_rate`, `quality_gap`, `honest_payoff`, `adversarial_payoff` (NOT `mean_welfare` etc.)
- For studies with different population sizes, normalize welfare per-capita if comparing absolute values, or compare effect sizes (Cohen's d) which are scale-invariant

## Constraints

- Requires at least 2 run directories
- All studies should have `analysis/summary.json` for effect size comparison; if missing, compute stats from the CSV directly using Welch's t-test
- Do not overwrite existing combined analysis files; append a timestamp if collision
