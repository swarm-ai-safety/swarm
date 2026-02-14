# /plot

Generate standard plots from a previously-exported run folder.

## Usage

`/plot <run_dir> [metric]`

Examples:
- `/plot runs/20260208-120102_baseline_seed42`
- `/plot runs/20260208-120102_baseline_seed42 toxicity_rate`

## Inputs (expected)

- `<run_dir>/history.json` (from `python -m swarm run ... --export-json ...`)
or
- `<run_dir>/csv/` (from `--export-csv`)

## Outputs

Write plots under:
- `<run_dir>/plots/`

At minimum generate:
- Time series: toxicity, quality_gap, welfare
- Acceptance rate over time (if available)

If optional deps are missing (e.g. `swarm-safety[analysis]`), fall back to:
- Writing `<run_dir>/plots/README.txt` with install instructions and what would be plotted.

## Sweep data detection

If the input path points to a **sweep CSV** (contains columns like `governance.*` or multiple `run_index` values), generate sweep-specific plots instead of time-series plots:

- **Grouped bar charts**: welfare and toxicity by each swept parameter, with individual data points overlaid and ±1 SD error bars
- **Box plots**: welfare distribution per configuration
- **Heatmap**: mean welfare/toxicity/quality_gap across the parameter grid (if 2 swept parameters)
- **Agent payoff comparison**: honest/opportunistic/adversarial payoffs by config
- **Effect size plot**: governance lever boost with 95% CI and significance annotations

Detect sweep data by checking for:
- Multiple unique values in columns matching `governance.*`
- A `run_index` column with values > 0
- No `epoch` column (sweep CSVs aggregate across epochs)

## Concrete implementation

For single-run time-series data:

- `python examples/plot_run.py <run_dir> [--metric <metric>]`

For sweep data:

- `python examples/plot_sweep.py <csv_path> [--output-dir <dir>]`

The sweep script auto-detects `governance.*` columns with multiple unique values and generates the 6 standard plot types listed above.
