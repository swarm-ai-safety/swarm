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

## Concrete implementation

This repo provides a plotting script:

- `python examples/plot_run.py <run_dir> [--metric <metric>]`
