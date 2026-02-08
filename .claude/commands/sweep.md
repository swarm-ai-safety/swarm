# /sweep

Run a parameter sweep and write results to a standardized run folder.

## Usage

`/sweep [scenario_path] [output_name]`

Examples:
- `/sweep` (defaults to `scenarios/baseline.yaml`)
- `/sweep scenarios/boundary_test.yaml boundary_sweep`

## Behavior

1) Create `runs/<YYYYMMDD-HHMMSS>_sweep/`.
2) Execute the built-in sweep example:
- `python examples/parameter_sweep.py --scenario <scenario_path> --output <run_dir>/<output_name>.csv`
3) If pandas+pyarrow are available, also write Parquet:
- `<run_dir>/<output_name>.parquet`
4) Print a compact table of the best/worst configs by:
- `mean_welfare` (higher is better)
- `mean_toxicity` (lower is better)

## Notes

- Keep sweeps small by default; prefer fewer epochs/runs unless explicitly requested.

