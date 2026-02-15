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
4) **Auto-generate `summary.json`**: After the CSV is written, read it with pandas, group by parameter columns, and compute per-config summary statistics:
```python
import pandas as pd, json
from pathlib import Path

df = pd.read_csv(csv_path)
run_dir = Path(run_dir)

# Identify parameter columns (governance.* or any column with few unique values)
param_cols = [c for c in df.columns if c.startswith("governance.")]
grouped = df.groupby(param_cols)

configs = []
for name, group in grouped:
    config = dict(zip(param_cols, name if isinstance(name, tuple) else [name]))
    config["n_runs"] = len(group)
    for metric in ["welfare", "toxicity_rate", "quality_gap", "honest_payoff", "adversarial_payoff", "opportunistic_payoff"]:
        if metric in group.columns:
            config[f"mean_{metric}"] = round(group[metric].mean(), 4)
            config[f"std_{metric}"] = round(group[metric].std(), 4)
    configs.append(config)

summary = {
    "scenario": scenario_id,
    "seed_base": seed,
    "epochs": epochs,
    "runs_per_config": runs_per_config,
    "total_runs": len(df),
    "param_combinations": len(configs),
    "configs": configs,
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
```
This eliminates the manual post-processing step that was previously required before running `/council_review`.

5) Print a compact table of the best/worst configs by:
- `mean_welfare` (higher is better)
- `mean_toxicity` (lower is better)

## Notes

- Keep sweeps small by default; prefer fewer epochs/runs unless explicitly requested.
- The `summary.json` file is consumed by `/council_review` — always generate it.

