# /run_scenario

Run a single SWARM scenario with a single seed and standardized, reproducible run folder. Optionally generate plots immediately after. Distinct from `/sweep` (parameter grid across multiple configs), `/benchmark` (standardized multi-condition evaluation suite), and `/full_study` (end-to-end pipeline including analysis and paper).

Consolidates the former `/run_and_plot` command (now `/run_scenario --plot`).

## Usage

`/run_scenario <scenario_path_or_id> [seed] [epochs] [steps] [--plot]`

Examples:
- `/run_scenario baseline`
- `/run_scenario scenarios/collusion_detection.yaml 123`
- `/run_scenario emergent_capabilities 42 15 20`
- `/run_scenario baseline --plot` (run + generate plots)
- `/run_scenario scenarios/strict_governance.yaml 42 15 12 --plot`

## Argument parsing

Parse `$ARGUMENTS` to extract:
- `--plot`: After the run completes, generate plots from the results (see Phase 2 below)
- Positional args: `<scenario_path_or_id> [seed] [epochs] [steps]`

## Behavior

### Phase 1: Run the scenario

1) Resolve `<scenario_path_or_id>`:
- If it contains `/` or ends with `.yaml`, treat it as a path.
- Otherwise resolve to `scenarios/<id>.yaml`.

2) Create a run directory:
- `runs/<YYYYMMDD-HHMMSS>_<scenario_id>_seed<seed>/`

3) Run the scenario via the project CLI (preferred):
- `python -m swarm run <scenario.yaml> --seed <seed> --epochs <epochs> --steps <steps> --export-json <run_dir>/history.json --export-csv <run_dir>/csv --export-dolt`

4) If the scenario YAML declares `outputs.event_log` or `outputs.metrics_csv`, copy those artifacts into `<run_dir>/artifacts/` (do not modify the scenario file in-place).

### Phase 2: Generate plots (only if `--plot` is set)

5) Run the plotting script against the run directory:
- `python examples/plot_run.py <run_dir>`

6) Display the generated plot images inline (read each PNG).

### Phase 3: Summary

7) Print a short, PR-ready summary:
- Scenario id, seed, epochs, steps
- Total interactions, accepted interactions, avg toxicity, final welfare
- Paths written under `runs/...`
- List of generated plots (if `--plot` was used)
- Success criteria pass/fail (if `--plot` was used)

## Constraints / invariants

- Never overwrite an existing `runs/<...>/` directory; if it exists, create a new run id.
- Keep the run folder self-contained enough to reproduce plots later (history JSON + CSV exports at minimum).
- If plotting fails (missing deps), still report the run results and write a fallback README in the plots directory.

## Migration from old commands

| Old command | Equivalent |
|---|---|
| `/run_and_plot baseline` | `/run_scenario baseline --plot` |
| `/run_and_plot scenarios/strict_governance.yaml 42 15 12` | `/run_scenario scenarios/strict_governance.yaml 42 15 12 --plot` |
