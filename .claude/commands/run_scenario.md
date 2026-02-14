# /run_scenario

Run a SWARM scenario with a standardized, reproducible run folder.

## Usage

`/run_scenario <scenario_path_or_id> [seed] [epochs] [steps]`

Examples:
- `/run_scenario baseline`
- `/run_scenario scenarios/collusion_detection.yaml 123`
- `/run_scenario emergent_capabilities 42 15 20`

## Behavior

1) Resolve `<scenario_path_or_id>`:
- If it contains `/` or ends with `.yaml`, treat it as a path.
- Otherwise resolve to `scenarios/<id>.yaml`.

2) Create a run directory:
- `runs/<YYYYMMDD-HHMMSS>_<scenario_id>_seed<seed>/`

3) Run the scenario via the project CLI (preferred):
- `python -m swarm run <scenario.yaml> --seed <seed> --epochs <epochs> --steps <steps> --export-json <run_dir>/history.json --export-csv <run_dir>/csv --export-dolt`

4) If the scenario YAML declares `outputs.event_log` or `outputs.metrics_csv`, copy those artifacts into `<run_dir>/artifacts/` (do not modify the scenario file in-place).

5) Print a short, PR-ready summary:
- Scenario id, seed, epochs, steps
- Total interactions, accepted interactions, avg toxicity, final welfare
- Paths written under `runs/...`

## Constraints / invariants

- Never overwrite an existing `runs/<...>/` directory; if it exists, create a new run id.
- Keep the run folder self-contained enough to reproduce plots later (history JSON + CSV exports at minimum).

