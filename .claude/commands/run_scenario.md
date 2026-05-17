# /run_scenario

Run a single SWARM scenario with a single seed and standardized, reproducible run folder. Optionally generate plots immediately after. Distinct from `/sweep` (parameter grid across multiple configs), `/benchmark` (standardized multi-condition evaluation suite), and `/full_study` (end-to-end pipeline including analysis and paper).

Consolidates the former `/run_and_plot` command (now `/run_scenario --plot`).

## Usage

`/run_scenario <scenario_path_or_id> [seed] [epochs] [steps] [--plot] [--engine swarm|miroshark] [--scale N] [--max-rounds N]`

Examples:
- `/run_scenario baseline`
- `/run_scenario scenarios/collusion_detection.yaml 123`
- `/run_scenario emergent_capabilities 42 15 20`
- `/run_scenario baseline --plot` (run + generate plots)
- `/run_scenario scenarios/strict_governance.yaml 42 15 12 --plot`
- `/run_scenario adversarial_redteam --engine miroshark --scale 5 --max-rounds 10` (run via the MiroShark social-cascade engine)

## Argument parsing

Parse `$ARGUMENTS` to extract:
- `--plot`: After the run completes, generate plots from the results (see Phase 2 below)
- `--engine <swarm|miroshark>`: Which engine to run on. Default: `swarm` (the in-tree game-theoretic engine). `miroshark` translates the scenario into a MiroShark seed briefing and runs through the MiroShark backend at `$MIROSHARK_API_URL` (default `http://localhost:5001`).
- `--scale N` (only with `--engine miroshark`): Multiply each agent-type count by `N` before generating the briefing roster. Default 20.
- `--max-rounds N` (only with `--engine miroshark`): Cap the MiroShark simulation length. Default 30.
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

### Phase 1b: MiroShark engine (only if `--engine miroshark` is set)

Skip Phase 1's `python -m swarm run` and instead invoke the SWARM→MiroShark bridge:

- `python -m swarm.bridges.miroshark <scenario.yaml> --scale <scale> --max-rounds <max_rounds> --runs-root runs`

The bridge will:
1. Translate the scenario YAML into a markdown seed briefing (with a named-agent roster scaled by `--scale`).
2. POST the seed to `/api/graph/ontology/generate`, then `/api/graph/build`, on the MiroShark backend.
3. `/api/simulation/create` → `/prepare` (poll) → `/start` → poll `run-status` to completion.
4. Save into `runs/<YYYYMMDD-HHMMSS>_<scenario_id>_miroshark/`: `seed_document.md`, `scenario.json`, `config.json`, `ontology.json`, `graph_build.json`, `prepare.json`, `start.json`, `run_final_status.json`, `export.json`.

Requires the MiroShark backend reachable at `$MIROSHARK_API_URL` (default `http://localhost:5001`). Skip `--plot` when using `--engine miroshark` — the SWARM plotting script does not consume MiroShark exports.

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
