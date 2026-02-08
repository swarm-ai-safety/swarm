# /add_scenario

Scaffold a new YAML scenario with SWARM conventions.

## Usage

`/add_scenario <scenario_id> [motif]`

## Behavior

1) Create `scenarios/<scenario_id>.yaml` with:
- `scenario_id`, `description`, `motif`
- `agents` minimal list
- `simulation` (include `seed`)
- `governance` section (explicitly set the main toggles)
- `rate_limits` section
- `payoff` section
- `outputs` section that writes to `logs/<scenario_id>_events.jsonl` and `logs/<scenario_id>_metrics.csv`

2) Ensure the scenario is loadable:
- `python -c "from pathlib import Path; from swarm.scenarios.loader import load_scenario; load_scenario(Path('scenarios/<scenario_id>.yaml')); print('OK')"`

3) Add a short note to `docs/scenarios.md` describing what mechanism the scenario isolates.

