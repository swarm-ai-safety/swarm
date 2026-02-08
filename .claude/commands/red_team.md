# /red_team

Run a red-team evaluation (adaptive/adversarial stress testing) and summarize failure modes.

## Usage

`/red_team [mode]`

Modes:
- `quick` (default): minimal set of attacks and one governance config
- `full`: full attack library (may be slow)

## Behavior

1) Create `runs/<YYYYMMDD-HHMMSS>_redteam/`.
2) Run the red-team evaluator from `swarm.redteam`:
- Instantiate `RedTeamEvaluator(governance_config=...)`
- Run quick/full evaluation
3) Write:
- `<run_dir>/report.json` (machine-readable)
- `<run_dir>/report.txt` (human summary)
4) Print a short summary suitable for an issue:
- Robustness score + grade
- Top vulnerabilities (severity + affected lever)
- Most effective attack vectors
- Recommended mitigations / next experiments

