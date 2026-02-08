# Claude Code (Project Template)

This repository ships a shared Claude Code setup so contributors (human or agentic) run simulations and report results consistently.

## What’s included

- Slash commands: `.claude/commands/`
  - `/run_scenario` – run a scenario into a standardized `runs/` folder
  - `/sweep` – parameter sweeps with run-folder outputs
  - `/plot` – generate standard plots for issues/PRs (`python examples/plot_run.py ...`)
  - `/add_scenario` – scaffold a YAML scenario
  - `/add_metric` – implement and wire a metric
  - `/red_team` – red-team evaluation summary outputs
  - `/install_hooks` – install optional git hooks
- Research-role “specialist agents”: `.claude/agents/`
- Optional git hooks (research hygiene): `.claude/hooks/`
- MCP config stub: `.mcp.json`

## Permissions profiles

The default `.claude/settings.json` is a **safe baseline** (read-only tooling: `git status`, `git diff`, `ruff`, `mypy`).

If you want the full power-user profile (tests, installs, scenario runs, git push), replace it with:

- `.claude/settings.power.json`

(You can copy it over `.claude/settings.json` locally. Do not commit local overrides unless you intend them to be the new default.)

## Security notes

- Treat prompt audit logs as secrets if you enable them. Store them under `runs/` when possible and keep permissions tight.
- When using the Claude Code bridge service, always set `SWARM_BRIDGE_API_KEY` and keep `HOST` on loopback.

## Run folder convention

Write experiment artifacts to:

- `runs/<timestamp>_<scenario>_seed<seed>/history.json`
- `runs/<timestamp>_<scenario>_seed<seed>/csv/`
- `runs/<timestamp>_<scenario>_seed<seed>/plots/`

`runs/` is ignored by git (see `.gitignore`). For scenarios that write to `logs/`, prefer copying the relevant artifacts into the run folder when reporting results.

## PR results snippet

The repository PR template (`.github/PULL_REQUEST_TEMPLATE.md`) includes a “Results (SWARM)” section that references the `runs/` convention.

## Optional git hooks

Hooks live in `.claude/hooks/` and are installed into `.git/hooks/` via `/install_hooks`.

- `pre-commit`: ruff, mypy, YAML scenario load sanity, a fast smoke sim
- `pre-push`: `make ci`

Emergency bypass:

- `SKIP_SWARM_HOOKS=1 git commit ...`

## MCP integrations

`.mcp.json` contains safe-by-default placeholders for common MCP servers (e.g. GitHub, SQLite).

- Use environment variables (e.g. `GITHUB_TOKEN`, `SWARM_RUNS_DB_PATH`) rather than committing secrets.
- Versions are pinned in the template for supply-chain safety (GitHub via `npx`, SQLite via `uvx`).
- If you don’t use MCP, you can ignore this file.
