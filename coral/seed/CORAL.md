# CORAL Agent Instructions — SWARM

You are working on the **SWARM** distributional-safety simulation framework.
Read `CLAUDE.md` at the repo root for full architecture details, commands, and invariants.

## Quick Reference

```bash
# Install
python -m pip install -e ".[dev,runtime]"

# Test
python -m pytest tests/ -v

# Lint
ruff check swarm/ tests/

# Type check
python -m mypy swarm/

# Run a scenario
python -m swarm run scenarios/baseline.yaml --seed 42 --epochs 10 --steps 10
```

## Key Invariants (do NOT break)

- `p` (probability of beneficial interaction) must stay in `[0, 1]` everywhere.
- Event logs (`*.jsonl`) are append-only and replayable.
- Runs must be reproducible from: scenario YAML + seed + exported history/CSVs.

## Evaluation Criteria

Your changes are graded on a composite score:

| Check             | Weight | Tool                              |
|-------------------|--------|-----------------------------------|
| Test pass rate    | 60%    | `python -m pytest tests/ -v`      |
| Lint cleanliness  | 20%    | `ruff check swarm/ tests/`        |
| Type safety       | 20%    | `python -m mypy swarm/`           |

Run `uv run coral eval -m "description of change"` to evaluate your current state.

## Architecture Overview

```
Observables -> ProxyComputer -> v_hat -> sigmoid -> p -> SoftPayoffEngine -> payoffs
                                                    |
                                              SoftMetrics -> toxicity, quality gap, etc.
```

- **`swarm/core/proxy.py`** — `ProxyComputer`: observables -> `v_hat` -> `p`
- **`swarm/core/payoff.py`** — `SoftPayoffEngine`: soft-label payoff calculations
- **`swarm/metrics/soft_metrics.py`** — `SoftMetrics`: probabilistic quality metrics
- **`swarm/metrics/reporters.py`** — Dual soft/hard metric reporting
- **`swarm/logging/event_log.py`** — Append-only JSONL event logger

## Sharing Knowledge

- Write notes to `.coral/public/notes/` to share insights with other agents.
- Check `.coral/public/attempts/` to see what other agents have tried.
- Check `.coral/public/skills/` for reusable patterns discovered during the run.
