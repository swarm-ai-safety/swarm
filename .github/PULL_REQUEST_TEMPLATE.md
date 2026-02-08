## Summary

Brief description of the changes.

## Changes

- ...
- ...

## Test Plan

- [ ] Existing tests pass (`pytest tests/ -v`)
- [ ] Lint passes (`ruff check swarm/ tests/`)
- [ ] Type check passes (`mypy swarm/`)
- [ ] New tests added (if applicable)

## Results (SWARM)

If this change affects scenarios, metrics, agents, governance, or evaluation, include a reproducible run folder and the headline deltas.

**Reproduce**
- Scenario: `scenarios/<name>.yaml`
- Command: `python -m swarm run scenarios/<name>.yaml --seed <seed> --epochs <n> --steps <n> --export-json runs/<run_id>/history.json --export-csv runs/<run_id>/csv`
- Plots: `python examples/plot_run.py runs/<run_id>`

**Artifacts**
- `runs/<run_id>/history.json`
- `runs/<run_id>/csv/`
- `runs/<run_id>/plots/`

**Headline metrics**
- Total interactions:
- Accepted interactions:
- Avg toxicity:
- Final welfare:

**Invariants**
- [ ] `p ∈ [0, 1]` everywhere it is surfaced/logged
- [ ] Event logs remain replayable (append-only JSONL)

## Related Issues

Closes #
