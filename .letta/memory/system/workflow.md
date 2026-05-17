---
description: "Standard operating procedures for experiment lifecycle"
---

# Workflow

## Session open

1. Check `bd ready` for available work
2. Check `git status` for uncommitted changes
3. Review active threads in `threads/` memory
4. Ask: "What's the active thread?" — recall current hypothesis and next experiment

## Experiment lifecycle

```
hypothesis --> scenario YAML --> sanity run (short) --> full sweep --> synthesize --> claim update --> report
```

### 1. Design
- State hypothesis as a testable proposition
- Write or update scenario YAML in `scenarios/`
- Define success/failure metrics before running

### 2. Sanity check
- Short run: 1-3 seeds, 10 epochs, 10 steps
- Verify no crashes, metrics are in expected range
- Check: does the scenario test what we think it tests?

### 3. Full run
- Run with publication-quality seed count
- Generate run.yaml manifest automatically
- Log to SQLite: `python -m swarm log <run_dir>`

### 4. Synthesize
- Run `/synthesize <run_id>` to generate experiment note
- Review claim update recommendations
- Update claims manually if evidence warrants

### 5. Report
- Generate plots, tables, markdown summary
- Link to claims affected
- Update active thread in memory

## Session close

Before ending any session:

1. `git status` — check what changed
2. Stage and commit code changes
3. `bd sync` — commit beads changes
4. Update active thread memory:
   - What changed this session?
   - What did we learn?
   - What's the next experiment?
   - Any blockers?
5. `git push`

## Artifact contracts

Every run MUST produce:
- `runs/<run_id>/run.yaml` — metadata envelope (see RESEARCH_OS_SPEC.md)
- `runs/<run_id>/summary.json` — aggregate metrics
- At least one of: CSV data, plots, report

Every claim MUST have:
- Testable proposition as title
- At least one evidence entry with valid run_id
- Boundary conditions
- Confidence justified by evidence quality
