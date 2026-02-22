---
name: experiment-loop
description: Run the full experiment lifecycle — design, sanity check, full sweep, synthesize, and claim review. Given a hypothesis, proposes sweep matrix, runs the experiment, generates artifacts, and updates the research log.
version: "1.0"
metadata:
  author: swarm-research-os
  generated_from: research-os-v0.1
allowed-tools: Read Write Edit Glob Grep Bash
---

## EXECUTE NOW

**Hypothesis: $ARGUMENTS**

If no hypothesis provided, check memory at `.letta/memory/threads/current.md` for the active thread.

---

## Step 1: Design

1. State the hypothesis as a testable proposition
2. Identify which governance knobs and scenario parameters are relevant
3. Propose the sweep matrix:
   - Parameter(s) to sweep and their values
   - Number of seeds (10 for exploratory, 50 for publication)
   - Epochs and steps per epoch
4. Check if a similar scenario already exists in `scenarios/`:
   - If yes, propose modifications
   - If no, draft a new scenario YAML
5. Define success/failure metrics before running

Present the design and wait for user approval before proceeding.

## Step 2: Sanity check

Run a short version first:
```bash
python -m swarm run scenarios/<name>.yaml --seed 42 --epochs 10 --steps 10
```

Verify:
- No crashes or exceptions
- Metrics are in expected range
- The scenario tests what we think it tests

If sanity check fails, diagnose and fix before proceeding.

## Step 3: Full run

Run the full experiment:
```bash
python -m swarm sweep scenarios/<name>.yaml --seeds <N>
```

Or for single runs:
```bash
python -m swarm run scenarios/<name>.yaml --seed 42 --epochs <E> --steps <S>
```

After completion:
1. Verify `run.yaml` was generated in the output directory
2. Log to SQLite: check if `/log_run` should be invoked
3. Copy run folder to swarm-artifacts if publication-quality

## Step 4: Synthesize

1. Run the synthesize skill on the completed run
2. Review claim update recommendations
3. Present findings to user

## Step 5: Update memory

Append to `.letta/memory/threads/research-log.md`:
```
## {date} — {hypothesis short name}

**Ran:** {experiment description}
**Found:** {key results with effect sizes}
**Learned:** {what this changes about our understanding}
**Next:** {what to investigate next}
**Run pointers:** {run_id}
```

Update `.letta/memory/threads/current.md` with next steps.
Update `.letta/memory/runs/latest.md` with the new run pointer.
