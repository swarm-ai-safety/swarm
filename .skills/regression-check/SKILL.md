---
name: regression-check
description: Re-run the baseline suite and compare against the last known-good run. Flags metric drift, test failures, or behavioral changes.
version: "1.0"
metadata:
  author: swarm-research-os
  generated_from: research-os-v0.1
allowed-tools: Read Bash Glob Grep
---

## EXECUTE NOW

**Mode: $ARGUMENTS**

Parse:
- `tests` — run pytest suite only
- `baseline` — re-run baseline scenario and compare metrics
- `full` or empty — run both tests and baseline comparison

---

## Step 1: Run tests

```bash
python -m pytest tests/ -v --tb=short
```

Report pass/fail count. If any failures, list them.

## Step 2: Run baseline scenario

```bash
python -m swarm run scenarios/baseline.yaml --seed 42 --epochs 10 --steps 10
```

## Step 3: Compare against last known-good

Read the most recent baseline run from memory at `.letta/memory/runs/latest.md`.

If a previous baseline exists, compare:
- Welfare: delta and percentage change
- Toxicity: delta
- Acceptance rate: delta
- Quality gap: delta

Flag any metric that changed by more than 10% as a potential regression.

## Step 4: Report

```
## Regression Check

### Tests
- {passed}/{total} passed
- Failures: {list or "none"}

### Baseline comparison
| Metric | Previous | Current | Delta | Status |
|--------|----------|---------|-------|--------|
| Welfare | ... | ... | ... | OK/DRIFT |
| Toxicity | ... | ... | ... | OK/DRIFT |

### Verdict
{CLEAN — no regressions detected}
{REGRESSED — {details}}
```
