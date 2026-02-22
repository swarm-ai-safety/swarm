---
name: sanity-check
description: Quick validation run before committing to a full sweep. Runs a scenario with minimal seeds/epochs, checks for crashes and metric sanity, reports go/no-go.
version: "1.0"
metadata:
  author: swarm-research-os
  generated_from: research-os-v0.1
allowed-tools: Read Bash Glob
---

## EXECUTE NOW

**Scenario: $ARGUMENTS**

If no scenario provided, ask which scenario to check.

---

## Step 1: Validate scenario file

1. Check that the scenario file exists in `scenarios/`
2. Read and parse the YAML
3. Report: agent types, governance config, topology, epoch/step counts

## Step 2: Run short

```bash
python -m swarm run scenarios/$SCENARIO --seed 42 --epochs 5 --steps 5
```

## Step 3: Check results

1. Verify the run completed without exceptions
2. Read the output summary
3. Check metric sanity:
   - `p` values in [0, 1]
   - Welfare is finite and reasonable
   - Toxicity rate is in [0, 1]
   - Acceptance rate is in [0, 1]
   - No NaN or Inf values

## Step 4: Report

```
## Sanity Check: {scenario}

Status: GO / NO-GO
Runtime: {seconds}s

Metrics:
- Welfare: {value}
- Toxicity: {value}
- Acceptance rate: {value}
- Quality gap: {value}

{If NO-GO: describe the issue and suggest fix}
{If GO: ready for full sweep with --seeds N}
```
