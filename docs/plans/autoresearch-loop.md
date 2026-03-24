# Building a SWARM "Autoresearch" Loop

> Status: Implemented MVP CLI as `python -m swarm autoresearch` for local governance-loop optimization.

This document adapts the core idea behind Karpathy's `autoresearch` pattern to SWARM's multi-agent governance setting.

## What it is

An autoresearch loop is a tight optimization cycle where an agent:

1. Reads a human-written objective.
2. Proposes and applies a small code/config change.
3. Runs a bounded experiment.
4. Scores the result against a target metric.
5. Commits the change with rationale.
6. Repeats.

In SWARM, this maps naturally to scenario/governance iteration rather than only training-script iteration.

## SWARM-specific mapping

- **Objective spec**: `program.md` or scenario-local objective file.
- **Editable surface**: `scenarios/*.yaml`, `swarm/governance/*`, optional `swarm/agents/*` toggles.
- **Execution**: `python -m swarm run <scenario> --epochs <n> --steps <n> --seed <s>`.
- **Evaluation**: track target metrics from exported JSON/CSV (`toxicity_rate`, `quality_gap`, `total_welfare`, `illusion_delta` where available).
- **Versioning**: one git commit per loop with machine-generated rationale + metric delta.

## Recommended first implementation (MVP)

### Inputs

- `program.md` with:
  - objective statement
  - primary metric and direction
  - guardrail metrics and hard constraints
  - time budget / max iterations
- A baseline scenario YAML.

### Loop steps

1. Run baseline and capture score.
2. Ask an LLM to propose a minimal patch (single mechanism change).
3. Apply patch in a temporary branch/worktree.
4. Run a short evaluation (e.g., 3 epochs x 5 steps, fixed seed set).
5. Compare against acceptance policy:
   - improve primary metric
   - do not violate guardrails
6. If accepted:
   - commit with a structured message containing metric diff
   - optionally run a longer confirmation eval
7. Repeat until budget reached.

## Acceptance policy template

```text
PRIMARY: minimize quality_gap
REQUIRE: quality_gap improves by >= 0.02 absolute
GUARDRAILS:
- toxicity_rate must not increase by > 0.01
- total_welfare must not decrease by > 5%
```

## Why this fits SWARM

Compared to pure training-loss optimization, SWARM needs mechanism-level clarity and reproducibility:

- Changes can target governance levers directly.
- Multi-metric acceptance prevents narrow over-optimization.
- Scenario-level iteration keeps experiments cheap and interpretable.

## Safety and rigor requirements

- Use fixed seeds (or a fixed small seed panel) in the inner loop.
- Keep inner-loop runs short; require periodic longer validation runs.
- Store all run artifacts (`--export-json`, `--export-csv`) per iteration.
- Reject edits touching unrelated files.
- Use one-change-per-commit to keep causal attribution clear.

## Suggested CLI extension

Current CLI (available now):

```bash
python -m swarm autoresearch \
  --objective program.md \
  --scenario scenarios/baseline.yaml \
  --iterations 20 \
  --eval-epochs 3 \
  --eval-steps 5 \
  --seeds 7,11,19 \
  --export-root runs/autoresearch
```

## Implemented MVP scope

- Objective parsing from YAML/JSON or markdown fenced YAML (`--objective`).
- Local mutation/evaluation loop over governance levers.
- Acceptance policy with primary metric + guardrail regressions.
- Structured ledger output to `runs/autoresearch/summary.json`.
- Optional `--auto-commit` for committing the summary artifact.
