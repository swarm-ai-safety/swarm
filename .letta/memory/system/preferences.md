---
description: "Stable user preferences — coding style, plotting conventions, workflow habits"
---

# Preferences

## Coding style

- Python 3.11+, type hints on public APIs
- ruff for linting, mypy for type checking
- Prefer deterministic tests (set seeds, constrain inputs) over loose assertions
- No over-engineering — minimum complexity for current task

## Plotting conventions

- matplotlib with seaborn styling
- Save to `runs/<run_id>/plots/` as PNG
- Include axis labels, titles, and legends
- Use consistent color palette: honest=blue, adversarial=red, adaptive=orange
- Error bars or confidence intervals on all aggregate plots

## Naming conventions

- Run folders: `YYYYMMDD-HHMMSS_<slug>`
- Claim files: `claim-<kebab-case-proposition>.md` (max 80 chars)
- Scenario files: `<descriptive_name>.yaml`
- Branch names: `feature/<short-description>` or `session/pane-N`

## Default sweep ranges

- Seeds: 10 for exploratory, 50 for publication-quality
- Adversarial fraction: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
- Transaction tax: [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15]
- Epochs: 10 (default), 50 (long-horizon studies)

## What counts as publishable

- Bonferroni-corrected significance with effect size (Cohen's d)
- Replicated across >= 2 independent seed sets
- Boundary conditions documented
- Reproduction command included
- p-hacking audit passed
