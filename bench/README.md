# SWARM SkillsBench

A benchmark for evaluating whether curated skills (slash commands) help AI agents perform SWARM AI safety research tasks, built on the [SkillsBench](https://arxiv.org/abs/2602.12670) methodology.

## Overview

This benchmark contains 10 tasks of varying difficulty that test an agent's ability to:
- Run simulations and export artifacts
- Conduct parameter sweeps
- Perform rigorous statistical analysis
- Generate visualizations
- Scaffold research papers
- Compose multiple skills into end-to-end studies

Each task is packaged as a Docker container with instructions, a verification test suite, and an oracle solution.

## Task Inventory

| # | Task | Difficulty | Skills Tested |
|---|------|-----------|---------------|
| 1 | `swarm-run-scenario` | Easy | run-scenario |
| 2 | `swarm-multi-seed-repro` | Easy | run-scenario |
| 3 | `swarm-sweep-analysis` | Medium | parameter-sweep |
| 4 | `swarm-sweep-to-plots` | Medium | plotting |
| 5 | `swarm-toxicity-diagnosis` | Medium | statistical-analysis |
| 6 | `swarm-statistical-rigor` | Hard | statistical-analysis |
| 7 | `swarm-adverse-selection` | Hard | run-scenario, statistical-analysis |
| 8 | `swarm-governance-tuning` | Hard | parameter-sweep, statistical-analysis |
| 9 | `swarm-paper-scaffold` | Hard | paper-writing |
| 10 | `swarm-end-to-end-study` | Hard | all 5 skills |

## Skills (5 total)

Skills are packaged as `SKILL.md` files following the SkillsBench format:

1. **run-scenario** — Execute SWARM simulations with standardized output
2. **parameter-sweep** — Run parameter grid searches with summary generation
3. **statistical-analysis** — Rigorous stats (t-tests, effect sizes, corrections)
4. **plotting** — Generate publication-quality visualizations
5. **paper-writing** — Scaffold research papers from run data

## Running the Benchmark

### Prerequisites

- [Harbor CLI](https://github.com/av/harbor) installed
- Docker running
- Python 3.12+

### Quick Start

```bash
# Install dependencies
cd bench && pip install -e .

# Validate task format
harbor tasks check tasks/

# Run a single task (no skills)
harbor trials start --tasks tasks/swarm-run-scenario --agent claude-code

# Run with skills
harbor trials start --tasks tasks/swarm-run-scenario --agent claude-code --skills skills/

# Run full benchmark
harbor trials start --tasks tasks/ --agent claude-code --skills skills/ --runs 3
```

### Analyzing Results

```bash
# Compare pass rates
harbor trials report --compare no-skills,with-skills

# Per-task breakdown
harbor trials report --by-task
```

## Task Structure

Each task directory contains:

```
tasks/<task-name>/
├── instruction.md          # Task prompt given to the agent
├── task.toml               # Metadata (difficulty, skills, timeout)
├── environment/
│   └── Dockerfile          # Docker build context
├── solution/
│   └── solve.sh            # Oracle solution (must pass all tests)
└── tests/
    ├── test.sh             # Test runner entry point
    └── test_outputs.py     # Python verification script
```

## Pre-loaded Fixtures

Medium and hard tasks use pre-generated data in `fixtures/`:
- `sweep_results.csv` — 80-row sweep CSV (5 tax rates x 4 seeds x 4 agent types)
- `sweep_results_small.csv` — 12-row sweep CSV for medium tasks
- `runs.db` — SQLite database with 5 scenarios x 3 seeds

These are copied into Docker containers during build.

## Development

```bash
# Test an oracle solution locally
cd tasks/swarm-run-scenario
bash solution/solve.sh
python tests/test_outputs.py

# Regenerate fixtures
python fixtures/generate_fixtures.py
```
