---
description: "Repository layout and how to navigate the SWARM codebase"
---

# Repository Map

## Main repo: distributional-agi-safety

```
swarm/                          # Core simulation framework
  core/proxy.py                 # ProxyComputer: observables -> v_hat -> p
  core/payoff.py                # SoftPayoffEngine: p -> payoffs
  metrics/soft_metrics.py       # Probabilistic metrics (toxicity, quality gap)
  metrics/reporters.py          # Dual soft/hard metric reporting
  logging/event_log.py          # Append-only JSONL logger

scenarios/                      # ~80 scenario YAMLs
tests/                          # pytest suite
examples/                       # 50+ runner scripts, Jupyter notebooks
runs/                           # Run output (gitignored, uses UUIDs locally)
docs/                           # Blog, research posts

.claude/commands/               # 55 slash commands
.claude/agents/                 # 6 specialist agents
.beads/                         # Issue/task tracking (beads)
```

## Artifacts repo: swarm-artifacts

```
/Users/raelisavitt/swarm-artifacts/
  runs/                         # 117 indexed runs (YYYYMMDD-HHMMSS_slug format)
  run-index.yaml                # Auto-generated catalog of all runs
  vault/                        # Knowledge vault (claims, experiments, governance, etc.)
  schemas/                      # JSON Schema for run.yaml, claims, sweeps
  skills/                       # Agent Skills (claim, synthesize, vault-init, verify)
  scripts/                      # Automation (validate, index, backfill)
  research/                     # Papers, posts, notes
  RESEARCH_OS_SPEC.md           # Artifact contracts spec v0.1
```

## Key commands

```bash
python -m swarm run scenarios/<name>.yaml --seed 42 --epochs 10 --steps 10
python -m swarm sweep scenarios/sweeps/<name>.yaml --seeds 50
python -m pytest tests/ -v
ruff check swarm/ tests/
```

## Data flow

```
Observables -> ProxyComputer -> v_hat -> sigmoid -> p -> SoftPayoffEngine -> payoffs
                                                    |
                                              SoftMetrics -> toxicity, quality gap
```

## Domain vocabulary

- **p**: P(v=+1), probability interaction is beneficial, always in [0,1]
- **v_hat**: raw proxy score before sigmoid, in [-1,+1]
- **adverse selection**: low-quality interactions preferentially accepted (quality_gap < 0)
- **externality internalization**: rho parameters control cost-bearing
