---
description: "Pointers to recent runs — updated each session, never stores raw data"
---

# Recent Run Pointers

## Latest runs (by date)

| Date | Run ID | Type | Key Finding |
|------|--------|------|-------------|
| 2026-02-21 | redteam_contract_screening_full | redteam | full governance stack prevents most attack types |
| 2026-02-21 | langgraph_governed | sweep | trust boundaries modify but never deny handoffs |
| 2026-02-21 | contract_screening_sweep | sweep | screening achieves perfect type separation |
| 2026-02-17 | memori_study | study | (check run.yaml for details) |
| 2026-02-14 | kernel_v4_code_sweep | sweep | (check run.yaml for details) |

## Run index location

- Full catalog: `/Users/raelisavitt/swarm-artifacts/run-index.yaml` (117 runs)
- Local runs: `runs/` in main repo (gitignored, UUID-based)
- Tracked runs: `swarm-artifacts/runs/` (YYYYMMDD-HHMMSS_slug format)

## How to query

```bash
# Search by tag
grep -A5 "collusion" /Users/raelisavitt/swarm-artifacts/run-index.yaml

# Search by date range
grep "date: '2026-02-2" /Users/raelisavitt/swarm-artifacts/run-index.yaml

# Read a specific run's metadata
cat /Users/raelisavitt/swarm-artifacts/runs/<run_id>/run.yaml
```

## Active claims

Pointers only — source of truth is `swarm-artifacts/vault/claims/`:

- claim-collusion-detection-reduces-ring-damage-75pct
- claim-sybil-attacks-resist-full-governance-stack
- claim-contract-screening-achieves-perfect-type-separation
- claim-trust-boundaries-modify-but-never-deny-handoffs
- claim-full-governance-stack-prevents-most-attack-types
