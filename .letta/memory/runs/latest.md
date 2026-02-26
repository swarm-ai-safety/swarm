---
description: "Pointers to recent runs — updated each session, never stores raw data"
---

# Recent Run Pointers

## Latest runs (by date)

| Date | Run ID | Type | Key Finding |
|------|--------|------|-------------|
| 2026-02-25 | 20260224-220829_mesa_governance_study | mesa_sweep | rho sweet spot [0.3,0.7]: toxicity ↓34% (sig), welfare cost non-sig |
| 2026-02-22 | evo_game_prisoners_seed999 | evo_game | Sharp phase transition pattern, final welfare 34.24 |
| 2026-02-22 | evo_game_prisoners_seed314 | evo_game | Most volatile — welfare peaked then fell to 14.99 |
| 2026-02-22 | evo_game_prisoners_seed777 | evo_game | Gradual ramp, epoch 3 acceptance crash (37.5%) |
| 2026-02-22 | evo_game_prisoners_seed123 | evo_game | Gradual ramp, no early crash, final welfare 25.62 |
| 2026-02-22 | evo_game_prisoners_seed42 | evo_game | Sharp phase transition, TFT 3-5x reputation, welfare 32.43 |
| 2026-02-21 | redteam_contract_screening_full | redteam | full governance stack prevents most attack types |
| 2026-02-21 | langgraph_governed | sweep | trust boundaries modify but never deny handoffs |
| 2026-02-21 | contract_screening_sweep | sweep | screening achieves perfect type separation |
| 2026-02-17 | memori_study | study | (check run.yaml for details) |

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
- claim-reputation-mediated-cooperation-sustains-in-dominant-defect (NEW — 5-seed evo game study)
