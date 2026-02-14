# Self-Modification Governance Implementation Checklist

This checklist translates the governance architecture in
`self-modification-governance-byline.md` into concrete engineering work items.

## Scope and assumptions

- Runtime: SWARM orchestrator + governance engine.
- Goal: ship a minimally auditable two-gate modification loop before adding
  full compositional simulation and advanced rollout automation.
- Current calibration proxy mapping:
  - `tau_min` proxy -> `governance.refinery_p_threshold`
  - `K_max` proxy -> `governance.memory_write_rate_limit_per_epoch`

## Phase 0: Hardening prerequisites

- [ ] Define trust boundaries in code ownership:
  - [ ] immutable governance policy surfaces
  - [ ] mutable agent/runtime surfaces
- [ ] Add signed policy bundle loading path (hash + signer + version).
- [ ] Add policy-hash and artifact-hash fields to run metadata.
- [ ] Add failure mode: any attestation mismatch blocks promotion.

## Phase 1: Byline provenance foundation

- [ ] Define canonical Byline schema (JSON) with:
  - [ ] identity fields (`event_id`, `agent_id`, `run_id`, `timestamp`)
  - [ ] decision context references
  - [ ] verification outputs
  - [ ] lifecycle state transitions
  - [ ] integrity chain fields (`entry_hash`, `prev_hash`, signatures)
- [ ] Implement append-only storage API.
- [ ] Implement strict lifecycle transition validator.
- [ ] Add replay tool: reconstruct full modification history by `run_id`.
- [ ] Add tests:
  - [ ] schema validation
  - [ ] transition invalidation
  - [ ] hash-chain integrity detection

## Phase 2: Gate 1 (`tau_min`) implementation

- [ ] Implement deterministic `tau` computation from held-out metrics.
- [ ] Add metric-direction registry (`higher-is-better` vs `lower-is-better`).
- [ ] Add confidence-adjusted margin term (`eps_j` treatment).
- [ ] Implement policy thresholds by risk tier.
- [ ] Log gate decision artifacts:
  - [ ] raw metric values
  - [ ] normalized margins
  - [ ] final `tau` and threshold
- [ ] Add tests:
  - [ ] deterministic replay consistency
  - [ ] threshold boundary behavior
  - [ ] fail-closed on missing metrics

## Phase 3: Gate 2 (`K_max`) implementation

- [ ] Define deterministic complexity-weight model for modifications.
- [ ] Implement rolling-window budget accounting.
- [ ] Enforce cap with explicit outcomes:
  - [ ] permit
  - [ ] deny
  - [ ] escalate-to-human
- [ ] Add per-agent budget telemetry and reset/consolidation policy.
- [ ] Add tests:
  - [ ] cap exceed behavior
  - [ ] rolling-window expiration behavior
  - [ ] no negative/overflow accounting

## Phase 4: Deterministic risk-tier classifier

- [ ] Implement rule-based classifier over:
  - [ ] file/path surfaces
  - [ ] permission policy diffs
  - [ ] tool manifest diffs
  - [ ] dependency diffs
- [ ] Emit signed classifier report with matched rules.
- [ ] Add anti-bypass behavior:
  - [ ] unknown scope -> stricter tier
  - [ ] classifier failure -> stricter tier
  - [ ] transitive ownership escalation
- [ ] Add regression fixtures for tier assignment.

## Phase 5: Promotion workflow and rollout safety

- [ ] Implement staged path:
  - [ ] sandboxed test
  - [ ] shadow
  - [ ] canary-1
  - [ ] canary-2
  - [ ] full promotion
- [ ] Implement automatic rollback triggers with fixed thresholds.
- [ ] Implement kill-switch:
  - [ ] stop promotions
  - [ ] freeze affected populations
  - [ ] open incident record
- [ ] Add incident runbook and rollback SLO checks.

## Phase 6: Calibration and reproducibility

- [ ] Check in calibration script and runbook command.
- [ ] Store calibration artifacts under `runs/<timestamp>_tau_k_calibration/`.
- [ ] Require:
  - [ ] `runs.csv`
  - [ ] `summary.json`
  - [ ] `recommendation.json`
- [ ] Pin seed list and scenario in docs.
- [ ] Add smoke check that verifies both gates were exercised:
  - [ ] non-zero gate-hit count for `K_max` arm
  - [ ] non-zero rejection delta for stricter `tau` candidates

## Phase 7: Release criteria

- [ ] Byline completeness >= 99.9% for modification events.
- [ ] Deterministic replay success >= 95% on sampled events.
- [ ] Mean rollback latency < 10 minutes in fault-injection tests.
- [ ] No unresolved critical governance incident older than 24 hours.
- [ ] Documentation updated:
  - [ ] architecture doc
  - [ ] operator runbook
  - [ ] calibration instructions

## Current calibration snapshot

Latest run (seeded sweep):

- Artifacts:
  - `runs/20260214-020518_tau_k_calibration/runs.csv`
  - `runs/20260214-020518_tau_k_calibration/summary.json`
  - `runs/20260214-020518_tau_k_calibration/recommendation.json`
- Recommended values from that run:
  - `tau_min = 0.55`
  - `K_max = 6`

Reproduce:

```bash
python scripts/calibrate_tau_k_memory.py
```
