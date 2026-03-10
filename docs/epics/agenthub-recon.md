---
description: "Research-scout integration plan for using AgentHub as a collaboration substrate for SWARM research communities."
---

# AgentHub Recon for SWARM (Research Scout)

*Prepared 2026-03-10 · Role: Research Scout*

## Source Basis and Confidence

This recon is based on a user-provided AgentHub project description (architecture + quick-start snippets), not direct source inspection from this runtime.

- **Observed (from provided description):** AgentHub is an agent-first collaboration platform with a bare git repo + message board, built as one Go server binary (`agenthub-server`), one thin CLI (`ah`), one SQLite DB, and one bare git repo on disk.
- **Inferred for SWARM integration:** Treat AgentHub as a coordination substrate above SWARM experiment execution and artifact governance.

## What AgentHub Is (Observed)

### Core product model

AgentHub is designed as a stripped-down GitHub-like environment for AI agents with:

- no main branch,
- no PRs,
- no merges,
- and a sprawling DAG of commits plus a message board for coordination.

The platform is intentionally generic: it does not encode research norms; those come from agent instructions/culture.

### Stated first use case

- Organization layer for autoresearch-style research agents.
- Community model where distributed contributors run agents and publish their results into a shared agent collaboration hub.

### Architecture primitives

- **Git layer**
  - Agents push code via git bundles.
  - Server validates/unbundles into a bare repo.
  - Agents can browse DAG topology (children/leaves/lineage) and diff commits.
- **Message board**
  - Channels + posts + threaded replies.
  - Unopinionated content (results, hypotheses, failures, coordination notes).
- **Auth/defense**
  - Per-agent API keys.
  - Rate limiting.
  - Bundle size limits.
- **Operational shape**
  - One Go server binary (`agenthub-server`).
  - One SQLite DB.
  - One bare repo.
  - Thin CLI (`ah`) over HTTP API.

## SWARM Fit: Why This Is Interesting

AgentHub complements SWARM at the **community orchestration layer**, while SWARM remains the **experiment/governance/measurement layer**.

- AgentHub can coordinate many autonomous research workers.
- SWARM can standardize scenario execution, governance interventions, metrics, and reproducible artifacts.

In short:
- **AgentHub:** who collaborates, what gets proposed, where hypotheses are negotiated.
- **SWARM:** how interventions are tested and how outcomes are measured credibly.

## Integration Strategy

## 1) Keep systems loosely coupled

Use SWARM as an execution backend and provenance engine; use AgentHub for branch-DAG collaboration and threaded coordination.

## 2) Add a dedicated bridge package

Recommended landing zone:

- `swarm/bridges/agenthub/`
  - `config.py`
  - `client.py`
  - `events.py`
  - `mapper.py`
  - `bridge.py`
  - `__init__.py`

## 3) Define adapter contract

Map AgentHub primitives into SWARM research primitives:

1. **Bundle/commit events** → SWARM run intents + provenance events.
2. **DAG lineage metadata** → experiment ancestry graph (baseline/intervention descendants).
3. **Channel posts/replies** → structured experiment notes (hypothesis, setup, anomalies, conclusions).
4. **Agent identity/API key** → SWARM actor IDs + governance policy scope.
5. **Server guardrails (rate/size limits)** → SWARM resource-control observables for robustness studies.

## 4) Minimal end-to-end flow

1. Agent posts hypothesis + intended scenario config on AgentHub message board.
2. Agent commits scenario/intervention change into AgentHub DAG.
3. Bridge ingests commit metadata and runs corresponding SWARM scenario with fixed seeds.
4. SWARM exports metrics/history/artifacts.
5. Bridge posts a structured "results receipt" reply (metrics deltas + artifact pointers + seed list).

## Experiment Plan (Baseline vs Intervention)

### Objective

Test whether AgentHub-mediated collaboration improves research throughput without degrading governance/safety outcomes.

### Baseline

Current SWARM workflow without AgentHub coordination (single-runner or existing bridge workflow).

### Intervention

SWARM runs triggered/organized through AgentHub commit+message workflows.

### Candidate scenarios

- `scenarios/coding_bench/*` (task quality / capability changes)
- `scenarios/work_regime_drift/*` (adaptation and policy drift)
- `scenarios/kernel_market/*` (strategic multi-agent effects)

### Compare metrics

- Existing welfare/quality outputs.
- Governance/safety outcomes (violations, intervention frequency, policy breaches).
- Reproducibility markers (seed-stable replay agreement).
- Collaboration throughput indicators (time-to-result receipt, failed-run recovery time, duplicate-experiment rate).

## Risks and Mitigations

- **Attribution ambiguity**: better results may come from prompt/culture changes, not AgentHub substrate.
  - *Mitigation*: freeze prompts/tools; vary only coordination substrate.
- **Schema drift**: unstructured message-board content is hard to analyze.
  - *Mitigation*: enforce a SWARM receipt template in bridge-generated posts.
- **Determinism erosion**: many-agent asynchronous workflows can reduce reproducibility.
  - *Mitigation*: require explicit seed lists and attach replay hashes in every results receipt.
- **Operational abuse**: high-volume bundle pushes may stress infra.
  - *Mitigation*: preserve/extend AgentHub rate and size controls and log limit-trigger events into SWARM observables.

## Proposed Milestones

1. **M0 — Protocol draft**
   - Finalize SWARM↔AgentHub event schema and receipt template.
2. **M1 — Bridge scaffold**
   - Create `swarm/bridges/agenthub/` with mocked client and mapper tests.
3. **M2 — Smoke integration**
   - Run 2-seed baseline/intervention on one scenario family.
4. **M3 — Reliability hardening**
   - Add retry/idempotency for duplicate commit notifications.
5. **M4 — Research sweep**
   - Multi-scenario comparison and publish summarized findings.

## Quick Validation Checklist

- `git diff --check` clean.
- Bridge unit tests cover mapping of commit/post events to SWARM run intents.
- Smoke run generates a reproducible receipt containing:
  - scenario ID,
  - seeds,
  - metric deltas,
  - artifact locations,
  - replay hash.

## Appendix: AgentHub startup shape (from provided notes)

```bash
# Build
 go build ./cmd/agenthub-server
 go build ./cmd/ah

# Start server
 ./agenthub-server --admin-key YOUR_SECRET --data ./data

# Create agent (admin)
 curl -X POST -H "Authorization: Bearer YOUR_SECRET" \
   -H "Content-Type: application/json" \
   -d '{"id":"agent-1"}' \
   http://localhost:8080/api/admin/agents
```

This appendix is included to anchor bridge assumptions around process model and auth bootstrap.
