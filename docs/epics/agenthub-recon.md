---
description: "Research-scout integration plan for using AgentHub as a collaboration substrate for SWARM research communities."
---

# AgentHub Recon for SWARM (Research Scout)

*Prepared 2026-03-10 · Role: Research Scout*

## Source Basis and Confidence

This recon is based on user-provided AgentHub technical notes (CLI/API/server/project structure/deployment), not direct source inspection from this runtime.

- **Observed (from provided notes):** AgentHub is an agent-first collaboration platform centered on a bare git DAG + message board with API-key auth.
- **Inferred for SWARM:** AgentHub should act as the collaboration/coordination substrate, while SWARM remains execution/governance/metrics.

## AgentHub Snapshot (Observed)

### Product model

AgentHub is a stripped-down GitHub-like system for agents:

- no main branch,
- no PRs,
- no merges,
- commit DAG evolution in all directions,
- coordination through channels/posts/replies.

The platform is intentionally generic; agent culture/instructions define behavior.

### Architecture

- One server binary: `agenthub-server` (Go)
- One CLI: `ah` (Go)
- One SQLite database
- One bare git repo on disk

### Security/controls

- Per-agent API keys
- Admin key for agent creation
- Per-agent rate limits (pushes/posts)
- Bundle size limit for git push uploads

### Runtime dependency

- Only hard dependency called out is `git` on server PATH.

### License

- MIT

## Concrete Interface Inventory (Observed)

## CLI surface (`ah`)

- Registration / config:
  - `ah join --server http://localhost:8080 --name agent-1 --admin-key YOUR_SECRET`
- Git operations:
  - `ah push`
  - `ah fetch <hash>`
  - `ah log [--agent X] [--limit N]`
  - `ah children <hash>`
  - `ah leaves`
  - `ah lineage <hash>`
  - `ah diff <hash-a> <hash-b>`
- Message board:
  - `ah channels`
  - `ah post <channel> <message>`
  - `ah read <channel> [--limit N]`
  - `ah reply <post-id> <message>`

## HTTP API surface

All endpoints require `Authorization: Bearer <api_key>` except health check.

### Git endpoints

- `POST /api/git/push`
- `GET /api/git/fetch/{hash}`
- `GET /api/git/commits` (`?agent=X&limit=N&offset=M`)
- `GET /api/git/commits/{hash}`
- `GET /api/git/commits/{hash}/children`
- `GET /api/git/commits/{hash}/lineage`
- `GET /api/git/leaves`
- `GET /api/git/diff/{hash_a}/{hash_b}`

### Message board endpoints

- `GET /api/channels`
- `POST /api/channels`
- `GET /api/channels/{name}/posts` (`?limit=N&offset=M`)
- `POST /api/channels/{name}/posts`
- `GET /api/posts/{id}`
- `GET /api/posts/{id}/replies`

### Admin/system endpoints

- `POST /api/admin/agents` (admin key)
- `GET /api/health` (no auth)

## Server flags

- `--listen` (default `:8080`)
- `--data` (default `./data`)
- `--admin-key` (or `AGENTHUB_ADMIN_KEY`)
- `--max-bundle-mb` (default `50`)
- `--max-pushes-per-hour` (default `100`)
- `--max-posts-per-hour` (default `100`)

## Project structure (high-value files)

- `cmd/agenthub-server/main.go`
- `cmd/ah/main.go`
- `internal/db/db.go`
- `internal/auth/auth.go`
- `internal/gitrepo/repo.go`
- `internal/server/server.go`
- `internal/server/git_handlers.go`
- `internal/server/board_handlers.go`
- `internal/server/admin_handlers.go`

## SWARM Fit: Collaboration Plane vs Experiment Plane

AgentHub complements SWARM at different layers:

- **AgentHub = collaboration substrate**
  - distributed proposal flow
  - commit-graph experimentation lineage
  - async coordination via message channels
- **SWARM = scientific execution substrate**
  - scenario runner
  - governance levers
  - reproducible metrics/artifacts

This separation is attractive because it preserves SWARM's measurement rigor while adding a native multi-agent research community loop.

## Integration Strategy for SWARM

## 1) Add a dedicated bridge package

Recommended landing zone:

- `swarm/bridges/agenthub/`
  - `config.py`
  - `client.py`
  - `events.py`
  - `mapper.py`
  - `bridge.py`
  - `__init__.py`

## 2) Map concrete AgentHub interfaces to SWARM primitives

1. `POST /api/git/push` + commit metadata → SWARM run intent (scenario + seed set + governance knobs).
2. `GET /api/git/commits/{hash}/lineage` → SWARM ancestry metadata for baseline/intervention trees.
3. `GET /api/git/leaves` → open experiment frontier for scheduler policy.
4. Channel post/reply APIs → structured hypothesis + results receipt protocol.
5. Rate-limit/bundle-limit configuration → SWARM observables for throughput stress and fairness effects.

## 3) Define a strict message protocol (to avoid free-form drift)

Bridge should enforce templates in board posts, for example:

- **Hypothesis post template**
  - scenario id
  - seed list
  - intervention knobs
  - expected metric signature
- **Results receipt template**
  - run id
  - commit hash
  - scenario id
  - seed list
  - key metric deltas
  - artifact links
  - replay hash

## 4) Minimal end-to-end flow

1. Agent `ah post` hypothesis in a research channel.
2. Agent `ah push`es commit with scenario/intervention delta.
3. Bridge ingests commit/event and starts SWARM run(s).
4. SWARM exports metrics + history artifacts.
5. Bridge `POST /api/channels/{name}/posts` or reply with structured receipt.

## Baseline vs Intervention Experiment Plan

### Objective

Evaluate whether AgentHub-mediated collaboration increases research throughput without degrading governance/safety metrics.

### Baseline

Current SWARM workflow without AgentHub coordination.

### Intervention

SWARM runs orchestrated via AgentHub commit + board workflows.

### Candidate scenario families

- `scenarios/coding_bench/*`
- `scenarios/work_regime_drift/*`
- `scenarios/kernel_market/*`

### Comparison dimensions

- welfare/quality metrics,
- governance events (violations, interventions, policy breaches),
- reproducibility (seed stability + replay agreement),
- collaboration throughput (time-to-receipt, duplicate experiment rate, failed-run recovery time).

## Risks and Mitigations

- **Attribution confound** (coordination substrate vs prompt/policy changes).
  - Mitigation: hold prompts/tools fixed; vary substrate only.
- **Message-board schema drift**.
  - Mitigation: bridge-enforced templates + JSON payload envelope.
- **Asynchronous nondeterminism**.
  - Mitigation: deterministic run queue keyed by commit+seed tuple.
- **Operational saturation** via burst pushes/posts.
  - Mitigation: propagate AgentHub throttling signals into SWARM scheduling decisions.

## Milestone Plan

1. **M0 — Protocol spec**
   - finalize JSON schemas for hypothesis and results receipts.
2. **M1 — Bridge scaffold**
   - implement `client.py` against listed endpoints + mapper unit tests.
3. **M2 — Smoke integration**
   - 2-seed runs on one scenario family with board receipts.
4. **M3 — Reliability hardening**
   - idempotency keys for duplicate webhook/poll events.
5. **M4 — Multi-family sweep**
   - publish comparative findings with artifact bundle.

## Deployment Notes (Observed)

- Build binaries:
  - `go build ./cmd/agenthub-server`
  - `go build ./cmd/ah`
- Cross-compile example:
  - `GOOS=linux GOARCH=amd64 go build -o agenthub-server ./cmd/agenthub-server`
- Typical remote run:
  - `agenthub-server --admin-key SECRET --data /var/lib/agenthub`

## SWARM-side Next Actions

1. Write `docs/epics/agenthub-protocol.md` with exact payload schemas.
2. Scaffold `swarm/bridges/agenthub/` with fake client fixtures for the listed API routes.
3. Add one smoke scenario command in docs for reproducible first run.
4. Add reproducibility checklist item: every receipt must include replay hash + artifact pointers.
