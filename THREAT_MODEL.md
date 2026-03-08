# Threat Model

Attack surfaces and mitigations for the SWARM multi-agent research system.

## Scope

This covers threats to the local development and research environment — multi-session Claude Code operations, shared state, and tool integrations. It does not cover threats to the simulation itself (adversarial agents within the simulated economy are the Adversary Designer's domain).

---

## 1. Shared SQLite (runs.db) Concurrent Access

**Surface**: All sessions read/write `runs/runs.db` for inter-session coordination (`agent_messages` table) and run logging. SQLite has limited concurrent write support.

**Threats**:
- Write contention causing `SQLITE_BUSY` errors and lost messages
- A rogue session corrupting the database (no per-session permissions)
- Message spoofing: any session can send messages as any `from_agent`

**Mitigations** (current):
- SQLite WAL mode (if enabled) allows concurrent reads
- Convention-based `from_agent` = `SESSION_ID`

**Mitigations** (recommended):
- [ ] Enable WAL mode explicitly on database creation
- [ ] Add a `session_token` column to `agent_messages` validated against `SESSION_ID` env var
- [ ] Add write timeout/retry logic in MCP server config

---

## 2. Memory File Integrity

**Surface**: `.letta/memory/` files are read/write by all sessions. No access control.

**Threats**:
- Concurrent session overwrites (two sessions updating `current.md` simultaneously)
- Research log corruption (append-only invariant violated by a careless edit)
- Context poisoning: a session writes misleading content to `current.md` that steers future sessions

**Mitigations** (current):
- Append-only convention for `research-log.md` (documented, not enforced)
- Post-write hook warns on foundational section edits

**Mitigations** (recommended):
- [ ] Add post-write hook check: warn if `.letta/memory/threads/research-log.md` is modified via `Edit` with `old_string` that removes existing content
- [ ] Add file-level locking for memory writes (flock or advisory lock)

---

## 3. Hook Bypass (SKIP_SWARM_HOOKS)

**Surface**: Setting `SKIP_SWARM_HOOKS=1` disables all verification hooks with no audit trail.

**Threats**:
- Secrets committed to repo undetected
- Broken invariants (p outside [0,1], non-append-only log edits)
- Bypassed lint/type checks allowing broken code to land

**Mitigations** (current):
- Documented as "emergency only" in AGENTS.md and install_hooks.md

**Mitigations** (implemented in this PR):
- [x] Hook bypass events logged to `.claude/hook_bypass.log` with timestamp, session ID, and file path
- [ ] Pre-push hook that warns if `hook_bypass.log` has entries since last push

---

## 4. MCP Server Exposure

**Surface**: `.mcp.json` defines 4 MCP servers globally. All sessions see all servers regardless of agent role.

**Threats**:
- Read-only roles (Research Scout) having write access to `sqlite_runs`
- Token leakage if environment variables are shared across sessions
- Poisoned MCP responses (tool output injection)

**Mitigations** (current):
- Separate environment variables per server (`GITHUB_TOKEN`, `PERPLEXITY_API_KEY`)
- Agent `.md` files now declare tool allowlists (advisory)

**Mitigations** (recommended):
- [ ] Enforce tool allowlists at the MCP routing layer (not just advisory)
- [ ] Rotate tokens per-session where possible
- [ ] Add output validation for MCP tool responses in critical paths

---

## 5. Repository Config Injection

**Surface**: `CLAUDE.md`, `.claude/agents/*.md`, and `.claude/commands/*.md` are instructions that Claude Code follows. A malicious PR could inject instructions.

**Threats**:
- Malicious `CLAUDE.md` changes that weaken safety invariants
- Agent role definitions modified to expand scope (e.g., Research Scout gains write access)
- Command definitions modified to exfiltrate data

**Mitigations** (current):
- Core principles guard in post-write hook warns on foundational section edits
- PR review process (human-in-the-loop)

**Mitigations** (recommended):
- [ ] Add CODEOWNERS rule: changes to `CLAUDE.md`, `AGENTS.md`, `.claude/agents/*`, `.claude/hooks/*` require explicit reviewer approval
- [ ] Pre-commit check that diffs agent files against a known-good hash

---

## 6. Worktree Isolation Gaps

**Surface**: Git worktrees isolate branches and index, but share the filesystem and environment.

**Threats**:
- Cross-session env var pollution (all sessions inherit parent shell)
- Shared `runs/` directory allows one session to read/modify another's run data
- Timing side-channels via filesystem (session A can observe session B's file creation patterns)

**Mitigations** (current):
- `detect-session.sh` exports per-session env vars
- Each worktree has its own branch

**Mitigations** (recommended):
- [ ] Run sessions in separate containers (Docker) for full isolation
- [ ] Per-session `runs/` subdirectory: `runs/$SESSION_ID/`

---

## 7. Transitive Prompt Injection

**Surface**: Research Scout fetches external content (repos, docs, web pages). Fetched content could contain prompt injection.

**Threats**:
- External repo README containing instructions that override CLAUDE.md
- Fetched documentation containing hidden instructions
- GitHub issue/PR comments with injection payloads

**Mitigations** (current):
- Research Scout is read-only (cannot modify local files)
- Tool allowlists restrict what each role can do

**Mitigations** (recommended):
- [ ] Flag fetched content as untrusted in agent context
- [ ] Research Scout findings should be summarized, not raw-pasted, to reduce injection surface

---

## Threat Priority Matrix

| Threat | Likelihood | Impact | Current mitigation | Priority |
|---|---|---|---|---|
| SQLite contention | High (15+ sessions) | Medium (lost messages) | Minimal | **P1** |
| Hook bypass abuse | Medium | High (broken invariants) | Now logged | **P2** |
| Memory file corruption | Medium | Medium (stale context) | Convention only | **P2** |
| MCP over-exposure | Low | Medium (scope creep) | Advisory allowlists | **P3** |
| Config injection | Low (requires PR merge) | High (full compromise) | Post-write hook | **P3** |
| Worktree gaps | Low | Low (research context) | Env vars | **P4** |
| Prompt injection | Medium | Low (read-only Scout) | Role constraints | **P4** |
