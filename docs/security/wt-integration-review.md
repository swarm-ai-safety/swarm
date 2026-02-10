# Security & Integration Review: `abhi-arya1/wt`

**Repository:** https://github.com/abhi-arya1/wt
**Package:** `@abhi-arya1/wt` (npm)
**Version reviewed:** 0.0.9
**Date:** 2026-02-10
**Status:** No existing integration found in swarm codebase

---

## 1. What is `wt`?

`wt` is a TypeScript CLI tool (built on Bun) that creates isolated git worktree
sandboxes, either locally or on remote hosts via SSH. It enables parallel branch
development in disposable environments without touching the main working tree.

**Core capabilities:**
- Creates bare mirror clones of git repositories
- Spins up isolated worktrees for parallel development
- Supports remote (SSH) sandbox creation with tmux session management
- Automatic `.env*` file copying to sandboxes
- Garbage collection for stale sandboxes
- Structured `--json` output for agent/tool integration

**Runtime:** Bun v1.3.3+
**Dependencies:** commander, @inquirer/prompts, chalk, proper-lockfile, nanoid

---

## 2. Current Integration Status

**No integration exists.** A search of the swarm codebase found zero references to
`wt`, `worktree`, or `abhi-arya1`. The swarm project has no dependency on this
tool in `pyproject.toml`, `requirements*.txt`, `.mcp.json`, or any import path.

### Potential integration surface

If `wt` were integrated with swarm, the most likely touchpoints would be:

| Swarm component | Integration scenario | Risk level |
|---|---|---|
| `swarm/bridges/claude_code/` | Using `wt` to create sandboxed agent workspaces | Medium |
| `.mcp.json` | Adding `wt` as an MCP tool server | Medium |
| `swarm/research/agentrxiv_server.py` | Using `wt` to isolate research agent environments | Low |
| `swarm/agents/llm_agent.py` | Spawning agent code in isolated worktrees | Medium-High |
| CI/CD (`.github/workflows/`) | Parallel test environments | Low |

---

## 3. Security Findings in `wt`

### HIGH RISK

#### H1: Automatic `.env*` file propagation (Information Disclosure)

Every `wt up` command silently copies **all** `.env*` files from CWD to the new
sandbox, including to remote SSH hosts via SCP. The glob `.env*` matches
`.env`, `.env.local`, `.env.production`, `.env.secret`, etc.

- **No user confirmation** before copying
- **No filtering** of sensitive variables
- **Silent failure** (SCP uses `.nothrow()`)
- **Remote exposure**: secrets transmitted to and stored on remote hosts

**Swarm impact:** If `wt` is used to create sandboxes for swarm agents, credentials
in `.env` files (ANTHROPIC_API_KEY, OPENAI_API_KEY, GITHUB_TOKEN, etc.) would be
automatically propagated to every sandbox, including remote hosts. This violates
swarm's credential isolation model.

**Recommendation:** If integrating, override or disable `.env` propagation. Use
explicit environment injection per sandbox instead.

---

### MEDIUM RISK

#### M1: SSH `StrictHostKeyChecking=accept-new` (TOFU model)

`wt` hardcodes `StrictHostKeyChecking=accept-new` for all SSH connections. This
automatically trusts host keys on first connection, making the initial connection
vulnerable to MITM attacks. Subsequent connections are verified.

**Swarm impact:** If swarm orchestrates remote sandboxes via `wt`, the first
connection to a new host could be intercepted without user awareness.

#### M2: SSH command string serialization

Remote command execution serializes command arrays into single-quoted shell
strings for SSH. The quoting logic (`q()` function) appears correct (POSIX
single-quote escaping), and inputs are validated against shell metacharacters.
However, this string-based SSH execution is inherently fragile — any quoting
flaw would enable remote code execution.

Defense in depth is present: input validation + POSIX quoting. But this remains
a critical trust boundary.

#### M3: `rm -rf` without path guards

Sandbox removal executes `rm -rf <path>` without sanity checks (e.g., verifying
the path contains `/sandboxes/`). A bug in path construction or a corrupted
config file could lead to destructive deletion.

#### M4: Trailing space in quote function

The `q()` shell quoting function appends a trailing space to every quoted value.
This is cosmetic but could cause subtle bugs in contexts where trailing
whitespace matters (e.g., filenames, path comparisons).

---

### LOW RISK

#### L1: Full `process.env` passed to SSH subprocess

The `execInteractive` method in the SSH backend passes the full `process.env` to
the SSH subprocess. While SSH does not forward env by default, the local SSH
process has access to all parent environment variables.

#### L2: `_env` parameter ignored in SSH interactive mode

The `execInteractive` SSH method ignores its `_env` parameter (underscore prefix),
unlike the local backend which correctly merges it. Environment customization
doesn't work for interactive SSH sessions.

---

### POSITIVE Security Patterns

| Pattern | Assessment |
|---|---|
| Shell injection defense | Defense in depth: strict input validation + POSIX quoting |
| Subprocess execution (local) | Safe: Bun tagged templates + `Bun.spawn` arrays |
| Config file permissions | 600/700 with proper-lockfile for concurrency |
| Git credential stripping | URLs sanitized before config persistence |
| Agent name validation | Strict `^[a-zA-Z0-9._-]+$` regex + `q()` quoting |
| YAML loading (swarm side) | `yaml.safe_load()` only — no code injection risk |
| No hardcoded secrets | Environment variable sourcing throughout |

---

## 4. Integration Risk Assessment for Swarm

### Would `wt` be a safe tool for swarm agent sandboxing?

**Conditionally.** The tool is well-engineered for human CLI use, with solid shell
injection defenses. However, several issues arise when used programmatically by
AI agents:

| Concern | Detail |
|---|---|
| **Credential leakage** | Automatic `.env*` copying would propagate API keys to agent sandboxes |
| **Agent autonomy** | `wt` executes arbitrary commands in sandboxes (`wt run`); agents could escalate |
| **Remote access** | SSH sandbox creation gives agents network access to remote hosts |
| **No audit trail** | `wt` has no built-in logging compatible with swarm's event log format |
| **No resource limits** | No cgroup, memory, or CPU limits on sandbox processes |
| **Trust boundary** | `wt` trusts the CLI caller completely — no per-sandbox permission model |

### Comparison with swarm's existing boundaries

Swarm already has:
- `FlowTracker` for information flow monitoring
- `ExternalWorld` simulation for boundary interactions
- `BoundaryHandler` for sandbox permeability control
- Claude Code bridge with risk-scored tool usage

`wt` would bypass all of these boundary controls unless explicitly wrapped.

---

## 5. Recommendations

### If integrating `wt` with swarm:

1. **Wrap, don't expose directly.** Create a swarm-side adapter that:
   - Filters or disables `.env` propagation
   - Logs all sandbox creation/deletion to swarm's event log
   - Enforces allowlists on commands run inside sandboxes
   - Routes all interactions through `FlowTracker`

2. **Disable remote SSH sandboxes** for agent-initiated use. Only allow local
   worktrees unless explicitly authorized per scenario.

3. **Pin the version.** `wt` is at 0.0.9 — pre-1.0, API unstable. Pin to an
   exact version and audit each update.

4. **Add resource limits.** `wt` provides no resource isolation. Combine with
   cgroups or container-based limits if used for agent sandboxing.

5. **Validate the trust model.** `wt` assumes a trusted CLI operator. Swarm
   agents are not fully trusted. Add a permission layer between agent requests
   and `wt` execution.

### If NOT integrating:

No action required. The swarm codebase has zero coupling to `wt`. The existing
boundary and sandbox simulation infrastructure (`ExternalWorld`, `FlowTracker`,
`BoundaryHandler`) provides appropriate abstraction for swarm's needs.

---

## 6. Summary

| Category | Rating |
|---|---|
| Code quality | Good — well-structured TypeScript, defense-in-depth patterns |
| Shell injection safety | Good — input validation + POSIX quoting |
| Credential handling | **Risky** — automatic `.env*` propagation |
| Suitability for agent use | **Conditional** — needs wrapping and access control |
| Current integration risk | **None** — no coupling exists |
| Recommended action | Do not integrate without an adapter layer |
