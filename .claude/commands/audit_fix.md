# /audit_fix

Deep codebase audit: spawn category-specific agents, triage findings by severity, fix critical issues.

## Usage

`/audit_fix [--scan-only] [--categories=security,invariants,...] [path]`

- `/audit_fix` — full scan + triage + fix cycle on the whole codebase
- `/audit_fix --scan-only` — scan and triage only, do not apply fixes
- `/audit_fix --categories=security,threading` — limit to specific categories
- `/audit_fix swarm/agents/` — scope scan to a specific directory

## Phase 1: Scan

Spawn up to 10 parallel Explore agents, one per category. Each agent searches the codebase (or specified path) for issues in its domain:

| # | Category | What to look for |
|---|---|---|
| 1 | **Deprecated APIs** | Removed stdlib modules, deprecated function calls, legacy patterns |
| 2 | **Error handling** | Bare `except:`, `except Exception: pass`, swallowed errors, missing logging |
| 3 | **Type safety** | Missing type annotations on public APIs, `Any` overuse, unchecked casts |
| 4 | **Security** | SQL injection (f-string interpolation in queries), hardcoded secrets, command injection, path traversal |
| 5 | **Dead code** | Unused imports, unreachable branches, commented-out code blocks, unused functions |
| 6 | **Test quality** | Missing assertions, tests that can't fail, no edge-case coverage for critical paths |
| 7 | **Concurrency** | Shared mutable state without locks, race conditions, missing thread safety |
| 8 | **Invariant violations** | `p` outside `[0,1]`, unseeded `random.*()` calls, destructive operations on append-only data |
| 9 | **Config/schema** | Missing validation, undocumented required fields, schema drift between YAML and code |
| 10 | **Documentation drift** | CLAUDE.md, docstrings, or READMEs that contradict current code behavior |

If `--categories` is specified, only spawn agents for the listed categories.

## Phase 2: Triage

Consolidate all agent findings into a severity-ranked table:

| Severity | Criteria |
|---|---|
| **Critical** | Breaks reproducibility, violates safety invariants, security vulnerability, data loss risk |
| **High** | Silent failures, race conditions, test gaps on critical paths |
| **Medium** | Code quality, dead code, documentation drift, minor type issues |
| **Low** | Style, naming, minor cleanup |

Present the table to the user:

```
| # | Severity | Category | File:Line | Description |
|---|----------|----------|-----------|-------------|
| 1 | Critical | Security | beads.py:42 | SQL injection via f-string |
| 2 | Critical | Invariant | base.py:89 | Unseeded random.random() |
| ... |
```

If `--scan-only` was specified, stop here.

## Phase 3: Fix

For each **Critical** and **High** severity issue:

1. Group related fixes that touch the same files to minimize conflicts.
2. Spawn parallel fix agents (one per group) using the Task tool.
3. Each fix agent:
   - Reads the affected file(s)
   - Applies the minimal fix (no over-engineering)
   - Verifies the fix doesn't break imports: `python -c "import swarm"`
4. After all fix agents complete, run the full test suite: `python -m pytest tests/ -v`
5. Report results: which issues were fixed, which need manual attention.

**Medium** and **Low** issues are reported but not auto-fixed.

## Constraints

- Do NOT refactor or "improve" code beyond the identified issue.
- Do NOT add comments, docstrings, or type annotations to unchanged code.
- Do NOT modify tests unless the test itself is the issue (e.g. missing assertion).
- If a fix touches more than 3 files, flag it for manual review instead of auto-fixing.
- If tests fail after fixes, revert the failing change and report it.
- Respect the project's safety invariants (see CLAUDE.md): `p` in `[0,1]`, append-only logs, reproducible runs.
