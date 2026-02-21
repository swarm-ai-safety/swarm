# /guard

Pre-work safety checks: verify you're on the right branch and/or check that files haven't been modified by concurrent sessions. Consolidates the former `/branch_guard` and `/session_guard` commands.

## Usage

```
/guard [expected-branch]
/guard --files <file1> [file2] ...
/guard [expected-branch] --files <file1> [file2] ...
```

Examples:
- `/guard` (show current branch and context)
- `/guard main` (verify you're on main)
- `/guard feat/isometric-viz` (verify you're on the expected branch)
- `/guard --files swarm/core/orchestrator.py` (check if files were modified by other sessions)
- `/guard main --files swarm/core/orchestrator.py swarm/logging/event_log.py` (both checks)

## Argument parsing

- `--files <file1> [file2] ...`: Check listed files for concurrent modification (session guard mode)
- Any non-flag token that doesn't look like a file path: treated as expected branch name
- If neither branch nor `--files` is given: show current branch info (default)

---

## Branch check (default, or when branch name is given)

### 1) Check current branch
- Run `git branch --show-current`.
- Run `git status --short` to show working tree state.

### 2) Compare against expected
- If an expected branch is specified and the current branch doesn't match:
  - Warn clearly: "WARNING: On branch `<current>`, expected `<expected>`"
  - Show uncommitted changes on current branch (if any).
  - Ask: "Switch to `<expected>`?" before doing anything.
- If they match, confirm: "On correct branch: `<expected>`"

### 3) If no expected branch given
- Show the current branch, its tracking remote, and how many commits ahead/behind.
- Show any in-progress beads (`bd list --status=in_progress`) to remind what work is active.
- This gives enough context to decide if you're in the right place.

### 4) Stale branch detection
- If the current branch has no commits ahead of its upstream AND there are no local changes, note: "Branch is clean and up-to-date — nothing in progress here."

---

## File freshness check (`--files`)

Prevents silent edit reversions in multi-worktree environments by checking whether files were modified by another concurrent session since you last read them.

### Phase 1: Detect concurrent modifications

For each file provided:

1. Run `git log --oneline -1 -- <file>` to get the most recent commit touching that file.
2. Run `git log --oneline -3 HEAD` to get recent HEAD commits.
3. If the file's last commit is **not** in the current session's recent commits (i.e., it was changed by another branch/session since this session started), flag it as **externally modified**.
4. Also run `git diff -- <file>` to check for unstaged changes (another session may have written to the file without committing).

### Phase 2: Report

```
Guard: 2 files checked
──────────────────────────────
  swarm/core/orchestrator.py   SAFE (last modified by you: 2e8e6d6)
  swarm/logging/event_log.py   WARNING: modified by commit abc1234 (not yours)
──────────────────────────────
```

If any file is flagged:
- Show the diff between your expected state and the current state
- Recommend re-reading the file before editing
- If using the Edit tool, the file MUST be re-read or the edit will fail

### Phase 3: Auto-recover (optional)

If all files are flagged as externally modified:
- Offer to re-read all flagged files in parallel
- After re-reading, report that files are ready for editing

---

## Combined mode

When both a branch name and `--files` are given, run both checks and present a unified report:

```
Guard Check
══════════════════════════════════════════
  Branch:  main — correct

  Files:
    swarm/core/orchestrator.py   SAFE
    swarm/logging/event_log.py   WARNING: modified externally
══════════════════════════════════════════
```

## When to use

- **Session start**: `/guard main` to confirm you're on the right branch
- **Before multi-file edits** in a concurrent session environment: `/guard --files <files>`
- **After a test failure** that might be caused by another session modifying source files
- **After the Edit tool reports "File has been modified since read"** — this is the symptom that `/guard --files` prevents

## Integration with other commands

- `/ship` does not need this — it operates on already-staged changes
- `/rename_symbol` should call `/guard --files` on all target files during Phase 2 (Apply)

## Constraints

- Never switch branches automatically without asking.
- If there are uncommitted changes, warn before any branch switch.
- This is a read-only check — it never modifies files or git state.

## Migration from old commands

| Old command | Equivalent |
|---|---|
| `/branch_guard` | `/guard` |
| `/branch_guard main` | `/guard main` |
| `/session_guard swarm/core/orchestrator.py` | `/guard --files swarm/core/orchestrator.py` |
