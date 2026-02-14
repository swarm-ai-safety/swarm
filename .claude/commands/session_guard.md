# /session_guard

Check whether files you're about to edit have been modified by another concurrent session since you last read them. Prevents silent edit reversions in multi-worktree environments.

## Usage

`/session_guard <file1> [file2] ...`

Examples:
- `/session_guard swarm/core/orchestrator.py`
- `/session_guard swarm/core/orchestrator.py swarm/logging/event_log.py`

## Behavior

### Phase 1: Detect concurrent modifications

For each file provided:

1. Run `git log --oneline -1 -- <file>` to get the most recent commit touching that file.
2. Run `git log --oneline -3 HEAD` to get recent HEAD commits.
3. If the file's last commit is **not** in the current session's recent commits (i.e., it was changed by another branch/session since this session started), flag it as **externally modified**.
4. Also run `git diff -- <file>` to check for unstaged changes (another session may have written to the file without committing).

### Phase 2: Report

```
Session Guard: 2 files checked
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

## When to use

- **Before multi-file edits** in a concurrent session environment
- **After a test failure** that might be caused by another session modifying source files
- **After the Edit tool reports "File has been modified since read"** — this is the symptom that `/session_guard` prevents

## Integration with other commands

- `/commit_push` does not need this — it operates on already-staged changes
- `/rename_symbol` should call `/session_guard` on all target files during Phase 2 (Apply)

## Why this exists

In multi-session worktree environments (15+ concurrent Claude Code sessions), files like `orchestrator.py` are frequently modified by multiple sessions. The Edit tool silently fails or reverts when the file changes between read and edit. This command surfaces the conflict before wasted work.

**Observed in sessions**: 2 sessions had edits to `orchestrator.py` silently reverted, requiring 3+ retry tool calls each time to detect and recover.
