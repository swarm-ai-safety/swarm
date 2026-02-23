# /status

Quick orientation command for session start or mid-session check-in. Answers "where am I?" in one shot.

Use `--full` for a complete session warmup (status + healthcheck + tests). Consolidates the former `/warmup` command.

## Usage

```
/status [--full] [--skip-tests] [--research]
```

Examples:
- `/status` (quick orientation)
- `/status --full` (status + healthcheck + tests)
- `/status --full --skip-tests` (status + healthcheck, no tests)
- `/status --research` (standard status + research context)

## Argument parsing

- `--full`: Run the complete warmup sequence (see Full Mode below)
- `--skip-tests`: Skip the test suite in `--full` mode
- `--research`: Append research context from the memory system (see Research Mode below)

---

## Default behavior

Run all of the following in parallel and present a single consolidated summary:

### 0. Session Detection (run first)

Detect worktree context:
```bash
git rev-parse --git-dir
git rev-parse --git-common-dir
git rev-parse --show-toplevel
```

If `--git-dir` and `--git-common-dir` resolve to different paths, this is a linked worktree. Extract the session identity from the worktree directory name (e.g. `session-2` -> `pane-2`). Include this at the top of the output:

```
  Session:     pane-2 (session/pane-2)
  Main repo:   /path/to/main/repo
```

If not in a worktree, omit the session line.

### 1. Branch & Remote State
```bash
git branch --show-current
git log --oneline origin/$(git branch --show-current)..HEAD  # unpushed commits
git log --oneline HEAD..origin/$(git branch --show-current)  # commits behind remote
```

Report:
- Current branch name
- N commits ahead of remote (unpushed)
- N commits behind remote (need pull)
- Whether branch tracks a remote

### 2. Working Tree
```bash
git status --short
git diff --stat  # unstaged changes summary
git diff --cached --stat  # staged changes summary
git stash list  # any stashed work
```

Report:
- N files modified (unstaged)
- N files staged
- N untracked files
- N stashes

### 3. Open PRs for This Branch
```bash
gh pr list --head $(git branch --show-current) --state open
```

Report:
- PR number, title, status, and checks state (if any)
- Or "No open PR for this branch"

### 4. Recent Activity
```bash
git log --oneline -5
```

Report the last 5 commits for context.

### 5. Pre-commit Readiness (if files are staged)

Only if there are staged files:
- Count staged `.py` files
- Flag any staged files matching gitignore patterns (would fail `git add`)
- Flag any files that look like secrets/credentials

### 5b. Provenance & Docs Compliance

If `.claude/provenance.jsonl` exists and is non-empty, read the last 5 entries and report:
- Total commits tracked in those entries
- How many co-staged docs files (entries where `files_changed` includes CHANGELOG.md, README.md, or AGENTS.md)
- How many had unresolved docs reminders (`docs_reminders_unresolved > 0`)
- Whether the most recent entry has unresolved reminders

If `.claude/docs_reminders.log` exists and is non-empty, also list pending reminders (each line is a reminder).

If neither file exists or both are empty, skip this section silently (no error).

## Output Format

```
Session Status
══════════════════════════════════════════
  Session:     pane-2 (session/pane-2)      ← only if in a worktree
  Main repo:   /path/to/main/repo           ← only if in a worktree

  Branch:      feature/my-branch
  Remote:      2 ahead, 0 behind origin
  Stashes:     0

  Working tree:
    Staged:    3 files (+45 / -12)
    Modified:  1 file (unstaged)
    Untracked: 2 files

  Open PR:     #42 "Add feature X" — checks passing
               (or: No open PR)

  Recent commits:
    abc1234 Most recent commit message
    def5678 Previous commit message
    ...

  Pre-commit:  3 staged .py files — ready for /preflight

  Provenance:  5 commits tracked
    Docs co-staged: 3/5 (60%)
    Unresolved reminders: 1 (last commit)

  Pending reminders: 2                     ← only if .claude/docs_reminders.log exists
    - new_module.py: Update CHANGELOG.md [Unreleased]
    - new_agent.md: Update AGENTS.md
══════════════════════════════════════════
```

---

## `--full` mode (session warmup)

Runs the complete session opening sequence: status + healthcheck + tests.

### Step 1: Status

Run the default behavior above. Present the consolidated summary.

### Step 2: Healthcheck

Run the full `/healthcheck` scan (duplicate defs, dead imports, import conflicts, merge artifacts, stale re-exports). Present the results.

### Step 3: Tests (skip if `--skip-tests`)

Run the test suite:

```bash
python -m pytest tests/ -x -q
```

Report pass/fail count and duration.

### Output Format

```
Session Warmup
══════════════════════════════════════════

  1/3  STATUS
  Branch:      main
  Remote:      0 ahead, 0 behind — in sync
  Working tree: clean
  ...

  2/3  HEALTHCHECK
  Duplicates: 0 | Dead imports: 0 | Merge artifacts: 0
  Status: CLEAN

  3/3  TESTS
  2202 passed in 28.1s

══════════════════════════════════════════
  Ready to work.
```

## `--research` mode (research context)

Appends research context to the standard status output. Can be combined with `--full`.

After running the default status (or full warmup), read the following memory files and display a compact "Research Context" block:

### Step R1: Active Thread

Read `.letta/memory/threads/current.md`. Display:
- Active hypothesis (first heading or bold line)
- Current blockers (if any)
- Next planned experiment

If the file doesn't exist or is empty, show "No active research thread."

### Step R2: Recent Runs

Read `.letta/memory/runs/latest.md`. Display the last 5 run pointers (one line each: run ID, scenario, key result).

If the file doesn't exist or is empty, show "No recent runs."

### Step R3: Governance Knobs

Read `.letta/memory/project/governance-knobs.md`. Display a 3-5 line summary of high-leverage knobs and their current settings.

If the file doesn't exist or is empty, skip this section silently.

### Output Format (appended to standard status)

```
  Research Context
  ────────────────────────────────────
  Active thread:
    Hypothesis: <hypothesis text>
    Blockers:   <blockers or "none">
    Next:       <next experiment>

  Recent runs:
    1. <run_id> — <scenario> — <key result>
    2. ...

  Governance knobs:
    <knob>: <setting> — <note>
    ...
  ────────────────────────────────────
```

## Why This Exists

Session continuations from compacted context lose track of git state. Without `/status`, orientation requires 10-15 manual git commands and risks wasted work (e.g. trying to push commits that are already merged, creating branches that already exist).

Run `/status` at the start of any resumed session or when unsure of current state. Use `/status --full` at the very start of a new session for complete orientation.

## Relation to Other Commands

- `/status` tells you where you are — run it first
- `/preflight` checks if staged code is ready to commit — run it before committing
- `/pr` creates a pull request — run it after pushing
- `/sync --cleanup` tidies up after merge — run it last

## Migration from old commands

| Old command | Equivalent |
|---|---|
| `/warmup` | `/status --full` |
| `/warmup --skip-tests` | `/status --full --skip-tests` |
| `letta-os.sh thread` | `/status --research` |
