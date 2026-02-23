# /ship

Commit and push in one shot. Consolidates the former `/commit_push`, `/fix_commit`, `/sweep_and_ship`, and `/close_and_ship` commands.

## Usage

`/ship [options] [commit message]`

Examples:
- `/ship` (auto-generates commit message from diff)
- `/ship "Fix typo in governance config"`
- `/ship --fix "Add PettingZoo bridge"` (auto-fix lint/mypy before committing)
- `/ship --all "Batch commit session output"` (stage everything safe)
- `/ship --close b6t b7x "Finish live integration tests"` (close beads + commit + push)
- `/ship --fix --close b6t` (auto-fix, close beads, commit, push)
- `/ship --research-close` (update research memory, then commit + push)
- `/ship --research-close "Finished governance knob sweep"` (with explicit message)

## Argument parsing

Parse `$ARGUMENTS` to extract:
- `--fix`: Enable ruff/mypy auto-fix with retry loop (up to 3 attempts)
- `--all`: Stage all modified + untracked files (excluding secrets/junk), not just already-staged ones
- `--close <bead-ids...>`: Close specified beads before committing. Bead IDs are tokens matching `distributional-agi-safety-*` or similar short IDs. Collect all bead IDs that follow `--close` until the next flag or quoted string.
- `--research-close`: Run the research session close ritual (Phase 0) before committing. Updates memory files with session summary, active thread, and run pointers.
- `--no-push`: Commit only, skip push step
- Everything else in quotes (or the remaining non-flag text) is the commit message.

If no flags are given, behavior matches the original `/ship`: commit staged changes and push.

## Behavior

### Phase 0: Research Close (only if `--research-close` flag is set)

Runs the research session close ritual before any git operations.

**Step 0a: Inventory changes**
- Run `git status` and `git diff --stat` to see what changed this session.

**Step 0b: Summarize session**
- Generate a concise session summary: what files changed, what was learned, what's next.
- Format as an append-only entry with ISO timestamp header.

**Step 0c: Append to research log**
- Append the session summary to `.letta/memory/threads/research-log.md`.
- Each entry format:
  ```
  ## <ISO-8601 timestamp>

  **Changed:** <brief list of key changes>
  **Learned:** <key insight or finding>
  **Next:** <next planned action>
  ```
- This file is append-only — never overwrite existing entries.

**Step 0d: Update active thread**
- Read `.letta/memory/threads/current.md`.
- Update it to reflect the current state: active hypothesis, blockers, next experiment.
- If the hypothesis was resolved, note the outcome and set the next one (or mark as "No active hypothesis").

**Step 0e: Update run pointers**
- If new runs were completed during this session (check `runs/` directory for recent timestamps), add them to `.letta/memory/runs/latest.md`.
- Keep only the 10 most recent entries.

**Step 0f: Stage memory files**
- Stage all modified files under `.letta/memory/` (never stage secrets or credentials).
- Proceed to Phase 1 with these files included in the commit.

### Phase 1: Pre-flight

1. Run `git status` to identify staged, unstaged, and untracked files.
2. If nothing to commit, say so and stop.
3. **Already-committed detection**: Before staging, run `git diff HEAD -- <files>` for tracked files. If ALL target files already match HEAD (parallel session committed them), report "already committed" and skip to Phase 5 (push).

### Phase 1.5: Docs compliance check

Before staging, check whether new files will need documentation updates:

1. Identify new (untracked or newly added) files in `swarm/`, `scenarios/`, `.claude/commands/`, `.claude/agents/`.
2. If any new files are found, check whether `CHANGELOG.md` has been modified (staged or unstaged via `git diff` and `git diff --cached`).
3. If CHANGELOG.md has NOT been modified:
   - Show the list of new files.
   - Ask the user: **"N new file(s) detected but CHANGELOG.md not updated. Stage CHANGELOG too, skip docs check, or abort?"**
     - **Stage CHANGELOG**: Open/update CHANGELOG.md with draft entries (see Phase 2 auto-draft), stage it, and continue.
     - **Skip docs**: Set `SKIP_DOCS_CHECK=1` in the environment so the pre-commit hook won't block, and continue.
     - **Abort**: Stop the `/ship` command.
4. If CHANGELOG.md IS already modified, continue silently.

This catches missing docs before the pre-commit hook, with a friendlier interactive UX.

### Phase 2: Stage

**If `--all` flag is set:**
- Stage all modified + untracked files, listing them explicitly (never `git add -A` or `git add .`).
- Exclude files matching: `.DS_Store`, `*.db`, `.env*`, `credentials*`, `*_token*`, `*_secret*`, `*.pem`, `*.key`.
- If ALL files are excluded, report "Nothing safe to ship" and exit.
- **Auto-draft CHANGELOG entries**: If new files are detected (untracked files being staged) and CHANGELOG.md has no pending changes:
  1. Read the current CHANGELOG.md. Find the `## [Unreleased]` section.
  2. Auto-append draft entries based on file paths. Categorize:
     - `swarm/agents/*` or `.claude/agents/*` → "### Added" with "New agent: `<filename>`"
     - `swarm/scenarios/*` or `scenarios/*` → "### Added" with "New scenario: `<filename>`"
     - `.claude/commands/*` → "### Added" with "New command: `<filename>`"
     - `swarm/**/*.py` → "### Added" or "### Changed" with "New/updated module: `<path>`"
  3. Stage the updated CHANGELOG.md.
  4. Show the user what was auto-drafted so they can review in the commit diff.

**Otherwise (default):**
- If there are already staged changes, use those.
- If nothing is staged but there are unstaged/untracked changes, show them and ask the user what to stage.
- Never stage `.DS_Store`, `*.db`, `.env`, or credential files.

### Phase 3: Fix (only if `--fix` flag is set)

**Step 3a: Ruff auto-fix**
- Run `ruff check --fix` on all staged `.py` files.
- Common fixes: F401 (unused imports), I001 (import sorting), C420, B019.
- Re-stage any modified files.

**Step 3b: Mypy mechanical fixes**
- Run `mypy` on staged `swarm/` files. Apply mechanical fixes for `no-any-return`:

| Return type | Pattern | Fix |
|---|---|---|
| `bool` | `return self._rng.random() < X` | `return bool(self._rng.random() < X)` |
| `bool` | `return expr > X` | `return bool(expr > X)` |
| `str` | `return self._rng.choice(items)` | `result: str = self._rng.choice(items); return result` |
| `float` | `return expr` | `return float(expr)` |
| `dict` | `return obj` | `return dict(obj)` |
| Custom type | `return self._rng.choice(items)` | `typed_local: Type = self._rng.choice(items); return typed_local` |

- Re-stage fixed files. Re-run mypy. Repeat up to 3 times for cascading errors.
- For non-mechanical errors, report them and stop.

**Step 3c: Syntax check**
- Run `python -m py_compile` on each staged `.py` file.
- If any fail, report the file and error. Do NOT auto-fix syntax errors. Suggest `git checkout -- <file>` or `git reset HEAD <file>`.

### Phase 4: Commit

**4a: Pre-stage lint check** (always, even without `--fix`):
- Run `ruff check` on staged `.py` files.
- If `--fix` is NOT set and errors are found, show them all upfront, fix them, re-check, then proceed.
- This avoids the iterative fix-one-discover-next loop from the pre-commit hook.

**4b: Index race guard** (CRITICAL for multi-session repos):
- After staging, run `git diff --cached --name-only` to get the ACTUAL staged files.
- Compare against the files YOU explicitly staged.
- If there are unexpected files (from a concurrent session):
  1. Save your intended file list.
  2. Run `git reset HEAD` to clear the entire index.
  3. Re-stage ONLY your intended files.
  4. Re-verify with `git diff --cached --name-only`.

**4c: Close beads** (only if `--close` flag is set):
- Run `bd close <id1> <id2> ...` to close all specified issues.
- If any close fails, report the error but continue.
- Run `bd sync` (or `bd --sandbox sync` in session worktrees).
- Stage `.beads/issues.jsonl` if modified.

**4d: Commit**:
- If a commit message was provided, use it.
- If `--close` was used and no message provided, auto-generate from closed bead titles (e.g. "Close b6t: AWM live integration tests").
- Otherwise, analyze the diff and draft a concise commit message.
- Always append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.
- Let pre-commit hooks run (never bypass with `--no-verify`).

**4e: Handle commit failure**:
- If `--fix` is set and failure is from fixable lint/mypy: go back to Phase 3. Max 3 total attempts.
- If commit exits non-zero but hooks passed: check if a parallel session committed the same files. If staged diff is now empty, skip to Phase 5.
- If pre-commit hook fails with test failures: report but do NOT auto-fix tests.

### Phase 5: Push (skip if `--no-push` flag is set)

- Push to the current branch's upstream (`git push`).
- If no upstream is set, push with `-u origin <current-branch>`.
- **If push fails with non-fast-forward**:
  1. Check for uncommitted changes. If any, `git stash push -m "ship: auto-stash"`.
  2. Run `git pull --rebase origin <current-branch>`.
  3. If stashed, `git stash pop`.
  4. Retry push once. If it fails again, stop and report.
  5. If merge conflicts arise, stop and report — never force-push.

### Phase 6: Beads sync

- If beads are configured (`.beads/` directory exists):
  - Detect worktree context: compare `git rev-parse --git-dir` vs `--git-common-dir`. If they differ and branch starts with `session/`, use `bd --sandbox sync`.
  - Otherwise: `bd sync`.

### Phase 7: Status report

Print a compact summary:
```
Shipped:
  Commit:    <sha> <first line of commit message>
  Branch:    <branch> -> origin/<branch>
  Closed:    <bead-ids> (if --close was used)
  Auto-fix:  N ruff fixes, M mypy fixes (if --fix was used)
  Attempts:  K (if --fix retried)
  Remaining: <N files modified, M untracked> (or "clean")
```

## Session worktrees

When running inside a session worktree (branch `session/pane-*`), `/ship` commits and pushes to the **session branch**, not directly to `main`. Use `/merge_session` afterward to merge into main.

## Constraints

- Never force-push.
- Never commit files likely containing secrets.
- Never `--no-verify` to skip hooks.
- Never auto-fix test failures or syntax errors — these need human review.
- Maximum 3 commit attempts (with `--fix`) before giving up.
- If pre-commit hooks fail, stop and report.
- For full branch + PR workflows, use `/pr` instead.

## Migration from old commands

| Old command | Equivalent |
|---|---|
| `/commit_push` | `/ship` |
| `/commit_push "msg"` | `/ship "msg"` |
| `/fix_commit "msg"` | `/ship --fix "msg"` |
| `/fix_commit "msg" file1 file2` | `git add file1 file2` then `/ship --fix "msg"` |
| `/sweep_and_ship` | `/ship --all` |
| `/sweep_and_ship "msg"` | `/ship --all "msg"` |
| `/close_and_ship b6t b7x` | `/ship --close b6t b7x` |
| `/close_and_ship b6t "msg"` | `/ship --close b6t "msg"` |
