# /ship

Commit and push in one shot. For quick direct-to-branch pushes without the full PR workflow.

## Usage

`/ship [commit message]`

Examples:
- `/ship` (auto-generates commit message from diff)
- `/ship "Fix typo in governance config"`

## Behavior

1) Check for changes:
- Run `git status` to identify staged, unstaged, and untracked files.
- If nothing to commit, say so and stop.

2) Already-committed early detection:
- Before staging, identify the files you intend to commit.
- Run `git diff HEAD -- <files>` for tracked files and check `git ls-files <file>` for untracked files.
- If ALL target files already match HEAD (i.e. a parallel session committed them), report "already committed" and skip to step 7 (push check). This avoids a wasted stage+commit cycle.

3) Stage changes:
- If there are already staged changes, use those.
- If nothing is staged but there are unstaged/untracked changes, show them and ask the user what to stage.
- Never stage `.DS_Store`, `*.db`, `.env`, or credential files (the pre-commit hook will catch these too, but avoid staging them in the first place).

4) Pre-stage lint check (`.py` files only):
- Before committing, run `ruff check` on all staged `.py` files.
- If errors are found, show **all** of them upfront and fix them before attempting `git commit`. This avoids the iterative fix-one-discover-next loop where the pre-commit hook blocks, you fix one error, re-commit, and hit the next.
- After fixing, re-run `ruff check` to confirm clean, then proceed.

5) Index race guard (CRITICAL for multi-session repos):
- After staging, run `git diff --cached --name-only` to get the ACTUAL list of staged files.
- Compare against the files YOU explicitly staged.
- If there are unexpected files (staged by another concurrent session), warn the user and list them.
- **Safe unstage strategy**: Do NOT use `git reset HEAD <unexpected-files>` — this can nuke your intended staged files when the index is shared. Instead:
  1. Save your intended file list.
  2. Run `git reset HEAD` to clear the entire index.
  3. Re-stage ONLY your intended files with `git add <my-files>`.
  4. Re-verify with `git diff --cached --name-only`.
- This prevents the shared git index race condition where parallel Claude/Codex sessions pollute each other's commits.

6) Commit (with parallel session awareness):
- If `<commit message>` is provided, use it.
- Otherwise, analyze the diff and draft a concise commit message.
- Always append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.
- Let the pre-commit hook run (do not bypass with `SKIP_SWARM_HOOKS=1` unless the hook itself is being modified).
- **If the commit exits non-zero but pre-commit hooks passed**: A parallel session likely committed the same files during the hook run. Run `git log --oneline -1` and `git diff --cached --stat` to check:
  - If staged diff is now empty (all files match HEAD), report "parallel session committed these files" and skip to step 7 (push check).
  - If staged diff is smaller (some files remain), report which files were taken by the parallel session, then re-commit only the remaining changes.
  - Do NOT re-run `git add` on files that are already at HEAD — this stages empty diffs that produce confusing "nothing to commit" errors.

7) Push (with auto-rebase on conflict):
- Push to the current branch's upstream (`git push`).
- If no upstream is set, push with `-u origin <current-branch>`.
- **If push fails with non-fast-forward** (common in multi-session repos):
  1. Check for uncommitted changes. If any exist, `git stash` them first.
  2. Run `git pull --rebase origin <current-branch>`.
  3. If there were stashed changes, `git stash pop`.
  4. If the rebase succeeds with no conflicts, retry `git push`.
  5. If there are merge conflicts, stop and report them — do not force-push.
  6. If the commit is already on remote (e.g. another session pushed it), report success and skip the push.
- Never retry more than once. If the second push also fails, stop and report.

8) Post-ship status report:
- Print a compact summary so the user doesn't need to run `/status` afterward:
  ```
  Shipped:
    Commit:    <sha> <first line of commit message>
    Branch:    <branch>
    Remote:    origin/<branch> — pushed (rebased N if applicable)
    Remaining: <N files modified, M untracked> (or "clean")
  ```
- Run `git status --short` and `git diff --stat` to populate the "Remaining" line.
- If everything is clean, say "Working tree clean".

## Constraints

- Never force-push.
- Never commit files likely containing secrets.
- If the pre-commit hook fails, stop and report — do not bypass.
- For full branch + PR workflows, use `/pr` instead.
