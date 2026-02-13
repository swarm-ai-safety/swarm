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

2) Stage changes:
- If there are already staged changes, use those.
- If nothing is staged but there are unstaged/untracked changes, show them and ask the user what to stage.
- Never stage `.DS_Store`, `*.db`, `.env`, or credential files (the pre-commit hook will catch these too, but avoid staging them in the first place).

3) Pre-stage lint check (`.py` files only):
- Before committing, run `ruff check` on all staged `.py` files.
- If errors are found, show **all** of them upfront and fix them before attempting `git commit`. This avoids the iterative fix-one-discover-next loop where the pre-commit hook blocks, you fix one error, re-commit, and hit the next.
- After fixing, re-run `ruff check` to confirm clean, then proceed.

4) Index race guard (CRITICAL for multi-session repos):
- After staging, run `git diff --cached --name-only` to get the ACTUAL list of staged files.
- Compare against the files YOU explicitly staged.
- If there are unexpected files (staged by another concurrent session), warn the user and list them.
- Unstage unexpected files with `git reset HEAD <file>` before committing.
- This prevents the shared git index race condition where parallel Claude/Codex sessions pollute each other's commits.

5) Commit:
- If `<commit message>` is provided, use it.
- Otherwise, analyze the diff and draft a concise commit message.
- Always append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.
- Let the pre-commit hook run (do not bypass with `SKIP_SWARM_HOOKS=1` unless the hook itself is being modified).

6) Push (with auto-rebase on conflict):
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

7) Print confirmation: commit SHA, branch, and remote status.

## Constraints

- Never force-push.
- Never commit files likely containing secrets.
- If the pre-commit hook fails, stop and report — do not bypass.
- For full branch + PR workflows, use `/pr` instead.
