# /commit_push

Stage, commit, and push changes in one step.

## Usage

`/commit_push [message]`

If no message is provided, auto-generate one from the staged changes.

## Behavior

1. Run `git status` and `git diff --stat` to see what changed.
2. Stage all relevant files (exclude `.DS_Store`, `*.db`, `.env`, credentials).
3. If the user provided a commit message via `$ARGUMENTS`, use it. Otherwise, draft a concise commit message from the diff summary.
4. Commit with the message (include `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`).
5. Push to the current branch's remote.
6. **If push fails with "non-fast-forward" (behind remote)**:
   - If there are unstaged changes, stash them first: `git stash push -m "commit_push: auto-stash"`
   - Run `git pull --rebase`
   - If stashed, pop: `git stash pop`
   - Retry push (once). If it fails again, report the error and stop.
7. Run beads sync if beads are configured:
   - Detect worktree context: run `git rev-parse --git-dir` vs `git rev-parse --git-common-dir`. If they differ and the current branch starts with `session/`, use `bd --sandbox sync` to avoid daemon contention with other sessions.
   - Otherwise: `bd sync`.
8. Print the final `git log --oneline -1` to confirm.

## Constraints

- Do NOT force push.
- Do NOT stage files matching: `.env*`, `credentials*`, `*.db`, `.DS_Store`.
- If there are no changes, say so and stop.
- If push fails (e.g. no remote, auth), report the error clearly.
