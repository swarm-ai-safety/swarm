# /fix_pr

Create a PR from uncommitted changes on main (or any branch) by branching, staging specific files, committing, pushing, and opening a PR — all in one step.

Unlike `/pr` which expects you to describe what changed, `/fix_pr` is for when you've already made edits and just want to ship them on a clean branch.

## Usage

`/fix_pr [branch-name]`

Examples:
- `/fix_pr` (auto-generates branch name from changes)
- `/fix_pr fix/circuit-breaker-reset`

## Behavior

1. **Sanity check**:
   - Run `git status` to identify modified and untracked files.
   - If there are no changes, abort with a message.

2. **Identify relevant files**:
   - Show the user the list of modified/untracked files.
   - Ask which files to include if it's ambiguous (e.g. mix of unrelated changes).
   - Exclude: `.DS_Store`, `*.db`, `.env*`, `credentials*`.

3. **Create a clean branch**:
   - Stash all changes: `git stash push --include-untracked -m "fix_pr: temp stash"`.
   - Ensure main is up to date: `git checkout main && git pull origin main` (skip if in a session worktree — use `git fetch origin main` and branch from `origin/main` instead).
   - Create the feature branch: `git checkout -b <branch-name>` (or `git checkout -b <branch-name> origin/main` in a worktree).
   - Pop the stash: `git stash pop`.

4. **Stage and commit**:
   - Stage only the relevant files (not `git add -A`).
   - Analyze the diff and draft a concise commit message.
   - Always append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.
   - Commit. If the pre-commit hook fails on a lint issue introduced by the changes, fix it and re-commit. If it fails on a pre-existing/flaky test, note it and use `--no-verify` with an explanation.

5. **Push and open PR**:
   - `git push -u origin <branch-name>`
   - `gh pr create --head <branch-name> --base main --title "..." --body "..."`
   - Body format: `## Summary` (bullet points), `## Test plan` (checklist), Claude Code footer.

6. **Return to main**:
   - `git checkout main` (leave the feature branch pushed).
   - Print the PR URL.

## Constraints

- Never force-push.
- Never commit `.DS_Store`, `*.db`, or files likely containing secrets.
- If tests fail during pre-commit, stop and report unless the failure is clearly pre-existing.
- Do not merge the PR automatically.
- If the branch name already exists on the remote, abort and ask for a different name.
