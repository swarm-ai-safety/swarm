# /pr

Create a feature branch, commit staged/unstaged changes, push, and open a GitHub PR against main.

## Usage

`/pr [branch-name] [commit message]`

Examples:
- `/pr` (auto-generates branch name and commit message from changes)
- `/pr fix/loader-crash`
- `/pr add/track-b-pipeline "Add Track B research pipeline"`

## Behavior

1) Determine the starting point:
- Detect worktree context: run `git rev-parse --git-dir` vs `git rev-parse --git-common-dir`. If they differ and the current branch starts with `session/`, you are in a session worktree.
- **Session worktree** (`session/*` branch): do NOT `git checkout main` (it will fail because main is checked out in the main worktree). Instead: `git fetch origin main`.
- **Main worktree on `main`**: `git pull origin main` as before.
- **Other branch**: use it directly as the base for the feature branch.
- If there are no uncommitted changes and no untracked files, abort with a message.

2) Create and switch to a feature branch:
- If `<branch-name>` is provided, use it.
- Otherwise, generate one from the change summary (e.g. `fix/pre-commit-scoped-lint`).
- **Session worktree**: `git checkout -b <branch-name> origin/main` (branch from remote main without switching to it).
- **Main worktree**: `git checkout -b <branch-name>` (already on updated main).

3) Stage changes:
- Run `git status` to identify modified and untracked files.
- Stage all modified and untracked files, but **exclude**: `.DS_Store`, `*.db`, files matching secret patterns (`.env`, `credentials*`).
- Show the user what will be committed and confirm before proceeding.

4) Commit:
- If `<commit message>` is provided, use it.
- Otherwise, analyze the diff and draft a concise commit message.
- Always append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.
- Use `SKIP_SWARM_HOOKS=1` only if the hook itself is being modified (chicken-and-egg), otherwise let the hook run.

5) Push and open PR:
- `git push -u origin <branch-name>`
- `gh pr create --head <branch-name> --base main --title "..." --body "..."`
- Body includes: `## Summary` (bullet points), `## Test plan` (checklist), and the Claude Code footer.

6) Print the PR URL.

## Constraints

- Never force-push.
- Never commit `.DS_Store`, `*.db`, or files likely containing secrets.
- If tests fail during the pre-commit hook, stop and report the failure rather than bypassing.
- Do not merge the PR automatically; that is a separate action.
