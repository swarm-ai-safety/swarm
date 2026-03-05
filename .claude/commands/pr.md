# /pr

Create a feature branch, commit staged/unstaged changes, push, and open a GitHub PR against main — use when you have local changes not yet on any branch and want a reviewable PR. Distinct from `/ship` (commit + push to current branch, no PR) and `/fix_pr` (resolve conflicts or run quality gates on an existing PR).

Also handles branch lifecycle cleanup via `--cleanup`.

## Usage

`/pr [branch-name] [commit message]`
`/pr --cleanup`

Examples:
- `/pr` (auto-generates branch name and commit message from changes)
- `/pr fix/loader-crash`
- `/pr add/track-b-pipeline "Add Track B research pipeline"`
- `/pr --cleanup` (delete all merged/stale remote branches)

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

---

## `--cleanup` mode

Delete all merged or stale remote branches. Keeps `main` and `gh-pages`.

### Behavior

1) **List remote branches**:
```bash
gh api repos/<owner>/<repo>/branches --paginate --jq '.[].name'
```

2) **Identify protected branches**: Always keep `main`, `gh-pages`, and any branch with an open PR.

3) **Check for open PRs**:
```bash
gh pr list --state open --json headRefName --jq '.[].headRefName'
```
Any branch with an open PR is excluded from deletion.

4) **Show candidates**: List all branches that will be deleted, with their last commit date if available. Report the count.

5) **Confirm with user**: Ask before proceeding. Show the count and list.

6) **Delete branches**:
```bash
# For each branch to delete:
gh api -X DELETE "repos/<owner>/<repo>/git/refs/heads/<branch>"
```

7) **Report**:
```
Branch cleanup:
  Deleted: N branches
  Kept:    main, gh-pages, <any with open PRs>
```

### Edge cases
- If no stale branches exist, report "All clean — no stale branches found."
- If deletion fails for a specific branch (e.g. protected), report it and continue with the rest.
- Never delete `main` or `gh-pages` under any circumstances.

---

## Constraints

- Never force-push.
- Never commit `.DS_Store`, `*.db`, or files likely containing secrets.
- If tests fail during the pre-commit hook, stop and report the failure rather than bypassing.
- Do not merge the PR automatically; that is a separate action.
