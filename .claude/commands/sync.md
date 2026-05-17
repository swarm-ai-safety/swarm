# /sync

Atomic fetch-rebase-push with auto-stash. Resolves the common "push rejected because remote advanced" problem in one step, even when you have unstaged changes — automatically stashes dirty working tree before rebase and restores after push. Also handles post-merge branch cleanup via `--cleanup`. Use instead of manual `git pull --rebase && git push` sequences. Consolidates the former `/cleanup_branch` command (now `/sync --cleanup`).

## Usage

`/sync [branch] [--cleanup]`

Examples:
- `/sync` (syncs current branch)
- `/sync main` (syncs the specified branch)
- `/sync --cleanup` (switch to main, pull, delete current branch locally + remotely)
- `/sync --cleanup fix/pre-commit-scoped-lint` (delete a specific merged branch)

## Argument parsing

Parse `$ARGUMENTS` to extract:
- `--cleanup`: Post-merge branch cleanup mode (see Phase B below)
- Any remaining non-flag token is the branch name. If omitted, use current branch.

---

## Default behavior (no flags): fetch-rebase-push

### 1) Stash if dirty
If the working tree has uncommitted changes, run `git stash --include-untracked`. Note stash was created so it can be restored at the end.

### 2) Fetch
Run `git fetch origin`.

### 3) Check divergence
Run `git rev-list --left-right --count HEAD...origin/<branch>` to see how far ahead/behind we are.
- If already up to date (0 behind, 0 ahead or only ahead): skip rebase, go to push.
- If behind: rebase onto origin/<branch>.
- If diverged (both ahead and behind): rebase onto origin/<branch>.

### 4) Rebase
Run `git rebase origin/<branch>`.
- If rebase succeeds: continue to push.
- If rebase has conflicts: abort the rebase (`git rebase --abort`), restore stash if created, and report the conflicting files. Do NOT attempt to resolve conflicts automatically.

### 5) Push
Run `git push origin <branch>`.
- If no upstream is set, use `git push -u origin <branch>`.

### 6) Restore stash
If step 1 stashed changes, run `git stash pop`.

### 7) Report
```
Synced:
  Branch:  <branch>
  Rebased: <N> commits from origin/<branch>
  Pushed:  <M> local commits
  Status:  clean | stash restored
```

---

## `--cleanup` mode: post-merge branch deletion

Use after a PR has been merged to clean up the branch.

### 1) Identify branch to clean up
- If a branch name is provided, use it.
- Otherwise, if currently on a non-main branch, use that branch.
- If already on `main`, check `gh pr list --state merged --limit 5` for recently merged branches and offer a choice.

### 2) Switch to main and pull
- `git checkout main && git pull origin main`
- If there are uncommitted changes on the feature branch, stash them first and pop after checkout.

### 3) Delete the local branch
- Use `git branch -D <branch>` (force-delete is safe because the PR was squash-merged, so git won't see it as "fully merged").

### 4) Delete the remote branch
- `git push origin --delete <branch>`
- If the remote branch was already deleted (e.g. GitHub auto-delete), skip gracefully.

### 5) Print confirmation
Branch name deleted, current HEAD on main.

## Constraints

- Never force-push. If the push still fails after rebase, report the error and stop.
- Never resolve merge conflicts automatically — abort and report.
- Never modify git config.
- If the branch has no remote tracking branch and no `origin/<branch>` exists, report the error and stop.
- Safe to run repeatedly — if already synced, it's a no-op.
- Never delete `main` or `master` (in `--cleanup` mode).
- If `--cleanup` and the branch has unmerged commits that don't appear in any merged PR, warn the user before deleting.

## Migration from old commands

| Old command | Equivalent |
|---|---|
| `/cleanup_branch` | `/sync --cleanup` |
| `/cleanup_branch fix/foo` | `/sync --cleanup fix/foo` |
