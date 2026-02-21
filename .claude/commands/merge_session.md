# /merge_session

Merge session branch(es) into `main` via rebase + fast-forward push. Use from inside a session worktree pane to land your work, or with `--all` to batch merge all sessions from the main repo.

Consolidates the former `/merge_all_sessions` command (now `/merge_session --all`).

## Usage

`/merge_session [--all] [--cleanup]`

Examples:
- `/merge_session` (merge current session branch into main)
- `/merge_session --all` (batch merge all session branches)
- `/merge_session --all --cleanup` (merge all + remove worktrees + delete branches)

## Argument parsing

Parse `$ARGUMENTS` to extract:
- `--all`: Merge ALL `session/pane-*` branches sequentially (run from main repo, not a worktree)
- `--cleanup`: After merging, remove worktrees and delete branches for successfully merged sessions

---

## Default behavior (single session merge)

### 1) Verify branch
Run `git branch --show-current`. If it does not match `session/*`, print an error and stop:
```
Error: Not on a session branch (current: <branch>). /merge_session only works from session/* branches.
```

### 2) Check working tree
Run `git status --porcelain`. If there is any output, print an error and stop:
```
Error: Working tree is dirty. Commit or stash changes before merging.
```
List the dirty files so the user knows what to deal with.

### 3) Fetch latest main
```bash
git fetch origin main
```

### 4) Rebase onto main
```bash
git rebase origin/main
```
If the rebase fails with conflicts:
- Run `git rebase --abort`
- Run `git diff --name-only --diff-filter=U` to list conflicting files
- Print the list and stop:
  ```
  Rebase conflict — aborted. Conflicting files:
    <file1>
    <file2>
  Resolve manually or coordinate with the other session.
  ```

### 5) Push to main
```bash
git push origin HEAD:main
```
If the push is rejected (non-fast-forward):
- Fetch again: `git fetch origin main`
- Rebase again: `git rebase origin/main` (abort on conflict as above)
- Retry push once: `git push origin HEAD:main`
- If it fails again, report the error and stop.

### 6) Report success
```
Merged session branch to main:
  Branch: <branch>
  Commits: <N> commits rebased
  HEAD:   <short_hash> <message>
```

### 7) Cleanup (only if `--cleanup` is set)
```bash
git worktree remove --force .worktrees/session-<N>
git branch -D session/pane-<N>
git worktree prune
```

---

## `--all` mode (batch merge)

Run from the main repo (not a worktree) to land all session work at once.

### 1) Enumerate branches
List all branches matching `session/pane-*`, sorted numerically by pane number.

### 2) For each branch, in order:

**a.** Check how many commits the branch is ahead of `origin/main`:
```bash
git rev-list --count origin/main..<branch>
```
If 0 → mark as SKIPPED ("nothing to merge") and continue.

**b.** Check if the corresponding worktree (`.worktrees/session-<N>`) has a dirty working tree:
```bash
git -C .worktrees/session-<N> status --porcelain
```
If dirty → mark as SKIPPED ("dirty worktree") and continue.

**c.** Check out the branch in a temporary detached state and rebase:
```bash
git fetch origin main
git checkout <branch>
git rebase origin/main
```
If conflict → `git rebase --abort`, `git checkout -`, mark as CONFLICT and continue.

**d.** Push:
```bash
git push origin HEAD:main
```
If rejected → fetch, rebase, retry once. If still fails → mark as FAILED, `git checkout -`, continue.

**e.** Mark as SUCCESS. Return to previous branch: `git checkout -`

### 3) Report results table
```
Session Merge Results
═══════════════════════════════════════════
session/pane-1   SUCCESS    3 commits
session/pane-2   SKIPPED    nothing to merge
session/pane-3   CONFLICT   swarm/core/proxy.py
session/pane-4   SUCCESS    1 commit
═══════════════════════════════════════════
```

### 4) Cleanup (only if `--cleanup` is also set)
For each SUCCESS branch, remove the worktree and delete the branch:
```bash
git worktree remove --force .worktrees/session-<N>
git branch -D session/pane-<N>
```
Prune worktree refs: `git worktree prune`

## Constraints

- Never force-push.
- Always rebase, never merge commit (keeps linear history).
- Never push to any remote branch other than `main`.
- If anything goes wrong, leave the branch in a clean state (rebase --abort if needed).
- Process branches in numeric order (pane-1 before pane-2, etc.) in `--all` mode.
- Skip rather than fail on dirty worktrees in `--all` mode.
- Leave CONFLICT and FAILED branches untouched for manual resolution.

## Migration from old commands

| Old command | Equivalent |
|---|---|
| `/merge_all_sessions` | `/merge_session --all` |
| `/merge_all_sessions --cleanup` | `/merge_session --all --cleanup` |
