# /status

Quick orientation command for session start or mid-session check-in. Answers "where am I?" in one shot.

## Usage

`/status`

## Behavior

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

  Open PR:     #42 "Add feature X" — checks passing ✓
               (or: No open PR)

  Recent commits:
    abc1234 Most recent commit message
    def5678 Previous commit message
    ...

  Pre-commit:  3 staged .py files — ready for /preflight
══════════════════════════════════════════
```

## Why This Exists

Session continuations from compacted context lose track of git state. Without `/status`, orientation requires 10-15 manual git commands and risks wasted work (e.g. trying to push commits that are already merged, creating branches that already exist).

Run `/status` at the start of any resumed session or when unsure of current state.

## Relation to Other Commands

- `/status` tells you where you are — run it first
- `/preflight` checks if staged code is ready to commit — run it before committing
- `/pr` creates a pull request — run it after pushing
- `/cleanup_branch` tidies up after merge — run it last
