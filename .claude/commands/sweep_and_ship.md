# /sweep_and_ship

Commit and push all uncommitted changes (modified + untracked) in a single shot. Designed for sweeping in output from concurrent sessions.

## Usage

`/sweep_and_ship [commit message]`

Examples:
- `/sweep_and_ship` (auto-generates commit message from changed files)
- `/sweep_and_ship "Add run artifacts and paper updates"`

## Behavior

1) Run `git status --short` to see what's pending. If working tree is clean, report "Nothing to ship" and exit.

2) Categorize changes:
   - Modified files (unstaged)
   - Staged files
   - Untracked files/directories

3) **Safety checks** before committing:
   - Skip any files matching `.gitignore` patterns (they won't stage anyway)
   - Warn and exclude files that look like secrets (`.env`, `credentials`, `*_token*`, `*_secret*`, `*.pem`, `*.key`)
   - Warn and exclude `.db` files (matched by gitignore)
   - If ALL files are excluded, report "Nothing safe to ship" and exit

4) Stage all safe files with `git add` (list files explicitly, never `git add -A`)

5) Generate commit message if not provided:
   - Scan staged files and group by category:
     - `docs/` → "documentation"
     - `tests/` → "tests"
     - `swarm/` → "source updates"
     - other → list filenames
   - Format: `Add <category1>, <category2>, and <category3>`
   - Append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`

6) Commit with the message.

7) Push to origin.

8) Print summary:
   ```
   Swept and shipped:
     Committed: N files (+X / -Y)
     Pushed:    <branch> → origin/<branch>
     Hash:      <short_hash> <first line of message>
   ```

## When to use this

- After a concurrent Claude Code or Codex session has produced output you want to commit
- When you have accumulated misc changes (run artifacts, docs, plots) and want to batch-ship them
- Any time the working tree has changes you've reviewed and want committed without running full `/preflight`

## When NOT to use this

- For your own code changes that need review — use `/preflight` then `/ship` instead
- When you want to commit specific files only — use `git add <files>` manually
- When you haven't looked at what changed — run `/status` first

## Session worktrees

When running inside a session worktree (branch `session/pane-*`), `/sweep_and_ship` commits and pushes to the **session branch**, not directly to `main`. This is the expected workflow:

1. `/sweep_and_ship` — commits to your session branch and pushes it to origin
2. `/merge_session` — rebases your session branch onto main and fast-forward pushes to main

This keeps each session's work isolated until you're ready to merge.

## Constraints

- Never use `git add -A` or `git add .` — always list files explicitly
- Never commit files that look like secrets
- Always include Co-Authored-By trailer
- If push fails (e.g. behind remote), pull first then retry push
