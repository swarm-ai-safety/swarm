# /stage

Smart git staging with pre-flight validation. Prevents the add-verify-reset-readd cycle.

## Usage

`/stage <files...>` or `/stage --all` or `/stage --interactive`

Examples:
- `/stage .claude/commands/status.md .claude/commands/stage.md`
- `/stage swarm/core/proxy.py tests/test_proxy.py`
- `/stage --all` (stage all modified + untracked, with safety checks)

## Behavior

### Step 1: Pre-validate files

Before running `git add`, check each file for problems:

a) **Gitignore check**: run `git check-ignore <file>` for each file. If any are ignored, warn and skip them:
   ```
   SKIPPED (gitignored): swarm/bridges/__init__.py
   ```

b) **Secrets scan**: run the secret patterns from `.claude/hooks/pre-commit` on each file's diff. If any match, warn and skip:
   ```
   BLOCKED (potential secret): config/api_keys.yaml
   ```

c) **Large file check**: warn if any file is >1MB (likely binary/artifact):
   ```
   WARNING (large file, 2.3MB): data/model_weights.bin
   ```

d) **Runtime artifact check**: warn if file matches common artifact patterns (`*.db`, `*.log`, `.DS_Store`, `__pycache__/`, `node_modules/`):
   ```
   WARNING (runtime artifact): sqlite_mcp_server.db
   ```

### Step 2: Clear stale staging

Check if there are already-staged files from a prior attempt. If so, report them:
```
Previously staged (from prior attempt):
  .claude/commands/pr.md
  .claude/commands/sync.md

Options:
  - Keep: add new files on top of existing staging
  - Reset: clear staging and start fresh with only the requested files
```

Default to **Keep** unless the user says otherwise.

### Step 3: Stage validated files

Run `git add` only on files that passed validation.

### Step 4: Report

Print a summary of what happened:

```
Staging Summary
────────────────────────────────
  Staged:   4 files (+128 / -23)
    .claude/commands/status.md (new)
    .claude/commands/stage.md (new)
    .claude/hooks/post_write_lint_check.sh (new)
    .claude/settings.json (modified)

  Skipped:  1 file
    swarm/bridges/__init__.py (gitignored)

  Warnings: 0
────────────────────────────────
  Ready for: /preflight → commit
```

### `--all` mode

When `/stage --all` is used:
1. Collect all modified (tracked) files and all untracked files
2. Run the same validation (gitignore, secrets, large files, artifacts)
3. Stage everything that passes
4. Report what was staged vs skipped

This is safer than `git add -A` because it pre-filters dangerous files.

### `--interactive` mode

List all modified + untracked files with checkboxes and let the user select which to stage. Still runs validation on selected files.

## Why This Exists

The manual staging cycle (`git add` → `git diff --cached` → find unexpected files → `git reset HEAD` → re-add) repeated 3 times in a single session. Common failure modes:

1. **Gitignored files** cause `git add` to silently fail or error
2. **Stale staged files** from prior failed commits contaminate the next attempt
3. **Accidental staging** of `.DS_Store`, `*.db`, credentials, or large binaries
4. **No feedback** — `git add` produces no output on success, so you always need a follow-up `git diff --cached`

`/stage` combines validation + staging + reporting into one step.

## Relation to Other Commands

- `/stage` prepares files for commit — run it after editing
- `/preflight` checks if staged code passes lint/tests — run it after staging
- Then commit and `/pr` to push
