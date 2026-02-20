# /sync_artifacts

Stage, commit, and push changes in the `swarm-artifacts` repo. Handles pre-commit validation and Co-Authored-By automatically.

## Usage

`/sync_artifacts [message]`

Examples:
- `/sync_artifacts` (auto-generates commit message from diff)
- `/sync_artifacts "Add new claim cards from collusion study"`

## Behavior

### 1. Locate the artifacts repo

Check these paths in order:
1. `$SWARM_ARTIFACTS_DIR` environment variable (if set)
2. `/Users/raelisavitt/swarm-artifacts` (default)
3. `../swarm-artifacts` relative to the main repo root

If none exist, report: "Cannot find swarm-artifacts repo. Set $SWARM_ARTIFACTS_DIR or clone it."

### 2. Check for changes

```bash
cd <artifacts-repo>
git status --short
```

If no changes (staged or unstaged), report "swarm-artifacts is clean, nothing to sync." and stop.

### 3. Show what changed

```bash
git diff --stat        # unstaged
git diff --cached --stat  # staged
```

Report the summary to the user.

### 4. Stage files

Stage all changed files, excluding:
- `.DS_Store`
- `*.db`
- `.env*`
- `credentials*`

```bash
git add -A -- ':!.DS_Store' ':!*.db' ':!.env*' ':!credentials*'
```

### 5. Commit

If the user provided a message via `$ARGUMENTS`, use it. Otherwise, auto-generate from the staged diff:
- Count files by category (vault/, scripts/, schemas/, runs/)
- Summarize: "Update N vault notes, M scripts" or similar

Always append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.

```bash
git commit -m "$(cat <<'EOF'
<message>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

If pre-commit hooks fail:
- Read the error output
- If it's an index consistency error (notes not in `_index.md`), auto-fix by adding the missing references to `vault/_index.md`, re-stage, and retry the commit
- If it's a validation error (schema, YAML), report the errors and stop

### 6. Push

```bash
git push
```

If push fails with non-fast-forward:
- `git pull --rebase`
- Retry push once
- If still fails, report error and stop

### 7. Report

```
swarm-artifacts synced:
  Commit:  <short-sha> <message>
  Files:   <N> changed
  Push:    origin/main
```

## Constraints

- Never force push.
- Never modify files in the main distributional-agi-safety repo — this command only operates on swarm-artifacts.
- Always return to the original working directory after finishing.
- If pre-commit index fixes are applied, mention what was auto-fixed.
