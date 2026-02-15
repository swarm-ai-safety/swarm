# /fix_commit

Auto-fix lint/mypy issues and commit with retry loop. Collapses the "commit → pre-commit fails → fix → re-stage → commit again" cycle into one command.

## Usage

`/fix_commit <message> [files...]`

Examples:
- `/fix_commit "Add PettingZoo bridge" swarm/bridges/pettingzoo/ tests/test_pettingzoo_bridge.py`
- `/fix_commit "Fix agent RNG seeding"` (operates on already-staged files)

## Behavior

### Step 0: Stage files

If file paths are provided in `$ARGUMENTS` after the quoted message, stage them with `git add`. If no files are given, use whatever is already staged.

Verify staged files exist: `git diff --cached --name-only`. If empty, abort with "Nothing staged."

### Step 1: Ruff auto-fix

Run `ruff check --fix` on all staged `.py` files. Common fixes:
- F401: Remove unused imports (e.g. `import random` after switching to `self._rng`)
- I001: Import sorting
- C420: `dict.fromkeys` suggestions
- B019: `lru_cache` on methods

Re-stage any modified files: `git add <fixed_files>`.

### Step 2: Mypy mechanical fixes

Run `mypy` on staged `swarm/` files. If there are `no-any-return` errors, apply these mechanical fixes:

| Return type | Pattern | Fix |
|---|---|---|
| `bool` | `return self._rng.random() < X` | `return bool(self._rng.random() < X)` |
| `bool` | `return expr > X` | `return bool(expr > X)` |
| `str` | `return self._rng.choice(items)` | `result: str = self._rng.choice(items); return result` |
| `float` | `return expr` | `return float(expr)` |
| `dict` | `return obj` | `return dict(obj)` |
| `AttackStrategy` / custom | `return self._rng.choice(items)` | `typed_local: Type = self._rng.choice(items); return typed_local` |

Re-stage fixed files. Re-run mypy to check. If new errors appear in newly-checked files (mypy cascade), repeat this step up to 3 times.

For errors that don't match the mechanical patterns above, report them and stop — these need human judgement.

### Step 3: Syntax check

Run `python -m py_compile` on each staged `.py` file. If any fail (e.g. `IndentationError` from a dirty worktree), report the file and error. Do NOT attempt to fix syntax errors — these usually mean a dirty file from another session was accidentally staged. Suggest `git checkout -- <file>` or `git reset HEAD <file>`.

### Step 4: Commit

Run `git commit -m "<message>\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"`.

### Step 5: Handle test failures

If the pre-commit pytest hook fails:

1. Parse the failure output to identify which test(s) failed.
2. Classify the failure:
   - **Staged-file regression**: A test that imports or directly tests staged code failed. Report to user — this needs manual investigation.
   - **Statistical/flaky test**: A test with `assert X >= Y * Z` or `assert X < threshold` that shifted due to RNG/seeding changes. Report the specific assertion values and suggest widening the tolerance.
   - **Unrelated test**: A test that doesn't involve any staged files at all. Report it but note it's likely a pre-existing issue.
3. Do NOT auto-fix test failures — always report and let the user decide.

### Step 6: Retry (up to 2 times)

If the commit failed due to fixable lint/mypy issues that appeared after step 4 (e.g. mypy cascade from newly-checked files), go back to step 1 and retry. Maximum 2 retries (3 total attempts).

If the commit fails on the 3rd attempt, report all remaining issues and stop.

### Step 7: Confirm

Print the final `git log --oneline -1` and a summary:

```
fix_commit: SUCCESS
  Ruff auto-fixes:  N
  Mypy fixes:       M
  Commit attempts:  K
  Commit:           <sha> <message>
```

## Constraints

- Never auto-fix test failures or syntax errors — these need human review.
- Never `--no-verify` to skip hooks.
- Maximum 3 commit attempts before giving up.
- If a staged file has a `SyntaxError` or `IndentationError`, check `git diff HEAD -- <file>` to see if the error is in YOUR changes or pre-existing. If pre-existing, suggest unstaging the file.
- Do NOT stage files matching: `.env*`, `credentials*`, `*.db`, `.DS_Store`.

## Difference from other commands

| Command | What it does | When to use |
|---|---|---|
| `/lint_fix` | Ruff fix on unstaged files, no commit | Right after writing new code |
| `/preflight` | Run all checks on staged files, no commit | Before committing, to see all issues at once |
| `/fix_commit` | Fix + stage + commit with retry loop | When you're ready to commit and want auto-fix |
| `/commit_push` | Stage + commit + push, no auto-fix | When code is already clean |

## Origin

Created from `/retro` analysis of a session where 8 commit retries (35+ tool calls) were spent fixing ruff, mypy, and flaky test issues one at a time due to the pre-commit hook's serial fail-fast behavior.
