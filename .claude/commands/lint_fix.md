# /lint-fix

Run ruff auto-fix on a directory after multi-file creation, then report remaining issues. Fills the gap between "just wrote code" and "ready to stage."

## Usage

`/lint-fix <path> [--test]`

Examples:
- `/lint-fix swarm/bridges/live_swe/`
- `/lint-fix swarm/research/ --test`

## Behavior

1) Run `ruff check --fix <path>`. Report what was auto-fixed (import sorting, unused imports, etc.).

2) Re-run `ruff check <path>`. Report any remaining unfixable issues.

3) If `--test` flag is given, detect and run the associated test file:
   - For `swarm/bridges/foo/`, look for `tests/test_foo_bridge.py` or `tests/test_foo.py`
   - For `swarm/core/foo.py`, look for `tests/test_foo.py`
   - Run `python -m pytest <test_file> -v`

4) Print a summary:
```
Lint-Fix Results
─────────────────────────────
  Auto-fixed:     N issues (import sort, unused imports, ...)
  Remaining:      M issues (or "clean")
  Tests:          PASS / FAIL / SKIP (if --test)
─────────────────────────────
```

## Why this exists

After creating multiple files, ruff consistently finds trivially fixable issues (F401 unused imports, I001 import sorting, F841 unused variables). Running `ruff check --fix` first, then checking for remaining issues, collapses a 4-step manual cycle (lint → edit → lint → edit) into one command.

This pattern appeared in 2 consecutive sessions and errored both times.

## Difference from /preflight

`/preflight` operates on **staged** files and mirrors the pre-commit hook. `/lint-fix` operates on **unstaged/new** files immediately after creation, before staging. Use `/lint-fix` right after writing code, then `/preflight` after staging.
