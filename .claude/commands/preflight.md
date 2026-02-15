# /preflight

Run all pre-commit checks on staged files **without committing**. Shows all issues at once and optionally auto-fixes what it can.

## Usage

`/preflight [--fix]`

Examples:
- `/preflight` — check only, report all issues
- `/preflight --fix` — auto-fix ruff/isort issues, then report remaining

## Behavior

1) Get the list of staged Python files:
```bash
git diff --cached --name-only --diff-filter=ACM | grep '\.py$'
```
If no staged `.py` files, report "No staged Python files to check" and exit.

2) Run **all** checks, collecting results before reporting (do NOT stop on first failure):

   a) **Syntax check** — for each staged `.py` file, run `python -m py_compile <file>`. This catches `IndentationError`, `SyntaxError`, and other parse-level errors that concurrent sessions may introduce. Report any failures immediately — these block everything else.

   ```bash
   for f in $staged_py_files; do
       python -m py_compile "$f" 2>&1 || echo "SYNTAX ERROR: $f"
   done
   ```

   b) **Secrets scan** — run the same secret-pattern check from `.claude/hooks/pre-commit` on the staged diff. Report any matches.

   c) **Ruff lint** — `ruff check <staged_files>`. If `--fix` flag was given, first run `ruff check --fix <staged_files>`, then re-stage the fixed files with `git add`, then re-check to report remaining unfixable issues.

   d) **Mypy type check** — only for staged files under `swarm/`: `mypy --follow-imports=skip <swarm_files>`. Report type errors (cannot auto-fix).

   e) **Pytest** — `python -m pytest tests/ -x -q --tb=short`. Report pass/fail count.

3) Print a summary table:
```
Preflight Results
─────────────────────────────
  Syntax check:   PASS / FAIL (N files with errors)
  Secrets scan:   PASS / FAIL (N matches)
  Ruff lint:      PASS / FAIL (N issues, M auto-fixed)
  Mypy:           PASS / FAIL / SKIP (N errors)
  Tests:          PASS / FAIL (N passed, M failed)
─────────────────────────────
  Verdict:        READY TO COMMIT / N issues remain
```

4) If `--fix` was used and files were modified, remind to review changes and confirm they are re-staged.

## Why this exists

The pre-commit hook runs checks serially and stops on first failure. When ruff fails, you never see mypy or test errors. This command shows ALL issues at once and can auto-fix the mechanical ones (import ordering, trailing whitespace, unused imports), saving multiple commit-retry cycles.

The syntax check (step 2a) is especially important in multi-session workflows where concurrent sessions may introduce `IndentationError` or import ordering issues by editing shared files. Catching these before `git commit` avoids wasted pre-commit hook cycles.

## Mirror of pre-commit

This command intentionally mirrors the checks in `.claude/hooks/pre-commit` so there are no surprises at commit time. If you add a new check to the pre-commit hook, add it here too.
