# /preflight

Run all pre-commit checks on staged files **without committing**. Shows all issues at once and optionally auto-fixes what it can.

Use `--secrets [path]` to run the secrets scanner standalone (replaces the former `/scan_secrets` command).

## Usage

```
/preflight [--fix]
/preflight --secrets [path]
```

Examples:
- `/preflight` — check only, report all issues
- `/preflight --fix` — auto-fix ruff/isort issues, then report remaining
- `/preflight --secrets` — scan all tracked + untracked files for secrets
- `/preflight --secrets swarm/bridges/` — scan a specific directory
- `/preflight --secrets swarm/bridges/gastown/bridge.py` — scan a single file

## Argument parsing

- `--fix`: Auto-fix ruff/isort issues before reporting.
- `--secrets [path]`: Run secrets scanner only (skip all other checks). If no path given, scans the whole repo. Equivalent to `bash .claude/hooks/scan_secrets.sh [path]`.

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
  Docs compliance: OK / WARN (details)
─────────────────────────────
  Verdict:        READY TO COMMIT / N issues remain
```

The "Docs compliance" row checks two things:
- Are there outstanding entries in `.claude/docs_reminders.log`? If so, report count.
- Are new files being staged in key directories (`swarm/`, `scenarios/`, `.claude/commands/`, `.claude/agents/`) without `CHANGELOG.md` co-staged? If so, warn.

Report as:
- `OK` — no outstanding reminders and either CHANGELOG is co-staged or no key-directory files are staged
- `WARN (N outstanding reminders)` — reminders log is non-empty
- `WARN (CHANGELOG not staged)` — key-directory files staged without CHANGELOG
- `WARN (N outstanding reminders, CHANGELOG not staged)` — both

This row is **advisory only** — it does not affect the verdict. If `.claude/docs_reminders.log` doesn't exist, treat as no outstanding reminders. If no key-directory files are staged, skip the CHANGELOG check.

4) If `--fix` was used and files were modified, remind to review changes and confirm they are re-staged.

## Why this exists

The pre-commit hook runs checks serially and stops on first failure. When ruff fails, you never see mypy or test errors. This command shows ALL issues at once and can auto-fix the mechanical ones (import ordering, trailing whitespace, unused imports), saving multiple commit-retry cycles.

The syntax check (step 2a) is especially important in multi-session workflows where concurrent sessions may introduce `IndentationError` or import ordering issues by editing shared files. Catching these before `git commit` avoids wasted pre-commit hook cycles.

## `--secrets` mode

Run the secrets scanner standalone without the full preflight pipeline.

1) Run the secrets scanner script:
   ```
   bash .claude/hooks/scan_secrets.sh [path]
   ```

2) The scanner checks for:
   - Platform API keys: agentxiv (`ax_`), clawxiv (`clx_`), moltbook (`moltbook_sk_`), moltipedia, wikimolt (`wm_`), clawchan, clawk
   - Cloud provider keys: OpenAI (`sk-`), AWS (`AKIA`), GitHub (`ghp_`, `gho_`), Anthropic (`sk-ant-`)
   - Generic patterns: hardcoded `API_KEY = "..."`, `Bearer` tokens, private keys (`-----BEGIN`)

3) Report results:
   - If clean: print confirmation with count of files scanned
   - If secrets found: list each match with file, line number, matched pattern, and a redacted preview
   - Suggest remediation: use `os.environ.get()`, `~/.config/<platform>/credentials.json`, or `${VAR}` interpolation

## Mirror of pre-commit

This command intentionally mirrors the checks in `.claude/hooks/pre-commit` so there are no surprises at commit time. If you add a new check to the pre-commit hook, add it here too.

## Migration from old commands

| Old command | Equivalent |
|---|---|
| `/scan_secrets` | `/preflight --secrets` |
| `/scan_secrets scripts/` | `/preflight --secrets scripts/` |
