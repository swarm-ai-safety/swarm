# /rename_symbol

Rename a Python symbol (class, function, variable, constant) across the entire codebase in one shot. Handles imports, type annotations, string references, docstrings, tests, and documentation.

## Usage

`/rename_symbol <old_name> <new_name> [--scope <path>]`

Examples:
- `/rename_symbol TestableClaim VerifiableClaim`
- `/rename_symbol compute_v_hat compute_proxy_score --scope swarm/core/`

## Behavior

### Phase 1: Discovery

1. Run `grep -r <old_name>` across `swarm/`, `tests/`, `examples/`, `scripts/`, `docs/`, and project root files (`*.md`, `*.yaml`).
2. Exclude `__pycache__`, `.git`, `node_modules`, `runs/`, `logs/`, `.eggs/`.
3. If `--scope` is given, restrict search to that path (but always also check `tests/` and `__init__.py` re-exports).
4. Present a summary table:

```
Symbol Rename: OldName → NewName
──────────────────────────────────
  swarm/core/foo.py          3 occurrences
  swarm/core/__init__.py     2 occurrences (re-export)
  tests/test_foo.py          8 occurrences
  CLAUDE.md                  1 occurrence
──────────────────────────────────
  Total: 14 occurrences in 4 files
```

### Phase 2: Apply

For each file with occurrences:

1. Read the file.
2. Apply `replace_all` for `<old_name>` → `<new_name>`.
3. **Important**: also rename test classes that embed the old name (e.g., `TestTestableClaim` → `TestVerifiableClaim`).
4. **Important**: update string literals like `"OldName"` in `__all__` lists, `from_dict` return annotations, and YAML references.

Apply edits file-by-file (not in parallel) to avoid the "file not read" error.

### Phase 3: Verify

1. Run `grep <old_name>` again to confirm zero remaining occurrences.
2. Run `ruff check` on all modified Python files.
3. Run `python -m pytest <affected_test_files> -x -q --tb=short` to verify nothing broke.
4. If any test fails, report the failure and stop — do not attempt auto-fix.

### Phase 4: Report

```
Rename Complete: OldName → NewName
──────────────────────────────────
  Files modified:  4
  Occurrences:     14
  Remaining:       0
  Lint:            clean
  Tests:           32 passed
──────────────────────────────────
```

## Edge Cases

- **Self-referencing strings**: Classes often reference their own name in `from_dict` return type hints (e.g., `-> "ClassName"`). The `replace_all` flag handles these, but verify with the post-rename grep.
- **Partial matches**: If `<old_name>` is a substring of another symbol (e.g., renaming `Claim` would match `ClaimResult`), warn the user and ask for confirmation before proceeding.
- **Re-exports in `__init__.py`**: Always check `__init__.py` files for both import lines and `__all__` lists.
- **Concurrent sessions**: If running in a multi-session worktree, check `git log --oneline -1 -- <file>` before editing to detect if another session has already modified the file.

## Why this exists

Symbol renames touch many files (source, tests, re-exports, docs) and are error-prone when done manually. In the session that motivated this command, a 2-symbol rename required 12+ tool calls across 3 retry cycles due to missed self-references, files not being read before edit, and concurrent session interference. This command collapses that into a single invocation with built-in verification.

## Constraints

- Only renames exact symbol matches (word-boundary aware where possible).
- Does not rename file paths — only content within files.
- Does not auto-commit. Use `/commit_push` afterward.
