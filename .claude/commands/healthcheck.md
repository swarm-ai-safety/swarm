# /healthcheck

Scan the codebase for code health issues that arise from parallel commits, merges, or external changes. Complements `/preflight` (which checks before commit) by checking after external changes arrive.

## Usage

`/healthcheck` or `/healthcheck --fix` or `/healthcheck --hf-space`

## Behavior

### Step 1: Duplicate Definitions (AST-based)

Use Python's `ast` module to find duplicate function/class definitions **at the same scope** (same class or module level). This avoids false positives from same-named methods on different classes (e.g. multiple classes each defining `__init__` or `to_dict`).

```python
import ast, os, collections

issues = []
for root_dir in ['swarm', 'tests']:
    for dirpath, dirs, files in os.walk(root_dir):
        for f in sorted(files):
            if not f.endswith('.py'):
                continue
            path = os.path.join(dirpath, f)
            with open(path) as fh:
                try:
                    tree = ast.parse(fh.read(), filename=path)
                except SyntaxError:
                    continue

            # Check module-level duplicates
            mod_names = collections.defaultdict(list)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mod_names[node.name].append(node.lineno)
                elif isinstance(node, ast.ClassDef):
                    # Check class-level duplicates
                    cls_names = collections.defaultdict(list)
                    for item in ast.iter_child_nodes(node):
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            cls_names[item.name].append(item.lineno)
                    for name, linenos in sorted(cls_names.items()):
                        if len(linenos) > 1:
                            issues.append((path, node.name, name, linenos))

            for name, linenos in sorted(mod_names.items()):
                if len(linenos) > 1:
                    issues.append((path, '<module>', name, linenos))

for path, scope, name, linenos in issues:
    lines_str = ', '.join(str(l) for l in linenos)
    print(f'SHADOW: {path}  {scope}.{name}  (lines {lines_str})')
```

For each duplicate found, report:
```
SHADOW: swarm/research/swarm_papers/track_a.py  <module>._build_related_work  (lines 57, 2258)
  The second definition silently shadows the first.
  Keep the most robust version (longest, or the one with helper calls)
```

### Step 2: Dead Imports

Run ruff's F401 (unused import) check across modified files:

```bash
ruff check swarm/ tests/ --select F401
```

Report any unused imports that may have been left behind by external edits.

### Step 3: Import Conflicts

Check for the same name imported from different modules in the same file:

```bash
# Scan for "from X import Name" and "from Y import Name" in same file
```

### Step 4: Merge Artifacts

Scan for leftover merge conflict markers:

```bash
grep -rn "^<<<<<<< \|^=======$\|^>>>>>>> " swarm/ tests/ scripts/
```

### Step 5: Stale Re-exports

Check `__init__.py` files for names that no longer exist in their source modules.

### Step 6: Report

```
Healthcheck Results
══════════════════════════════════════════
  Duplicates:     1 found (1 file)
  Dead imports:   0
  Import conflicts: 0
  Merge artifacts:  0
  Stale re-exports: 0
══════════════════════════════════════════
```

### `--hf-space` mode

Check the status of the HF Spaces deployment (rsavitt/swarm-sandbox).

1. **Health endpoint**:
   ```bash
   HTTP=$(curl -s -o /dev/null -w "%{http_code}" https://rsavitt-swarm-sandbox.hf.space/_stcore/health)
   ```

2. **Runtime status** (requires HF token at `~/.cache/huggingface/token`):
   ```bash
   curl -s -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
     "https://huggingface.co/api/spaces/rsavitt/swarm-sandbox/runtime"
   ```
   Extract `stage` and `sha` fields.

3. **Report**:
   ```
   HF Space: rsavitt/swarm-sandbox
     URL:    https://rsavitt-swarm-sandbox.hf.space
     Stage:  RUNNING
     SHA:    8f894e2...
     Health: 200 OK
   ```

   If stage is `BUILDING`, report that and suggest waiting. If `ERROR` or health != 200, suggest running `/deploy_demo`.

### `--fix` mode

When `/healthcheck --fix` is used:
- For duplicate definitions: keep the most robust version (prefer the one with helper function calls, proper escaping, or more complete logic). Remove the others.
- For dead imports: run `ruff check --fix --select F401`
- For merge artifacts and import conflicts: report only (manual fix required)

## Why This Exists

When multiple sessions (Claude Code, Codex, manual edits) commit to the same branch in parallel, HEAD race conditions can cause:
1. **Duplicate function definitions** — the same function added by two commits, both land
2. **Dead imports** — one commit removes a function, another adds an import for it
3. **Merge conflict markers** — incomplete merges leave `<<<<<<<` in files

These issues pass all existing checks (ruff won't flag duplicate `def` names, mypy crashes before reaching them, tests may still pass if Python uses the last definition). `/healthcheck` catches them.

## When to Run

- After `git pull` when other sessions have been active
- At the start of a resumed session (after `/status`)
- After merging a PR
- When tests pass but behavior seems wrong

## Relation to Other Commands

- `/status` shows git state — run it first
- `/healthcheck` checks code health after external changes
- `/preflight` checks staged code before commit
- `/stage` validates and stages files
