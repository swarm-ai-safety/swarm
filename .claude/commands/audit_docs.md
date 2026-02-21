# /audit_docs

Audit all project metadata files for stale counts, version mismatches, broken references, and missing entries. Reports discrepancies without auto-fixing.

Use `--nav-only` to just check mkdocs.yml nav completeness (replaces the former `/check_nav` command).

## Usage

```
/audit_docs [--nav-only]
```

Examples:
- `/audit_docs` (full audit)
- `/audit_docs --nav-only` (just check mkdocs nav)

## Argument parsing

- `--nav-only`: Only run Phase 6 (mkdocs nav check) and report. Skip all other phases.

---

## `--nav-only` mode

Compare mkdocs.yml nav entries against actual files in `docs/`, reporting missing pages and orphaned files.

### 1. Extract nav paths

Read `mkdocs.yml` and extract all file paths from the `nav:` section (e.g. `getting-started/installation.md`, `blog/ecosystem-collapse.md`).

### 2. List actual docs

Glob `docs/**/*.md` to get all markdown files under `docs/`.

### 3. Compare

Report three categories:

#### Missing from nav
Files that exist in `docs/` but are NOT referenced in `mkdocs.yml` nav. Exclude:
- Files inside `docs/overrides/`
- Files inside `docs/posts/` (old blog location, if any)
- Files inside `docs/papers/` (research papers, not site pages)
- Files inside `docs/design/` (internal design docs)

#### Missing from disk
Nav entries that point to files that do NOT exist on disk.

#### Orphaned directories
Directories under `docs/` that contain `.md` files but have zero files in the nav.

### 4. Output format

```
mkdocs.yml Nav Check
====================

Nav entries: 62
Docs files:  85 (excluding overrides, posts, papers, design)

Missing from nav (should probably be added):
  - bridges/claude_code.md
  - bridges/prime_intellect.md

Missing from disk (broken links):
  (none)

Orphaned directories:
  (none)

Summary: 2 files not in nav, 0 broken links
```

---

## Full audit (default)

### Scope

Check these files against the actual codebase:

### 1. Version consistency

Read the version from `pyproject.toml` and verify it matches:
- `CITATION.cff` (`version:` field)
- `skill.md` (frontmatter `version:` field)
- `CHANGELOG.md` (first versioned `## [x.y.z]` entry should match)

Report any mismatches.

### 2. README.md counts

Count actual files and compare against README claims:
- Test count and file count (`tests/` — count `test_*` functions and `test_*.py` files)
- Scenario count (`scenarios/*.yaml` + `scenarios/*.yml`)
- Agent modules (`swarm/agents/*.py` excluding `__init__.py`)
- Example scripts (`examples/*.py` at root level)
- Governance modules (`swarm/governance/*.py` excluding `__init__.py`)
- Bridge files (`swarm/bridges/**/*.py`)
- Module counts for each directory listed in the Directory Structure section
- LLM provider count (check `LLMProvider` enum in `swarm/agents/llm_config.py`)

### 3. AGENTS.md vs .claude/agents/

- List all `.md` files in `.claude/agents/`
- Check each has a corresponding section in `AGENTS.md`
- Check the "How To Choose" section lists all agents

### 4. CHANGELOG.md

- Check `[Unreleased]` section exists
- Verify recent commits (since last versioned entry) are reflected in Unreleased

### 5. SECURITY.md

- Check supported versions table references current major.minor

### 6. mkdocs.yml nav completeness

Run the `--nav-only` logic above and include results.

### 7. Broken file references

For each checked file, verify that referenced file paths actually exist:
- README.md example file links
- CLAUDE.md referenced source files and test fixtures
- CONTRIBUTING.md project structure
- MEMORY_TESTING.md test files and baselines
- mkdocs.yml nav entries

## Output format

```
Audit Report
============

Version: pyproject.toml=1.7.0
  CITATION.cff:  1.7.0  OK
  skill.md:      1.7.0  OK
  CHANGELOG.md:  1.7.0  OK

Counts (README.md):
  Tests:       README says 4556, actual 4556  OK
  Scenarios:   README says 78, actual 80      STALE
  ...

AGENTS.md:
  All 7 agents listed  OK

mkdocs.yml:
  Missing from nav: docs/bridges/new_bridge.md
  Orphaned nav entry: (none)

Broken references:
  CLAUDE.md:  tests/fixtures/interactions.py  OK
  ...

Summary: 2 issues found
```

## Constraints

- Read-only: do not edit any files, only report discrepancies
- Use parallel tool calls where possible (e.g. glob multiple directories at once)
- For test count, count `def test_` patterns rather than running pytest --collect-only (faster)

## Migration from old commands

| Old command | Equivalent |
|---|---|
| `/check_nav` | `/audit_docs --nav-only` |
