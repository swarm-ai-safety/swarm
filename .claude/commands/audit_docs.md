# /audit_docs

Audit all project metadata files for stale counts, version mismatches, broken references, and missing entries. Reports discrepancies without auto-fixing.

## Scope

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

Run the `/check_nav` logic (see that command) and include results.

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
