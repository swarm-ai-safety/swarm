# /check_nav

Compare mkdocs.yml nav entries against actual files in `docs/`, reporting missing pages and orphaned files.

## Steps

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

## Constraints

- Read-only: do not edit mkdocs.yml, only report
- If all nav entries match, say so and exit
