# /scrub_id

Find and remove private infrastructure IDs (run IDs, eval IDs, dashboard links) from the repo and redeploy if public-facing files were changed.

## Usage

`/scrub_id <pattern_or_id>`

Examples:
- `/scrub_id abc123def456`
- `/scrub_id "app.primeintellect.ai"`
- `/scrub_id id_one id_two id_three`

## Behavior

### Phase 1: Search

1. For each provided ID or pattern, grep the entire repo (excluding `.git/`, `external/`, `runs/`, `node_modules/`, `.venv/`, `__pycache__/`).
2. Also grep for URL patterns containing the ID (e.g. dashboard links embedding the ID).
3. Present all matches with file path, line number, and surrounding context (3 lines).

### Phase 2: Classify matches

Categorize each match as:
- **Public** — files under `docs/`, blog posts, README, any `.md` served by the site. These are highest priority.
- **Internal** — command templates (`.claude/commands/`), scripts, config. Lower priority but still worth cleaning.
- **Data** — run artifacts, logs, CSVs. Usually gitignored; flag but don't auto-edit.

### Phase 3: Edit

For each match, propose a replacement:
- Dashboard URLs: remove the entire link, keep descriptive text (e.g. "Full logs on the Prime dashboard" becomes "Trained on Prime Intellect")
- Bare run/eval IDs: replace with a generic placeholder (e.g. `"your-external-run-id"`) or remove the line
- IDs in code examples: replace with `"your-run-id"` or similar

Ask the user to confirm before editing. Apply edits.

### Phase 4: Verify

After edits:
1. Re-grep for each pattern to confirm zero remaining matches.
2. If any public-facing files (`docs/**`, `*.md` in repo root) were modified, prompt to run `/deploy_blog`.

### Phase 5: Commit

Stage the changed files and commit with message:
```
Remove private infrastructure IDs from <list of affected areas>
```

Push if the user confirms.

## Also check the live site

If the scrubbed content was in a blog post or docs page, use WebFetch to verify the live site at `https://www.swarm-ai.org/` still shows the old (cached) content, and confirm that a redeploy will fix it.

## Constraints

- Never auto-edit without showing the proposed changes first.
- Never remove entire paragraphs — only the specific IDs and their surrounding links.
- Preserve useful metadata (model name, environment, batch size, wall-clock time) when removing IDs.
- After scrubbing, the pre-commit hook (section 1b in `.claude/hooks/pre-commit`) should catch any re-introduction of these patterns.
- **Never include real IDs in this command file or its examples.** Use obviously fake placeholders only.
