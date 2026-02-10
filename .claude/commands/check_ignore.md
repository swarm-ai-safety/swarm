# /check-ignore

Verify that `.gitignore` won't hide files in a target directory before you create them. Prevents the "wrote 7 files but git can't see them" footgun.

## Usage

`/check-ignore <path>`

Examples:
- `/check-ignore swarm/bridges/new_bridge/`
- `/check-ignore swarm/research/new_module/`

## Behavior

1) Run `git check-ignore -v <path>/__init__.py` (use a synthetic filename if the dir doesn't exist yet).

2) **If ignored**, report:
   - Which `.gitignore` rule is responsible (file, line number, pattern)
   - Whether the pattern is missing a leading `/` anchor (most common cause)
   - Suggest the fix (e.g. `bridges/` → `/bridges/`)
   - Ask the user whether to apply the fix

3) **If not ignored**, report "Path is visible to git" and proceed.

4) If the fix is applied, verify by re-running `git check-ignore` to confirm the path is now visible.

## Why this exists

`.gitignore` patterns without a leading `/` match anywhere in the tree. A rule like `bridges/` intended for a root-level directory will also match `swarm/bridges/live_swe/`. Files already tracked before the rule was added remain tracked, but new files in the same parent are silently invisible. This wastes time debugging "why doesn't git status show my files."

## Known patterns in this repo

The following `.gitignore` entries are anchored to root and should stay that way:
- `/bridges/` — root-level relocated directory (not `swarm/bridges/`)
- `submissions/` — root-level submissions dir
- `swarm-api/` — root-level API dir

If you add a new gitignore rule for a directory name that also exists under `swarm/`, anchor it with a leading `/`.
