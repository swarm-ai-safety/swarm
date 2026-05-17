# /bump_version

Update the project version across all files that track it. Takes a version string as argument.

## Usage

```
/bump_version 1.8.0
```

If no argument is provided, read the current version from `pyproject.toml` and report it, then ask the user what the new version should be.

## Files to update

1. **`pyproject.toml`** — `version = "X.Y.Z"` under `[project]`
2. **`CITATION.cff`** — `version: "X.Y.Z"`
3. **`skill.md`** — `version: X.Y.Z` in frontmatter

## Steps

1. Read all three files to get current versions
2. Report current state:
   ```
   Current versions:
     pyproject.toml:  1.7.0
     CITATION.cff:    1.7.0
     skill.md:        1.7.0
   ```
3. Edit each file to the new version
4. Report what changed:
   ```
   Updated to 1.8.0:
     pyproject.toml   1.7.0 → 1.8.0
     CITATION.cff     1.7.0 → 1.8.0
     skill.md         1.7.0 → 1.8.0
   ```

## Constraints

- Do NOT commit or push — the user may want to bundle this with other release changes
- Do NOT update CHANGELOG.md — the `/release` command handles that
- Validate the version looks like semver (X.Y.Z) before applying
- If any file already has the target version, skip it and note "already at X.Y.Z"
