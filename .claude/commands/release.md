# /release

Tag a commit and create a GitHub release with auto-generated notes.

## Usage

`/release [version] [--commit SHA]`

Examples:
- `/release` (auto-increment from latest tag)
- `/release v1.1.0` (explicit version)
- `/release v1.0.1 --commit abc1234` (tag a specific commit)

## Behavior

1) **Determine version**:
- If `<version>` is provided, use it (must start with `v`).
- Otherwise, find the latest tag with `git tag -l 'v*' --sort=-v:refname | head -1`.
  - If no tags exist, default to `v0.1.0`.
  - Otherwise, bump the minor version (e.g. `v1.0.0` → `v1.1.0`).
  - Ask the user to confirm or override.

2) **Determine commit**:
- If `--commit SHA` is provided, use that commit.
- Otherwise, use HEAD.
- Show the commit message and ask the user to confirm.

3) **Generate release notes**:
- List all commits since the previous tag (or all commits if first release): `git log <prev-tag>..HEAD --oneline`
- Group commits by prefix pattern:
  - `Add/Create` → "New"
  - `Fix` → "Fixes"
  - `Update/Refactor/Enhance` → "Improvements"
  - `Remove/Delete` → "Removed"
  - Everything else → "Other"
- Include a "Quick Start" section with install + run instructions.

4) **Update CHANGELOG.md**:
- Read `CHANGELOG.md`.
- Convert the `## [Unreleased]` section into a versioned entry: `## [<version>] - <YYYY-MM-DD>`.
- Organize entries under Keep a Changelog categories (`### Added`, `### Changed`, `### Fixed`, `### Removed`). Use the grouped commits from step 3 as a starting point, but **read the actual diffs** to write human-quality descriptions rather than echoing raw commit messages. Highlight:
  - New modules, handlers, bridges, or agent types
  - New scenario configs or governance levers
  - Test count changes (e.g. "2922 tests" → "2950 tests")
  - Breaking changes or renamed APIs
- Add a fresh empty `## [Unreleased]` section above the new entry.
- Commit the CHANGELOG update: `git add CHANGELOG.md && git commit -m "Update CHANGELOG for <version>"`.
- If pyproject.toml version doesn't match `<version>` (without the `v` prefix), update it and amend the commit.

5) **Create tag and release**:
- `git tag <version> <commit> -m "<version>: <summary>"`
- `git push origin <version>`
- `gh release create <version> --title "<version>: <summary>" --notes "<generated notes>"`

6) **Print the release URL**.

## Constraints

- Never overwrite an existing tag. If the version already exists, abort and tell the user.
- Never force-push tags.
- Always confirm the version and commit with the user before tagging.
- If `gh` CLI is not available, fall back to just creating the local tag and pushing it.
