# /release

Cut a `swarm-safety` release end-to-end: bump versions, finalize the CHANGELOG via a Release PR, tag `main`, and shepherd the `release.yml` pipeline through to the **PyPI publish** and GitHub release — then verify it actually landed.

Pushing a `v*` tag triggers `.github/workflows/release.yml`: `validate → test → build → publish (PyPI, OIDC trusted publishing) → github-release`. This command's job is to make that pipeline succeed on the first tag and to recover cleanly if it doesn't.

## Usage

`/release [version] [--commit SHA]`

Examples:
- `/release` (minor-bump from the latest tag, confirm with user)
- `/release 1.9.0`
- `/release --recover` (a tag exists but its run failed before publishing — re-validate, fix, re-point the tag)

## Critical invariants (these silently broke 1.6.0–1.8.0)

Before tagging, confirm all three or the publish will fail *after* a 10–60 min CI run:

1. **`release.yml` test extras** — the test job must install `.[dev,runtime,api,analysis,gamescape]` (matches `ci.yml`). A bare `.[dev,runtime]` misses `matplotlib` etc. → 14 collection errors → build/publish silently skipped.
2. **No direct-URL deps in `pyproject.toml`** — PyPI rejects PEP 508 direct refs (`pkg @ git+https://…`) with a 400 *after* build. Keep such deps out of declared extras (document a manual install instead).
3. **`release.yml` test runs `-n auto --dist=loadfile --timeout=300`** — serial + no timeout means ~60 min and a networked test can hang forever.

## Steps

1. **Pre-flight**: be on `main`, clean tree, synced (`git fetch origin main`; `HEAD == origin/main`). Determine the version: default = minor bump of `git tag -l 'v*' --sort=-v:refname | head -1`; confirm with the user. Must be valid semver.

2. **Bump versions** — run `/bump_version <X.Y.Z>` (updates `pyproject.toml`, `CITATION.cff`, `skill.md`). All three MUST match the tag, or `validate` fails.

3. **Finalize CHANGELOG** — roll `## [Unreleased]` → `## [X.Y.Z] - <YYYY-MM-DD>`, add a fresh empty `## [Unreleased]`. Read the diffs since the last tag and write human-quality entries under `### Added/Changed/Fixed/Removed`.

4. **Local pre-validation (BEFORE tagging — this catches the publish-blockers cheaply):**
   ```bash
   python -m build && python -m twine check dist/*
   ```
   Then inspect the built metadata for direct-URL deps that PyPI will reject:
   ```bash
   python -c "import zipfile,glob; w=glob.glob('dist/*.whl')[0]; m=[n for n in zipfile.ZipFile(w).namelist() if n.endswith('METADATA')][0]; t=zipfile.ZipFile(w).read(m).decode(); print('DIRECT-URL DEPS:', [l for l in t.splitlines() if 'git+' in l or '@ http' in l] or 'none')"
   ```
   If `twine check` fails or any direct-URL dep prints, fix `pyproject.toml` before proceeding (invariant 2).

5. **Release PR** — open `Release vX.Y.Z: finalize CHANGELOG, bump version` with the bump + CHANGELOG changes. Wait for CI green, then merge to `main` (squash). `git fetch origin main`.

6. **Tag & push** — tag the merged `main` commit and push:
   ```bash
   git tag -a vX.Y.Z origin/main -m "swarm-safety X.Y.Z"
   git push origin vX.Y.Z
   ```
   Do **NOT** run `gh release create` yourself — `release.yml`'s `github-release` job owns it. The tag version must equal `pyproject.toml`'s.

7. **Watch the pipeline** — `gh run watch` (or poll) the `release.yml` run. Expect `test` ~10 min. Report each job: `validate → test → build → publish → github-release`.

8. **Verify it landed**:
   ```bash
   curl -s https://pypi.org/pypi/swarm-safety/json | python -c "import sys,json;print('PyPI latest:', json.load(sys.stdin)['info']['version'])"
   # fresh-venv install + import as a smoke test:
   python -m venv /tmp/rel-check && /tmp/rel-check/bin/pip install -q "swarm-safety==X.Y.Z" && /tmp/rel-check/bin/python -c "import swarm; print('import OK')"
   gh release view vX.Y.Z --json tagName,url --jq '.tagName + " " + .url'
   ```

## Recovery (`--recover`): a tag exists but the run failed before publishing

If `publish` did **not** run (failure at `validate`/`test`/`build`), the version was never uploaded, so the tag can be safely re-pointed:
1. Diagnose: `gh run view <id> --log-failed`. Fix the cause on `main` via a PR (e.g. a `release.yml` extras/timeout fix, or a `pyproject` direct-dep) and merge.
2. Re-point the tag to the fixed commit:
   ```bash
   git push origin :refs/tags/vX.Y.Z   # delete remote tag
   git tag -d vX.Y.Z
   git fetch origin main
   git tag -a vX.Y.Z origin/main -m "swarm-safety X.Y.Z"
   git push origin vX.Y.Z              # re-triggers release.yml
   ```
3. Re-watch (step 7) and verify (step 8).

If `publish` itself failed with a PyPI **400 direct-dependency** error → invariant 2 (fix the extra, recover as above). If it failed because the **version already exists on PyPI**, that version is burned — bump to the next patch and start over (PyPI never lets a version be reused).

## Constraints

- **Re-pointing a tag is allowed ONLY if that version never published to PyPI.** Check first (`curl …/pypi/swarm-safety/json`). Never move a tag whose version is already live — bump instead.
- Never manually `gh release create` — `release.yml` owns the GitHub release (doing both double-creates).
- Never force-push `main`. Always confirm the version with the user before tagging.
- Pre-validate locally (step 4) before every tag — a failed tagged run costs a full CI cycle.
