#!/usr/bin/env bash
set -euo pipefail

PY=3.12
VENV=.venv-build

uv python install "$PY"
uv venv --python "$PY" "$VENV"
uv pip install --python "$VENV/bin/python" \
  mkdocs-material \
  'mkdocstrings[python]' \
  pymdown-extensions \
  mkdocs-git-revision-date-localized-plugin \
  mkdocs-rss-plugin

(cd viz && npm install && npm run build:deploy)

"$VENV/bin/mkdocs" build

# Generate a same-origin backfill snapshot for the gitlawb live dashboard.
# Fail-safe: a node outage writes an empty snapshot rather than failing the build.
"$VENV/bin/python" scripts/gen_gitlawb_snapshot.py site/bridges/gitlawb_snapshot.json
