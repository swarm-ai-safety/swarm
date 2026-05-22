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

# gitlawb dashboard backfill snapshot.
# Prefer the SoftMetrics-scored snapshot committed by the scheduled bridge run
# (.github/workflows/gitlawb-snapshot.yml), which mkdocs copies into site/. Only
# generate a raw fallback here if no committed snapshot exists, so a build never
# clobbers scored data. Fail-safe: a node outage writes an empty snapshot.
if [ ! -f site/bridges/gitlawb_snapshot.json ]; then
  "$VENV/bin/python" scripts/gen_gitlawb_snapshot.py site/bridges/gitlawb_snapshot.json
else
  echo "gitlawb snapshot: using committed scored snapshot from docs/bridges/"
fi
