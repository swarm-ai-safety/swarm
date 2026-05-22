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
# mkdocs has copied the committed scored snapshot (docs/bridges/gitlawb_snapshot.json)
# into site/. Refresh it with freshly SoftMetrics-scored data using an isolated venv
# (swarm core is light: numpy/pydantic/pandas + gql). On ANY failure, keep the
# committed snapshot so the deploy never breaks and scored data is preserved.
# A scheduled Vercel deploy hook (.github/workflows/gitlawb-snapshot.yml) re-runs
# this build to keep the snapshot fresh without pushing to a protected branch.
SNAP=site/bridges/gitlawb_snapshot.json
SNAP_VENV=.venv-snapshot
if uv venv --python "$PY" "$SNAP_VENV" \
  && uv pip install --python "$SNAP_VENV/bin/python" -e ".[runtime,gitlawb]" \
  && "$SNAP_VENV/bin/python" scripts/gen_gitlawb_snapshot.py "$SNAP.tmp" \
  && "$SNAP_VENV/bin/python" -c "import json,sys; sys.exit(0 if json.load(open('$SNAP.tmp'))['scored'] else 1)"; then
  mv "$SNAP.tmp" "$SNAP"
  echo "gitlawb snapshot: refreshed with build-time scored data"
else
  rm -f "$SNAP.tmp"
  if [ ! -f "$SNAP" ]; then
    "$VENV/bin/python" scripts/gen_gitlawb_snapshot.py "$SNAP" || true
  fi
  echo "gitlawb snapshot: kept existing snapshot (build-time scoring unavailable)"
fi
