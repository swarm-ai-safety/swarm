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
