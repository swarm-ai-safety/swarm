#!/usr/bin/env bash
# Copy research docs (theory.md, papers.md) from swarm-artifacts into
# docs/research/ so mkdocs can render them. docs/research/ is gitignored
# in this repo; the canonical copies live in swarm-artifacts.
#
# Usage:
#   scripts/sync-docs-from-artifacts.sh            # uses $SWARM_ARTIFACTS or ../swarm-artifacts
#   SWARM_ARTIFACTS=/path/to/swarm-artifacts scripts/sync-docs-from-artifacts.sh

set -euo pipefail

ARTIFACTS="${SWARM_ARTIFACTS:-$(cd "$(dirname "$0")/.." && pwd)/../swarm-artifacts}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/docs/research"

FILES=(theory.md papers.md)

if [[ ! -d "$ARTIFACTS" ]]; then
  echo "warn: swarm-artifacts not found at $ARTIFACTS — skipping sync (docs/research/ left as-is)" >&2
  echo "      set SWARM_ARTIFACTS=/path/to/swarm-artifacts to override" >&2
  exit 0
fi

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  src="$ARTIFACTS/research/$f"
  if [[ -f "$src" ]]; then
    cp "$src" "$DEST/$f"
    echo "synced: $src -> $DEST/$f"
  else
    echo "warn: $src missing — $DEST/$f left as-is" >&2
  fi
done
