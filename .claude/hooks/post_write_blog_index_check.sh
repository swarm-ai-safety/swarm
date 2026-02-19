#!/usr/bin/env bash
set -euo pipefail
# Post-tool-use hook: warns if a new blog post was written but not added
# to the mkdocs.yml navigation. Advisory only — never blocks.

if [ -z "${ARGUMENTS:-}" ]; then
    exit 0
fi

FILE_PATH=$(echo "$ARGUMENTS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path',''))" 2>/dev/null || true)

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Only check files under docs/blog/ that are markdown (not index.md)
case "$FILE_PATH" in
    */docs/blog/*.md) ;;
    *) exit 0 ;;
esac

BASENAME=$(basename "$FILE_PATH")
if [ "$BASENAME" = "index.md" ]; then
    exit 0
fi

# Check if this blog post filename appears in mkdocs.yml nav
MKDOCS="$(git rev-parse --show-toplevel 2>/dev/null || echo ".")/mkdocs.yml"
if [ ! -f "$MKDOCS" ]; then
    exit 0
fi

if ! grep -qF "blog/$BASENAME" "$MKDOCS"; then
    echo "WARNING: Blog post '$BASENAME' is not in mkdocs.yml navigation."
    echo "Add an entry under the Blog section in mkdocs.yml, e.g.:"
    echo "    - \"Post Title\": blog/$BASENAME"
fi

exit 0
