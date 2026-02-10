#!/usr/bin/env bash
set -euo pipefail
# Post-tool-use hook: warns if a written file looks like a runtime artifact
# that should be gitignored. Advisory only — never blocks.

if [ -z "${ARGUMENTS:-}" ]; then
    exit 0
fi

FILE_PATH=$(echo "$ARGUMENTS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path',''))" 2>/dev/null || true)

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

BASENAME=$(basename "$FILE_PATH")

# 1. If already gitignored, print informational note
if git check-ignore -q "$FILE_PATH" 2>/dev/null; then
    echo "Note: $FILE_PATH is gitignored and won't be committed."
    exit 0
fi

# 2. Check basename against patterns that SHOULD typically be ignored
SHOULD_IGNORE=false
case "$BASENAME" in
    .DS_Store|Thumbs.db)
        SHOULD_IGNORE=true ;;
    *.db|*.sqlite|*.sqlite3)
        SHOULD_IGNORE=true ;;
    *.log)
        SHOULD_IGNORE=true ;;
    *.pyc)
        SHOULD_IGNORE=true ;;
esac

# Check if file lives inside a directory that should be ignored
case "$FILE_PATH" in
    */node_modules/*)
        SHOULD_IGNORE=true ;;
    */__pycache__/*)
        SHOULD_IGNORE=true ;;
    */.env.*|*/.env)
        SHOULD_IGNORE=true ;;
esac

if [ "$SHOULD_IGNORE" = true ]; then
    echo "Warning: '$FILE_PATH' looks like a runtime artifact. Consider adding a pattern for it to .gitignore."
fi

exit 0
