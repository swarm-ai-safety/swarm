#!/usr/bin/env bash
set -euo pipefail
# ──────────────────────────────────────────────────────────────
# post_write_check.sh — Consolidated post-Write/Edit hook
#
# Runs all write-time checks in sequence:
#   1. Secrets scan (code/config files only)
#   2. Gitignore coverage (advisory)
#   3. Ruff lint (Python files only)
#   4. Blog nav check (docs/blog/*.md only)
#
# Called automatically by Claude Code after Write/Edit tool calls.
# Consolidates: post_write_secrets_check.sh, post_write_gitignore_check.sh,
#               post_write_lint_check.sh, post_write_blog_index_check.sh
# ──────────────────────────────────────────────────────────────

if [ -z "${ARGUMENTS:-}" ]; then
    exit 0
fi

FILE_PATH=$(echo "$ARGUMENTS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path',''))" 2>/dev/null || true)

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

HOOK_DIR="$(dirname "$0")"
BASENAME=$(basename "$FILE_PATH")

# ── 1. Secrets scan (code/config files only) ──
if [ -f "$FILE_PATH" ]; then
    case "$FILE_PATH" in
        *.py|*.js|*.ts|*.yaml|*.yml|*.json|*.sh|*.env|*.md|*.toml|*.cfg|*.ini)
            bash "$HOOK_DIR/scan_secrets.sh" "$FILE_PATH"
            ;;
    esac
fi

# ── 2. Gitignore coverage (advisory) ──
if git check-ignore -q "$FILE_PATH" 2>/dev/null; then
    echo "Note: $FILE_PATH is gitignored and won't be committed."
else
    SHOULD_IGNORE=false
    case "$BASENAME" in
        .DS_Store|Thumbs.db)    SHOULD_IGNORE=true ;;
        *.db|*.sqlite|*.sqlite3) SHOULD_IGNORE=true ;;
        *.log)                   SHOULD_IGNORE=true ;;
        *.pyc)                   SHOULD_IGNORE=true ;;
    esac
    case "$FILE_PATH" in
        */node_modules/*)  SHOULD_IGNORE=true ;;
        */__pycache__/*)   SHOULD_IGNORE=true ;;
        */.env.*|*/.env)   SHOULD_IGNORE=true ;;
    esac
    if [ "$SHOULD_IGNORE" = true ]; then
        echo "Warning: '$FILE_PATH' looks like a runtime artifact. Consider adding a pattern for it to .gitignore."
    fi
fi

# ── 3. Ruff lint (Python files only) ──
case "$FILE_PATH" in
    *.py)
        if [ -f "$FILE_PATH" ] && command -v ruff >/dev/null 2>&1; then
            ISSUES=$(ruff check "$FILE_PATH" 2>&1 || true)
            if [ -n "$ISSUES" ]; then
                echo "[swarm post-write-lint] Issues in $FILE_PATH:"
                echo "$ISSUES"
                echo ""
                echo "[swarm post-write-lint] Fix with: ruff check --fix $FILE_PATH"
            fi
        fi
        ;;
esac

# ── 4. Blog nav check (docs/blog/*.md only, not index.md) ──
case "$FILE_PATH" in
    */docs/blog/*.md)
        if [ "$BASENAME" != "index.md" ]; then
            MKDOCS="$(git rev-parse --show-toplevel 2>/dev/null || echo ".")/mkdocs.yml"
            if [ -f "$MKDOCS" ] && ! grep -qF "blog/$BASENAME" "$MKDOCS"; then
                echo "WARNING: Blog post '$BASENAME' is not in mkdocs.yml navigation."
                echo "Add an entry under the Blog section in mkdocs.yml, e.g.:"
                echo "    - \"Post Title\": blog/$BASENAME"
            fi
        fi
        ;;
esac

exit 0
