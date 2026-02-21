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
#   5. Documentation reminder (new files in key directories)
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

# ── 5. Documentation reminder (major changes) ──
# Advisory: remind about CHANGELOG/README/docs when creating or modifying
# files in key directories. Never blocks.
IS_NEW=false
if ! git ls-files --error-unmatch "$FILE_PATH" >/dev/null 2>&1; then
    IS_NEW=true
fi

DOCS_HINT=""
case "$FILE_PATH" in
    */swarm/agents/*.py|*/swarm/governance/*.py|*/swarm/core/*.py|*/swarm/bridges/*/*.py)
        if [ "$IS_NEW" = true ]; then
            DOCS_HINT="New module in swarm/. Update CHANGELOG.md [Unreleased] and README.md if it adds a user-facing feature."
        fi
        ;;
    */scenarios/*.yaml|*/scenarios/*.yml)
        if [ "$IS_NEW" = true ]; then
            DOCS_HINT="New scenario. Update CHANGELOG.md [Unreleased] and README.md scenario count."
        fi
        ;;
    */.claude/commands/*.md)
        if [ "$IS_NEW" = true ]; then
            DOCS_HINT="New slash command. Update CHANGELOG.md [Unreleased]. Remember: prefer extending existing commands over creating new ones (see CLAUDE.md)."
        fi
        ;;
    */.claude/agents/*.md)
        if [ "$IS_NEW" = true ]; then
            DOCS_HINT="New agent. Update AGENTS.md and CHANGELOG.md [Unreleased]."
        fi
        ;;
    */swarm/api/*.py|*/swarm/api/**/*.py)
        DOCS_HINT="API change. Update CHANGELOG.md [Unreleased]. If adding/removing endpoints, update API docs."
        ;;
esac

if [ -n "$DOCS_HINT" ]; then
    echo "[swarm docs-reminder] $DOCS_HINT"
    # Log the reminder for compliance tracking at commit time
    REMINDER_LOG="$(git rev-parse --show-toplevel 2>/dev/null || echo ".")/.claude/docs_reminders.log"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)|$FILE_PATH|$DOCS_HINT" >> "$REMINDER_LOG"
fi

exit 0
