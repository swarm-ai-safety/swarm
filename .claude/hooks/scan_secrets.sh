#!/usr/bin/env bash
set -euo pipefail
# ──────────────────────────────────────────────────────────────
# scan_secrets.sh — Scan files for hardcoded API keys & secrets
#
# Usage:
#   bash .claude/hooks/scan_secrets.sh           # scan whole repo
#   bash .claude/hooks/scan_secrets.sh scripts/  # scan a directory
#   bash .claude/hooks/scan_secrets.sh file.py   # scan one file
#
# All regex patterns use POSIX ERE compatible with macOS grep -E
# (no {N} quantifiers — use character repetition instead).
# ──────────────────────────────────────────────────────────────

TARGET="${1:-.}"

# ── Secret patterns ──────────────────────────────────────────
# Each entry: "label:::regex"
# Patterns avoid {N} quantifiers for macOS grep -E compatibility.
# We match the distinctive prefix + enough trailing chars to avoid
# false positives, using [class][class]+ instead of [class]{N}.
PATTERNS=(
    # Platform-specific keys (prefix + 16+ hex chars)
    "AgentXiv API key:::ax_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]+"
    "ClawXiv API key:::clx_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]+"
    "Moltbook API key:::moltbook_sk_[A-Za-z0-9_][A-Za-z0-9_][A-Za-z0-9_][A-Za-z0-9_][A-Za-z0-9_][A-Za-z0-9_][A-Za-z0-9_][A-Za-z0-9_]+"
    "Moltipedia API key:::moltipedia_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]+"
    "Wikimolt API key:::wm_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]+"
    "Clawchan API key:::clawchan_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]+"
    "Clawk API key:::clawk_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]+"

    # Cloud / LLM provider keys
    "OpenAI API key:::sk-[A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9]+"
    "Anthropic API key:::sk-ant-[A-Za-z0-9_-][A-Za-z0-9_-][A-Za-z0-9_-][A-Za-z0-9_-][A-Za-z0-9_-][A-Za-z0-9_-][A-Za-z0-9_-][A-Za-z0-9_-][A-Za-z0-9_-][A-Za-z0-9_-]+"
    "AWS access key:::AKIA[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]"
    "GitHub personal token:::ghp_[A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9]+"
    "GitHub OAuth token:::gho_[A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9]+"
    "Stripe live key:::sk_live_[A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9]+"

    # Generic patterns
    "Bearer token in string:::Bearer [A-Za-z0-9_.-][A-Za-z0-9_.-][A-Za-z0-9_.-][A-Za-z0-9_.-][A-Za-z0-9_.-][A-Za-z0-9_.-][A-Za-z0-9_.-][A-Za-z0-9_.-][A-Za-z0-9_.-][A-Za-z0-9_.-]+"
    "PEM private key:::-----BEGIN.*PRIVATE KEY-----"
)

# ── Scan ─────────────────────────────────────────────────────

FOUND=0
FILES_SCANNED=0
ISSUES=()

for entry in "${PATTERNS[@]}"; do
    LABEL="${entry%%:::*}"
    REGEX="${entry##*:::}"

    # Use grep -E (POSIX ERE) with file type filtering; skip binary files.
    # Use -e to pass the pattern explicitly (avoids patterns starting with
    # "--" being parsed as flags, e.g. "-----BEGIN PRIVATE KEY-----").
    MATCHES=$(grep -rnI \
        --include="*.py" --include="*.js" --include="*.ts" \
        --include="*.yaml" --include="*.yml" --include="*.json" \
        --include="*.sh" --include="*.env" --include="*.md" \
        --include="*.toml" --include="*.cfg" --include="*.ini" \
        --exclude-dir=".git" --exclude-dir="node_modules" \
        --exclude-dir=".venv" --exclude-dir="venv" \
        --exclude-dir="__pycache__" --exclude-dir=".mypy_cache" \
        --exclude-dir="external" \
        --exclude-dir="credentials" --exclude-dir=".credentials" \
        -E -e "$REGEX" "$TARGET" 2>/dev/null || true)

    if [ -n "$MATCHES" ]; then
        # Filter out known false positives (test fixtures, docs, this script).
        # Use grep -F (fixed strings) where possible to avoid regex pitfalls.
        FILTERED=$(echo "$MATCHES" | \
            grep -Fv 'clx_test' | \
            grep -Fv 'clx_YOUR' | \
            grep -Fv 'YOUR_TOKEN' | \
            grep -Fv 'YOUR_API_KEY' | \
            grep -Fv 'your_key_here' | \
            grep -Fv 'fake-key' | \
            grep -Fv 'test-key' | \
            grep -Fv '1234567890' | \
            grep -Fv '[REDACTED]' | \
            grep -Fv 'scan_secrets' | \
            grep -Fv 'post_write_check' | \
            grep -Fv 'pre-commit:' | \
            grep -Fv 'example' | \
            grep -Fv 'placeholder' | \
            grep -Fv 'os.environ' | \
            grep -Fv 'getenv' | \
            grep -Fv '${' | \
            grep -Fv 'PATTERNS' | \
            grep -Fv 'REGEX' || true)

        if [ -n "$FILTERED" ]; then
            FOUND=1
            while IFS= read -r line; do
                # Show file:line but redact the secret value
                FILE_LINE="${line%%:*}:$(echo "${line#*:}" | cut -d: -f1)"
                ISSUES+=("  [$LABEL] $FILE_LINE")
            done <<< "$FILTERED"
        fi
    fi
done

# Count files scanned
if [ -f "$TARGET" ]; then
    FILES_SCANNED=1
elif [ -d "$TARGET" ]; then
    FILES_SCANNED=$(find "$TARGET" \
        \( -name "*.py" -o -name "*.js" -o -name "*.ts" \
           -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" \
           -o -name "*.sh" -o -name "*.env" -o -name "*.md" \
           -o -name "*.toml" \) \
        -not -path "*/.git/*" \
        -not -path "*/node_modules/*" \
        -not -path "*/.venv/*" \
        -not -path "*/__pycache__/*" \
        -not -path "*/external/*" \
        2>/dev/null | wc -l | tr -d ' ')
fi

# ── Report ───────────────────────────────────────────────────

echo "========================================"
echo "  SWARM Secrets Scanner"
echo "========================================"
echo ""
echo "Target:  $TARGET"
echo "Files:   $FILES_SCANNED scanned"
echo ""

if [ $FOUND -eq 1 ]; then
    echo "STATUS:  SECRETS DETECTED"
    echo ""
    echo "Issues:"
    for issue in "${ISSUES[@]}"; do
        echo "$issue"
    done
    echo ""
    echo "----------------------------------------"
    echo "Remediation:"
    echo "  - Use environment variables:"
    echo "      api_key = os.environ.get('CLAWXIV_API_KEY')"
    echo "  - Use credential files (gitignored):"
    echo "      ~/.config/<platform>/credentials.json"
    echo "  - Use variable interpolation in config:"
    echo "      \${GITHUB_TOKEN}"
    echo "----------------------------------------"
    exit 1
else
    echo "STATUS:  CLEAN"
    echo "No hardcoded secrets detected."
    exit 0
fi
