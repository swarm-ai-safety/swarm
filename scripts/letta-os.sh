#!/usr/bin/env bash
# letta-os.sh — Headless wrapper for Claude Code as SWARM Research OS operator
#
# Usage:
#   ./scripts/letta-os.sh <command> [args...]
#
# Commands:
#   query <tag|date|type|claim> <value>   Search run history
#   synthesize <run_id>                   Generate vault notes from a run
#   sanity <scenario.yaml>                Quick validation run
#   regression [tests|baseline|full]      Regression check (default: full)
#   thread                                Show active research thread
#   log                                   Show recent research log entries
#   claims [list|status]                  Show claim inventory or health
#   close                                 End-of-session ritual
#   ask <question>                        Free-form question to the operator
#
# Options:
#   --json          Output as JSON
#   --new           Start a new conversation
#   --dry-run       Show the claude command without executing
#
# Examples:
#   ./scripts/letta-os.sh query tag governance
#   ./scripts/letta-os.sh synthesize 20260221-081106_redteam_contract_screening_no_collusion
#   ./scripts/letta-os.sh sanity scenarios/baseline.yaml
#   ./scripts/letta-os.sh regression
#   ./scripts/letta-os.sh thread
#   ./scripts/letta-os.sh ask "Which governance knobs are high leverage?"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Parse global flags ---
OUTPUT_FLAGS=()
EXTRA_FLAGS=()
DRY_RUN=false

args=()
for arg in "$@"; do
    case "$arg" in
        --json)    OUTPUT_FLAGS=(--output-format json) ;;
        --new)     EXTRA_FLAGS+=("--new") ;;
        --dry-run) DRY_RUN=true ;;
        *)         args+=("$arg") ;;
    esac
done

set -- "${args[@]:-}"

command="${1:-help}"
shift || true

run_operator() {
    local prompt="$1"
    local cmd=(claude -p "$prompt" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} ${OUTPUT_FLAGS[@]+"${OUTPUT_FLAGS[@]}"})

    if $DRY_RUN; then
        echo "[dry-run] ${cmd[*]}"
        return 0
    fi

    cd "$REPO_ROOT"
    exec "${cmd[@]}"
}

case "$command" in
    query|q)
        if [ $# -lt 1 ]; then
            echo "Usage: letta-os.sh query <tag|date|type|claim|recent> [value]" >&2
            exit 1
        fi
        query_type="$1"
        shift
        value="${*:-}"
        run_operator "Read .skills/run-query/SKILL.md and follow its instructions to query: $query_type $value"
        ;;

    synthesize|synth)
        if [ $# -lt 1 ]; then
            echo "Usage: letta-os.sh synthesize <run_id>" >&2
            exit 1
        fi
        run_operator "Read .skills/synthesize/SKILL.md and follow its instructions to synthesize run: $*"
        ;;

    sanity)
        if [ $# -lt 1 ]; then
            echo "Usage: letta-os.sh sanity <scenario.yaml>" >&2
            exit 1
        fi
        run_operator "Read .skills/sanity-check/SKILL.md and follow its instructions on scenario: $*"
        ;;

    regression|reg)
        mode="${1:-full}"
        run_operator "Read .skills/regression-check/SKILL.md and follow its instructions in mode: $mode"
        ;;

    thread|t)
        run_operator "Run /status --research. Read .letta/memory/threads/current.md and .letta/memory/runs/latest.md and display the active research context."
        ;;

    log|l)
        run_operator "Read .letta/memory/threads/research-log.md and show the last 5 entries."
        ;;

    claims|c)
        subcommand="${1:-list}"
        run_operator "Read .skills/claim/SKILL.md and follow its instructions: $subcommand"
        ;;

    close)
        run_operator "Run /ship --research-close. Summarize what changed this session, update memory files, commit and push."
        ;;

    ask|a)
        if [ $# -lt 1 ]; then
            echo "Usage: letta-os.sh ask <question>" >&2
            exit 1
        fi
        run_operator "$*"
        ;;

    help|--help|-h)
        # Print the header comment block as usage
        awk '/^#!/{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"
        ;;

    *)
        echo "Unknown command: $command" >&2
        echo "Run './scripts/letta-os.sh help' for usage." >&2
        exit 1
        ;;
esac
