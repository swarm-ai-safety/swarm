#!/usr/bin/env bash
# letta-os.sh — Headless wrapper for Letta Code as SWARM Research OS operator
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
#   --stream        Stream output (for long tasks)
#   --new           Start a new conversation
#   --dry-run       Show the letta command without executing
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
EXTRA_FLAGS=("--yolo")
DRY_RUN=false

# Auto-detect local Letta server and set model accordingly
if [ "${LETTA_BASE_URL:-}" = "http://localhost:8283" ]; then
    EXTRA_FLAGS+=("--model" "local-ollama/qwen2.5:14b")
fi

args=()
for arg in "$@"; do
    case "$arg" in
        --json)   OUTPUT_FLAGS=(--output-format json) ;;
        --stream) OUTPUT_FLAGS=(--output-format stream-json) ;;
        --new)    EXTRA_FLAGS+=("--new") ;;
        --dry-run) DRY_RUN=true ;;
        --model=*) EXTRA_FLAGS+=("--model" "${arg#--model=}") ;;
        *)        args+=("$arg") ;;
    esac
done

set -- "${args[@]:-}"

command="${1:-help}"
shift || true

run_letta() {
    local prompt="$1"
    local cmd=(letta -p "$prompt" "${EXTRA_FLAGS[@]}" ${OUTPUT_FLAGS[@]+"${OUTPUT_FLAGS[@]}"})

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
        run_letta "Use the run-query skill: $query_type $value"
        ;;

    synthesize|synth)
        if [ $# -lt 1 ]; then
            echo "Usage: letta-os.sh synthesize <run_id>" >&2
            exit 1
        fi
        run_letta "Use the synthesize skill on run: $*"
        ;;

    sanity)
        if [ $# -lt 1 ]; then
            echo "Usage: letta-os.sh sanity <scenario.yaml>" >&2
            exit 1
        fi
        run_letta "Use the sanity-check skill on scenario: $*"
        ;;

    regression|reg)
        mode="${1:-full}"
        run_letta "Use the regression-check skill in mode: $mode"
        ;;

    thread|t)
        run_letta "What's the active research thread? Read .letta/memory/threads/current.md and summarize."
        ;;

    log|l)
        run_letta "Show the last 5 entries from the research log at .letta/memory/threads/research-log.md"
        ;;

    claims|c)
        subcommand="${1:-list}"
        run_letta "Use the claim skill: $subcommand"
        ;;

    close)
        run_letta "Use the session-close skill. Summarize what changed, update memory, commit and push."
        ;;

    ask|a)
        if [ $# -lt 1 ]; then
            echo "Usage: letta-os.sh ask <question>" >&2
            exit 1
        fi
        run_letta "$*"
        ;;

    help|--help|-h)
        # Print the header comment block as usage
        sed -n '2,/^[^#]/{ /^#/s/^# \?//p; }' "$0"
        ;;

    *)
        echo "Unknown command: $command" >&2
        echo "Run './scripts/letta-os.sh help' for usage." >&2
        exit 1
        ;;
esac
