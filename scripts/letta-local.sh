#!/usr/bin/env bash
# letta-local.sh — Start/stop the local Letta stack (Ollama + Letta server + Letta Code)
#
# Usage:
#   ./scripts/letta-local.sh start     Start Ollama, Letta server, open Letta Code
#   ./scripts/letta-local.sh stop      Stop Letta server container
#   ./scripts/letta-local.sh status    Check if everything is running
#   ./scripts/letta-local.sh logs      Tail Letta server logs
#
# Prerequisites:
#   - Ollama installed with glm-4.7-flash:q8_0 pulled
#   - Docker running
#   - letta CLI installed (npm install -g @letta-ai/letta-code)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$REPO_ROOT/.letta/docker-compose.yml"
CONTAINER_NAME="letta-server"
MODEL="glm-4.7-flash:q8_0"

check_deps() {
    local missing=()
    command -v ollama  >/dev/null || missing+=("ollama")
    command -v docker  >/dev/null || missing+=("docker")
    command -v letta   >/dev/null || missing+=("letta (npm install -g @letta-ai/letta-code)")

    if [ ${#missing[@]} -gt 0 ]; then
        echo "Missing dependencies: ${missing[*]}" >&2
        exit 1
    fi
}

check_model() {
    if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
        echo "Model $MODEL not found. Pull it with:" >&2
        echo "  ollama pull $MODEL" >&2
        exit 1
    fi
}

start_stack() {
    check_deps
    check_model

    # 1. Ensure Ollama is serving
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Starting Ollama..."
        ollama serve &>/dev/null &
        sleep 2
    fi
    echo "Ollama: running ($(ollama list | grep "$MODEL" | awk '{print $1}'))"

    # 2. Start Letta server via Docker Compose
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "$CONTAINER_NAME"; then
        echo "Letta server: already running"
    else
        echo "Starting Letta server..."
        docker compose -f "$COMPOSE_FILE" -p letta up -d 2>&1
        echo "Waiting for server to be ready..."
        local retries=0
        while ! curl -s http://localhost:8283/v1/health >/dev/null 2>&1; do
            retries=$((retries + 1))
            if [ $retries -gt 30 ]; then
                echo "Letta server failed to start. Check: docker compose -f $COMPOSE_FILE logs" >&2
                exit 1
            fi
            sleep 2
        done
    fi
    echo "Letta server: running at http://localhost:8283"

    # 3. Set env and launch Letta Code
    echo ""
    echo "Stack is ready. Launch Letta Code with:"
    echo ""
    echo "  export LETTA_BASE_URL=http://localhost:8283"
    echo "  cd $REPO_ROOT && letta"
    echo ""
    echo "Or non-interactively:"
    echo ""
    echo "  export LETTA_BASE_URL=http://localhost:8283"
    echo "  ./scripts/letta-os.sh thread"
}

stop_stack() {
    echo "Stopping Letta server..."
    docker compose -f "$COMPOSE_FILE" -p letta down 2>&1
    echo "Letta server stopped."
    echo "(Ollama is still running — stop it with: pkill ollama)"
}

show_status() {
    echo "=== Ollama ==="
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Status: running"
        ollama list 2>/dev/null | grep -E "glm|llama" || echo "(no relevant models)"
    else
        echo "Status: not running"
    fi

    echo ""
    echo "=== Letta Server ==="
    if curl -s http://localhost:8283/v1/health >/dev/null 2>&1; then
        echo "Status: running at http://localhost:8283"
    elif docker ps --format '{{.Names}}' 2>/dev/null | grep -q "$CONTAINER_NAME"; then
        echo "Status: container running but not healthy"
    else
        echo "Status: not running"
    fi

    echo ""
    echo "=== Letta Code ==="
    if command -v letta >/dev/null 2>&1; then
        echo "Installed: $(letta --version 2>/dev/null || echo 'yes')"
    else
        echo "Not installed"
    fi

    echo ""
    echo "=== LETTA_BASE_URL ==="
    echo "${LETTA_BASE_URL:-not set (will use Letta Platform)}"
}

show_logs() {
    docker compose -f "$COMPOSE_FILE" -p letta logs -f 2>&1
}

case "${1:-help}" in
    start)   start_stack ;;
    stop)    stop_stack ;;
    status)  show_status ;;
    logs)    show_logs ;;
    help|--help|-h)
        sed -n '2,/^[^#]/{ /^#/s/^# \?//p; }' "$0"
        ;;
    *)
        echo "Unknown command: $1" >&2
        echo "Usage: letta-local.sh {start|stop|status|logs}" >&2
        exit 1
        ;;
esac
