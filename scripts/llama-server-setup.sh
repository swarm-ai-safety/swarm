#!/usr/bin/env bash
# llama-server-setup.sh — Download a GGUF model and manage a local llama-server.
#
# Usage:
#   ./scripts/llama-server-setup.sh download   # fetch a recommended small model
#   ./scripts/llama-server-setup.sh start      # start llama-server
#   ./scripts/llama-server-setup.sh stop       # stop llama-server
#   ./scripts/llama-server-setup.sh status     # check if server is running
#
# Environment variables (override defaults):
#   LLAMA_MODEL_URL   — HuggingFace direct-download URL for the GGUF file
#   LLAMA_MODEL_PATH  — Local path to store/load the model
#   LLAMA_PORT        — Port for llama-server (default: 8080)
#   LLAMA_THREADS     — Number of CPU threads (default: auto)
#   LLAMA_CTX         — Context window size (default: 4096)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"

# Defaults — Llama 3.2 3B Instruct Q4_K_M (~2 GB, good for MacBook CPU)
DEFAULT_MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
DEFAULT_MODEL_NAME="Llama-3.2-3B-Instruct-Q4_K_M.gguf"

MODEL_URL="${LLAMA_MODEL_URL:-$DEFAULT_MODEL_URL}"
MODEL_PATH="${LLAMA_MODEL_PATH:-${MODELS_DIR}/${DEFAULT_MODEL_NAME}}"
PORT="${LLAMA_PORT:-8080}"
THREADS="${LLAMA_THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
CTX="${LLAMA_CTX:-4096}"
PID_FILE="${MODELS_DIR}/.llama-server.pid"

_info()  { echo "==> $*"; }
_error() { echo "ERROR: $*" >&2; }

cmd_download() {
    mkdir -p "$MODELS_DIR"

    if [ -f "$MODEL_PATH" ]; then
        _info "Model already exists: $MODEL_PATH"
        _info "Size: $(du -h "$MODEL_PATH" | cut -f1)"
        return 0
    fi

    _info "Downloading model to $MODEL_PATH ..."
    _info "URL: $MODEL_URL"
    _info "This may take a few minutes (~2 GB)."

    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
    elif command -v wget &>/dev/null; then
        wget --show-progress -O "$MODEL_PATH" "$MODEL_URL"
    else
        _error "Neither curl nor wget found. Install one and retry."
        exit 1
    fi

    _info "Download complete: $(du -h "$MODEL_PATH" | cut -f1)"
}

cmd_start() {
    if ! command -v llama-server &>/dev/null; then
        _error "llama-server not found on PATH."
        echo ""
        echo "Install options:"
        echo "  macOS:   brew install llama.cpp"
        echo "  build:   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build --config Release -t llama-server"
        echo "  binary:  download from https://github.com/ggml-org/llama.cpp/releases"
        exit 1
    fi

    if [ ! -f "$MODEL_PATH" ]; then
        _error "Model not found at $MODEL_PATH"
        echo "Run: $0 download"
        exit 1
    fi

    # Check if already running
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        _info "llama-server already running (PID $(cat "$PID_FILE")) on port $PORT"
        return 0
    fi

    _info "Starting llama-server on port $PORT with $THREADS threads ..."
    _info "Model: $MODEL_PATH"
    _info "Context: $CTX tokens"

    llama-server \
        -m "$MODEL_PATH" \
        --port "$PORT" \
        --ctx-size "$CTX" \
        --threads "$THREADS" \
        --log-disable \
        &

    local pid=$!
    echo "$pid" > "$PID_FILE"

    # Wait for health endpoint
    _info "Waiting for server to be ready ..."
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            _info "llama-server is ready (PID $pid, port $PORT)"
            return 0
        fi
        sleep 1
    done

    _error "Server did not become healthy within 30 seconds."
    kill "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    exit 1
}

cmd_stop() {
    if [ ! -f "$PID_FILE" ]; then
        _info "No PID file found; server not running."
        return 0
    fi

    local pid
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
        _info "Stopping llama-server (PID $pid) ..."
        kill "$pid"
        rm -f "$PID_FILE"
        _info "Stopped."
    else
        _info "Process $pid not running; cleaning up PID file."
        rm -f "$PID_FILE"
    fi
}

cmd_status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        _info "llama-server running (PID $(cat "$PID_FILE"))"
    else
        _info "llama-server not running"
    fi

    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        _info "Health endpoint OK on port $PORT"
    else
        _info "Health endpoint unreachable on port $PORT"
    fi
}

# --- Main ---
case "${1:-help}" in
    download) cmd_download ;;
    start)    cmd_start ;;
    stop)     cmd_stop ;;
    status)   cmd_status ;;
    *)
        echo "Usage: $0 {download|start|stop|status}"
        echo ""
        echo "  download  Fetch a recommended GGUF model (~2 GB)"
        echo "  start     Start llama-server (downloads model if missing)"
        echo "  stop      Stop the running llama-server"
        echo "  status    Check server status"
        exit 1
        ;;
esac
