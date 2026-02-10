#!/usr/bin/env bash
# Launch multiple Claude Code sessions in a tmux layout.
#
# Usage:
#   ./scripts/claude-tmux.sh          # 2 panes (default)
#   ./scripts/claude-tmux.sh 3        # 3 panes
#   ./scripts/claude-tmux.sh kill     # kill the session
#
# The session is named "claude" and uses the repo root as the working directory.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="claude"
PANES="${1:-2}"

if [[ "${PANES}" == "kill" ]]; then
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed session '$SESSION'" || echo "No session '$SESSION' to kill"
    exit 0
fi

# If session already exists, just attach
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Attaching..."
    exec tmux attach-session -t "$SESSION"
fi

# Create session with first pane
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT"
tmux rename-window -t "$SESSION":1 "claude-1"
tmux send-keys -t "$SESSION":1 "claude" Enter

# Create additional panes
for ((i = 2; i <= PANES; i++)); do
    if (( i % 2 == 0 )); then
        tmux split-window -h -t "$SESSION" -c "$REPO_ROOT"
    else
        tmux split-window -v -t "$SESSION" -c "$REPO_ROOT"
    fi
    tmux send-keys -t "$SESSION" "claude" Enter
done

# Even out the layout
tmux select-layout -t "$SESSION" tiled

# Attach
exec tmux attach-session -t "$SESSION"
