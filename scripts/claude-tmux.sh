#!/usr/bin/env bash
# Launch multiple Claude Code sessions in a tmux layout, each in its own
# git worktree with a dedicated session branch.
#
# Usage:
#   ./scripts/claude-tmux.sh          # 2 panes (default)
#   ./scripts/claude-tmux.sh 3        # 3 panes
#   ./scripts/claude-tmux.sh kill     # kill the tmux session (worktrees kept)
#   ./scripts/claude-tmux.sh cleanup  # remove all session worktrees + branches
#
# Each pane gets:
#   worktree: .worktrees/session-<N>
#   branch:   session/pane-<N>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="claude"
WORKTREE_DIR="$REPO_ROOT/.worktrees"
ARG="${1:-2}"

# ── helpers ──────────────────────────────────────────────────────────────

setup_worktree() {
    local n="$1"
    local wt="$WORKTREE_DIR/session-$n"
    local branch="session/pane-$n"

    # Ensure we have the latest main
    git -C "$REPO_ROOT" fetch origin main --quiet 2>/dev/null || true

    if [ -d "$wt" ]; then
        # Worktree already exists — make sure the branch is checked out
        echo "Reusing worktree session-$n (branch $branch)"
        return 0
    fi

    mkdir -p "$WORKTREE_DIR"

    # Create or reset the branch to origin/main
    if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
        git -C "$REPO_ROOT" branch -f "$branch" origin/main 2>/dev/null || true
    fi

    # Create the worktree (creates the branch if it doesn't exist)
    git -C "$REPO_ROOT" worktree add -B "$branch" "$wt" origin/main --quiet
    echo "Created worktree session-$n (branch $branch)"
}

do_cleanup() {
    echo "Cleaning up session worktrees and branches..."
    local found=0

    # Remove worktrees
    for wt in "$WORKTREE_DIR"/session-*; do
        [ -d "$wt" ] || continue
        found=1
        local name
        name="$(basename "$wt")"
        echo "  Removing worktree $name"
        git -C "$REPO_ROOT" worktree remove --force "$wt" 2>/dev/null || rm -rf "$wt"
    done

    # Prune stale worktree refs
    git -C "$REPO_ROOT" worktree prune 2>/dev/null || true

    # Delete session branches
    for branch in $(git -C "$REPO_ROOT" branch --list 'session/pane-*' --format='%(refname:short)'); do
        found=1
        echo "  Deleting branch $branch"
        git -C "$REPO_ROOT" branch -D "$branch" 2>/dev/null || true
    done

    # Remove the .worktrees dir if empty
    rmdir "$WORKTREE_DIR" 2>/dev/null || true

    if [ "$found" -eq 0 ]; then
        echo "No session worktrees or branches found."
    else
        echo "Cleanup complete."
    fi
}

# ── subcommands ──────────────────────────────────────────────────────────

if [[ "$ARG" == "kill" ]]; then
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed session '$SESSION'" || echo "No session '$SESSION' to kill"
    exit 0
fi

if [[ "$ARG" == "cleanup" ]]; then
    # Kill tmux session first if it exists
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed tmux session '$SESSION'" || true
    do_cleanup
    exit 0
fi

PANES="$ARG"

# If session already exists, just attach
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Attaching..."
    exec tmux attach-session -t "$SESSION"
fi

# ── set up worktrees ─────────────────────────────────────────────────────

for ((i = 1; i <= PANES; i++)); do
    setup_worktree "$i"
done

# ── create tmux session ─────────────────────────────────────────────────

DETECT_SCRIPT="$REPO_ROOT/scripts/detect-session.sh"

WT1="$WORKTREE_DIR/session-1"
tmux new-session -d -s "$SESSION" -c "$WT1"
tmux rename-window -t "$SESSION":1 "claude-1"
tmux send-keys -t "$SESSION":1 "source '$DETECT_SCRIPT' && claude" Enter

for ((i = 2; i <= PANES; i++)); do
    WTi="$WORKTREE_DIR/session-$i"
    if (( i % 2 == 0 )); then
        tmux split-window -h -t "$SESSION" -c "$WTi"
    else
        tmux split-window -v -t "$SESSION" -c "$WTi"
    fi
    tmux send-keys -t "$SESSION" "source '$DETECT_SCRIPT' && claude" Enter
done

# Even out the layout
tmux select-layout -t "$SESSION" tiled

# Attach
exec tmux attach-session -t "$SESSION"
