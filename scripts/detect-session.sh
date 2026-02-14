#!/usr/bin/env bash
# detect-session.sh — Auto-detect git worktree session context.
#
# Source this file to export env vars describing the current session:
#   IS_SESSION_WORKTREE  (true/false)
#   SESSION_ID           (e.g. "session-2", or "" if not in a session worktree)
#   WORKTREE_ID          (e.g. "pane-2", or "")
#   SESSION_BRANCH       (e.g. "session/pane-2", or current branch)
#   MAIN_REPO_ROOT       (path to the main repo, even from inside a worktree)
#
# Usage:
#   source scripts/detect-session.sh
#   if [ "$IS_SESSION_WORKTREE" = "true" ]; then ...
#
# Pure read-only: no side effects beyond env vars.

_git_dir="$(git rev-parse --git-dir 2>/dev/null)"
_git_common_dir="$(git rev-parse --git-common-dir 2>/dev/null)"

if [ -z "$_git_dir" ]; then
    # Not inside a git repo at all
    export IS_SESSION_WORKTREE="false"
    export SESSION_ID=""
    export WORKTREE_ID=""
    export SESSION_BRANCH=""
    export MAIN_REPO_ROOT=""
    unset _git_dir _git_common_dir
    return 0 2>/dev/null || exit 0
fi

# Resolve to absolute paths for comparison
_git_dir="$(cd "$_git_dir" && pwd)"
_git_common_dir="$(cd "$_git_common_dir" && pwd)"

# Main repo root is always the parent of the common git dir
export MAIN_REPO_ROOT="$(dirname "$_git_common_dir")"

_current_branch="$(git branch --show-current 2>/dev/null)"
export SESSION_BRANCH="${_current_branch:-detached}"

# A worktree's .git dir differs from the common dir (main repo's .git)
if [ "$_git_dir" != "$_git_common_dir" ]; then
    # We're in a linked worktree — check if it's a session worktree
    _wt_dirname="$(basename "$(git rev-parse --show-toplevel 2>/dev/null)")"

    if [[ "$_wt_dirname" == session-* ]]; then
        export IS_SESSION_WORKTREE="true"
        export SESSION_ID="$_wt_dirname"
        # Extract pane number: session-2 -> pane-2
        _pane_num="${_wt_dirname#session-}"
        export WORKTREE_ID="pane-$_pane_num"
    else
        # Linked worktree but not a session one
        export IS_SESSION_WORKTREE="false"
        export SESSION_ID=""
        export WORKTREE_ID=""
    fi
else
    # Main worktree (not a linked worktree)
    export IS_SESSION_WORKTREE="false"
    export SESSION_ID=""
    export WORKTREE_ID=""
fi

unset _git_dir _git_common_dir _current_branch _wt_dirname _pane_num
