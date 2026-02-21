# /tmux

Show tmux hotkey reference and session management commands. Quick cheat sheet for working with multiple Claude Code panes.

## Behavior

Print the following reference card:

```
tmux Hotkeys (prefix = Ctrl-b)
══════════════════════════════════════════════════

  Panes
  ─────────────────────────────────
  Ctrl-b |         Split vertical (side by side)
  Ctrl-b -         Split horizontal (top/bottom)
  Ctrl-b arrow     Move between panes
  Ctrl-b z         Zoom/unzoom pane (fullscreen)
  Ctrl-b x         Close pane
  Ctrl-b q         Show pane numbers (tap # to jump)
  Ctrl-b Shift-arrow  Resize pane

  Windows (tabs)
  ─────────────────────────────────
  Ctrl-b c         New window
  Ctrl-b n / p     Next / previous window
  Ctrl-b 1-9       Jump to window by number
  Ctrl-b ,         Rename window
  Ctrl-b &         Close window

  Sessions
  ─────────────────────────────────
  Ctrl-b d         Detach (session stays running)
  Ctrl-b s         List / switch sessions
  Ctrl-b $         Rename session

  Scroll / Copy
  ─────────────────────────────────
  Ctrl-b [         Enter scroll mode (arrows / PgUp)
  q                Exit scroll mode

  Mouse
  ─────────────────────────────────
  Click pane       Switch to it
  Drag border      Resize pane
  Scroll wheel     Scroll history

══════════════════════════════════════════════════

  Launch scripts
  ─────────────────────────────────
  ./scripts/claude-tmux.sh          2 panes (default)
  ./scripts/claude-tmux.sh 3        3 panes
  ./scripts/claude-tmux.sh kill     Kill tmux session (worktrees kept)
  ./scripts/claude-tmux.sh cleanup  Remove all session worktrees + branches

  Session worktrees
  ─────────────────────────────────
  Each pane runs in its own git worktree:
    .worktrees/session-<N>  →  branch session/pane-<N>

  This eliminates staging-area races and push conflicts
  between concurrent sessions.

  Workflow:
    /ship --all           Commit to session branch
    /merge_session       Rebase + push session branch to main
    /merge_session --all Batch merge all session branches
```

Also run `tmux ls 2>/dev/null || echo "No active tmux sessions"` to show current sessions.
