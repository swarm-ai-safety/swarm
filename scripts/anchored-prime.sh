#!/usr/bin/env bash
# anchored-prime.sh — Anchored Iterative Summary for context recovery
#
# Generates a structured summary optimized for post-compaction context reload.
# Based on the "Anchored Iterative Summarization" pattern from
# Agent-Skills-for-Context-Engineering (context-compression skill).
#
# Design: Each section is an explicit anchor that forces preservation of
# high-signal information. Sections are populated from live sources
# (git, beads, .letta/memory) so the summary is always current.
#
# Usage:
#   ./scripts/anchored-prime.sh           # Full anchored summary
#   ./scripts/anchored-prime.sh --brief   # Minimal (for MCP / low-token contexts)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEMORY="$REPO_ROOT/.letta/memory"
BRIEF=false

[[ "${1:-}" == "--brief" ]] && BRIEF=true

# ── Helper: safe file read (returns empty string if missing) ──
read_file() { cat "$1" 2>/dev/null || true; }

# ── Helper: extract yaml frontmatter value ──
frontmatter_val() {
  local file="$1" key="$2"
  sed -n "/^---$/,/^---$/p" "$file" 2>/dev/null | grep "^${key}:" | head -1 | sed "s/^${key}: *//"
}

# ── Section 1: Session Intent ──
# Source: .letta/memory/threads/current.md
emit_session_intent() {
  local current="$MEMORY/threads/current.md"
  if [[ -f "$current" ]]; then
    local hypothesis
    hypothesis=$(awk '/^## Current hypothesis$/{found=1;next} /^##/{found=0} found' "$current" | sed '/^$/d')
    local testing
    testing=$(awk '/^## What we.*testing$/{found=1;next} /^##/{found=0} found' "$current" | sed '/^$/d')
    echo "## Session Intent"
    echo ""
    echo "**Hypothesis:** $hypothesis"
    echo ""
    if [[ -n "$testing" ]]; then
      echo "**Testing:** $testing"
      echo ""
    fi
  else
    echo "## Session Intent"
    echo ""
    echo "(No active thread — run \`/status --research\` to set one)"
    echo ""
  fi
}

# ── Section 2: Files Modified (this session / uncommitted) ──
emit_files_modified() {
  echo "## Files Modified"
  echo ""
  local staged unstaged
  staged=$(git -C "$REPO_ROOT" diff --cached --name-status 2>/dev/null || true)
  unstaged=$(git -C "$REPO_ROOT" diff --name-status 2>/dev/null || true)
  local untracked
  untracked=$(git -C "$REPO_ROOT" ls-files --others --exclude-standard 2>/dev/null | head -20 || true)

  if [[ -z "$staged" && -z "$unstaged" && -z "$untracked" ]]; then
    echo "Clean working tree — no uncommitted changes."
    echo ""
    return
  fi

  if [[ -n "$staged" ]]; then
    echo "**Staged:**"
    echo '```'
    echo "$staged"
    echo '```'
  fi
  if [[ -n "$unstaged" ]]; then
    echo "**Unstaged:**"
    echo '```'
    echo "$unstaged"
    echo '```'
  fi
  if [[ -n "$untracked" ]]; then
    echo "**Untracked:**"
    echo '```'
    echo "$untracked"
    echo '```'
  fi
  echo ""
}

# ── Section 3: Decisions & Findings ──
# Source: last entry in research-log.md
emit_decisions() {
  local log="$MEMORY/threads/research-log.md"
  echo "## Last Session Findings"
  echo ""
  if [[ -f "$log" ]]; then
    # Extract the last session entry (last ## block)
    local last_entry
    last_entry=$(awk '/^## [0-9]/{found=1; block=""} found{block=block"\n"$0} END{print block}' "$log" | tail -n +2)
    if [[ -n "$last_entry" ]]; then
      echo "$last_entry"
    else
      echo "(No session entries yet)"
    fi
  else
    echo "(No research log found)"
  fi
  echo ""
}

# ── Section 4: Active Work (beads in-progress) ──
emit_active_work() {
  echo "## Active Work"
  echo ""
  local in_progress
  in_progress=$(bd list --status=in_progress 2>/dev/null || echo "(beads unavailable)")
  if [[ -z "$in_progress" || "$in_progress" == "No issues found." ]]; then
    echo "No in-progress beads. Run \`bd ready\` to find available work."
  else
    echo "$in_progress"
  fi
  echo ""
}

# ── Section 5: Recent Runs ──
emit_recent_runs() {
  local latest="$MEMORY/runs/latest.md"
  echo "## Recent Runs"
  echo ""
  if [[ -f "$latest" ]]; then
    # Extract just the table (skip frontmatter and headers)
    sed -n '/^| Date/,/^$/p' "$latest" | head -12
  else
    echo "(No run pointers found)"
  fi
  echo ""
}

# ── Section 6: Next Steps ──
# Source: current.md "Next experiment" section
emit_next_steps() {
  local current="$MEMORY/threads/current.md"
  echo "## Next Steps"
  echo ""
  if [[ -f "$current" ]]; then
    local next_section
    next_section=$(awk '/^## Next experiment$/{found=1;next} /^##/{found=0} found' "$current" | sed '/^$/d')
    local blockers
    blockers=$(awk '/^## Blockers$/{found=1;next} /^##/{found=0} found' "$current" | sed '/^$/d')
    if [[ -n "$next_section" ]]; then
      echo "$next_section"
      echo ""
    fi
    if [[ -n "$blockers" && "$blockers" != "None currently." ]]; then
      echo "**Blockers:** $blockers"
      echo ""
    fi
  else
    echo "(Check \`bd ready\` for available work)"
    echo ""
  fi
}

# ── Section 7: Beads Workflow Reminder (compact) ──
emit_workflow_reminder() {
  echo "## Workflow Reminder"
  echo ""
  echo "- Track tasks: \`bd create\`, \`bd ready\`, \`bd close\`"
  echo "- Session close: \`git add → bd sync → git commit → bd sync → git push\`"
  echo "- Never skip the push. Work is not done until pushed."
  echo ""
}

# ── Assemble ──
echo "# Anchored Context Summary"
echo ""
echo "> Post-compaction context recovery. Run \`bd prime\` to regenerate."
echo "> Sections are anchors — each preserves a critical information dimension."
echo ""

if $BRIEF; then
  emit_session_intent
  emit_active_work
  emit_next_steps
  emit_workflow_reminder
else
  emit_session_intent
  emit_files_modified
  emit_decisions
  emit_active_work
  emit_recent_runs
  emit_next_steps
  emit_workflow_reminder
fi
