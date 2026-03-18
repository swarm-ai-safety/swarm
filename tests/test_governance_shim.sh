#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# test_governance_shim.sh — Tests for governance_shim.sh hook
#
# Verifies that each hook event handler emits valid JSONL to the sink
# and returns empty JSON to stdout (no approve/block).
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SHIM="$REPO_ROOT/.claude/hooks/governance_shim.sh"

# Use a temp file as sink
SINK=$(mktemp /tmp/governance_shim_test.XXXXXX.jsonl)
export GOVERNANCE_SHIM_SINK="$SINK"
export SESSION_ID="test-session"
export WORKTREE_ID="test-worktree"

PASS=0
FAIL=0

cleanup() {
  rm -f "$SINK"
}
trap cleanup EXIT

assert_jsonl_line() {
  local desc="$1"
  local line_num="$2"
  local expected_type="$3"

  local line
  line=$(sed -n "${line_num}p" "$SINK")

  if [ -z "$line" ]; then
    echo "FAIL: $desc — no line $line_num in sink"
    FAIL=$((FAIL + 1))
    return
  fi

  # Validate it's valid JSON
  if ! echo "$line" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    echo "FAIL: $desc — invalid JSON: $line"
    FAIL=$((FAIL + 1))
    return
  fi

  # Check event_type
  local actual_type
  actual_type=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('event_type',''))")
  if [ "$actual_type" != "$expected_type" ]; then
    echo "FAIL: $desc — expected event_type=$expected_type, got $actual_type"
    FAIL=$((FAIL + 1))
    return
  fi

  # Check required fields
  local has_event_id
  has_event_id=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if d.get('event_id') else 'no')")
  if [ "$has_event_id" != "yes" ]; then
    echo "FAIL: $desc — missing event_id"
    FAIL=$((FAIL + 1))
    return
  fi

  local has_timestamp
  has_timestamp=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if d.get('timestamp') else 'no')")
  if [ "$has_timestamp" != "yes" ]; then
    echo "FAIL: $desc — missing timestamp"
    FAIL=$((FAIL + 1))
    return
  fi

  echo "PASS: $desc"
  PASS=$((PASS + 1))
}

assert_stdout_empty_json() {
  local desc="$1"
  local stdout="$2"

  # Should be {} or {"systemMessage": ...}
  if ! echo "$stdout" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    echo "FAIL: $desc — stdout not valid JSON: $stdout"
    FAIL=$((FAIL + 1))
    return
  fi
  echo "PASS: $desc (stdout valid JSON)"
  PASS=$((PASS + 1))
}

# ── Test 1: session_start ──
echo "--- Test: session_start ---"
stdout=$(bash "$SHIM" session_start < /dev/null)
assert_jsonl_line "session_start emits governance.session_init" 1 "governance.session_init"
assert_stdout_empty_json "session_start returns valid JSON" "$stdout"

# ── Test 2: pre_tool_use ──
echo "--- Test: pre_tool_use ---"
stdout=$(echo '{"tool_name": "Bash", "tool_input": {"command": "git status"}}' | bash "$SHIM" pre_tool_use)
assert_jsonl_line "pre_tool_use emits governance.tool_invocation" 2 "governance.tool_invocation"
assert_stdout_empty_json "pre_tool_use returns valid JSON" "$stdout"

# ── Test 3: post_tool_use (normal) ──
echo "--- Test: post_tool_use (normal) ---"
stdout=$(echo '{"tool_name": "Bash", "tool_input": {"command": "git status"}, "exit_code": 0, "tool_response": "ok"}' | bash "$SHIM" post_tool_use)
assert_jsonl_line "post_tool_use emits governance.tool_outcome" 3 "governance.tool_outcome"
assert_stdout_empty_json "post_tool_use returns valid JSON" "$stdout"

# Check no deviation for normal command
line3=$(sed -n '3p' "$SINK")
deviation=$(echo "$line3" | python3 -c "import sys,json; print(json.load(sys.stdin).get('payload',{}).get('deviation',''))")
if [ "$deviation" = "none" ]; then
  echo "PASS: normal command has deviation=none"
  PASS=$((PASS + 1))
else
  echo "FAIL: normal command deviation=$deviation, expected none"
  FAIL=$((FAIL + 1))
fi

# ── Test 4: post_tool_use (destructive — deviation detection) ──
echo "--- Test: post_tool_use (destructive) ---"
export GOVERNANCE_TASK_SPEC="implement feature X"
stdout=$(echo '{"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}, "exit_code": 0, "tool_response": ""}' | bash "$SHIM" post_tool_use)
assert_jsonl_line "post_tool_use detects destructive op" 4 "governance.tool_outcome"

line4=$(sed -n '4p' "$SINK")
deviation=$(echo "$line4" | python3 -c "import sys,json; print(json.load(sys.stdin).get('payload',{}).get('deviation',''))")
if [ "$deviation" = "destructive_operation" ]; then
  echo "PASS: destructive command flagged as deviation"
  PASS=$((PASS + 1))
else
  echo "FAIL: destructive command deviation=$deviation, expected destructive_operation"
  FAIL=$((FAIL + 1))
fi
unset GOVERNANCE_TASK_SPEC

# ── Test 5: subagent_stop ──
echo "--- Test: subagent_stop ---"
stdout=$(echo '{"agent_id": "sub-1", "reason": "completed"}' | bash "$SHIM" subagent_stop)
assert_jsonl_line "subagent_stop emits governance.task_checkpoint" 5 "governance.task_checkpoint"
assert_stdout_empty_json "subagent_stop returns valid JSON" "$stdout"

# Check verdict is pass for completed
line5=$(sed -n '5p' "$SINK")
verdict=$(echo "$line5" | python3 -c "import sys,json; print(json.load(sys.stdin).get('payload',{}).get('review_verdict',''))")
if [ "$verdict" = "pass" ]; then
  echo "PASS: completed subagent has verdict=pass"
  PASS=$((PASS + 1))
else
  echo "FAIL: completed subagent verdict=$verdict, expected pass"
  FAIL=$((FAIL + 1))
fi

# ── Test 6: subagent_stop with error ──
echo "--- Test: subagent_stop (error) ---"
stdout=$(echo '{"agent_id": "sub-2", "reason": "error"}' | bash "$SHIM" subagent_stop)
line6=$(sed -n '6p' "$SINK")
verdict=$(echo "$line6" | python3 -c "import sys,json; print(json.load(sys.stdin).get('payload',{}).get('review_verdict',''))")
if [ "$verdict" = "fail" ]; then
  echo "PASS: errored subagent has verdict=fail"
  PASS=$((PASS + 1))
else
  echo "FAIL: errored subagent verdict=$verdict, expected fail"
  FAIL=$((FAIL + 1))
fi

# ── Test 7: unknown hook event ──
echo "--- Test: unknown event ---"
stdout=$(bash "$SHIM" unknown_event < /dev/null 2>&1)
# Should not crash, should output something parseable
if [ $? -eq 0 ]; then
  echo "PASS: unknown event exits cleanly"
  PASS=$((PASS + 1))
else
  echo "FAIL: unknown event caused non-zero exit"
  FAIL=$((FAIL + 1))
fi

# ── Summary ──
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
