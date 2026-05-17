#!/usr/bin/env bash
# Complete a milestone on an AgentArxiv research object
# Reads API key from .env (AGENTARXIV_API_KEY) or environment
#
# Usage:
#   bash scripts/agentarxiv_milestone.sh <milestone_id> <evidence> [artifact_url]
#
# Examples:
#   bash scripts/agentarxiv_milestone.sh cmlyijyh8001k3wdc... "Assumptions listed in paper."
#   bash scripts/agentarxiv_milestone.sh cmlyijyh8001m3wdc... "SWARM framework" "https://github.com/..."
#
# Milestone types (for reference):
#   CLAIM_STATED, ASSUMPTIONS_LISTED, TEST_PLAN, RUNNABLE_ARTIFACT,
#   INITIAL_RESULTS, INDEPENDENT_REPLICATION, CONCLUSION_UPDATE

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env if present
if [ -f "$REPO_ROOT/.env" ]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi

if [ -z "${AGENTARXIV_API_KEY:-}" ]; then
  echo "Error: AGENTARXIV_API_KEY not set. Add it to .env or export it."
  exit 1
fi

MILESTONE_ID="${1:?Usage: agentarxiv_milestone.sh <milestone_id> <evidence> [artifact_url]}"
EVIDENCE="${2:?Usage: agentarxiv_milestone.sh <milestone_id> <evidence> [artifact_url]}"
ARTIFACT_URL="${3:-}"

# Build payload
if [ -n "$ARTIFACT_URL" ]; then
  PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'completed': True,
    'evidence': '''$EVIDENCE''',
    'artifactUrl': '''$ARTIFACT_URL'''
}))
")
else
  PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'completed': True,
    'evidence': '''$EVIDENCE'''
}))
")
fi

echo "Completing milestone: $MILESTONE_ID"
RESPONSE=$(curl -s -w "\n%{http_code}" -X PATCH \
  "https://www.agentarxiv.org/api/v1/milestones/$MILESTONE_ID" \
  -H "Authorization: Bearer $AGENTARXIV_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
  echo "$BODY" | python3 -m json.tool
  PROGRESS=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('data',{}).get('progressScore','?'))")
  echo ""
  echo "Progress: ${PROGRESS}%"
else
  echo "Error: HTTP $HTTP_CODE"
  echo "$BODY"
  exit 1
fi
