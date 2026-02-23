#!/usr/bin/env bash
# Submit a paper to AgentArxiv and convert to a research object
# Reads API key from .env (AGENTARXIV_API_KEY) or environment
# Run: bash scripts/agentarxiv_submit.sh [paper_json]
#
# Examples:
#   bash scripts/agentarxiv_submit.sh                          # uses default paper.json
#   bash scripts/agentarxiv_submit.sh scripts/my_paper.json    # custom paper

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

PAPER_JSON="${1:-$SCRIPT_DIR/agentarxiv_paper.json}"
if [ ! -f "$PAPER_JSON" ]; then
  echo "Error: paper JSON not found: $PAPER_JSON"
  exit 1
fi

# Step 1: Publish the paper
echo "=== Publishing paper ==="
PAPER_RESPONSE=$(curl -s -X POST https://www.agentarxiv.org/api/v1/papers \
  -H "Authorization: Bearer $AGENTARXIV_API_KEY" \
  -H "Content-Type: application/json" \
  -d @"$PAPER_JSON")

echo "$PAPER_RESPONSE" | python3 -m json.tool

# Extract paper ID (nested under .data.id)
PAPER_ID=$(echo "$PAPER_RESPONSE" | python3 -c "
import sys, json
r = json.load(sys.stdin)
pid = r.get('data', {}).get('id') or r.get('id', '')
print(pid)
")

if [ -z "$PAPER_ID" ]; then
  echo "Error: could not extract paper ID from response"
  exit 1
fi

echo ""
echo "Paper ID: $PAPER_ID"
echo ""

# Step 2: Prompt for research object conversion
read -p "Convert to research object? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
  echo "Skipping research object. Paper ID: $PAPER_ID"
  exit 0
fi

RO_JSON="${PAPER_JSON%.json}_research_object.json"
if [ -f "$RO_JSON" ]; then
  echo "=== Converting to research object (from $RO_JSON) ==="
  # Inject paperId into the research object JSON
  RO_PAYLOAD=$(python3 -c "
import json, sys
with open('$RO_JSON') as f:
    ro = json.load(f)
ro['paperId'] = '$PAPER_ID'
json.dump(ro, sys.stdout)
")
  curl -s -X POST https://www.agentarxiv.org/api/v1/research-objects \
    -H "Authorization: Bearer $AGENTARXIV_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$RO_PAYLOAD" | python3 -m json.tool
else
  echo "No research object JSON found at $RO_JSON — skipping."
  echo "Create it and re-run, or use agentarxiv_milestone.sh to update milestones."
fi

echo ""
echo "=== Done ==="
