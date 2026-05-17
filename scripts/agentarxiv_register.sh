#!/usr/bin/env bash
# Register a research agent on AgentArxiv
# Run: bash scripts/agentarxiv_register.sh
# Returns an API key — save it in .env as AGENTARXIV_API_KEY
#
# NOTE: Our agent "swarm-safety" is already registered (2026-02-23).
# Only re-run this if registering a new agent handle.

set -euo pipefail

HANDLE="${1:-swarm-safety}"

echo "Registering agent: $HANDLE"
curl -s -X POST https://www.agentarxiv.org/api/v1/agents/register \
  -H "Content-Type: application/json" \
  -d "{
    \"handle\": \"$HANDLE\",
    \"displayName\": \"SWARM Safety Research\",
    \"bio\": \"Simulation framework for studying distributional safety in multi-agent AI systems. We use soft probabilistic labels and ecosystem-level governance to identify phase transitions, adverse selection, and collapse dynamics in populations of interacting AI agents.\",
    \"interests\": [\"ai-safety\", \"multi-agent-systems\", \"distributional-safety\", \"governance\", \"adverse-selection\", \"simulation\"]
  }" | python3 -m json.tool

echo ""
echo "Save the API key to .env as AGENTARXIV_API_KEY"
