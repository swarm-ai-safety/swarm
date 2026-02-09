#!/usr/bin/env bash
set -euo pipefail

SCENARIO=${1:-scenarios/claude_code_demo.yaml}
BASE_URL=${BASE_URL:-http://localhost:3100}
API_PREFIX=${API_PREFIX:-/api}

python scripts/run_claude_code_scenario.py \
  --scenario "$SCENARIO" \
  --base-url "$BASE_URL" \
  --api-prefix "$API_PREFIX" \
  --auto-approve
