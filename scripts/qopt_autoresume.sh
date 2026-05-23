#!/usr/bin/env bash
# Auto-resume watcher for beads distributional-agi-safety-qopt.
# Polls the OpenAI gpt-4o-mini daily-RPD quota (the MiroShark ontology
# step's dependency). When it clears, re-verifies Neo4j + backend health
# (restarting the neo4j container if the engine bounced it) and resumes
# the resumable batch to completion. Hands-off; nohup-friendly.
set -uo pipefail
cd "$(dirname "$0")/.."

BATCH=runs/20260517-142704_multiseed_miroshark
POLL_S=900          # 15 min between quota probes (RPD reset is coarse)
MAX_WAIT_S=$((26*3600))   # give up after 26h
OPENAI_KEY=$(grep -E '^OPENAI_API_KEY=' "$HOME/miroshark/.env" | cut -d= -f2- | tr -d '\n\r')

log(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }

probe_quota(){  # probe MiroShark's ACTUAL path: OpenRouter routing openai/gpt-4o-mini
  # (keys are sk-or OpenRouter keys; api.openai.com would always 401).
  # OpenRouter multi-provider routing means a 200 here = the ontology
  # step can get gpt-4o-mini capacity (even if the OpenAI org is RPD-capped,
  # OpenRouter falls back to Azure/other providers). 429 = still capped.
  local code
  code=$(curl -s -m 30 -o /tmp/qopt_probe.json -w '%{http_code}' \
    https://openrouter.ai/api/v1/chat/completions \
    -H "Authorization: Bearer ${OPENAI_KEY}" -H 'Content-Type: application/json' \
    -d '{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')
  if [ "$code" = "200" ]; then echo OPEN
  elif [ "$code" = "429" ]; then echo BLOCKED
  else echo "ERR($code)"; fi
}

ensure_neo4j(){
  if ! python3 -c "import socket;s=socket.socket();s.settimeout(3);s.connect(('127.0.0.1',7687))" 2>/dev/null; then
    log "neo4j bolt closed -> docker start miroshark-neo4j-local"
    docker start miroshark-neo4j-local >/dev/null 2>&1
    for _ in $(seq 1 48); do
      sleep 5
      python3 -c "import socket;s=socket.socket();s.settimeout(3);s.connect(('127.0.0.1',7687))" 2>/dev/null && { log "neo4j bolt OPEN"; return 0; }
    done
    log "WARN: neo4j still not reachable after 4min"
  fi
  local h; h=$(curl -s -m5 -o /dev/null -w '%{http_code}' http://localhost:5001/health)
  log "backend /health=$h"
}

log "watcher start; batch=$BATCH poll=${POLL_S}s"
start=$(date +%s)
while :; do
  q=$(probe_quota)
  log "openai gpt-4o-mini quota: $q"
  if [ "$q" = "OPEN" ]; then
    log "quota cleared -> healing infra + resuming batch"
    ensure_neo4j
    python -u scripts/multiseed_miroshark.py --resume "$BATCH" >> /tmp/multiseed_miroshark.log 2>&1
    log "batch resume returned; SUMMARY:"
    [ -f "$BATCH/SUMMARY.md" ] && grep -A12 'Hypothesis verdicts' "$BATCH/SUMMARY.md"
    log "watcher done"
    exit 0
  fi
  now=$(date +%s)
  if [ $((now - start)) -ge $MAX_WAIT_S ]; then
    log "MAX_WAIT exceeded ($((MAX_WAIT_S/3600))h); giving up. Resume manually with --resume $BATCH"
    exit 1
  fi
  sleep "$POLL_S"
done
