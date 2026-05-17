# Letta Code Configuration — SWARM Research OS

This directory configures Letta Code as the stateful research operator for SWARM.

## Architecture

```
SWARM (experiment engine)  <-->  Letta Code (stateful operator)  <-->  Git (provenance)
     scenarios, sweeps,              memory, decisions,                configs, results,
     metrics, artifacts              intent, pointers                  run manifests
```

## Memory Tiers

Letta's MemFS is organized into 4 tiers:

### `memory/system/` (pinned to context, rarely changes)
- `identity.md` — role definition, operating principles
- `preferences.md` — coding style, plotting conventions, sweep defaults
- `workflow.md` — experiment lifecycle, session protocols, artifact contracts

### `memory/project/` (changes slowly)
- `repo-map.md` — repository layout, key commands, data flow
- `scenario-families.md` — taxonomy of scenarios and what they test
- `governance-knobs.md` — all governance mechanisms and known effects

### `memory/threads/` (changes daily)
- `current.md` — active hypothesis, next experiment, blockers
- `research-log.md` — rolling session summaries (append-only)

### `memory/runs/` (changes constantly)
- `latest.md` — pointers to recent runs (never raw data)

## Skills

Project-scoped skills in `.skills/`:

| Skill | Description |
|-------|-------------|
| `experiment-loop` | Full lifecycle: hypothesis -> sweep -> synthesize -> claim review |
| `session-close` | End-of-session ritual: summarize, update memory, commit, push |
| `run-query` | Query run index by tag, date, type, or claim |
| `sanity-check` | Quick validation run before full sweep |
| `regression-check` | Re-run baseline and compare against last known-good |

Additional skills from swarm-artifacts (install manually):

| Skill | Description |
|-------|-------------|
| `claim` | Create or update claim cards with evidence |
| `synthesize` | Generate vault notes from run artifacts |
| `vault-init` | Initialize the knowledge vault structure |
| `verify` | Run vault integrity checks |

To install swarm-artifacts skills:
```bash
cp -r /path/to/swarm-artifacts/skills/* .skills/
```

## Setup

```bash
npm install -g @letta-ai/letta-code
cd /path/to/distributional-agi-safety
letta
```

On first run, Letta will discover `.letta/settings.json` and `.skills/`.
Run `/init` to index the codebase, then `/memory` to verify the memory structure.

## Day-to-day usage

```
# Session start
"What's the active thread?"          # Letta recalls from threads/current.md

# Run experiments
"Test whether tax > 10% causes..."   # Triggers experiment-loop skill
"Sanity check the new scenario"      # Triggers sanity-check skill

# Query history
"What runs tested collusion?"        # Triggers run-query skill
"Show claims about circuit breakers" # Queries vault/claims/

# Session end
"Close session"                      # Triggers session-close skill
```
