---
description: "Initial reconnaissance status for Agent-Field/SWE-AF and integration plan for SWARM."
---

# SWE-AF reconnaissance (Agent-Field/SWE-AF)

## Scope

This note tracks a **Research Scout** pass for: <https://github.com/Agent-Field/SWE-AF>.

Requested target:
- External repository architecture and transferable patterns for SWARM.

## Access status

Direct repository access from this execution environment is currently blocked by outbound network policy (HTTP 403 from CONNECT tunnel).

Commands attempted:

```bash
git clone --depth 1 https://github.com/Agent-Field/SWE-AF /tmp/SWE-AF
curl -I -L https://raw.githubusercontent.com/Agent-Field/SWE-AF/main/README.md
```

Both returned:

```text
CONNECT tunnel failed, response 403
```

## What is ready now

Even without source access, we can stage a concrete import plan so follow-up scouting is low-friction once connectivity is available.

### Planned scout checklist (first 10 files)

1. `README.md` (project scope + quickstart)
2. top-level build/runtime config (`pyproject.toml`, `requirements*.txt`, `package.json`, etc.)
3. agent runtime entrypoint (`src/*main*`, CLI, orchestration)
4. policy/prompts/config (`.claude/`, `.github/workflows/`, rules)
5. evaluation harness (`eval*`, `benchmark*`, `tests/`)
6. environment/task adapters
7. logging/trajectory schema
8. failure handling/retry logic
9. governance/safety constraints
10. reproducibility scripts

### SWARM mapping template

When access is restored, map findings to these local surfaces:
- Scenario design: `scenarios/*.yaml`
- Governance levers: `swarm/governance/*`
- Agent behavior and red-team behavior: `swarm/agents/*`, `swarm/redteam/*`
- Experiment/repro hygiene: `scripts/*`, `tests/*`, docs under `docs/research/*`

## Follow-up trigger

Run a full Research Scout pass as soon as one of the following is available:
- Network egress to GitHub is enabled, or
- A tarball/mirror of `Agent-Field/SWE-AF` is provided in this workspace.

At that point, replace this placeholder with a complete pattern audit:
- Relevant patterns (what/where/how/relevance/adoption effort)
- Concrete recommendations (ranked by impact)
- Not relevant findings (to avoid re-investigation)
