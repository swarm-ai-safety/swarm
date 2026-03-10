---
description: "Research-scout recon plan for evaluating Karpathy AgentHub as an execution/runtime layer for SWARM studies."
---

# AgentHub Recon for SWARM (Blocked External Access + Integration Plan)

*Prepared 2026-03-10 · Role: Research Scout*

## Status

I attempted to inspect `https://github.com/karpathy/agenthub` directly from this environment, but outbound access to GitHub is blocked (`CONNECT tunnel failed, response 403`).

Because source retrieval failed, this memo captures:

1. What was verified locally in SWARM for likely integration points.
2. A concrete adapter plan to execute once AgentHub source is reachable.
3. Risks and acceptance checks so this can convert quickly into an implementation task.

## Local SWARM Surfaces Ready for an Agent Runtime Bridge

The bridge architecture in SWARM is already modular and has multiple precedent integrations under `swarm/bridges/*` (e.g., `agent_lab`, `langgraph_swarm`, `autogpt`, `openclaw`, `ralph`, `concordia`). This means AgentHub integration should be implemented as another bridge package, not as changes to core orchestration.

### Recommended landing zone

- New package: `swarm/bridges/agenthub/`
- First-pass files to mirror existing bridge conventions:
  - `config.py`
  - `events.py`
  - `mapper.py`
  - `client.py`
  - `bridge.py`
  - `__init__.py`

### Why this fits SWARM research workflow

A bridge preserves SWARM's research value proposition:

- Keep SWARM metrics and governance levers as the "measurement/control" plane.
- Treat AgentHub as an external "execution policy" plane.
- Run baseline-vs-intervention experiments with identical SWARM scenario seeds while swapping the external runtime.

## Adapter Contract (What to map from AgentHub)

When source is accessible, extract and map these primitives:

1. **Task/episode lifecycle** → SWARM run/epoch/step loop.
2. **Agent action schema** → SWARM action + event taxonomy.
3. **Tool-call traces** → SWARM provenance/event log stream.
4. **Outcome/reward signals** → SWARM metric inputs.
5. **Memory/context state transitions** → SWARM per-agent state snapshots.

If AgentHub lacks one of these as first-class primitives, implement a normalization shim in `mapper.py` so SWARM still emits deterministic artifacts.

## Minimal Experiment Plan (once unblocked)

### Goal

Quantify whether AgentHub execution policies change governance outcomes relative to native SWARM agents under fixed seeds.

### Baseline vs intervention

- **Baseline**: existing SWARM scenario with native bridge/agents.
- **Intervention**: same scenario + `agenthub` bridge implementation.

### First scenarios to run

- `scenarios/coding_bench/*` for capability/task quality differences.
- `scenarios/work_regime_drift/*` for policy drift and adaptation behavior.
- `scenarios/kernel_market/*` for multi-agent strategic effects.

### Metrics to compare

- Welfare and quality metrics already exported by SWARM.
- Safety/governance metrics (e.g., policy violations, intervention frequency).
- Replay determinism checks across repeated seeds.

## Risks and Mitigations

- **Schema mismatch risk**: AgentHub action/tool payloads may not match SWARM abstractions.
  - *Mitigation*: strict typed mapper + explicit "unknown event" bucket.
- **Determinism risk**: runtime-level nondeterminism can invalidate comparisons.
  - *Mitigation*: fixed seed threading and replay verifier checks before study runs.
- **Attribution risk**: outcome deltas may come from prompt defaults, not runtime design.
  - *Mitigation*: freeze prompts/tools and vary only runtime adapter.

## Unblock Checklist (for next session with GitHub access)

1. Clone AgentHub and inventory core modules (`README`, runner, agent loop, tool interface, logging).
2. Produce source-backed mapping table (AgentHub primitive → SWARM primitive) with exact file paths.
3. Scaffold `swarm/bridges/agenthub/` with a smoke test scenario.
4. Run a 2-seed smoke benchmark and export artifacts.
5. Promote to larger sweeps only after deterministic replay passes.
