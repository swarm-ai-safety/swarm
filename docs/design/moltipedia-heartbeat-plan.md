# Plan: Model Moltipedia Heartbeat Loop in SWARM

## Context

The Moltipedia heartbeat loop is a real-world pattern where AI agents periodically check in with an external wiki platform, pull work, evaluate content against editorial policy, take actions (create/edit/object/flag), earn points, and save state. This is a natural case study for SWARM because it exhibits the exact multi-agent dynamics SWARM studies: competing agents, governance mechanisms, potential for collusion and point farming, and measurable quality outcomes.

## Concept Mapping

| Moltipedia | SWARM Abstraction |
|---|---|
| Heartbeat tick | One SWARM step within an epoch |
| Wiki pages | New `WikiPage` dataclass in `swarm/env/wiki.py` |
| Work queue (contested/random/search) | `WikiTaskPool` generating queues by type |
| Editorial policy evaluation | `MoltipediaObservableGenerator` mapping policy compliance to proxy signals |
| Edit/Create/Objection/Flag | New `ActionType` variants |
| Points (+25/+15/+20/+8) | `MoltipediaScorer` feeding into `SoftInteraction.tau` |
| Pair cap, page cooldown, daily cap, no self-fix | New `GovernanceLever` subclasses |
| Leaderboard | New `Leaderboard` tracked in `MoltipediaHandler` |
| SWARM epoch = 1 "day", step = 1 heartbeat tick | Temporal mapping |

## New Files

### 1. `swarm/env/wiki.py` — Wiki Page Model & Task Pool

- `PageStatus` enum: STUB, DRAFT, PUBLISHED, CONTESTED, LOCKED
- `EditType` enum: CREATE (+25), EDIT (+15), CONTESTED_RESOLVE (+20), POLICY_FIX (+8)
- `PolicyViolationType` enum: NEUTRALITY, SCOPE, SPECULATION, SOURCING, NONE
- `WikiPage` dataclass: page_id, title, content, status, quality_score, edit_history, cooldown_until, policy_violations
- `WikiTaskPool`: manages pages, generates work queues (contested, random, search), tracks leaderboard, seeds initial pages

### 2. `swarm/governance/moltipedia.py` — Anti-Spam Governance Levers

Four levers following the `GovernanceLever` pattern in `swarm/governance/levers.py`:

- **`PairCapLever`** — Max 2 scored fixes per agent pair per epoch. Tracks `Dict[Tuple[str,str], int]`, resets on `on_epoch_start`, enforces on `on_interaction` by zeroing points via `LeverEffect.cost_a`
- **`PageCooldownLever`** — Max 1 scored fix per page per N steps. Tracks `Dict[str, int]` of last scored step per page
- **`DailyPointCapLever`** — Max +24 policy-fix points per agent per epoch. Tracks `Dict[str, float]` of accumulated policy-fix points
- **`NoSelfFixLever`** — Cancels scoring when agent edits own page. Checks `interaction.initiator == page.created_by`

### 3. `swarm/agents/wiki_editor.py` — Agent Types

Subclass `BaseAgent` from `swarm/agents/base.py`:

- **`DiligentEditorAgent`** — Honest editor following editorial policy. Prioritizes contested > search > random. Takes highest-value action
- **`PointFarmerAgent`** — Opportunistic. Targets easy policy fixes, creates stubs to "improve" later, exploits cooldown gaps
- **`CollusiveEditorAgent`** — Adversarial. Pre-arranges edits with partner, alternates creating and fixing, stays under pair cap
- **`VandalAgent`** — Adversarial. Degrades page quality to create work for partners or disrupt the wiki

### 4. `swarm/core/moltipedia_handler.py` — Orchestrator Handler

Following the pattern of marketplace/boundary handlers:

- `MoltipediaScorer`: Computes raw points by edit type, subject to governance
- `MoltipediaHandler`: Manages WikiTaskPool lifecycle, builds wiki observation fields, executes wiki actions, computes scoring, updates leaderboard, emits events

### 5. `swarm/core/moltipedia_observables.py` — Observable Generator

Maps editorial quality to `ProxyObservables`:

| Observable | Moltipedia Source |
|---|---|
| `task_progress_delta` | Quality improvement from edit (delta in page quality_score) |
| `rework_count` | Subsequent reverts/corrections needed |
| `verifier_rejections` | Policy violations in the edit |
| `tool_misuse_flags` | Vandalism/spam markers |
| `engagement_delta` | Community response (built on vs reverted) |

High-quality edit: `(+0.6, 0, 0, 0, +0.4)` → p ~0.73. Manufactured fix: `(+0.1, 1, 0, 0, +0.05)` → p ~0.52. Vandalism: `(-0.5, 0, 2, 1, -0.3)` → p ~0.29.

### 6. `swarm/metrics/moltipedia_metrics.py` — Platform-Specific Metrics

- `point_concentration()` — Gini coefficient of point distribution
- `pair_farming_rate()` — Fraction of scored interactions between repeated pairs
- `policy_fix_exploitation_rate()` — Fraction of policy fixes that appear manufactured
- `content_quality_trend()` — Quality score trend across epochs
- `governance_effectiveness()` — How much governance reduced exploitative scoring

### 7. `scenarios/moltipedia_heartbeat.yaml` — Scenario Config

9 agents: 4 diligent editors, 2 point farmers, 2 collusive editors, 1 vandal. 50 initial pages. 20 epochs x 10 steps. All Moltipedia governance enabled plus SWARM collusion detection and circuit breakers. Success criteria: Gini < 0.6, pair farming rate < 15%, content quality > 0.55, honest agents in top half.

## Files to Modify

### `swarm/agents/base.py`
- Add `ActionType` variants: `CREATE_PAGE`, `EDIT_PAGE`, `FILE_OBJECTION`, `POLICY_FLAG`
- Add helper methods: `create_page_action()`, `create_edit_page_action()`, etc.
- Add `Observation` fields: `contested_pages`, `search_results`, `random_pages`, `leaderboard`, `agent_points`, `heartbeat_status`

### `swarm/governance/config.py`
- Add Pydantic fields: `moltipedia_pair_cap_enabled`, `moltipedia_pair_cap_max`, `moltipedia_page_cooldown_enabled`, `moltipedia_page_cooldown_steps`, `moltipedia_daily_cap_enabled`, `moltipedia_daily_policy_fix_cap`, `moltipedia_no_self_fix`

### `swarm/governance/engine.py`
- Import and register the four Moltipedia levers when config flags are enabled

### `swarm/core/orchestrator.py`
- Add `moltipedia_config` to `OrchestratorConfig`
- Init `MoltipediaHandler` in `__init__` when config present
- Inject wiki observation fields in `_build_observation()`
- Dispatch wiki action types in `_execute_action()`

### `swarm/scenarios/loader.py`
- Parse `moltipedia:` YAML section
- Register new agent types in `AGENT_TYPES`

### `swarm/models/events.py`
- Add event types: `PAGE_CREATED`, `PAGE_EDITED`, `OBJECTION_FILED`, `POLICY_VIOLATION_FLAGGED`, `POINTS_AWARDED`, `PAIR_CAP_TRIGGERED`, `COOLDOWN_TRIGGERED`, `DAILY_CAP_TRIGGERED`

## Implementation Order

1. `swarm/env/wiki.py` — Domain model (independently testable)
2. `swarm/agents/base.py` — New ActionType/Observation fields
3. `swarm/governance/config.py` — Config fields
4. `swarm/governance/moltipedia.py` — Governance levers
5. `swarm/governance/engine.py` — Register levers
6. `swarm/core/moltipedia_observables.py` — Observable generator
7. `swarm/core/moltipedia_handler.py` — Handler
8. `swarm/agents/wiki_editor.py` — Agent implementations
9. `swarm/core/orchestrator.py` — Wire handler in
10. `swarm/scenarios/loader.py` — YAML parsing + agent registration
11. `swarm/models/events.py` — Event types
12. `swarm/metrics/moltipedia_metrics.py` — Metrics
13. `scenarios/moltipedia_heartbeat.yaml` — Scenario
14. Tests

## Test Strategy

### Unit tests
- `tests/test_wiki.py` — WikiPage lifecycle, WikiTaskPool queues, cooldowns, leaderboard
- `tests/test_moltipedia_governance.py` — Each lever independently: pair cap resets/enforces, cooldown timing, daily cap limits, self-fix blocking

### Integration tests
- `tests/test_moltipedia_integration.py` — Full heartbeat cycle, multi-epoch simulation, governance prevents pair farming, governance prevents self-fix, daily cap limits policy-fix farming, collusion detected

### Safety property tests
- Scoring not gameable under governance (point farmers don't dominate)
- Content quality doesn't degrade over epochs
- Point distribution not overly concentrated (Gini check)
- Control test: governance OFF allows exploitation (validates governance matters)

### Scenario test
- `tests/test_moltipedia_scenario.py` — Load YAML, build orchestrator, run simulation, check success criteria

## Verification

```bash
# Run all Moltipedia tests
python -m pytest tests/test_wiki.py tests/test_moltipedia_governance.py tests/test_moltipedia_integration.py tests/test_moltipedia_scenario.py -v

# Run the scenario
swarm run scenarios/moltipedia_heartbeat.yaml

# Check metrics output
cat logs/moltipedia_metrics.csv
```
