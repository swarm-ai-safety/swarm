# SWARM-Ralph Bridge

Integrate SWARM governance and metrics with event streams exported from
[Ralph](https://github.com/snarktank/ralph).

## What this bridge does

The bridge ingests Ralph JSONL events and maps them to SWARM `SoftInteraction`
records so they can be scored by SWARM's proxy labels and downstream metrics.

## Quickstart

```python
from swarm.bridges.ralph import RalphBridge, RalphConfig

bridge = RalphBridge(
    RalphConfig(
        events_path="./ralph-events.jsonl",
        agent_role_map={"alice": "worker_alice"},
    )
)

new_interactions = bridge.poll()
print(len(new_interactions))
```

## Event format

Each line in `events_path` should be a JSON object:

```json
{
  "event_id": "evt-1",
  "event_type": "task:completed",
  "timestamp": "2026-02-10T00:00:00+00:00",
  "actor_id": "alice",
  "task_id": "task-42",
  "payload": {
    "task_progress_delta": 0.8,
    "rework_count": 0,
    "verifier_rejections": 0,
    "tool_misuse_flags": 0,
    "counterparty_engagement_delta": 0.4
  }
}
```

Unknown `event_type` values are accepted and mapped to a neutral `generic`
interaction.

## Notes

- `poll()` is incremental: only newly appended lines are processed.
- If the source file is rotated or truncated, the bridge automatically restarts from offset `0`.
- Malformed or non-object JSON lines are skipped.
- Field aliases are accepted for compatibility (`type`, `agent`, `job`, `id`, `time`).
- Override defaults by passing observables in the event payload.
