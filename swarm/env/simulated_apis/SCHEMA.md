# Simulated API Episode Log Schema

`swarm.env.simulated_apis` uses an append-only, JSON-serializable event model designed for:
- stable provenance identifiers (hash-based)
- parent-linking for causal chains
- post-hoc metric extraction

## Event envelope (all events)

- `event_hash` (string): SHA-256 hash of the canonical JSON payload
- `event_type` (string): one of the types below
- `timestamp` (string): ISO-8601 UTC timestamp
- `agent_id` (string|null): actor that emitted the event
- `parent_event_hash` (string|null): hash pointer for causal chain
- `provenance_id` (string): short handle (first 12 chars of `event_hash`)
- `payload` (object): event-type specific data

## Event types

### `api_call`

Payload fields:
- `domain` (string): e.g. `iam`, `payments`, `incident_response`
- `endpoint` (string)
- `params` (object)
- `cost` (int)
- `irreversible` (bool)
- `high_cost` (bool)
- `response` (object)

### `irreversible_proposed`

Payload fields:
- `request_id` (string): local request identifier (e.g. `req_1`)
- `endpoint` (string)
- `params` (object)

### `irreversible_vote`

Payload fields:
- `request_id` (string)
- `approve` (bool)
- `evidence_ids` (list[string]): provenance ids used as justification

