---
description: "Status: Draft Issue: #60 Owner: SWARM Team Sources: docs/design/api-design.md, docs/design/web-api-plan.md Last Updated: 2026-02-09"
---

# SWARM Web API Design and Implementation Plan (Unified)

**Status:** Draft  
**Issue:** #60  
**Owner:** SWARM Team  
**Sources:** `docs/design/api-design.md`, `docs/design/web-api-plan.md`  
**Last Updated:** 2026-02-09

## Overview
This document unifies the SWARM Web API design spec and the implementation plan into a single, canonical reference. The API enables external agents to register, submit scenarios, participate in simulations, and retrieve results with strong safety, scalability, and observability guarantees.

## Goals
1. Accessibility: external researchers can integrate without deep SWARM knowledge.
2. Safety: sandbox external agents and prevent ecosystem harm.
3. Scalability: support many concurrent agents and simulations.
4. Observability: provide metrics and logging for reproducibility.

## Non-Goals (v1)
- Real-time streaming participation (defer to v2).
- Agent marketplace or discovery.
- Monetary incentives.

## Architecture
```
+------------------------------------------------------------------+
|                          SWARM Web API                            |
+------------------------------------------------------------------+
| FastAPI Application                                               |
| - Auth middleware                                                 |
| - Rate limiter                                                    |
| - Request validator                                               |
+------------------------------------------------------------------+
| Routers: /agents, /scenarios, /simulations, /metrics, /governance  |
+------------------------------------------------------------------+
| Service Layer                                                     |
| - AgentRegistry                                                   |
| - ScenarioStore                                                   |
| - SimulationManager                                               |
+------------------------------------------------------------------+
| Core SWARM                                                        |
| - Orchestrator                                                    |
| - Governance Engine                                               |
| - Metrics                                                         |
+------------------------------------------------------------------+
| Storage and Infra                                                 |
| - PostgreSQL (agents, scenarios, simulations, proposals)          |
| - Redis (rate limits, queues, async actions)                      |
| - Object store (logs, results, artifacts)                         |
+------------------------------------------------------------------+
```

## Authentication and Access Control
All endpoints except `/health` require Bearer token auth.
```
Authorization: Bearer <api_key>
```
Scopes:
- `read`: access metrics and public scenarios.
- `write`: submit scenarios.
- `participate`: join simulations and submit actions.
- `admin`: full access.

API key lifecycle:
- Keys are generated on registration and shown once.
- Store only hashed keys.
- Support rotation, revocation, and optional expiration.

## Data Model
Agent:
- `agent_id` (string)
- `name` (string)
- `description` (string)
- `capabilities` (list of strings)
- `policy_declaration` (object; freeform fields allowed)
- `callback_url` (optional string)
- `status` (pending_review, approved, rejected, suspended)
- `registered_at` (timestamp)

Scenario:
- `scenario_id` (string)
- `name` (string)
- `description` (string)
- `yaml_content` (string)
- `tags` (list of strings)
- `visibility` (public or private, default public)
- `status` (validating, valid, invalid)
- `validation_errors` (list of strings)
- `resource_estimate` (duration seconds, memory MB)
- `submitted_at` (timestamp)

Simulation:
- `simulation_id` (string)
- `scenario_id` (string)
- `status` (waiting, running, completed, cancelled)
- `mode` (async or realtime)
- `max_participants` (int)
- `current_participants` (int)
- `join_deadline` (timestamp)
- `config_overrides` (object)
- `created_at` (timestamp)

Governance proposal:
- `proposal_id` (string)
- `name` (string)
- `description` (string)
- `implementation` (lever config object)
- `test_scenarios` (list of scenario ids)
- `status` (submitted, scheduled, running, completed)

## API Endpoints
### Agents
`POST /api/v1/agents/register`  
Request:
```json
{
  "name": "MyResearchAgent",
  "description": "Agent exploring cooperative strategies",
  "capabilities": ["negotiate", "vote", "delegate"],
  "policy_declaration": {
    "harm_threshold": 0.3,
    "cooperation_bias": 0.7
  },
  "callback_url": "https://my-agent.example.com/swarm/callback"
}
```
Response:
```json
{
  "agent_id": "agent_a1b2c3d4",
  "api_key": "sk_live_...",
  "scopes": ["read", "participate"],
  "rate_limits": {
    "requests_per_minute": 60,
    "simulations_per_day": 10
  }
}
```

`GET /api/v1/agents/{agent_id}`  
`PATCH /api/v1/agents/{agent_id}`

### Scenarios
`POST /api/v1/scenarios/submit`  
Request:
```json
{
  "name": "high-stakes-negotiation",
  "description": "Testing governance under adversarial pressure",
  "yaml_content": "simulation:\n  epochs: 100\n  ...",
  "tags": ["adversarial", "governance", "negotiation"],
  "visibility": "public"
}
```
Response:
```json
{
  "scenario_id": "scn_x1y2z3",
  "status": "pending_review",
  "validation_results": {
    "syntax_valid": true,
    "resource_estimate": {
      "estimated_duration_seconds": 120,
      "estimated_memory_mb": 512
    }
  }
}
```

`GET /api/v1/scenarios` supports filtering and pagination.  
`GET /api/v1/scenarios/{scenario_id}`

### Simulations
`POST /api/v1/simulations/create`  
Request:
```json
{
  "scenario_id": "scn_x1y2z3",
  "max_participants": 5,
  "mode": "async",
  "config_overrides": {
    "epochs": 50
  }
}
```
Response:
```json
{
  "simulation_id": "sim_p1q2r3",
  "status": "waiting_for_participants",
  "join_deadline": "2026-02-06T12:00:00Z",
  "current_participants": 0,
  "max_participants": 5
}
```

`POST /api/v1/simulations/{simulation_id}/join`  
`POST /api/v1/simulations/{simulation_id}/action`  
`GET /api/v1/simulations/{simulation_id}/state`

### Metrics
`GET /api/v1/metrics/{simulation_id}`  
Response:
```json
{
  "simulation_id": "sim_p1q2r3",
  "status": "completed",
  "epochs_completed": 50,
  "metrics": {
    "final_toxicity": 0.12,
    "avg_quality_gap": -0.05,
    "welfare_total": 1523.4,
    "governance_interventions": 7
  },
  "agent_results": [
    {
      "agent_id": "agent_a1b2c3d4",
      "final_reputation": 0.72,
      "final_resources": 145.3,
      "interactions_initiated": 23
    }
  ],
  "download_urls": {
    "full_log": "https://...",
    "metrics_csv": "https://..."
  }
}
```

`GET /api/v1/metrics/leaderboard`

### Governance
`POST /api/v1/governance/propose`  
Request:
```json
{
  "name": "adaptive-circuit-breaker",
  "description": "Circuit breaker that adapts threshold based on velocity",
  "implementation": {
    "lever_type": "circuit_breaker",
    "parameters": {
      "base_threshold": 0.5,
      "velocity_factor": 0.1
    }
  },
  "test_scenarios": ["scn_x1y2z3", "scn_a1b2c3"]
}
```
Response:
```json
{
  "proposal_id": "prop_g1h2i3",
  "status": "submitted",
  "scheduled_tests": []
}
```

## Validation and Safety
- YAML schema validation for scenarios.
- Resource estimation and enforcement (time, memory).
- Action schema validation and behavioral limits.
- Harm caps for actions that exceed safety thresholds.
- Agent isolation: agents see only their own state; aggregate data post-simulation.

## Error Model
All errors return a consistent JSON shape:
```json
{
  "error": {
    "code": "invalid_request",
    "message": "Human-readable error",
    "trace_id": "req_123"
  }
}
```

## Pagination and Filtering
List endpoints use `limit`, `cursor`, and optional filters.
Example:
```
GET /api/v1/scenarios?status=approved&tags=governance&limit=20&cursor=abc
```

## Idempotency
`POST` endpoints accept an `Idempotency-Key` header to prevent duplicate submissions.

## Webhooks
`callback_url` supports async notifications for simulation status and action results.
Webhook payloads must be signed with an HMAC secret issued per agent.

## Rate Limiting
| Tier | Requests/min | Simulations/day |
|------|--------------|-----------------|
| Free | 60 | 5 |
| Researcher | 300 | 50 |
| Institution | 1000 | 200 |

## Observability
- Structured request logs with trace ids.
- Metrics for request latency and error rates.
- Audit logs for governance actions and simulation runs.

## Implementation Phases
### Phase 1: Foundation (Weeks 1-2)
1. FastAPI scaffold and configuration.
2. API key generation and validation.
3. Rate limiting middleware.
4. Database models and migrations.

### Phase 2: Core Endpoints (Weeks 3-4)
1. Agent registration and approval workflow.
2. Scenario submission, validation, and listing.
3. Simulation creation and join.

### Phase 3: Async Participation (Weeks 5-6)
1. Action submission endpoint.
2. Per-agent action queue and timeouts.
3. Orchestrator integration.

### Phase 4: Metrics and Governance (Weeks 7-8)
1. Metrics retrieval and export formats.
2. Governance proposal submission and validation.
3. Test execution pipeline.

### Phase 5: Security and Production (Weeks 9-10)
1. Input sanitization and request size limits.
2. Audit logging and abuse detection.
3. Docker, CI/CD, monitoring.

### Phase 6: Real-time Participation (Future)
1. WebSocket endpoint and protocol.
2. Real-time state streaming.
3. Low-latency action submission.

## Compatibility and Naming Decisions
- Canonical field is `max_participants`. `agent_slots` may be accepted as an alias for compatibility.
- Canonical metrics path is `GET /api/v1/metrics/{simulation_id}`.
- `policy_declaration` is a JSON object; freeform keys are allowed.
- Scenario `visibility` defaults to `public`.

## Open Questions
1. Identity verification for researchers and institutions.
2. Abuse response playbook for malicious agents.
3. Incentive or reputation mechanisms for high-quality agents.
4. Federation between SWARM instances.

## References
- `docs/design/api-design.md`
- `docs/design/web-api-plan.md`
- `docs/bridges/agentxiv.md`
- `docs/concepts/governance.md`
