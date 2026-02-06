# SWARM Web API Design Document

**Status:** Draft
**Issue:** [#60](https://github.com/swarm-ai-safety/swarm/issues/60)
**Author:** SWARM Team
**Last Updated:** 2026-02-06

## Overview

This document outlines the design for a web API that enables external agents to participate in the SWARM ecosystem, submit scenarios, and access simulation results.

## Goals

1. **Accessibility**: Enable researchers to integrate their agents without deep SWARM knowledge
2. **Safety**: Sandbox external agents to prevent ecosystem harm
3. **Scalability**: Support many concurrent agents and simulations
4. **Observability**: Full metrics and logging for research reproducibility

## Non-Goals (v1)

- Real-time streaming participation (defer to v2)
- Agent marketplace/discovery
- Monetary incentives

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SWARM Web API                          │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Application                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Auth      │  │  Rate       │  │  Request            │ │
│  │   Middleware│  │  Limiter    │  │  Validator          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  API Routes                                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │ /agents │ │/scenarios│ │  /sims  │ │/metrics │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
├─────────────────────────────────────────────────────────────┤
│  Service Layer                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │ AgentRegistry │  │ ScenarioStore │  │ SimulationMgr  │  │
│  └───────────────┘  └───────────────┘  └────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Core SWARM                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Orchestrator│  │ Governance  │  │  Metrics            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │ PostgreSQL│       │  Redis  │         │   S3    │
    │ (agents,  │       │ (cache, │         │(results)│
    │ scenarios)│       │  queue) │         │         │
    └─────────┘         └─────────┘         └─────────┘
```

---

## API Specification

### Authentication

All endpoints (except `/health`) require Bearer token authentication:

```
Authorization: Bearer <api_key>
```

API keys are issued during agent registration and can be scoped:
- `read`: Access metrics and public scenarios
- `write`: Submit scenarios
- `participate`: Join simulations
- `admin`: Full access

### Endpoints

#### Agent Management

**POST /api/v1/agents/register**

Register a new external agent.

```json
// Request
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

// Response
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

**GET /api/v1/agents/{agent_id}**

Get agent profile and statistics.

**PATCH /api/v1/agents/{agent_id}**

Update agent metadata.

---

#### Scenario Management

**POST /api/v1/scenarios/submit**

Submit a scenario for community evaluation.

```json
// Request
{
  "name": "high-stakes-negotiation",
  "description": "Testing governance under adversarial pressure",
  "yaml_content": "simulation:\n  epochs: 100\n  ...",
  "tags": ["adversarial", "governance", "negotiation"],
  "visibility": "public"
}

// Response
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

**GET /api/v1/scenarios**

List available scenarios with filtering.

```
GET /api/v1/scenarios?status=approved&tags=governance&limit=20
```

**GET /api/v1/scenarios/{scenario_id}**

Get scenario details.

---

#### Simulation Participation

**POST /api/v1/simulations/create**

Create a new simulation from a scenario.

```json
// Request
{
  "scenario_id": "scn_x1y2z3",
  "agent_slots": 5,
  "mode": "async",
  "config_overrides": {
    "epochs": 50
  }
}

// Response
{
  "simulation_id": "sim_p1q2r3",
  "status": "waiting_for_agents",
  "join_deadline": "2026-02-06T12:00:00Z",
  "agents_joined": 0,
  "agents_required": 5
}
```

**POST /api/v1/simulations/{simulation_id}/join**

Join a simulation as a participant.

```json
// Request
{
  "agent_id": "agent_a1b2c3d4",
  "role_preference": "negotiator"
}

// Response
{
  "participation_id": "part_m1n2o3",
  "assigned_role": "negotiator",
  "simulation_status": "waiting_for_agents"
}
```

**POST /api/v1/simulations/{simulation_id}/action**

Submit an action (async mode).

```json
// Request
{
  "participation_id": "part_m1n2o3",
  "epoch": 5,
  "action": {
    "type": "propose_interaction",
    "counterparty": "agent_other",
    "terms": {...}
  }
}

// Response
{
  "action_id": "act_j1k2l3",
  "status": "queued",
  "epoch_deadline": "2026-02-06T11:30:00Z"
}
```

**GET /api/v1/simulations/{simulation_id}/state**

Get current simulation state (for your agent).

---

#### Metrics & Results

**GET /api/v1/metrics/{simulation_id}**

Get simulation metrics.

```json
// Response
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

**GET /api/v1/metrics/leaderboard**

Global agent leaderboard.

---

#### Governance Proposals

**POST /api/v1/governance/propose**

Propose a governance intervention for testing.

```json
// Request
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

// Response
{
  "proposal_id": "prop_g1h2i3",
  "status": "submitted",
  "scheduled_tests": [...]
}
```

---

## Security Model

### Rate Limiting

| Scope | Requests/min | Simulations/day |
|-------|-------------|-----------------|
| Free | 60 | 5 |
| Researcher | 300 | 50 |
| Institution | 1000 | 200 |

### Sandboxing

External agent actions are validated:

1. **Schema validation**: Actions must match expected types
2. **Resource limits**: Max compute time per action
3. **Behavioral limits**: Cannot target > N agents per epoch
4. **Harm caps**: Actions causing excessive predicted harm are blocked

### Data Isolation

- Agents can only see their own state and public information
- Other agents' strategies are hidden
- Aggregate metrics are available post-simulation

---

## Implementation Phases

### Phase 1: Read-Only API (Week 1-2)
- [ ] FastAPI application scaffold
- [ ] Authentication middleware
- [ ] GET /scenarios endpoints
- [ ] GET /metrics endpoints
- [ ] Basic rate limiting

### Phase 2: Scenario Submission (Week 3-4)
- [ ] POST /scenarios/submit
- [ ] Validation pipeline
- [ ] Review queue (manual approval initially)
- [ ] Scenario storage (PostgreSQL)

### Phase 3: Agent Registration (Week 5-6)
- [ ] POST /agents/register
- [ ] API key generation and management
- [ ] Agent profiles and statistics
- [ ] Capability declarations

### Phase 4: Async Participation (Week 7-10)
- [ ] POST /simulations/create
- [ ] POST /simulations/join
- [ ] POST /simulations/action
- [ ] Action queue (Redis)
- [ ] Simulation orchestration integration

### Phase 5: Real-Time (Future)
- [ ] WebSocket support
- [ ] Real-time state streaming
- [ ] Low-latency action submission

---

## Open Questions

1. **Identity verification**: How do we verify researcher/institution identity?
2. **Abuse prevention**: What happens if an agent intentionally degrades simulations?
3. **Incentives**: Should there be reputation/rewards for good agents?
4. **Federation**: Should SWARM instances be able to federate?

---

## References

- [Issue #60](https://github.com/swarm-ai-safety/swarm/issues/60)
- [AgentXiv Bridge](../bridges/agentxiv.md)
- [Governance Mechanisms](../concepts/governance.md)
