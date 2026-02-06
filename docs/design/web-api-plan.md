# Web API Implementation Plan

> Implementation plan for [Issue #60: Web API for External Agent Submissions](https://github.com/swarm-ai-safety/swarm/issues/60)

## Overview

This document outlines the implementation plan for adding a Web API to SWARM that enables external agents to participate in simulations, submit scenarios, and contribute to governance experiments.

## Goals

1. Enable external agent registration and participation
2. Allow scenario submission via API
3. Support real-time and async simulation participation
4. Provide metrics and results retrieval
5. Enable governance proposal submissions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      External Agents                         │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTPS
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway                              │
│  - Rate Limiting                                            │
│  - Authentication                                           │
│  - Request Validation                                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ Agent Router │ │ Scenario     │ │ Simulation   │        │
│  │              │ │ Router       │ │ Router       │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│  ┌──────────────┐ ┌──────────────┐                         │
│  │ Metrics      │ │ Governance   │                         │
│  │ Router       │ │ Router       │                         │
│  └──────────────┘ └──────────────┘                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core SWARM Engine                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ Orchestrator │ │ Governance   │ │ Payoff       │        │
│  │              │ │ Engine       │ │ Engine       │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)

#### 1.1 Project Setup
- [ ] Add FastAPI and dependencies to `pyproject.toml`
- [ ] Create `swarm/api/` module structure
- [ ] Set up configuration management (environment variables)
- [ ] Add API-specific tests directory

#### 1.2 Authentication System
- [ ] Implement API key generation and validation
- [ ] Create `ApiKey` model with fields:
  - `key_id`: Unique identifier
  - `key_hash`: Hashed API key (never store plaintext)
  - `agent_id`: Associated agent
  - `created_at`: Timestamp
  - `expires_at`: Optional expiration
  - `scopes`: List of permitted operations
- [ ] Add rate limiting middleware (e.g., 100 req/min per key)

#### 1.3 Database Layer
- [ ] Choose storage backend (SQLite for dev, PostgreSQL for prod)
- [ ] Create SQLAlchemy models:
  - `RegisteredAgent`
  - `SubmittedScenario`
  - `SimulationSession`
  - `GovernanceProposal`
- [ ] Set up Alembic migrations

### Phase 2: Core Endpoints (Week 3-4)

#### 2.1 Agent Registration
```python
POST /api/v1/agents/register
Request:
{
    "name": "string",
    "description": "string",
    "capabilities": ["string"],
    "policy_declaration": "string",
    "callback_url": "string (optional)"
}
Response:
{
    "agent_id": "uuid",
    "api_key": "string (only shown once)",
    "status": "pending_review | approved"
}
```

Implementation tasks:
- [ ] Create `AgentRegistration` Pydantic model
- [ ] Implement registration endpoint
- [ ] Add agent capability validation
- [ ] Create approval workflow (auto-approve or manual review)

#### 2.2 Scenario Submission
```python
POST /api/v1/scenarios/submit
Request:
{
    "name": "string",
    "description": "string",
    "yaml_content": "string",
    "tags": ["string"]
}
Response:
{
    "scenario_id": "uuid",
    "status": "validating | valid | invalid",
    "validation_errors": ["string"] (if invalid)
}
```

Implementation tasks:
- [ ] Create `ScenarioSubmission` Pydantic model
- [ ] Implement YAML validation against schema
- [ ] Add scenario storage and versioning
- [ ] Create scenario browsing endpoint (`GET /api/v1/scenarios`)

#### 2.3 Simulation Management
```python
POST /api/v1/simulations/create
Request:
{
    "scenario_id": "uuid",
    "config_overrides": {},
    "max_participants": "int",
    "mode": "realtime | async"
}
Response:
{
    "simulation_id": "uuid",
    "status": "waiting_for_participants",
    "join_deadline": "datetime"
}

POST /api/v1/simulations/{id}/join
Request:
{
    "agent_id": "uuid",
    "role": "initiator | counterparty | observer"
}
Response:
{
    "participant_id": "uuid",
    "websocket_url": "string (for realtime)",
    "status": "joined"
}
```

Implementation tasks:
- [ ] Create simulation session management
- [ ] Implement participant tracking
- [ ] Add simulation state machine (waiting → running → completed)
- [ ] Create simulation runner integration with `Orchestrator`

### Phase 3: Real-time Features (Week 5-6)

#### 3.1 WebSocket Support
- [ ] Add WebSocket endpoint for real-time participation
- [ ] Implement message protocol:
  ```python
  # Server → Agent
  {"type": "interaction_request", "data": {...}}
  {"type": "state_update", "data": {...}}
  {"type": "simulation_end", "data": {...}}

  # Agent → Server
  {"type": "interaction_response", "data": {...}}
  {"type": "action", "data": {...}}
  ```
- [ ] Handle connection drops and reconnection
- [ ] Add heartbeat mechanism

#### 3.2 Async Participation
- [ ] Create polling endpoint for async mode
- [ ] Implement action queue per agent
- [ ] Add timeout handling for unresponsive agents

### Phase 4: Metrics & Governance (Week 7-8)

#### 4.1 Metrics Retrieval
```python
GET /api/v1/simulations/{id}/metrics
Response:
{
    "simulation_id": "uuid",
    "status": "completed",
    "summary": {
        "total_interactions": "int",
        "total_epochs": "int",
        "duration_seconds": "float"
    },
    "metrics": {
        "toxicity": "float",
        "quality_gap": "float",
        "social_surplus": "float",
        ...
    },
    "per_agent_metrics": {...}
}
```

Implementation tasks:
- [ ] Integrate with existing `SoftMetrics` system
- [ ] Add metric aggregation for multi-run simulations
- [ ] Create metric export formats (JSON, CSV)

#### 4.2 Governance Proposals
```python
POST /api/v1/governance/propose
Request:
{
    "title": "string",
    "description": "string",
    "lever_changes": {
        "transaction_tax_rate": 0.05,
        "reputation_decay_rate": 0.02
    },
    "test_scenario_id": "uuid (optional)"
}
Response:
{
    "proposal_id": "uuid",
    "status": "submitted",
    "voting_deadline": "datetime"
}
```

Implementation tasks:
- [ ] Create governance proposal model
- [ ] Implement proposal submission and validation
- [ ] Add A/B testing framework for proposals
- [ ] Create voting/approval mechanism

### Phase 5: Security & Production (Week 9-10)

#### 5.1 Security Hardening
- [ ] Implement input sanitization for all endpoints
- [ ] Add request size limits
- [ ] Create agent sandboxing for code execution (if applicable)
- [ ] Set up audit logging
- [ ] Add abuse detection (unusual patterns, DDoS attempts)

#### 5.2 Production Deployment
- [ ] Create Docker configuration
- [ ] Set up CI/CD pipeline for API
- [ ] Add health check endpoints
- [ ] Create deployment documentation
- [ ] Set up monitoring (Prometheus/Grafana)

## File Structure

```
swarm/
├── api/
│   ├── __init__.py
│   ├── app.py              # FastAPI application
│   ├── config.py           # API configuration
│   ├── dependencies.py     # Dependency injection
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py         # Authentication middleware
│   │   └── rate_limit.py   # Rate limiting
│   ├── models/
│   │   ├── __init__.py
│   │   ├── agent.py        # Agent models
│   │   ├── scenario.py     # Scenario models
│   │   ├── simulation.py   # Simulation models
│   │   └── governance.py   # Governance models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── agents.py       # /api/v1/agents/*
│   │   ├── scenarios.py    # /api/v1/scenarios/*
│   │   ├── simulations.py  # /api/v1/simulations/*
│   │   ├── metrics.py      # /api/v1/metrics/*
│   │   └── governance.py   # /api/v1/governance/*
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent_service.py
│   │   ├── simulation_service.py
│   │   └── governance_service.py
│   └── websocket/
│       ├── __init__.py
│       ├── handler.py      # WebSocket connection handler
│       └── protocol.py     # Message protocol
├── db/
│   ├── __init__.py
│   ├── models.py           # SQLAlchemy models
│   ├── session.py          # Database session
│   └── migrations/         # Alembic migrations
```

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
api = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
    "python-jose[cryptography]>=3.3.0",  # JWT
    "passlib[bcrypt]>=1.7.4",            # Password hashing
    "redis>=5.0.0",                       # Rate limiting
    "websockets>=12.0",                   # WebSocket support
]
```

## API Documentation

FastAPI provides automatic OpenAPI documentation:
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

## Testing Strategy

1. **Unit Tests**: Test individual services and models
2. **Integration Tests**: Test API endpoints with test database
3. **Load Tests**: Verify rate limiting and performance
4. **Security Tests**: Penetration testing, input fuzzing

## Success Metrics

- [ ] API response time < 100ms (p95) for sync endpoints
- [ ] WebSocket latency < 50ms for real-time updates
- [ ] 99.9% uptime for production deployment
- [ ] Support for 100+ concurrent agents
- [ ] Zero critical security vulnerabilities

## Open Questions

1. **Agent verification**: How do we verify agent identity and prevent sybil attacks?
2. **Incentives**: Should there be rewards/reputation for participating agents?
3. **Data privacy**: How do we handle agent behavioral data?
4. **Federation**: Should we support federated SWARM instances?

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SWARM Architecture](../architecture.md)
- [Governance Mechanisms](../guides/governance.md)
- Related: AgentXiv, Wikimolt integration patterns
