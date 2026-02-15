"""Agent registration and management endpoints."""

import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from swarm.api.middleware import register_api_key
from swarm.api.models.agent import (
    AgentRegistration,
    AgentResponse,
    AgentStatus,
)

router = APIRouter()

# In-memory storage for development (replace with database in production)
_registered_agents: dict[str, AgentResponse] = {}

# Rate limiting for registration endpoint (per-IP).
# Maps IP -> list of registration timestamps within the window.
_registration_rate: dict[str, list[float]] = {}
_REGISTRATION_WINDOW = 60.0  # seconds
_REGISTRATION_LIMIT = 10  # max registrations per IP per window
MAX_REGISTERED_AGENTS = 10_000


@router.post("/register", response_model=AgentResponse)
async def register_agent(
    registration: AgentRegistration, request: Request
) -> AgentResponse:
    """Register a new agent to participate in SWARM simulations.

    Args:
        registration: Agent registration details.

    Returns:
        Registered agent info with API key.
    """
    # Per-IP rate limiting to prevent registration flooding (security fix 5.1)
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _registration_rate.get(client_ip, [])
    window = [t for t in window if now - t < _REGISTRATION_WINDOW]
    if len(window) >= _REGISTRATION_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Too many registrations. Try again later.",
        )
    window.append(now)
    _registration_rate[client_ip] = window

    # Cap total registered agents
    if len(_registered_agents) >= MAX_REGISTERED_AGENTS:
        raise HTTPException(
            status_code=429,
            detail="Registration capacity reached. Try again later.",
        )

    agent_id = str(uuid.uuid4())
    api_key = f"swarm_{uuid.uuid4().hex}"  # Simple key for now

    agent = AgentResponse(
        agent_id=agent_id,
        api_key=api_key,
        name=registration.name,
        description=registration.description,
        capabilities=registration.capabilities,
        status=AgentStatus.APPROVED,  # Auto-approve for now
        registered_at=datetime.now(timezone.utc),
    )

    _registered_agents[agent_id] = agent

    # Register the API key so it can be used for runs/posts endpoints
    register_api_key(api_key, agent_id)

    return agent


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str) -> AgentResponse:
    """Get agent details by ID.

    Args:
        agent_id: The agent's unique identifier.

    Returns:
        Agent details.

    Raises:
        HTTPException: If agent not found.
    """
    if agent_id not in _registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _registered_agents[agent_id]
    # Don't return the API key on subsequent requests
    return AgentResponse(
        agent_id=agent.agent_id,
        api_key="[REDACTED]",
        name=agent.name,
        description=agent.description,
        capabilities=agent.capabilities,
        status=agent.status,
        registered_at=agent.registered_at,
    )


@router.get("/", response_model=list[AgentResponse])
async def list_agents() -> list[AgentResponse]:
    """List all registered agents.

    Returns:
        List of registered agents (with redacted API keys).
    """
    return [
        AgentResponse(
            agent_id=a.agent_id,
            api_key="[REDACTED]",
            name=a.name,
            description=a.description,
            capabilities=a.capabilities,
            status=a.status,
            registered_at=a.registered_at,
        )
        for a in _registered_agents.values()
    ]
