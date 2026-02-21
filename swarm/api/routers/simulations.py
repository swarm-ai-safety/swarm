"""Simulation management endpoints."""

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from swarm.api.action_queue import AsyncActionQueue
from swarm.api.event_bus import SimEvent, SimEventType, event_bus
from swarm.api.middleware.auth import Scope, require_scope
from swarm.api.models.action import ActionSubmission
from swarm.api.models.simulation import (
    SimulationCreate,
    SimulationJoin,
    SimulationResponse,
    SimulationStatus,
)

router = APIRouter()

# In-memory storage for development
_simulations: dict[str, SimulationResponse] = {}
_participants: dict[str, list[dict]] = {}  # simulation_id -> list of participant dicts
_observations: dict[str, dict[str, dict]] = {}  # simulation_id -> agent_id -> observation
_action_queues: dict[str, AsyncActionQueue] = {}  # simulation_id -> queue


@router.post("/create", response_model=SimulationResponse)
async def create_simulation(request: SimulationCreate) -> SimulationResponse:
    """Create a new simulation session.

    Args:
        request: Simulation creation parameters.

    Returns:
        Created simulation details.
    """
    simulation_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    simulation = SimulationResponse(
        simulation_id=simulation_id,
        scenario_id=request.scenario_id,
        status=SimulationStatus.WAITING,
        mode=request.mode,
        max_participants=request.max_participants,
        current_participants=0,
        created_at=now,
        join_deadline=now + timedelta(minutes=30),
        config_overrides=request.config_overrides,
    )

    _simulations[simulation_id] = simulation
    _participants[simulation_id] = []
    return simulation


@router.post("/{simulation_id}/join")
async def join_simulation(simulation_id: str, request: SimulationJoin) -> dict:
    """Join an existing simulation as a participant.

    Args:
        simulation_id: The simulation to join.
        request: Join request with agent details.

    Returns:
        Participation confirmation.

    Raises:
        HTTPException: If simulation not found or join fails.
    """
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    simulation = _simulations[simulation_id]

    # Check join deadline before other checks
    if datetime.now(timezone.utc) > simulation.join_deadline:
        raise HTTPException(status_code=400, detail="Join deadline has passed")

    if simulation.status != SimulationStatus.WAITING:
        raise HTTPException(
            status_code=400, detail="Simulation is not accepting participants"
        )

    if simulation.current_participants >= simulation.max_participants:
        raise HTTPException(status_code=400, detail="Simulation is full")

    # Add participant with metadata
    now = datetime.now(timezone.utc)
    _participants[simulation_id].append({
        "agent_id": request.agent_id,
        "role": request.role,
        "joined_at": now.isoformat(),
    })
    simulation.current_participants += 1

    participant_id = str(uuid.uuid4())

    return {
        "participant_id": participant_id,
        "simulation_id": simulation_id,
        "agent_id": request.agent_id,
        "role": request.role,
        "status": "joined",
    }


@router.get("/{simulation_id}/state")
async def get_simulation_state(simulation_id: str) -> dict:
    """Get detailed simulation state.

    Args:
        simulation_id: The simulation's unique identifier.

    Returns:
        Detailed simulation state including participants and config.

    Raises:
        HTTPException: If simulation not found.
    """
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    simulation = _simulations[simulation_id]
    participants = _participants.get(simulation_id, [])

    now = datetime.now(timezone.utc)
    deadline = simulation.join_deadline
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    remaining = (deadline - now).total_seconds()
    time_remaining_seconds = max(0, remaining)

    return {
        "simulation_id": simulation.simulation_id,
        "status": simulation.status.value,
        "participants": participants,
        "config_overrides": simulation.config_overrides,
        "join_deadline": simulation.join_deadline.isoformat(),
        "time_remaining_seconds": time_remaining_seconds,
    }


@router.post("/{simulation_id}/start")
async def start_simulation(simulation_id: str) -> dict:
    """Start a simulation, transitioning from WAITING to RUNNING.

    Args:
        simulation_id: The simulation to start.

    Returns:
        Updated simulation status.

    Raises:
        HTTPException: If simulation not found, not in WAITING state,
            or doesn't have enough participants.
    """
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    simulation = _simulations[simulation_id]

    if simulation.status != SimulationStatus.WAITING:
        raise HTTPException(
            status_code=400, detail="Simulation is not in waiting state"
        )

    if simulation.current_participants < 2:
        raise HTTPException(
            status_code=400, detail="Not enough participants to start"
        )

    simulation.status = SimulationStatus.RUNNING

    return {
        "simulation_id": simulation_id,
        "status": "running",
    }


@router.get("/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(simulation_id: str) -> SimulationResponse:
    """Get simulation details by ID.

    Args:
        simulation_id: The simulation's unique identifier.

    Returns:
        Simulation details.

    Raises:
        HTTPException: If simulation not found.
    """
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return _simulations[simulation_id]


@router.get("/", response_model=list[SimulationResponse])
async def list_simulations(
    status: SimulationStatus | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[SimulationResponse]:
    """List all simulations with optional filtering and pagination.

    Args:
        status: Filter by simulation status.
        limit: Maximum number of results to return.
        offset: Number of results to skip.

    Returns:
        List of simulations.
    """
    simulations = list(_simulations.values())

    if status is not None:
        simulations = [s for s in simulations if s.status == status]

    return simulations[offset : offset + limit]


# ---------------------------------------------------------------------------
# Action submission
# ---------------------------------------------------------------------------


@router.post(
    "/{simulation_id}/actions",
    dependencies=[Depends(require_scope(Scope.PARTICIPATE))],
)
async def submit_action(simulation_id: str, action: ActionSubmission) -> dict:
    """Submit an action for a running simulation step.

    Args:
        simulation_id: The simulation to act in.
        action: The action to submit.

    Returns:
        Acceptance confirmation with action_id.

    Raises:
        HTTPException: If simulation not found, not running, or agent not a
            participant.
    """
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    simulation = _simulations[simulation_id]
    if simulation.status != SimulationStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Simulation is not running")

    # Verify agent is a participant
    participants = _participants.get(simulation_id, [])
    agent_ids = {p["agent_id"] for p in participants}
    if action.agent_id not in agent_ids:
        raise HTTPException(
            status_code=403, detail="Agent is not a participant in this simulation"
        )

    # Submit to the action queue
    queue = _action_queues.get(simulation_id)
    if queue is None:
        queue = AsyncActionQueue()
        _action_queues[simulation_id] = queue

    accepted = await queue.submit_action(
        action.agent_id,
        {
            "action_type": action.action_type.value,
            "payload": action.payload,
            "step": action.step,
        },
    )

    action_id = str(uuid.uuid4())
    return {
        "action_id": action_id,
        "simulation_id": simulation_id,
        "agent_id": action.agent_id,
        "status": "accepted" if accepted else "no_waiter",
    }


# ---------------------------------------------------------------------------
# Observation polling
# ---------------------------------------------------------------------------


@router.get(
    "/{simulation_id}/observation",
    dependencies=[Depends(require_scope(Scope.READ))],
)
async def get_observation(simulation_id: str, agent_id: str = Query(...)) -> dict:
    """Get the current observation for an agent in a simulation.

    Args:
        simulation_id: The simulation to query.
        agent_id: The agent whose observation to retrieve.

    Returns:
        The agent's current observation data.

    Raises:
        HTTPException: If simulation not found or agent not a participant.
    """
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    # Verify agent is a participant
    participants = _participants.get(simulation_id, [])
    agent_ids = {p["agent_id"] for p in participants}
    if agent_id not in agent_ids:
        raise HTTPException(
            status_code=403, detail="Agent is not a participant in this simulation"
        )

    sim_observations = _observations.get(simulation_id, {})
    obs = sim_observations.get(agent_id)
    if obs is None:
        return {
            "simulation_id": simulation_id,
            "agent_id": agent_id,
            "observation": None,
            "status": "no_observation",
        }

    return {
        "simulation_id": simulation_id,
        "agent_id": agent_id,
        "observation": obs,
        "status": "ready",
    }


# ---------------------------------------------------------------------------
# SSE event stream
# ---------------------------------------------------------------------------


@router.get("/{simulation_id}/events")
async def simulation_events(
    simulation_id: str,
    agent_id: str = Query(..., description="Agent ID for filtering events"),
):
    """Stream simulation events via Server-Sent Events.

    Args:
        simulation_id: The simulation to stream events from.
        agent_id: The agent to receive events for.

    Returns:
        SSE event stream.

    Raises:
        HTTPException: If simulation not found.
    """
    import asyncio

    from starlette.responses import StreamingResponse

    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    queue = event_bus.subscribe(simulation_id, agent_id)

    async def event_generator():
        try:
            while True:
                try:
                    event: SimEvent = await asyncio.wait_for(
                        queue.get(), timeout=15.0
                    )
                    data = {
                        "event_type": event.event_type.value,
                        "simulation_id": event.simulation_id,
                        "data": event.data,
                    }
                    import json

                    yield f"event: {event.event_type.value}\ndata: {json.dumps(data)}\n\n"

                    if event.event_type == SimEventType.SIMULATION_COMPLETE:
                        break
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield ": heartbeat\n\n"
        finally:
            event_bus.unsubscribe(simulation_id, agent_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
