"""Simulation management endpoints."""

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query

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
