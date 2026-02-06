"""Simulation management endpoints."""

import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException

from swarm.api.models.simulation import (
    SimulationCreate,
    SimulationJoin,
    SimulationResponse,
    SimulationStatus,
)

router = APIRouter()

# In-memory storage for development
_simulations: dict[str, SimulationResponse] = {}
_participants: dict[str, list[str]] = {}  # simulation_id -> list of agent_ids


@router.post("/create", response_model=SimulationResponse)
async def create_simulation(request: SimulationCreate) -> SimulationResponse:
    """Create a new simulation session.

    Args:
        request: Simulation creation parameters.

    Returns:
        Created simulation details.
    """
    simulation_id = str(uuid.uuid4())
    now = datetime.utcnow()

    simulation = SimulationResponse(
        simulation_id=simulation_id,
        scenario_id=request.scenario_id,
        status=SimulationStatus.WAITING,
        mode=request.mode,
        max_participants=request.max_participants,
        current_participants=0,
        created_at=now,
        join_deadline=now + timedelta(minutes=30),
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

    if simulation.status != SimulationStatus.WAITING:
        raise HTTPException(
            status_code=400, detail="Simulation is not accepting participants"
        )

    if simulation.current_participants >= simulation.max_participants:
        raise HTTPException(status_code=400, detail="Simulation is full")

    # Add participant
    _participants[simulation_id].append(request.agent_id)
    simulation.current_participants += 1

    participant_id = str(uuid.uuid4())

    return {
        "participant_id": participant_id,
        "simulation_id": simulation_id,
        "agent_id": request.agent_id,
        "role": request.role,
        "status": "joined",
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
async def list_simulations() -> list[SimulationResponse]:
    """List all simulations.

    Returns:
        List of simulations.
    """
    return list(_simulations.values())
