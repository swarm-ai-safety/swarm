"""Simulation management endpoints."""

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)

from swarm.api.action_queue import AsyncActionQueue
from swarm.api.event_bus import SimEvent, SimEventType, event_bus
from swarm.api.middleware.auth import Scope, _key_scopes, _lookup_key, require_scope
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
_action_history: dict[str, list[dict]] = {}  # simulation_id -> list of action records
_execution_state: dict[str, dict] = {}  # simulation_id -> {current_step, current_epoch, ...}
_simulation_results: dict[str, dict] = {}  # simulation_id -> final results/metrics
_ws_connections: dict[str, dict[str, WebSocket]] = {}  # simulation_id -> agent_id -> WebSocket

# Concurrency limits
MAX_ACTIVE_SIMULATIONS = 50
MAX_RESULTS_BYTES = 1_048_576  # 1 MiB cap on serialized results payload


@router.post(
    "/create",
    response_model=SimulationResponse,
    dependencies=[Depends(require_scope(Scope.WRITE))],
)
async def create_simulation(request: SimulationCreate) -> SimulationResponse:
    """Create a new simulation session.

    Args:
        request: Simulation creation parameters.

    Returns:
        Created simulation details.

    Raises:
        HTTPException: If concurrency limit exceeded.
    """
    # Enforce global concurrency limit
    active = sum(
        1
        for s in _simulations.values()
        if s.status in (SimulationStatus.WAITING, SimulationStatus.RUNNING)
    )
    if active >= MAX_ACTIVE_SIMULATIONS:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum active simulations ({MAX_ACTIVE_SIMULATIONS}) reached",
        )

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


@router.post(
    "/{simulation_id}/join",
    dependencies=[Depends(require_scope(Scope.PARTICIPATE))],
)
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


@router.get(
    "/{simulation_id}/state",
    dependencies=[Depends(require_scope(Scope.READ))],
)
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


@router.post(
    "/{simulation_id}/start",
    dependencies=[Depends(require_scope(Scope.PARTICIPATE))],
)
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


@router.get(
    "/{simulation_id}",
    response_model=SimulationResponse,
    dependencies=[Depends(require_scope(Scope.READ))],
)
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


@router.get(
    "/",
    response_model=list[SimulationResponse],
    dependencies=[Depends(require_scope(Scope.READ))],
)
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


@router.post("/{simulation_id}/actions")
async def submit_action(
    simulation_id: str,
    action: ActionSubmission,
    auth_agent_id: str = Depends(require_scope(Scope.PARTICIPATE)),
) -> dict:
    """Submit an action for a running simulation step.

    Args:
        simulation_id: The simulation to act in.
        action: The action to submit.
        auth_agent_id: Authenticated agent identity (from API key).

    Returns:
        Acceptance confirmation with action_id.

    Raises:
        HTTPException: If simulation not found, not running, or agent not a
            participant.
    """
    # Bind authenticated identity to the action — prevent impersonation
    if action.agent_id != auth_agent_id:
        raise HTTPException(
            status_code=403,
            detail="Cannot submit actions for another agent",
        )

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

    # Record action in history
    if simulation_id not in _action_history:
        _action_history[simulation_id] = []
    _action_history[simulation_id].append({
        "action_id": action_id,
        "agent_id": action.agent_id,
        "action_type": action.action_type.value,
        "step": action.step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
    })

    return {
        "action_id": action_id,
        "simulation_id": simulation_id,
        "agent_id": action.agent_id,
        "status": "accepted" if accepted else "no_waiter",
    }


# ---------------------------------------------------------------------------
# Observation polling
# ---------------------------------------------------------------------------


@router.get("/{simulation_id}/observation")
async def get_observation(
    simulation_id: str,
    agent_id: str = Query(...),
    auth_agent_id: str = Depends(require_scope(Scope.READ)),
) -> dict:
    """Get the current observation for an agent in a simulation.

    Args:
        simulation_id: The simulation to query.
        agent_id: The agent whose observation to retrieve.
        auth_agent_id: Authenticated agent identity (from API key).

    Returns:
        The agent's current observation data.

    Raises:
        HTTPException: If simulation not found or agent not a participant.
    """
    # Bind authenticated identity — prevent reading other agents' observations
    if agent_id != auth_agent_id:
        raise HTTPException(
            status_code=403,
            detail="Cannot read observations for another agent",
        )

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


@router.get(
    "/{simulation_id}/events",
    dependencies=[Depends(require_scope(Scope.READ))],
)
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


# ---------------------------------------------------------------------------
# Simulation completion and results
# ---------------------------------------------------------------------------


@router.post(
    "/{simulation_id}/complete",
    dependencies=[Depends(require_scope(Scope.PARTICIPATE))],
)
async def complete_simulation(
    simulation_id: str, results: dict | None = None
) -> dict:
    """Mark a simulation as completed and store its results.

    Args:
        simulation_id: The simulation to complete.
        results: Optional results/metrics dict to store.

    Returns:
        Updated simulation status.

    Raises:
        HTTPException: If simulation not found or not running.
    """
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    simulation = _simulations[simulation_id]
    if simulation.status != SimulationStatus.RUNNING:
        raise HTTPException(
            status_code=400, detail="Simulation is not running"
        )

    # Validate results payload size before accepting
    if results is not None:
        try:
            payload_size = len(json.dumps(results))
        except (TypeError, ValueError, OverflowError) as exc:
            raise HTTPException(
                status_code=422, detail="Results payload is not JSON-serializable"
            ) from exc
        if payload_size > MAX_RESULTS_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Results payload exceeds {MAX_RESULTS_BYTES} byte limit",
            )

    simulation.status = SimulationStatus.COMPLETED

    # Store results
    if results is not None:
        _simulation_results[simulation_id] = results

    # Clean up action queue
    queue = _action_queues.pop(simulation_id, None)
    if queue is not None:
        await queue.cancel_all()

    # Publish completion event
    await event_bus.publish(
        SimEvent(
            event_type=SimEventType.SIMULATION_COMPLETE,
            simulation_id=simulation_id,
            data={"results": results or {}},
        )
    )

    return {
        "simulation_id": simulation_id,
        "status": "completed",
    }


@router.get(
    "/{simulation_id}/results",
    dependencies=[Depends(require_scope(Scope.READ))],
)
async def get_simulation_results(simulation_id: str) -> dict:
    """Get results for a completed simulation.

    Args:
        simulation_id: The simulation to get results for.

    Returns:
        Simulation results including metrics and action history.

    Raises:
        HTTPException: If simulation not found or not completed.
    """
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    simulation = _simulations[simulation_id]
    if simulation.status != SimulationStatus.COMPLETED:
        raise HTTPException(
            status_code=400, detail="Simulation is not completed"
        )

    results = _simulation_results.get(simulation_id, {})
    history = _action_history.get(simulation_id, [])
    participants = _participants.get(simulation_id, [])

    return {
        "simulation_id": simulation_id,
        "status": "completed",
        "results": results,
        "action_count": len(history),
        "participant_count": len(participants),
        "participants": participants,
    }


# ---------------------------------------------------------------------------
# Execution state
# ---------------------------------------------------------------------------


@router.get(
    "/{simulation_id}/execution",
    dependencies=[Depends(require_scope(Scope.READ))],
)
async def get_execution_state(simulation_id: str) -> dict:
    """Get execution state for a running simulation.

    Returns per-agent action counts and the simulation's current
    step/epoch if tracked.

    Args:
        simulation_id: The simulation to query.

    Returns:
        Execution state including per-agent action counts.

    Raises:
        HTTPException: If simulation not found.
    """
    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    history = _action_history.get(simulation_id, [])
    exec_state = _execution_state.get(simulation_id, {})

    # Compute per-agent action counts
    agent_counts: dict[str, int] = {}
    for record in history:
        aid = record["agent_id"]
        agent_counts[aid] = agent_counts.get(aid, 0) + 1

    return {
        "simulation_id": simulation_id,
        "status": _simulations[simulation_id].status.value,
        "total_actions": len(history),
        "per_agent_actions": agent_counts,
        **exec_state,
    }


# ---------------------------------------------------------------------------
# WebSocket real-time participation
# ---------------------------------------------------------------------------


def _ws_authenticate(token: str | None) -> str | None:
    """Validate a bearer token and return agent_id, or None."""
    if token is None:
        return None
    raw = token[7:] if token.startswith("Bearer ") else token
    agent_id, key_hash = _lookup_key(raw)
    if agent_id is None:
        return None
    scopes = _key_scopes.get(key_hash, frozenset())
    from swarm.api.middleware.auth import Scope as _Scope

    if _Scope.PARTICIPATE not in scopes:
        return None
    return str(agent_id)


@router.websocket("/{simulation_id}/ws")
async def websocket_participate(websocket: WebSocket, simulation_id: str):
    """WebSocket endpoint for real-time simulation participation.

    Protocol:
        1. Client connects with ``?token=<api_key>`` query param.
        2. Server authenticates and verifies agent is a participant.
        3. Server sends events (observations, step completions) as JSON.
        4. Client sends actions as JSON: ``{"action_type": "...", "payload": {}, "step": N}``.
        5. Connection closes when simulation completes or client disconnects.
    """
    # Authenticate via query param (WebSocket doesn't support headers easily)
    token = websocket.query_params.get("token")
    agent_id = _ws_authenticate(token)
    if agent_id is None:
        await websocket.close(code=4001, reason="Authentication failed")
        return

    # Verify simulation exists
    if simulation_id not in _simulations:
        await websocket.close(code=4004, reason="Simulation not found")
        return

    # Verify agent is a participant
    participants = _participants.get(simulation_id, [])
    agent_ids = {p["agent_id"] for p in participants}
    if agent_id not in agent_ids:
        await websocket.close(code=4003, reason="Not a participant")
        return

    await websocket.accept()

    # Register connection
    if simulation_id not in _ws_connections:
        _ws_connections[simulation_id] = {}
    _ws_connections[simulation_id][agent_id] = websocket

    # Subscribe to events
    event_queue = event_bus.subscribe(simulation_id, agent_id)

    # Send welcome message
    await websocket.send_json({
        "type": "connected",
        "simulation_id": simulation_id,
        "agent_id": agent_id,
    })

    async def _forward_events():
        """Forward event bus events to the WebSocket client."""
        try:
            while True:
                event: SimEvent = await event_queue.get()
                await websocket.send_json({
                    "type": "event",
                    "event_type": event.event_type.value,
                    "simulation_id": event.simulation_id,
                    "data": event.data,
                })
                if event.event_type == SimEventType.SIMULATION_COMPLETE:
                    break
        except WebSocketDisconnect:
            # Client disconnected; stop forwarding events.
            pass
        except RuntimeError:
            # WebSocket closed while sending; stop forwarding events.
            pass

    forward_task = asyncio.create_task(_forward_events())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "detail": "Invalid JSON",
                })
                continue

            msg_type = msg.get("type", "action")

            if msg_type == "action":
                # Submit action to the queue
                simulation = _simulations.get(simulation_id)
                if simulation is None or simulation.status != SimulationStatus.RUNNING:
                    await websocket.send_json({
                        "type": "error",
                        "detail": "Simulation is not running",
                    })
                    continue

                queue = _action_queues.get(simulation_id)
                if queue is None:
                    queue = AsyncActionQueue()
                    _action_queues[simulation_id] = queue

                accepted = await queue.submit_action(
                    agent_id,
                    {
                        "action_type": msg.get("action_type", "noop"),
                        "payload": msg.get("payload", {}),
                        "step": msg.get("step", 0),
                    },
                )

                action_id = str(uuid.uuid4())

                # Record in history
                if simulation_id not in _action_history:
                    _action_history[simulation_id] = []
                _action_history[simulation_id].append({
                    "action_id": action_id,
                    "agent_id": agent_id,
                    "action_type": msg.get("action_type", "noop"),
                    "step": msg.get("step", 0),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "accepted": accepted,
                    "source": "websocket",
                })

                await websocket.send_json({
                    "type": "action_result",
                    "action_id": action_id,
                    "accepted": accepted,
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({
                    "type": "error",
                    "detail": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        pass
    finally:
        forward_task.cancel()
        try:
            await forward_task
        except asyncio.CancelledError:
            # Expected: ignore cancellation of the forwarder task during cleanup.
            pass
        event_bus.unsubscribe(simulation_id, agent_id, event_queue)
        conns = _ws_connections.get(simulation_id, {})
        conns.pop(agent_id, None)
        if not conns:
            _ws_connections.pop(simulation_id, None)
