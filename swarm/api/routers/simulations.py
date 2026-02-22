"""Simulation management endpoints."""

import asyncio
import json
import logging
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    SimulationResults,
    SimulationStatus,
)
from swarm.api.persistence import SimulationStore

logger = logging.getLogger(__name__)

router = APIRouter()

# Persistent storage (lazy-init to avoid SQLite lock contention at import time)
_store: SimulationStore | None = None


def _get_store() -> SimulationStore:
    """Lazy-init the simulation store singleton."""
    global _store
    if _store is None:
        _store = SimulationStore()
    return _store

# Runtime-only (ephemeral) state — not persisted
_observations: dict[str, dict[str, dict]] = {}  # simulation_id -> agent_id -> observation
_action_queues: dict[str, AsyncActionQueue] = {}  # simulation_id -> queue
_ws_connections: dict[str, dict[str, WebSocket]] = {}  # simulation_id -> agent_id -> WebSocket
_sim_tasks: dict[str, asyncio.Task] = {}  # simulation_id -> background task

# Concurrency limits
MAX_ACTIVE_SIMULATIONS = 50
MAX_RESULTS_BYTES = 1_048_576  # 1 MiB cap on serialized results payload
MAX_WS_MESSAGE_BYTES = 65_536  # 64 KiB cap on incoming WebSocket messages


# ---------------------------------------------------------------------------
# Orchestrator execution helpers
# ---------------------------------------------------------------------------


def _resolve_sim_scenario(scenario_id: str) -> Path:
    """Resolve a scenario_id to a YAML file path.

    Args:
        scenario_id: The scenario identifier (without extension).

    Returns:
        Path to the scenario YAML file.

    Raises:
        FileNotFoundError: If no matching scenario file exists.
    """
    scenarios_dir = Path(__file__).resolve().parents[3] / "scenarios"
    for ext in (".yaml", ".yml"):
        candidate = scenarios_dir / f"{scenario_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Scenario '{scenario_id}' not found")


async def _execute_simulation(simulation_id: str) -> None:
    """Run the orchestrator as a background task for a simulation.

    Loads the scenario, builds the orchestrator, registers API participants
    as external agents, and runs the simulation to completion.

    If the scenario file cannot be found, the task exits silently so that
    API-only workflows (no YAML backing) continue to work.

    Args:
        simulation_id: The simulation to execute.
    """
    # Resolve scenario early — if missing, bail without touching state.
    simulation = _get_store().get(simulation_id)
    if simulation is None:
        _sim_tasks.pop(simulation_id, None)
        return

    try:
        scenario_path = _resolve_sim_scenario(simulation.scenario_id)
    except FileNotFoundError:
        logger.warning(
            "Simulation %s: scenario '%s' not found, orchestrator not started",
            simulation_id,
            simulation.scenario_id,
        )
        _sim_tasks.pop(simulation_id, None)
        return

    try:
        from swarm.agents.honest import HonestAgent
        from swarm.api.models.simulation import SimulationOverrides
        from swarm.scenarios.loader import (
            apply_config_overrides,
            build_orchestrator,
            load_scenario,
        )

        scenario = load_scenario(scenario_path)

        # Apply validated config overrides if present
        raw_overrides = simulation.config_overrides
        if raw_overrides:
            if isinstance(raw_overrides, dict):
                parsed = SimulationOverrides.model_validate(raw_overrides)
            else:
                parsed = raw_overrides
            apply_config_overrides(scenario, parsed)

        orchestrator = build_orchestrator(scenario)

        # Register API participants as external agents
        participants = _get_store().get_participants(simulation_id)
        for participant in participants:
            agent = HonestAgent(
                agent_id=participant["agent_id"],
                name=participant.get("agent_id"),
            )
            agent.is_external = True
            orchestrator.register_agent(agent)

        # Wire the action queue
        queue = _action_queues.get(simulation_id)
        if queue is None:
            queue = AsyncActionQueue()
            _action_queues[simulation_id] = queue
        orchestrator.set_external_action_queue(queue)

        # Epoch callback: update execution state + publish events
        def _on_epoch_complete(epoch_metrics):
            now = datetime.now(timezone.utc).isoformat()
            exec_state = _get_store().get_execution_state(simulation_id)
            exec_state["current_epoch"] = epoch_metrics.epoch
            exec_state["last_activity"] = now
            _get_store().save_execution_state(simulation_id, exec_state)

            asyncio.ensure_future(event_bus.publish(
                SimEvent(
                    event_type=SimEventType.EPOCH_COMPLETE,
                    simulation_id=simulation_id,
                    data={
                        "epoch": epoch_metrics.epoch,
                        "toxicity_rate": epoch_metrics.toxicity_rate,
                        "avg_payoff": epoch_metrics.avg_payoff,
                    },
                )
            ))

        orchestrator.on_epoch_end(_on_epoch_complete)

        # Wrap _run_step_async to sync observations after each step
        original_run_step = orchestrator._run_step_async

        async def _wrapped_run_step():
            await original_run_step()
            ext_obs = orchestrator.get_external_observations()
            if ext_obs:
                _observations[simulation_id] = dict(ext_obs)

            # Update step counter in execution state
            step = orchestrator.state.current_step
            exec_state = _get_store().get_execution_state(simulation_id)
            exec_state["current_step"] = step
            exec_state["last_activity"] = datetime.now(timezone.utc).isoformat()
            _get_store().save_execution_state(simulation_id, exec_state)

            # Publish step event for SSE/WebSocket clients
            await event_bus.publish(
                SimEvent(
                    event_type=SimEventType.STEP_COMPLETE,
                    simulation_id=simulation_id,
                    data={"step": step, "observations": len(ext_obs) if ext_obs else 0},
                )
            )

        orchestrator._run_step_async = _wrapped_run_step  # type: ignore[method-assign]

        # Run the simulation
        metrics_list = await orchestrator.run_async()

        # Build results
        metrics_dicts = [m.model_dump() for m in metrics_list]
        n_interactions = len(orchestrator.state.completed_interactions)
        n_accepted = sum(
            1 for i in orchestrator.state.completed_interactions if i.accepted
        )

        # Extract summary values from last epoch metrics if available
        last_metrics = metrics_dicts[-1] if metrics_dicts else {}
        sim_results = SimulationResults(
            total_interactions=n_interactions,
            accepted_interactions=n_accepted,
            metrics_history=metrics_dicts,
            avg_toxicity=last_metrics.get("toxicity_rate", 0.0),
            avg_payoff=last_metrics.get("avg_payoff", 0.0),
            quality_gap=last_metrics.get("quality_gap", 0.0),
            n_epochs_completed=len(metrics_dicts),
            n_steps_completed=0,
            n_agents=len(participants),
        )
        results = sim_results.model_dump()

        # Mark completed
        simulation = _get_store().get(simulation_id)
        if simulation is not None:
            simulation.status = SimulationStatus.COMPLETED
            _get_store().save(simulation)
            _get_store().save_results(simulation_id, results)

        await event_bus.publish(
            SimEvent(
                event_type=SimEventType.SIMULATION_COMPLETE,
                simulation_id=simulation_id,
                data={"results": results},
            )
        )

    except asyncio.CancelledError:
        logger.info("Simulation %s cancelled", simulation_id)
        raise
    except Exception:
        logger.exception("Simulation %s failed", simulation_id)
        simulation = _get_store().get(simulation_id)
        if simulation is not None:
            simulation.status = SimulationStatus.CANCELLED
            _get_store().save(simulation)
            _get_store().save_results(simulation_id, {
                "error": traceback.format_exc(),
            })
    finally:
        _sim_tasks.pop(simulation_id, None)
        _observations.pop(simulation_id, None)
        queue = _action_queues.pop(simulation_id, None)
        if queue is not None:
            await queue.cancel_all()
        # Close any lingering WebSocket connections
        ws_conns = _ws_connections.pop(simulation_id, {})
        for ws in ws_conns.values():
            try:
                await ws.close(code=1000, reason="Simulation ended")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


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
    active = _get_store().count_active()
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
        config_overrides=request.config_overrides.model_dump(exclude_none=True),
    )

    _get_store().save(simulation)
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
    simulation = _get_store().get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    # Check join deadline before other checks
    if datetime.now(timezone.utc) > simulation.join_deadline:
        raise HTTPException(status_code=400, detail="Join deadline has passed")

    if simulation.status != SimulationStatus.WAITING:
        raise HTTPException(
            status_code=400, detail="Simulation is not accepting participants"
        )

    if simulation.current_participants >= simulation.max_participants:
        raise HTTPException(status_code=400, detail="Simulation is full")

    # Add participant
    now = datetime.now(timezone.utc)
    _get_store().add_participant(simulation_id, request.agent_id, request.role, now.isoformat())
    simulation.current_participants += 1
    _get_store().save(simulation)

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
    simulation = _get_store().get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    participants = _get_store().get_participants(simulation_id)

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
    simulation = _get_store().get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    if simulation.status != SimulationStatus.WAITING:
        raise HTTPException(
            status_code=400, detail="Simulation is not in waiting state"
        )

    if simulation.current_participants < 2:
        raise HTTPException(
            status_code=400, detail="Not enough participants to start"
        )

    simulation.status = SimulationStatus.RUNNING
    _get_store().save(simulation)

    now = datetime.now(timezone.utc).isoformat()
    _get_store().save_execution_state(simulation_id, {
        "current_step": 0,
        "current_epoch": 0,
        "started_at": now,
        "last_activity": now,
    })

    # Launch orchestrator as background task
    task = asyncio.create_task(_execute_simulation(simulation_id))
    _sim_tasks[simulation_id] = task

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
    simulation: SimulationResponse | None = _get_store().get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return simulation


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
    result: list[SimulationResponse] = list(_get_store().list_simulations(status=status, limit=limit, offset=offset))
    return result


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

    simulation = _get_store().get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    if simulation.status != SimulationStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Simulation is not running")

    # Verify agent is a participant
    participants = _get_store().get_participants(simulation_id)
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

    action_ts = datetime.now(timezone.utc).isoformat()

    # Record action in persistent store
    _get_store().add_action(simulation_id, {
        "action_id": action_id,
        "agent_id": action.agent_id,
        "action_type": action.action_type.value,
        "step": action.step,
        "timestamp": action_ts,
        "accepted": accepted,
    })

    # Update execution state with latest step and activity timestamp
    exec_state = _get_store().get_execution_state(simulation_id)
    exec_state["current_step"] = max(
        exec_state.get("current_step", 0), action.step,
    )
    exec_state["last_activity"] = action_ts
    _get_store().save_execution_state(simulation_id, exec_state)

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

    simulation = _get_store().get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    # Verify agent is a participant
    participants = _get_store().get_participants(simulation_id)
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

    simulation = _get_store().get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    try:
        queue = event_bus.subscribe(simulation_id, agent_id)
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

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
    simulation_id: str, results: SimulationResults | None = None
) -> dict:
    """Mark a simulation as completed and store its results.

    Args:
        simulation_id: The simulation to complete.
        results: Optional validated results to store.

    Returns:
        Updated simulation status.

    Raises:
        HTTPException: If simulation not found or not running.
    """
    simulation = _get_store().get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    if simulation.status != SimulationStatus.RUNNING:
        raise HTTPException(
            status_code=400, detail="Simulation is not running"
        )

    # Serialize and validate payload size before accepting
    results_dict: dict | None = None
    if results is not None:
        results_dict = results.model_dump()
        try:
            payload_size = len(json.dumps(results_dict))
        except (TypeError, ValueError, OverflowError) as exc:
            raise HTTPException(
                status_code=422, detail="Results payload is not JSON-serializable"
            ) from exc
        if payload_size > MAX_RESULTS_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Results payload exceeds {MAX_RESULTS_BYTES} byte limit",
            )

    # Cancel any running background task
    task = _sim_tasks.pop(simulation_id, None)
    if task is not None and not task.done():
        task.cancel()

    simulation.status = SimulationStatus.COMPLETED
    _get_store().save(simulation)

    # Store results
    if results_dict is not None:
        _get_store().save_results(simulation_id, results_dict)

    # Clean up action queue
    queue = _action_queues.pop(simulation_id, None)
    if queue is not None:
        await queue.cancel_all()

    # Publish completion event
    await event_bus.publish(
        SimEvent(
            event_type=SimEventType.SIMULATION_COMPLETE,
            simulation_id=simulation_id,
            data={"results": results_dict or {}},
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
    simulation = _get_store().get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    if simulation.status != SimulationStatus.COMPLETED:
        raise HTTPException(
            status_code=400, detail="Simulation is not completed"
        )

    results = _get_store().get_results(simulation_id) or {}
    history = _get_store().get_action_history(simulation_id)
    participants = _get_store().get_participants(simulation_id)

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
    simulation = _get_store().get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    history = _get_store().get_action_history(simulation_id)
    exec_state = _get_store().get_execution_state(simulation_id)

    # Compute per-agent action counts
    agent_counts: dict[str, int] = {}
    for record in history:
        aid = record["agent_id"]
        agent_counts[aid] = agent_counts.get(aid, 0) + 1

    return {
        "simulation_id": simulation_id,
        "status": simulation.status.value,
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
    simulation = _get_store().get(simulation_id)
    if simulation is None:
        await websocket.close(code=4004, reason="Simulation not found")
        return

    # Verify agent is a participant
    participants = _get_store().get_participants(simulation_id)
    agent_ids = {p["agent_id"] for p in participants}
    if agent_id not in agent_ids:
        await websocket.close(code=4003, reason="Not a participant")
        return

    await websocket.accept()

    # Register connection — close any stale connection for the same agent
    if simulation_id not in _ws_connections:
        _ws_connections[simulation_id] = {}
    old_ws = _ws_connections[simulation_id].get(agent_id)
    if old_ws is not None:
        try:
            await old_ws.close(code=4008, reason="Replaced by new connection")
        except RuntimeError:
            # Old connection already closed; ignore.
            pass
    _ws_connections[simulation_id][agent_id] = websocket

    # Subscribe to events
    try:
        event_queue = event_bus.subscribe(simulation_id, agent_id)
    except ValueError:
        await websocket.close(code=4029, reason="Too many subscriptions")
        _ws_connections[simulation_id].pop(agent_id, None)
        return

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
            if len(raw) > MAX_WS_MESSAGE_BYTES:
                await websocket.send_json({
                    "type": "error",
                    "detail": f"Message exceeds {MAX_WS_MESSAGE_BYTES} byte limit",
                })
                continue
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
                sim = _get_store().get(simulation_id)
                if sim is None or sim.status != SimulationStatus.RUNNING:
                    await websocket.send_json({
                        "type": "error",
                        "detail": "Simulation is not running",
                    })
                    continue

                action_queue = _action_queues.get(simulation_id)
                if action_queue is None:
                    action_queue = AsyncActionQueue()
                    _action_queues[simulation_id] = action_queue

                accepted = await action_queue.submit_action(
                    agent_id,
                    {
                        "action_type": msg.get("action_type", "noop"),
                        "payload": msg.get("payload", {}),
                        "step": msg.get("step", 0),
                    },
                )

                action_id = str(uuid.uuid4())

                ws_action_ts = datetime.now(timezone.utc).isoformat()
                ws_step = msg.get("step", 0)

                # Record in persistent store
                _get_store().add_action(simulation_id, {
                    "action_id": action_id,
                    "agent_id": agent_id,
                    "action_type": msg.get("action_type", "noop"),
                    "step": ws_step,
                    "timestamp": ws_action_ts,
                    "accepted": accepted,
                    "source": "websocket",
                })

                # Update execution state
                ws_exec = _get_store().get_execution_state(simulation_id)
                ws_exec["current_step"] = max(
                    ws_exec.get("current_step", 0), ws_step,
                )
                ws_exec["last_activity"] = ws_action_ts
                _get_store().save_execution_state(simulation_id, ws_exec)

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
