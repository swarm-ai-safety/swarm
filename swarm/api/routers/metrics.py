"""Metrics router — per-simulation metrics and analytics retrieval."""

from fastapi import APIRouter, Depends, HTTPException

from swarm.api.middleware.auth import Scope, require_scope

router = APIRouter()


@router.get(
    "/{simulation_id}",
    dependencies=[Depends(require_scope(Scope.READ))],
)
async def get_metrics(simulation_id: str) -> dict:
    """Get metrics for a simulation.

    Retrieves results from the simulation results store. Returns
    the stored metrics if the simulation has completed, or current
    execution state if still running.

    Args:
        simulation_id: The simulation to get metrics for.

    Returns:
        Simulation metrics and analytics.

    Raises:
        HTTPException: If simulation not found.
    """
    from swarm.api.routers.simulations import (
        _action_history,
        _execution_state,
        _participants,
        _simulation_results,
        _simulations,
    )

    if simulation_id not in _simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    simulation = _simulations[simulation_id]
    results = _simulation_results.get(simulation_id, {})
    history = _action_history.get(simulation_id, [])
    participants = _participants.get(simulation_id, [])
    exec_state = _execution_state.get(simulation_id, {})

    # Compute per-agent action counts
    agent_action_counts: dict[str, int] = {}
    for record in history:
        aid = record["agent_id"]
        agent_action_counts[aid] = agent_action_counts.get(aid, 0) + 1

    # Compute action type distribution
    action_type_counts: dict[str, int] = {}
    for record in history:
        at = record["action_type"]
        action_type_counts[at] = action_type_counts.get(at, 0) + 1

    return {
        "simulation_id": simulation_id,
        "status": simulation.status.value,
        "participant_count": len(participants),
        "total_actions": len(history),
        "per_agent_actions": agent_action_counts,
        "action_type_distribution": action_type_counts,
        "results": results,
        "execution_state": exec_state,
    }
