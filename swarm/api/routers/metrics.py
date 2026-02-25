"""Metrics router — per-simulation metrics and analytics retrieval."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from swarm.api.middleware.auth import Scope, require_scope

router = APIRouter()
logger = logging.getLogger(__name__)


def _build_soft_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Compute soft/hard metrics if interaction data is available in results.

    Looks for an ``"interactions"`` key containing a list of dicts that can
    be reconstructed into :class:`SoftInteraction` objects.  When present,
    runs :class:`MetricsReporter` and returns the structured summary dict.

    Returns an empty dict when interaction data is missing or unusable.
    """
    interactions_data = results.get("interactions")
    # Also check inside "extra" for SimulationResults-shaped payloads
    if not interactions_data and isinstance(results.get("extra"), dict):
        interactions_data = results["extra"].get("interactions")
    if not interactions_data or not isinstance(interactions_data, list):
        return {}

    # Lazy imports to keep startup fast and avoid circular deps
    from pydantic import ValidationError

    from swarm.metrics.reporters import MetricsReporter
    from swarm.models.interaction import SoftInteraction

    interactions: list[SoftInteraction] = []
    skipped_reasons: list[str] = []
    for entry in interactions_data:
        if not isinstance(entry, dict):
            skipped_reasons.append(f"not a dict (got {type(entry).__name__})")
            continue
        try:
            interactions.append(SoftInteraction(**entry))
        except (TypeError, ValidationError) as exc:
            # Use type name + truncated message to avoid leaking field values.
            truncated = str(exc)[:120]
            skipped_reasons.append(f"{type(exc).__name__}: {truncated}")

    if skipped_reasons:
        # Aggregate reasons by count and log only the top 5 distinct reasons to
        # bound log size and avoid leaking sensitive data from many entries.
        from collections import Counter

        reason_counts = Counter(skipped_reasons)
        top_reasons = reason_counts.most_common(5)
        logger.warning(
            "Skipped %d malformed interaction entries out of %d total. "
            "Top reasons (reason: count): %s",
            len(skipped_reasons),
            len(interactions_data),
            top_reasons,
        )

    if not interactions:
        return {}

    reporter = MetricsReporter()
    summary = reporter.summary(interactions)
    result: dict[str, Any] = summary.to_dict()
    return result


@router.get(
    "/{simulation_id}",
    dependencies=[Depends(require_scope(Scope.READ))],
)
async def get_metrics(simulation_id: str) -> dict:
    """Get metrics for a simulation.

    Retrieves results from the simulation results store. Returns
    the stored metrics if the simulation has completed, or current
    execution state if still running.

    When the stored results contain serialized interaction data (a list
    of dicts under the ``"interactions"`` key), full soft/hard metrics
    are computed via :class:`MetricsReporter` and included in the
    response.

    Args:
        simulation_id: The simulation to get metrics for.

    Returns:
        Simulation metrics and analytics.

    Raises:
        HTTPException: If simulation not found.
    """
    from swarm.api.routers.simulations import _get_store

    store = _get_store()
    simulation = store.get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    results = store.get_results(simulation_id) or {}
    history = store.get_action_history(simulation_id)
    participants = store.get_participants(simulation_id)
    exec_state = store.get_execution_state(simulation_id)

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

    response: dict[str, Any] = {
        "simulation_id": simulation_id,
        "status": simulation.status.value,
        "participant_count": len(participants),
        "total_actions": len(history),
        "per_agent_actions": agent_action_counts,
        "action_type_distribution": action_type_counts,
        "results": results,
        "execution_state": exec_state,
    }

    # Enrich with SoftMetrics / MetricsReporter output when possible
    computed = _build_soft_metrics(results)
    if computed:
        response.update(computed)

    return response
