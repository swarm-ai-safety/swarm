"""Scenario submission and management endpoints."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from swarm.api.models.scenario import (
    ScenarioResponse,
    ScenarioStatus,
    ScenarioSubmission,
)
from swarm.api.persistence import ScenarioStore
from swarm.api.validation import estimate_resources, validate_scenario_yaml

router = APIRouter()

_store = ScenarioStore()


@router.post("/submit", response_model=ScenarioResponse)
async def submit_scenario(submission: ScenarioSubmission) -> ScenarioResponse:
    """Submit a new scenario for validation and storage.

    Args:
        submission: Scenario submission details.

    Returns:
        Submitted scenario with validation status.
    """
    scenario_id = str(uuid.uuid4())

    # Validate YAML content (schema + type checks)
    validation_errors, parsed_config = validate_scenario_yaml(
        submission.yaml_content
    )

    status = ScenarioStatus.INVALID if validation_errors else ScenarioStatus.VALID

    # Compute resource estimate when we have a valid parsed config
    resource_estimate = None
    if parsed_config is not None and not validation_errors:
        resource_estimate = estimate_resources(parsed_config)

    scenario = ScenarioResponse(
        scenario_id=scenario_id,
        name=submission.name,
        description=submission.description,
        status=status,
        validation_errors=validation_errors,
        submitted_at=datetime.now(timezone.utc),
        tags=submission.tags,
        resource_estimate=resource_estimate,
    )

    _store.save(scenario)
    return scenario


@router.get("/{scenario_id}", response_model=ScenarioResponse)
async def get_scenario(scenario_id: str) -> ScenarioResponse:
    """Get scenario details by ID.

    Args:
        scenario_id: The scenario's unique identifier.

    Returns:
        Scenario details.

    Raises:
        HTTPException: If scenario not found.
    """
    scenario = _store.get(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario


@router.get("/", response_model=list[ScenarioResponse])
async def list_scenarios(
    status: ScenarioStatus | None = None,
    tag: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[ScenarioResponse]:
    """List all submitted scenarios with optional filtering and pagination.

    Args:
        status: Filter by scenario status (VALID, INVALID, VALIDATING).
        tag: Filter scenarios that have this tag.
        limit: Maximum number of results to return.
        offset: Number of results to skip.

    Returns:
        List of scenarios.
    """
    return _store.list_scenarios(status=status, tag=tag, limit=limit, offset=offset)
