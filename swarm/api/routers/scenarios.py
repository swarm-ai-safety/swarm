"""Scenario submission and management endpoints."""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from swarm.api.models.scenario import (
    ScenarioResponse,
    ScenarioStatus,
    ScenarioSubmission,
)

router = APIRouter()

# In-memory storage for development
_scenarios: dict[str, ScenarioResponse] = {}


@router.post("/submit", response_model=ScenarioResponse)
async def submit_scenario(submission: ScenarioSubmission) -> ScenarioResponse:
    """Submit a new scenario for validation and storage.

    Args:
        submission: Scenario submission details.

    Returns:
        Submitted scenario with validation status.
    """
    scenario_id = str(uuid.uuid4())

    # Basic validation (expand in production)
    validation_errors: list[str] = []
    if not submission.yaml_content.strip():
        validation_errors.append("YAML content cannot be empty")

    status = ScenarioStatus.INVALID if validation_errors else ScenarioStatus.VALID

    scenario = ScenarioResponse(
        scenario_id=scenario_id,
        name=submission.name,
        description=submission.description,
        status=status,
        validation_errors=validation_errors,
        submitted_at=datetime.now(UTC),
        tags=submission.tags,
    )

    _scenarios[scenario_id] = scenario
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
    if scenario_id not in _scenarios:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return _scenarios[scenario_id]


@router.get("/", response_model=list[ScenarioResponse])
async def list_scenarios() -> list[ScenarioResponse]:
    """List all submitted scenarios.

    Returns:
        List of scenarios.
    """
    return list(_scenarios.values())
