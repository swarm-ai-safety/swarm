"""Scenario-related API models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ScenarioStatus(str, Enum):
    """Scenario validation status."""

    VALIDATING = "validating"
    VALID = "valid"
    INVALID = "invalid"


class ScenarioSubmission(BaseModel):
    """Request model for scenario submission."""

    name: str = Field(..., description="Scenario name", min_length=1, max_length=100)
    description: str = Field(
        ..., description="Description of the scenario", max_length=2000
    )
    yaml_content: str = Field(
        ..., description="YAML content defining the scenario configuration"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing the scenario",
    )


class ScenarioResponse(BaseModel):
    """Response model for submitted scenario."""

    scenario_id: str = Field(..., description="Unique scenario identifier")
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    status: ScenarioStatus = Field(..., description="Validation status")
    validation_errors: list[str] = Field(
        default_factory=list, description="List of validation errors if invalid"
    )
    submitted_at: datetime = Field(..., description="Submission timestamp")
    tags: list[str] = Field(default_factory=list, description="Scenario tags")
