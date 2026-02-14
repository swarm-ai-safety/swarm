"""Run-related API models."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class RunStatus(str, Enum):
    """Run lifecycle status."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunVisibility(str, Enum):
    """Who can see run results."""

    PRIVATE = "private"
    PUBLIC = "public"


class RunCreate(BaseModel):
    """Request to kick off a SWARM run."""

    scenario_id: str = Field(
        ...,
        description="ID referencing a pre-approved scenario YAML (e.g. 'baseline')",
        min_length=1,
        max_length=100,
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Override params: seed, epochs, steps_per_epoch",
    )
    visibility: RunVisibility = Field(
        default=RunVisibility.PRIVATE,
        description="Result visibility (public requires trusted key)",
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Webhook URL to POST results when run completes (Pattern C). Must use HTTPS.",
        max_length=2048,
    )

    @field_validator("params")
    @classmethod
    def validate_params(cls, v: dict) -> dict:
        """Only allow known safe parameter keys, reject oversized payloads."""
        allowed = {"seed", "epochs", "steps_per_epoch"}
        unknown = set(v.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Unknown params: {unknown}. Allowed: {sorted(allowed)}"
            )
        # Enforce value ranges
        if "seed" in v and not isinstance(v["seed"], (int, float)):
            raise ValueError("seed must be numeric")
        if "epochs" in v:
            ep = int(v["epochs"])
            if not 1 <= ep <= 1000:
                raise ValueError("epochs must be between 1 and 1000")
        if "steps_per_epoch" in v:
            sp = int(v["steps_per_epoch"])
            if not 1 <= sp <= 1000:
                raise ValueError("steps_per_epoch must be between 1 and 1000")
        return v


class RunSummaryMetrics(BaseModel):
    """Key metrics from a completed run."""

    total_interactions: int = 0
    accepted_interactions: int = 0
    avg_toxicity: float = 0.0
    final_welfare: float = 0.0
    avg_payoff: float = 0.0
    quality_gap: float = 0.0
    n_agents: int = 0
    n_epochs_completed: int = 0


class RunResponse(BaseModel):
    """Response for a run (both creation and status queries)."""

    run_id: str = Field(..., description="Unique run identifier")
    scenario_id: str = Field(..., description="Scenario used")
    status: RunStatus = Field(..., description="Current run status")
    visibility: RunVisibility = Field(..., description="Result visibility")
    agent_id: str = Field(..., description="Agent that triggered the run")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="When run began executing")
    completed_at: Optional[datetime] = Field(None, description="When run finished")
    progress: Optional[float] = Field(
        None, description="Progress 0.0-1.0 (epochs completed / total)"
    )
    summary_metrics: Optional[RunSummaryMetrics] = Field(
        None, description="Summary metrics (available after completion)"
    )
    status_url: str = Field(..., description="URL to poll for status")
    public_url: Optional[str] = Field(
        None, description="Public results URL (if visibility=public)"
    )
    error: Optional[str] = Field(None, description="Error message if failed")


class RunKickoffResponse(BaseModel):
    """Minimal response returned immediately on run creation."""

    run_id: str
    status: RunStatus
    status_url: str
    public_url: Optional[str] = None
