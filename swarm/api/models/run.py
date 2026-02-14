"""Run-related API models."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


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
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Override params: seed, epochs, steps_per_epoch, payoff overrides",
    )
    visibility: RunVisibility = Field(
        default=RunVisibility.PRIVATE,
        description="Result visibility (public requires trusted key)",
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Webhook URL to POST results when run completes (Pattern C)",
    )


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
