"""Simulation-related API models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SimulationStatus(str, Enum):
    """Simulation status."""

    WAITING = "waiting_for_participants"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class SimulationMode(str, Enum):
    """Simulation execution mode."""

    REALTIME = "realtime"
    ASYNC = "async"


class SimulationCreate(BaseModel):
    """Request model for creating a simulation."""

    scenario_id: str = Field(..., description="ID of the scenario to run")
    config_overrides: dict = Field(
        default_factory=dict, description="Optional configuration overrides"
    )
    max_participants: int = Field(
        default=10, description="Maximum number of participants", ge=2, le=100
    )
    mode: SimulationMode = Field(
        default=SimulationMode.ASYNC, description="Execution mode"
    )


class SimulationJoin(BaseModel):
    """Request model for joining a simulation."""

    agent_id: str = Field(..., description="ID of the joining agent")
    role: str = Field(
        default="participant",
        description="Role in simulation (initiator, counterparty, observer)",
    )


class SimulationResults(BaseModel):
    """Validated results from a completed simulation."""

    total_interactions: int = Field(0, ge=0)
    accepted_interactions: int = Field(0, ge=0)
    avg_toxicity: float = Field(0.0, ge=0.0, le=1.0)
    final_welfare: float = 0.0
    avg_payoff: float = 0.0
    quality_gap: float = 0.0
    n_agents: int = Field(0, ge=0)
    n_epochs_completed: int = Field(0, ge=0)
    metrics_history: list[dict] = Field(default_factory=list)
    extra: dict = Field(
        default_factory=dict, description="Additional unstructured data"
    )


class SimulationResponse(BaseModel):
    """Response model for simulation."""

    simulation_id: str = Field(..., description="Unique simulation identifier")
    scenario_id: str = Field(..., description="Associated scenario ID")
    status: SimulationStatus = Field(..., description="Current status")
    mode: SimulationMode = Field(..., description="Execution mode")
    max_participants: int = Field(..., description="Maximum participants")
    current_participants: int = Field(..., description="Current participant count")
    created_at: datetime = Field(..., description="Creation timestamp")
    join_deadline: datetime = Field(..., description="Deadline to join")
    config_overrides: dict = Field(
        default_factory=dict, description="Configuration overrides"
    )
