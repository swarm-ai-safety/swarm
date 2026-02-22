"""Simulation-related API models."""

from __future__ import annotations

from dataclasses import fields as dc_fields
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class SimulationOverrides(BaseModel):
    """Optional per-section config overrides matching scenario YAML structure.

    Top-level fields (n_epochs, steps_per_epoch, seed) map to OrchestratorConfig.
    Section dicts (payoff, governance, rate_limits) are validated against their
    respective config classes.  ``extra="forbid"`` catches typos at creation time.
    """

    n_epochs: int | None = Field(None, ge=1)
    steps_per_epoch: int | None = Field(None, ge=1)
    seed: int | None = None

    payoff: dict[str, Any] | None = None
    governance: dict[str, Any] | None = None
    rate_limits: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_sections(self) -> SimulationOverrides:
        from swarm.core.payoff import PayoffConfig
        from swarm.env.state import RateLimits
        from swarm.governance.config import GovernanceConfig

        if self.payoff is not None:
            valid_keys = set(PayoffConfig.model_fields)
            unknown = set(self.payoff) - valid_keys
            if unknown:
                raise ValueError(
                    f"Unknown payoff fields: {', '.join(sorted(unknown))}"
                )
            PayoffConfig(**self.payoff)
        if self.governance is not None:
            valid_keys = set(GovernanceConfig.model_fields)
            unknown = set(self.governance) - valid_keys
            if unknown:
                raise ValueError(
                    f"Unknown governance fields: {', '.join(sorted(unknown))}"
                )
            GovernanceConfig(**self.governance)
        if self.rate_limits is not None:
            # RateLimits is a dataclass — reject unknown keys manually
            valid_keys = {f.name for f in dc_fields(RateLimits)}
            unknown = set(self.rate_limits) - valid_keys
            if unknown:
                raise ValueError(
                    f"Unknown rate_limits fields: {', '.join(sorted(unknown))}"
                )
            RateLimits(**self.rate_limits)
        return self


class SimulationCreate(BaseModel):
    """Request model for creating a simulation."""

    scenario_id: str = Field(..., description="ID of the scenario to run")
    config_overrides: SimulationOverrides = Field(
        default_factory=SimulationOverrides,
        description="Optional configuration overrides (validated against scenario schema)",
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
