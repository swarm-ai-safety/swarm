"""Action models for the SWARM API."""

from enum import Enum

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Types of actions an agent can submit."""

    NOOP = "noop"
    ACCEPT = "accept"
    REJECT = "reject"
    PROPOSE = "propose"
    COUNTER = "counter"


class ActionSubmission(BaseModel):
    """An action submitted by an agent for a simulation step."""

    agent_id: str = Field(..., description="ID of the acting agent")
    action_type: ActionType = Field(..., description="Type of action")
    payload: dict = Field(default_factory=dict, description="Action-specific data")
    step: int = Field(..., ge=0, description="Simulation step this action is for")
