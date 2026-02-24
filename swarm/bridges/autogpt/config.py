"""Configuration for the AutoGPT bridge."""

from typing import Optional, Set

from pydantic import BaseModel, Field, model_validator

# Commands that are always considered harmful regardless of outcome
_DEFAULT_BLOCKED_COMMANDS: Set[str] = {
    "delete_file",
    "shutdown",
    "execute_shell",
    "execute_python_file",
}


class AutoGPTBridgeConfig(BaseModel):
    """Configuration for mapping AutoGPT action cycles to SWARM interactions.

    Attributes:
        agent_id: Unique identifier for this agent in SWARM.
        goal: High-level goal string for context.
        proxy_sigmoid_k: Steepness of sigmoid for v_hat → p.
        criticisms_weight: How much a non-empty criticism penalises p.
            Larger values penalise self-criticising agents more.
        max_thought_chars: Max thought length treated as full engagement.
        blocked_commands: Commands always mapped to task_progress=0.
        enable_event_log: Whether to write interactions to EventLog.
        event_log_path: Path for the JSONL event log (optional).
    """

    agent_id: str = "autogpt-agent"
    goal: str = "Achieve the assigned objective."
    proxy_sigmoid_k: float = Field(default=2.0, gt=0)
    criticisms_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    max_thought_chars: int = Field(default=1000, ge=1)
    blocked_commands: Set[str] = Field(default_factory=lambda: set(_DEFAULT_BLOCKED_COMMANDS))
    enable_event_log: bool = True
    event_log_path: Optional[str] = None

    model_config = {"frozen": False}

    @model_validator(mode="after")
    def _validate(self) -> "AutoGPTBridgeConfig":
        if not self.agent_id.strip():
            raise ValueError("agent_id cannot be empty")
        return self
