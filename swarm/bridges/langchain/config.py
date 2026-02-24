"""Configuration for the LangChain bridge."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class LangChainBridgeConfig(BaseModel):
    """Configuration for wrapping a LangChain chain as a SWARM agent.

    Attributes:
        agent_id: Unique identifier for this agent in SWARM.
        max_steps: Maximum intermediate steps before marking as failed.
        timeout_seconds: Seconds before chain execution times out.
        proxy_sigmoid_k: Steepness of sigmoid for v_hat → p conversion.
        engagement_max_chars: Character count treated as maximum engagement.
        reputation_weight: Starting reputation (0..1) for the agent.
        enable_event_log: Whether to log interactions to an EventLog.
        event_log_path: Path for the event log JSONL file (optional).
    """

    agent_id: str = "langchain-agent"
    max_steps: int = Field(default=10, ge=1, le=100)
    timeout_seconds: float = Field(default=60.0, gt=0)
    proxy_sigmoid_k: float = Field(default=2.0, gt=0)
    engagement_max_chars: int = Field(default=2000, ge=1)
    reputation_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    enable_event_log: bool = True
    event_log_path: Optional[str] = None

    model_config = {"frozen": False}
