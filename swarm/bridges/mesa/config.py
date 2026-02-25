"""Configuration for the Mesa ABM bridge."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class MesaBridgeConfig(BaseModel):
    """Configuration for the Mesa ABM bridge.

    Attributes:
        model_id: Identifier for this Mesa model in SWARM.
        proxy_sigmoid_k: Steepness of sigmoid for v_hat → p.
        max_agents_per_step: Maximum agents sampled per step for interaction
            recording.  Reduces overhead for large models.
        agent_task_progress_attr: Mesa agent attribute name for task_progress.
        agent_rework_count_attr: Mesa agent attribute name for rework_count.
        agent_engagement_attr: Mesa agent attribute name for engagement.
        enable_event_log: Whether to write interactions to EventLog.
        event_log_path: Path for the JSONL event log (optional).
    """

    model_id: str = "mesa-model"
    proxy_sigmoid_k: float = Field(default=2.0, gt=0)
    max_agents_per_step: int = Field(default=100, ge=1)

    # Mesa agent attribute name overrides
    agent_task_progress_attr: str = "task_progress"
    agent_rework_count_attr: str = "rework_count"
    agent_engagement_attr: str = "engagement"

    enable_event_log: bool = True
    event_log_path: Optional[str] = None

    model_config = {"frozen": False}

    @model_validator(mode="after")
    def _validate(self) -> "MesaBridgeConfig":
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        return self
