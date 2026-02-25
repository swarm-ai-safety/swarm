"""Configuration for the Letta bridge."""

from typing import Optional

from pydantic import BaseModel, Field


class LettaConfig(BaseModel):
    """Configuration for the Letta (MemGPT) bridge.

    Parsed from the ``letta:`` section of a scenario YAML file.
    """

    enabled: bool = True

    # Server connection
    base_url: str = "http://localhost:8283"
    api_key: Optional[str] = None
    server_mode: str = "external"  # "external" | "docker" | "local"

    # Agent defaults
    default_model: str = "anthropic/claude-sonnet-4-20250514"
    default_persona: str = "You are a helpful agent in a multi-agent ecosystem."

    # Memory block limits
    core_memory_limit: int = Field(default=5000, ge=100)
    archival_memory_limit: int = Field(default=100000, ge=1000)

    # Shared governance block
    shared_governance_block: bool = True

    # Cost tracking
    cost_tracking: bool = True

    # Sleep-time consolidation
    sleep_time_enabled: bool = False
    sleep_time_interval_epochs: int = Field(default=5, ge=1)

    # Server lifecycle
    docker_image: str = "letta/letta:latest"
    docker_port: int = 8283
    startup_timeout: float = 60.0
    health_check_interval: float = 1.0

    # Timeout for send_message calls
    message_timeout: float = 30.0

    # Ollama local stack settings
    ollama_base_url: str = "http://localhost:11434"

    # Streaming token control.
    # None = auto (streaming enabled for cloud models, disabled for Ollama).
    # True = always stream.  False = never stream.
    stream_tokens: Optional[bool] = None
