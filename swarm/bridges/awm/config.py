"""Configuration for the AWM bridge."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class AWMConfig(BaseModel):
    """Configuration for the AWM (Agent World Model) bridge.

    Parsed from the ``awm:`` section of a scenario YAML file.
    """

    enabled: bool = True

    # Path to downloaded AWM environments (from download_awm_envs.sh)
    envs_path: Path = Path("external/awm-envs")

    # Which environment to use (e.g. "ecommerce_001", "project_mgmt_042")
    environment_id: str = "ecommerce_001"

    # Server configuration
    base_port: int = 9100  # Each agent gets base_port + agent_index
    host: str = "127.0.0.1"

    # Agent interaction limits
    max_steps_per_task: int = 20
    max_tasks_per_epoch: int = 5

    # Verification
    verification_timeout: float = 10.0  # seconds

    # Observable mapping weights
    error_weight: float = 1.0  # Weight for error-based rework_count
    misuse_weight: float = 1.0  # Weight for malformed tool calls

    # Confidence for binary_to_soft_p
    verification_confidence: float = 0.8

    # Optional seed for reproducibility
    seed: Optional[int] = None

    # Server startup timeout
    server_startup_timeout: float = 30.0

    # Whether to reset DB between epochs
    reset_between_epochs: bool = True

    # Number of concurrent servers (one per agent)
    max_concurrent_servers: int = Field(default=10, ge=1)

    # Phase 2: Live mode (False = simulation, True = real HTTP + subprocess)
    live_mode: bool = False
    server_command_template: str = (
        "{python} -m awm.server --host {host} --port {port} --env-path {env_path}"
    )
    health_check_interval: float = 0.5
