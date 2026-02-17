"""Configuration for the AWM bridge."""

import ipaddress
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr, field_validator


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

    # Phase 3: Multi-turn mode (False = batch for backward compat)
    step_mode: bool = False

    # Phase 4: Shared-state multi-agent coordination
    shared_database: bool = False
    isolation_level: str = "read_committed"  # "none" | "read_committed" | "serializable"
    conflict_probability: float = Field(default=0.3, ge=0.0, le=1.0)

    # Phase 3: LLM-based tool planning
    llm_planning: bool = False
    llm_provider: Optional[str] = None  # "anthropic", "openai", etc.
    llm_model: Optional[str] = None
    llm_api_key: Optional[SecretStr] = None
    llm_base_url: Optional[str] = None
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    llm_timeout: float = 30.0
    llm_max_retries: int = 2
    llm_system_prompt: Optional[str] = None
    llm_cost_tracking: bool = True
    llm_prompt_audit_path: Optional[str] = None
    llm_fallback_to_scripted: bool = True
    llm_max_calls_per_plan: int = 10

    @field_validator("llm_base_url")
    @classmethod
    def _validate_llm_base_url(cls, v: Optional[str]) -> Optional[str]:
        """Reject llm_base_url pointing at private/link-local networks."""
        if v is None:
            return v
        parsed = urlparse(v)
        hostname = parsed.hostname or ""
        try:
            addr = ipaddress.ip_address(hostname)
            if addr.is_private or addr.is_loopback or addr.is_link_local:
                raise ValueError(
                    f"llm_base_url must not target private/loopback "
                    f"addresses, got {hostname}"
                )
        except ValueError as exc:
            if "must not target" in str(exc):
                raise
            # hostname is a DNS name, not a literal IP — allow it
        return v
