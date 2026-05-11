"""Configuration for the SWARM-MiroShark bridge."""

from dataclasses import dataclass


@dataclass
class MirosharkConfig:
    api_url: str = "http://localhost:5001"
    scale: int = 20
    platform: str = "parallel"
    max_rounds: int = 30
    poll_interval_s: float = 5.0
    poll_timeout_s: float = 1800.0
    request_timeout_s: float = 60.0
    admin_token: str | None = None
