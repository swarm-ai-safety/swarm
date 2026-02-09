"""SWARM API configuration."""

from pydantic import BaseModel, Field


class APIConfig(BaseModel):
    """Configuration for the SWARM API."""

    host: str = Field(
        default="127.0.0.1",
        description="API host address (localhost only by default for security)",
    )
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Enable debug mode")
    cors_origins: list[str] = Field(
        default=["http://localhost:8000", "http://127.0.0.1:8000"],
        description="Allowed CORS origins (restrict in production)",
    )
    rate_limit_per_minute: int = Field(
        default=100, description="Rate limit per API key per minute"
    )
    auto_approve_agents: bool = Field(
        default=True,
        description="Automatically approve agent registrations (dev mode)",
    )
