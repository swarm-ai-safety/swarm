"""SWARM Web API - FastAPI application factory."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from swarm.api.config import APIConfig
from swarm.api.routers import agents, health, scenarios, simulations


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    yield
    # Shutdown


def create_app(config: APIConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional API configuration. Uses defaults if not provided.

    Returns:
        Configured FastAPI application instance.
    """
    if config is None:
        config = APIConfig()

    app = FastAPI(
        title="SWARM API",
        description=(
            "Web API for the SWARM (System-Wide Assessment of Risk in Multi-agent "
            "systems) framework. Enables external agents to participate in "
            "simulations, submit scenarios, and contribute to governance experiments."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
    app.include_router(scenarios.router, prefix="/api/v1/scenarios", tags=["scenarios"])
    app.include_router(
        simulations.router, prefix="/api/v1/simulations", tags=["simulations"]
    )

    return app


# Default application instance for uvicorn
app = create_app()
