"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Status indicating the API is healthy.
    """
    return {"status": "healthy"}


@router.get("/")
async def root() -> dict[str, str]:
    """Root endpoint.

    Returns:
        Welcome message and API info.
    """
    return {
        "name": "SWARM API",
        "version": "1.0.0",
        "docs": "/docs",
        "description": "System-Wide Assessment of Risk in Multi-agent systems",
    }
