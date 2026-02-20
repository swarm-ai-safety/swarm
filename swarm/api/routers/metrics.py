"""Metrics router stubs — per-simulation metrics retrieval."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/{simulation_id}")
async def get_metrics(simulation_id: str) -> dict:
    """Get metrics for a simulation (stub)."""
    return {"status": "not_implemented"}
