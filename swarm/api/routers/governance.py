"""Governance router stubs — proposal submission and listing."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/propose")
async def propose() -> dict:
    """Submit a governance proposal (stub)."""
    return {"status": "not_implemented"}


@router.get("/proposals")
async def list_proposals() -> dict:
    """List governance proposals (stub)."""
    return {"status": "not_implemented"}
