"""Governance router — proposal submission, listing, and voting."""

import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from swarm.api.middleware.auth import (
    Scope,
    get_request_key_hash,
    get_scopes_hash,
    require_scope,
)
from swarm.api.persistence import ProposalStore

router = APIRouter()


class ProposalStatus(str, Enum):
    """Status of a governance proposal."""

    DRAFT = "draft"
    OPEN = "open"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ProposalCreate(BaseModel):
    """Request model for creating a governance proposal."""

    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=5000)
    policy_declaration: dict = Field(
        default_factory=dict,
        description="Freeform JSON describing the policy to be tested",
    )
    target_scenarios: list[str] = Field(
        default_factory=list,
        description="Scenario IDs this proposal should be tested against",
    )


class ProposalResponse(BaseModel):
    """Response model for a governance proposal."""

    proposal_id: str
    title: str
    description: str
    policy_declaration: dict
    target_scenarios: list[str]
    status: ProposalStatus
    proposer_id: str
    created_at: datetime
    votes_for: int = 0
    votes_against: int = 0


_store: Optional[ProposalStore] = None
_store_lock = threading.Lock()


def _get_store() -> ProposalStore:
    """Return the proposal store singleton, creating it thread-safely on first use."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ProposalStore()
    return _store


@router.post(
    "/propose",
    response_model=ProposalResponse,
    dependencies=[Depends(require_scope(Scope.WRITE))],
)
async def create_proposal(
    request: ProposalCreate,
    agent_id: str = Depends(require_scope(Scope.WRITE)),
) -> ProposalResponse:
    """Submit a new governance proposal.

    Args:
        request: Proposal details.
        agent_id: Authenticated agent's ID (from auth).

    Returns:
        Created proposal.
    """
    proposal_id = str(uuid.uuid4())
    proposal = ProposalResponse(
        proposal_id=proposal_id,
        title=request.title,
        description=request.description,
        policy_declaration=request.policy_declaration,
        target_scenarios=request.target_scenarios,
        status=ProposalStatus.OPEN,
        proposer_id=agent_id,
        created_at=datetime.now(timezone.utc),
    )
    _get_store().save(proposal)
    return proposal


@router.get("/proposals", response_model=list[ProposalResponse])
async def list_proposals(
    status: ProposalStatus | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[ProposalResponse]:
    """List governance proposals with optional filtering.

    Args:
        status: Filter by proposal status.
        limit: Maximum results.
        offset: Skip count.

    Returns:
        List of proposals.
    """
    status_val = status.value if status is not None else None
    rows = _get_store().list_proposals(status=status_val, limit=limit, offset=offset)
    return [ProposalResponse(**r) for r in rows]


@router.get("/proposals/{proposal_id}", response_model=ProposalResponse)
async def get_proposal(proposal_id: str) -> ProposalResponse:
    """Get a proposal by ID.

    Args:
        proposal_id: The proposal's unique identifier.

    Returns:
        Proposal details.

    Raises:
        HTTPException: If proposal not found.
    """
    row = _get_store().get(proposal_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return ProposalResponse(**row)


@router.post("/proposals/{proposal_id}/vote")
async def vote_on_proposal(
    proposal_id: str,
    direction: int = Query(..., ge=-1, le=1),
    agent_id: str = Depends(require_scope(Scope.PARTICIPATE)),
) -> dict:
    """Vote on a governance proposal.

    Each agent may vote once per proposal.  Subsequent votes from the
    same agent are rejected with 409 Conflict.

    Args:
        proposal_id: The proposal to vote on.
        direction: +1 for, -1 against.
        agent_id: Authenticated agent identity (from API key).

    Returns:
        Updated vote counts.

    Raises:
        HTTPException: If proposal not found, not open, or already voted.
    """
    row = _get_store().get(proposal_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Proposal not found")

    if row["status"] != ProposalStatus.OPEN.value:
        raise HTTPException(
            status_code=400, detail="Proposal is not open for voting"
        )

    result = _get_store().vote(proposal_id, agent_id, direction)
    if result is None:
        raise HTTPException(
            status_code=409, detail="Agent has already voted on this proposal"
        )

    return result  # type: ignore[no-any-return]


@router.post("/proposals/{proposal_id}/finalize", response_model=ProposalResponse)
async def finalize_proposal(
    proposal_id: str,
    quorum: int = Query(1, ge=1),
    threshold: float = Query(0.5, gt=0.0, le=1.0),
    agent_id: str = Depends(require_scope(Scope.WRITE)),
) -> ProposalResponse:
    """Finalize voting on a proposal — transitions to ACCEPTED or REJECTED.

    Args:
        proposal_id: The proposal to finalize.
        quorum: Minimum total votes required.
        threshold: Fraction of for-votes needed to accept (> threshold).
        agent_id: Authenticated agent identity.
    """
    store = _get_store()
    row = store.get(proposal_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Proposal not found")

    if row["status"] != ProposalStatus.OPEN.value:
        raise HTTPException(status_code=400, detail="Proposal is not open")

    total = row["votes_for"] + row["votes_against"]
    if total < quorum:
        raise HTTPException(status_code=400, detail="Quorum not reached")

    new_status = (
        ProposalStatus.ACCEPTED if row["votes_for"] / total > threshold else ProposalStatus.REJECTED
    )
    store.close_proposal(proposal_id, new_status.value)

    updated = store.get(proposal_id)
    return ProposalResponse(**updated)  # type: ignore[arg-type]


@router.post("/proposals/{proposal_id}/close", response_model=ProposalResponse)
async def close_proposal(
    proposal_id: str,
    request: Request,
    agent_id: str = Depends(require_scope(Scope.WRITE)),
) -> ProposalResponse:
    """Close (withdraw) a proposal. Only the original proposer or ADMIN can close.

    Args:
        proposal_id: The proposal to close.
        request: FastAPI request (for scope lookup).
        agent_id: Authenticated agent identity.
    """
    store = _get_store()
    row = store.get(proposal_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Proposal not found")

    if row["status"] != ProposalStatus.OPEN.value:
        raise HTTPException(status_code=400, detail="Proposal is not open")

    key_hash = get_request_key_hash(request)
    scopes = get_scopes_hash(key_hash)
    is_proposer = row["proposer_id"] == agent_id
    is_admin = Scope.ADMIN in scopes

    if not is_proposer and not is_admin:
        raise HTTPException(
            status_code=403, detail="Only the proposer or an admin can close a proposal"
        )

    store.close_proposal(proposal_id, ProposalStatus.REJECTED.value)

    updated = store.get(proposal_id)
    return ProposalResponse(**updated)  # type: ignore[arg-type]


@router.get("/proposals/{proposal_id}/votes")
async def list_proposal_votes(proposal_id: str) -> list[dict]:
    """List all votes on a proposal.

    Args:
        proposal_id: The proposal to list votes for.

    Returns:
        List of vote records.
    """
    store = _get_store()
    row = store.get(proposal_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Proposal not found")
    return list(store.list_votes(proposal_id))
