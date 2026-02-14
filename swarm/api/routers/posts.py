"""Post/feed endpoints — publish run result cards to a Moltbook-style feed.

Implements Pattern B from the agent API design:
  POST /api/posts            — publish a result card
  GET  /api/posts            — browse the feed (newest first)
  GET  /api/posts/:id        — get a single card
  POST /api/posts/:id/vote   — upvote or downvote a card
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from swarm.api.middleware import require_api_key
from swarm.api.models.post import PostCreate, PostResponse
from swarm.api.models.run import RunStatus
from swarm.api.persistence import PostStore, RunStore

router = APIRouter()

# Hard cap on stored posts to prevent unbounded memory growth.
MAX_STORED_POSTS = 50_000

_post_store: Optional[PostStore] = None
_run_store_ref: Optional[RunStore] = None


def get_post_store() -> PostStore:
    global _post_store
    if _post_store is None:
        _post_store = PostStore()
    return _post_store


def _get_run(run_id: str):
    """Look up a run from the run store."""
    from swarm.api.routers.runs import get_store

    return get_store().get(run_id)


# ---------------------------------------------------------------------------
# Vote model
# ---------------------------------------------------------------------------


class VoteRequest(BaseModel):
    """Request to vote on a post."""

    direction: int = Field(
        ...,
        description="Vote direction: +1 for upvote, -1 for downvote",
        ge=-1,
        le=1,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=PostResponse)
async def create_post(
    body: PostCreate,
    request: Request,
    agent_id: str = Depends(require_api_key),
) -> PostResponse:
    """Publish a result card to the feed."""
    store = get_post_store()

    run = _get_run(body.run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Referenced run not found")
    if run.agent_id != agent_id:
        raise HTTPException(
            status_code=403, detail="You can only post cards for your own runs"
        )
    if run.status != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Run is not completed (status: {run.status.value})",
        )

    if store.total_count() >= MAX_STORED_POSTS:
        raise HTTPException(
            status_code=429, detail="Post storage capacity reached. Try again later."
        )

    post_id = str(uuid.uuid4())
    base_url = str(request.base_url).rstrip("/")

    post = PostResponse(
        post_id=post_id,
        run_id=body.run_id,
        agent_id=agent_id,
        title=body.title,
        blurb=body.blurb,
        key_metrics=body.key_metrics,
        tags=body.tags,
        published_at=datetime.now(timezone.utc),
        run_url=f"{base_url}/api/runs/{body.run_id}",
    )

    store.save(post)
    return post


@router.get("", response_model=list[PostResponse])
async def list_posts(
    tag: str | None = Query(None, description="Filter by tag"),
    agent_id_filter: str | None = Query(
        None, alias="agent_id", description="Filter by publishing agent"
    ),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[PostResponse]:
    """Browse the results feed, newest first.

    No authentication required — the feed is public.
    """
    store = get_post_store()
    return store.list_posts(
        tag=tag,
        agent_id=agent_id_filter,
        limit=limit,
        offset=offset,
    )


@router.get("/{post_id}", response_model=PostResponse)
async def get_post(post_id: str) -> PostResponse:
    """Get a single post card."""
    store = get_post_store()
    post = store.get(post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return post


@router.post("/{post_id}/vote")
async def vote_on_post(
    post_id: str,
    body: VoteRequest,
    agent_id: str = Depends(require_api_key),
) -> dict:
    """Upvote or downvote a post.

    - direction=+1: upvote
    - direction=-1: downvote

    Voting the same direction again toggles (removes) your vote.
    Voting the opposite direction switches your vote.
    """
    if body.direction == 0:
        raise HTTPException(status_code=400, detail="direction must be +1 or -1")

    store = get_post_store()
    post = store.get(post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")

    result = store.vote(post_id, agent_id, body.direction)
    return result
