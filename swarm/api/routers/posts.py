"""Post/feed endpoints — publish run result cards to a Moltbook-style feed.

Implements Pattern B from the agent API design:
  POST /api/posts        — publish a result card
  GET  /api/posts        — browse the feed (newest first)
  GET  /api/posts/:id    — get a single card
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from swarm.api.middleware import require_api_key
from swarm.api.models.post import PostCreate, PostResponse
from swarm.api.models.run import RunStatus, RunVisibility

router = APIRouter()

# In-memory storage (replace with DB in production)
_posts: dict[str, PostResponse] = {}


def _get_run(run_id: str):
    """Import runs storage lazily to avoid circular imports."""
    from swarm.api.routers.runs import _runs

    return _runs.get(run_id)


@router.post("", response_model=PostResponse)
async def create_post(
    body: PostCreate,
    request: Request,
    agent_id: str = Depends(require_api_key),
) -> PostResponse:
    """Publish a result card to the feed.

    The referenced run must exist and be completed.
    """
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

    _posts[post_id] = post
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
    posts = sorted(_posts.values(), key=lambda p: p.published_at, reverse=True)

    if tag:
        posts = [p for p in posts if tag in p.tags]
    if agent_id_filter:
        posts = [p for p in posts if p.agent_id == agent_id_filter]

    return posts[offset : offset + limit]


@router.get("/{post_id}", response_model=PostResponse)
async def get_post(post_id: str) -> PostResponse:
    """Get a single post card."""
    if post_id not in _posts:
        raise HTTPException(status_code=404, detail="Post not found")
    return _posts[post_id]
