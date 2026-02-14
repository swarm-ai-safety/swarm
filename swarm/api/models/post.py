"""Post/feed card models for the Moltbook-style results feed."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class PostCreate(BaseModel):
    """Request to publish a run result card to the feed."""

    run_id: str = Field(..., description="ID of the completed run")
    title: str = Field(
        ..., description="Card title", min_length=1, max_length=200
    )
    blurb: str = Field(
        ..., description="Short summary of results", max_length=2000
    )
    key_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Highlighted metrics to display on the card",
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags for filtering (e.g. 'baseline', 'sweep')"
    )


class PostResponse(BaseModel):
    """A published result card."""

    post_id: str = Field(..., description="Unique post identifier")
    run_id: str = Field(..., description="Associated run")
    agent_id: str = Field(..., description="Agent that published the card")
    title: str
    blurb: str
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    published_at: datetime
    run_url: Optional[str] = Field(None, description="Link to full run results")


class FeedQuery(BaseModel):
    """Query parameters for the feed."""

    tags: Optional[list[str]] = Field(None, description="Filter by tags")
    agent_id: Optional[str] = Field(None, description="Filter by publishing agent")
    limit: int = Field(default=20, ge=1, le=100, description="Max results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
