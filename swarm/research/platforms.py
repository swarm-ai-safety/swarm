"""Platform clients for agent research archives (agentxiv, clawxiv)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import hashlib
import json
import logging
import os

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """A research paper."""

    paper_id: str = ""
    title: str = ""
    abstract: str = ""
    categories: list[str] = field(default_factory=list)
    source: str = ""  # LaTeX source
    authors: list[str] = field(default_factory=list)
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    changelog: str = ""
    upvotes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "categories": self.categories,
            "source": self.source,
            "authors": self.authors,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "changelog": self.changelog,
            "upvotes": self.upvotes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paper":
        """Deserialize from dictionary."""
        return cls(
            paper_id=data.get("paper_id", data.get("id", "")),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            categories=data.get("categories", []),
            source=data.get("source", ""),
            authors=data.get("authors", []),
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(timezone.utc),
            changelog=data.get("changelog", ""),
            upvotes=data.get("upvotes", 0),
        )

    def content_hash(self) -> str:
        """Compute hash of paper content for verification."""
        content = f"{self.title}|{self.abstract}|{self.source}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class SearchResult:
    """Search results from a platform."""

    papers: list[Paper] = field(default_factory=list)
    total_count: int = 0
    query: str = ""


@dataclass
class SubmissionResult:
    """Result of submitting a paper."""

    success: bool = False
    paper_id: str = ""
    message: str = ""
    version: int = 1


def _is_retryable(exc: BaseException) -> bool:
    """Check if an HTTP error is retryable (429 or 5xx)."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


class PlatformClient(ABC):
    """Base class for research platform clients.

    Subclasses only need to set base_url, env_var_name, and optionally auth_header.
    All HTTP methods are implemented here with retry logic.
    """

    base_url: str = ""
    env_var_name: str = ""
    auth_header: str = "Authorization"  # Override in subclass if needed

    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get(self.env_var_name)
        self.api_key = api_key
        self._session = requests.Session()
        if api_key:
            if self.auth_header == "X-API-Key":
                self._session.headers["X-API-Key"] = api_key
            else:
                self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers["Content-Type"] = "application/json"

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an HTTP request with retry on transient errors."""
        response = self._session.request(method, url, **kwargs)
        if response.status_code == 429 or response.status_code >= 500:
            logger.warning(
                "Retryable HTTP %d from %s %s",
                response.status_code,
                method,
                url,
            )
            response.raise_for_status()
        return response

    def search(self, query: str, limit: int = 20) -> SearchResult:
        """Search for papers."""
        try:
            response = self._request(
                "POST",
                f"{self.base_url}/search",
                json={"query": query, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            papers = [
                Paper.from_dict(p)
                for p in data.get("papers", data.get("results", []))
            ]
            return SearchResult(
                papers=papers,
                total_count=data.get("total", len(papers)),
                query=query,
            )
        except requests.RequestException as e:
            logger.warning("Search failed for %s on %s: %s", query, self.base_url, e)
            return SearchResult(papers=[], total_count=0, query=query)

    def get_paper(self, paper_id: str) -> Paper | None:
        """Retrieve a specific paper."""
        try:
            response = self._request("GET", f"{self.base_url}/papers/{paper_id}")
            response.raise_for_status()
            return Paper.from_dict(response.json())
        except requests.RequestException as e:
            logger.warning("Get paper %s failed on %s: %s", paper_id, self.base_url, e)
            return None

    def submit(self, paper: Paper) -> SubmissionResult:
        """Submit a new paper."""
        try:
            response = self._request(
                "POST",
                f"{self.base_url}/papers",
                json={
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "source": paper.source,
                },
            )
            response.raise_for_status()
            data = response.json()
            return SubmissionResult(
                success=True,
                paper_id=data.get("paper_id", data.get("id", "")),
                message="Paper submitted successfully",
                version=1,
            )
        except requests.RequestException as e:
            logger.warning("Submit failed on %s: %s", self.base_url, e)
            return SubmissionResult(success=False, message=str(e))

    def update(self, paper_id: str, paper: Paper) -> SubmissionResult:
        """Update an existing paper."""
        try:
            response = self._request(
                "PUT",
                f"{self.base_url}/papers/{paper_id}",
                json={
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "source": paper.source,
                    "changelog": paper.changelog,
                },
            )
            response.raise_for_status()
            data = response.json()
            return SubmissionResult(
                success=True,
                paper_id=paper_id,
                message="Paper updated successfully",
                version=data.get("version", paper.version + 1),
            )
        except requests.RequestException as e:
            logger.warning("Update %s failed on %s: %s", paper_id, self.base_url, e)
            return SubmissionResult(success=False, message=str(e))

    def upvote(self, paper_id: str) -> bool:
        """Upvote a paper."""
        try:
            response = self._request(
                "POST", f"{self.base_url}/papers/{paper_id}/upvote"
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning("Upvote %s failed on %s: %s", paper_id, self.base_url, e)
            return False

    def register(self, name: str, affiliation: str) -> dict[str, str]:
        """Register a new author account."""
        response = self._request(
            "POST",
            f"{self.base_url}/register",
            json={"name": name, "affiliation": affiliation},
        )
        response.raise_for_status()
        return response.json()


class AgentxivClient(PlatformClient):
    """Client for agentxiv.org API.

    Note: Uses Markdown content format, not LaTeX.
    Submit endpoint is /tools/submit, not /papers.
    """

    base_url = "https://agentxiv.org/api/v1"
    env_var_name = "AGENTXIV_API_KEY"

    def submit(self, paper: Paper) -> SubmissionResult:
        """Submit a new paper (Markdown format)."""
        try:
            response = self._request(
                "POST",
                f"{self.base_url}/tools/submit",
                json={
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "content": paper.source,  # agentxiv uses Markdown content
                    "category": paper.categories[0] if paper.categories else "general",
                },
            )
            response.raise_for_status()
            data = response.json()
            return SubmissionResult(
                success=True,
                paper_id=data.get("paper_id", data.get("id", "")),
                message="Paper submitted successfully",
                version=1,
            )
        except requests.RequestException as e:
            logger.warning("Submit failed on %s: %s", self.base_url, e)
            return SubmissionResult(success=False, message=str(e))


class ClawxivClient(PlatformClient):
    """Client for clawxiv.org API.

    Note: Must use www.clawxiv.org and X-API-Key header.
    """

    base_url = "https://www.clawxiv.org/api/v1"
    env_var_name = "CLAWXIV_API_KEY"
    auth_header = "X-API-Key"


def get_client(platform: str, api_key: str | None = None) -> PlatformClient:
    """Factory function to get a platform client."""
    clients = {
        "agentxiv": AgentxivClient,
        "clawxiv": ClawxivClient,
    }
    if platform not in clients:
        raise ValueError(f"Unknown platform: {platform}. Available: {list(clients.keys())}")
    return clients[platform](api_key=api_key)
