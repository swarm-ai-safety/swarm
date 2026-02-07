"""Platform clients for agent research archives (agentxiv, clawxiv)."""

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests  # type: ignore[import-untyped]


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


class PlatformClient(ABC):
    """Abstract base class for research platform clients."""

    def __init__(self, api_key: str | None = None, base_url: str = ""):
        self.api_key = api_key
        self.base_url = base_url
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers["Content-Type"] = "application/json"

    @abstractmethod
    def search(self, query: str, limit: int = 20) -> SearchResult:
        """Search for papers."""
        pass

    @abstractmethod
    def get_paper(self, paper_id: str) -> Paper | None:
        """Retrieve a specific paper."""
        pass

    @abstractmethod
    def submit(self, paper: Paper) -> SubmissionResult:
        """Submit a new paper."""
        pass

    @abstractmethod
    def update(self, paper_id: str, paper: Paper) -> SubmissionResult:
        """Update an existing paper."""
        pass

    @abstractmethod
    def upvote(self, paper_id: str) -> bool:
        """Upvote a paper."""
        pass

    def register(self, name: str, affiliation: str) -> dict[str, str]:
        """Register a new author account."""
        response = self._session.post(
            f"{self.base_url}/register",
            json={"name": name, "affiliation": affiliation},
        )
        response.raise_for_status()
        return dict(response.json())


class AgentxivClient(PlatformClient):
    """Client for agentxiv.org API."""

    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("AGENTXIV_API_KEY")
        super().__init__(api_key=api_key, base_url="https://www.agentxiv.org/api")

    def search(self, query: str, limit: int = 20) -> SearchResult:
        """Search agentxiv papers."""
        try:
            response = self._session.post(
                f"{self.base_url}/search",
                json={"query": query, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            papers = [Paper.from_dict(p) for p in data.get("papers", data.get("results", []))]
            return SearchResult(
                papers=papers,
                total_count=data.get("total", len(papers)),
                query=query,
            )
        except requests.RequestException:
            return SearchResult(papers=[], total_count=0, query=query)

    def get_paper(self, paper_id: str) -> Paper | None:
        """Get a specific paper from agentxiv."""
        try:
            response = self._session.get(f"{self.base_url}/papers/{paper_id}")
            response.raise_for_status()
            return Paper.from_dict(response.json())
        except requests.RequestException:
            return None

    def submit(self, paper: Paper) -> SubmissionResult:
        """Submit a paper to agentxiv."""
        try:
            response = self._session.post(
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
            return SubmissionResult(
                success=False,
                message=str(e),
            )

    def update(self, paper_id: str, paper: Paper) -> SubmissionResult:
        """Update a paper on agentxiv."""
        try:
            response = self._session.put(
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
            return SubmissionResult(success=False, message=str(e))

    def upvote(self, paper_id: str) -> bool:
        """Upvote a paper on agentxiv."""
        try:
            response = self._session.post(f"{self.base_url}/papers/{paper_id}/upvote")
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False


class ClawxivClient(PlatformClient):
    """Client for clawxiv.org API."""

    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("CLAWXIV_API_KEY")
        super().__init__(api_key=api_key, base_url="https://clawxiv.org/api")

    def search(self, query: str, limit: int = 20) -> SearchResult:
        """Search clawxiv papers."""
        try:
            response = self._session.post(
                f"{self.base_url}/search",
                json={"query": query, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            papers = [Paper.from_dict(p) for p in data.get("papers", data.get("results", []))]
            return SearchResult(
                papers=papers,
                total_count=data.get("total", len(papers)),
                query=query,
            )
        except requests.RequestException:
            return SearchResult(papers=[], total_count=0, query=query)

    def get_paper(self, paper_id: str) -> Paper | None:
        """Get a specific paper from clawxiv."""
        try:
            response = self._session.get(f"{self.base_url}/papers/{paper_id}")
            response.raise_for_status()
            return Paper.from_dict(response.json())
        except requests.RequestException:
            return None

    def submit(self, paper: Paper) -> SubmissionResult:
        """Submit a paper to clawxiv."""
        try:
            response = self._session.post(
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
            return SubmissionResult(success=False, message=str(e))

    def update(self, paper_id: str, paper: Paper) -> SubmissionResult:
        """Update a paper on clawxiv."""
        try:
            response = self._session.put(
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
            return SubmissionResult(success=False, message=str(e))

    def upvote(self, paper_id: str) -> bool:
        """Upvote a paper on clawxiv."""
        try:
            response = self._session.post(f"{self.base_url}/papers/{paper_id}/upvote")
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False


def get_client(platform: str, api_key: str | None = None) -> PlatformClient:
    """Factory function to get a platform client."""
    clients = {
        "agentxiv": AgentxivClient,
        "clawxiv": ClawxivClient,
    }
    if platform not in clients:
        raise ValueError(f"Unknown platform: {platform}. Available: {list(clients.keys())}")
    return clients[platform](api_key=api_key)
