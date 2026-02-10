"""AgentRxiv integration helpers for SWARM paper runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from swarm.research.platforms import AgentRxivClient, Paper, SubmissionResult
from swarm.research.swarm_papers.memory import (
    MemoryArtifact,
    new_artifact_id,
    relevance_score,
)


@dataclass
class AgentRxivHit:
    paper: Paper
    score: float


class AgentRxivBridge:
    """Lightweight bridge for AgentRxiv retrieval + submission."""

    def __init__(self, base_url: str | None = None):
        self.client = AgentRxivClient(base_url=base_url)

    def available(self) -> bool:
        return self.client.health_check()

    def search(self, query: str, limit: int = 5) -> list[AgentRxivHit]:
        if not self.available():
            return []
        result = self.client.search(query, limit=limit)
        hits: list[AgentRxivHit] = []
        for paper in result.papers:
            text = " ".join([paper.title, paper.abstract])
            score = relevance_score(query, text)
            hits.append(AgentRxivHit(paper=paper, score=score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits

    def to_artifacts(
        self,
        query: str,
        limit: int = 5,
        *,
        min_score: float = 0.15,
    ) -> list[MemoryArtifact]:
        artifacts: list[MemoryArtifact] = []
        for hit in self.search(query, limit=limit):
            if hit.score < min_score:
                continue
            paper = hit.paper
            artifacts.append(
                MemoryArtifact(
                    artifact_id=new_artifact_id(),
                    title=paper.title or "AgentRxiv Paper",
                    summary=(paper.abstract or "").strip()[:400],
                    use_when="Related SWARM research context",
                    failure_modes=[],
                    metrics={"relevance": round(hit.score, 3)},
                    source="agentrxiv",
                    source_id=paper.paper_id,
                )
            )
        return artifacts

    def submit(self, paper: Paper, pdf_path: str) -> SubmissionResult:
        return self.client.submit(paper, pdf_path=pdf_path)

    def trigger_update(self) -> bool:
        return self.client.trigger_update()

    def related_work(self, query: str, limit: int = 5) -> list[Paper]:
        return [hit.paper for hit in self.search(query, limit=limit)]


def format_related_work(papers: Iterable[Paper]) -> str:
    lines = []
    for paper in papers:
        title = paper.title or "Untitled"
        paper_id = paper.paper_id or "unknown"
        lines.append(f"- {title} ({paper_id})")
    return "\n".join(lines)
