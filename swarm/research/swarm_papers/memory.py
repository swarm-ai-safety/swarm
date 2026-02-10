"""Memory artifacts and governance policies for SWARM paper runs."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def relevance_score(query: str, text: str) -> float:
    """Return a simple relevance score in [0, 1] based on token overlap."""
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(_tokenize(text))
    if not text_tokens:
        return 0.0
    overlap = query_tokens.intersection(text_tokens)
    return len(overlap) / max(len(query_tokens), 1)


@dataclass
class MemoryArtifact:
    """A compact memory item that can be retrieved during SWARM runs."""

    artifact_id: str
    title: str
    summary: str
    use_when: str
    failure_modes: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    source: str = "local"
    source_id: str = ""
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "title": self.title,
            "summary": self.summary,
            "use_when": self.use_when,
            "failure_modes": self.failure_modes,
            "metrics": self.metrics,
            "source": self.source,
            "source_id": self.source_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryArtifact":
        return cls(
            artifact_id=data.get("artifact_id", ""),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            use_when=data.get("use_when", ""),
            failure_modes=list(data.get("failure_modes", [])),
            metrics=dict(data.get("metrics", {})),
            source=data.get("source", "local"),
            source_id=data.get("source_id", ""),
            created_at=data.get("created_at", _now_iso()),
        )


@dataclass
class RetrievalPolicy:
    """Policy gates for memory retrieval."""

    max_items: int = 3
    min_score: float = 0.15
    allow_sources: set[str] = field(default_factory=lambda: {"local", "agentrxiv"})


@dataclass
class WritePolicy:
    """Policy gates for writing new memory artifacts."""

    min_accuracy: float = 0.55
    min_delta: float = 0.01
    min_tasks: int = 50
    require_audit: bool = True
    require_critic: bool = True
    max_critic_flag_rate: float = 0.25
    max_adversary_rate: float = 0.05
    adversary_conf_threshold: float = 0.8


@dataclass
class AuditReport:
    """Result of a memory audit."""

    passed: bool
    reasons: list[str] = field(default_factory=list)


class BasicAudit:
    """Lightweight audit hook to avoid low-signal memory writes."""

    def evaluate(
        self,
        *,
        artifact: MemoryArtifact,
        accuracy: float,
        delta_vs_baseline: float | None,
        n_tasks: int,
        critic_flag_rate: float | None,
        adversary_rate: float | None,
        policy: WritePolicy,
    ) -> AuditReport:
        reasons: list[str] = []

        if not artifact.title or not artifact.summary:
            reasons.append("artifact missing title or summary")
        if accuracy < policy.min_accuracy:
            reasons.append(f"accuracy below threshold: {accuracy:.3f}")
        if n_tasks < policy.min_tasks:
            reasons.append("insufficient tasks for confidence")
        if delta_vs_baseline is not None and delta_vs_baseline < policy.min_delta:
            reasons.append("delta vs baseline below threshold")
        if policy.require_critic and critic_flag_rate is None:
            reasons.append("critic gating required but unavailable")
        if critic_flag_rate is not None and critic_flag_rate > policy.max_critic_flag_rate:
            reasons.append("critic flag rate exceeds threshold")
        if adversary_rate is not None and adversary_rate > policy.max_adversary_rate:
            reasons.append("adversary detection rate exceeds threshold")

        return AuditReport(passed=not reasons, reasons=reasons)


class MemoryStore:
    """Append-only JSONL store for memory artifacts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[MemoryArtifact]:
        if not self.path.exists():
            return []
        artifacts = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    artifacts.append(MemoryArtifact.from_dict(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        return artifacts

    def append(self, artifact: MemoryArtifact) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(artifact.to_dict()) + "\n")

    def search(
        self,
        query: str,
        *,
        policy: RetrievalPolicy | None = None,
    ) -> list[MemoryArtifact]:
        policy = policy or RetrievalPolicy()
        artifacts = [a for a in self.load() if a.source in policy.allow_sources]

        scored: list[tuple[float, MemoryArtifact]] = []
        for artifact in artifacts:
            text = " ".join(
                [artifact.title, artifact.summary, artifact.use_when]
                + artifact.failure_modes
            )
            score = relevance_score(query, text)
            if score >= policy.min_score:
                scored.append((score, artifact))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [artifact for _, artifact in scored[: policy.max_items]]


def new_artifact_id() -> str:
    return uuid.uuid4().hex[:12]


def summarize_artifacts(artifacts: Iterable[MemoryArtifact]) -> str:
    """Render a short bullet list of memory artifacts for prompts."""
    lines = []
    for artifact in artifacts:
        lines.append(
            f"- {artifact.title}: {artifact.summary} (Use when: {artifact.use_when})"
        )
    return "\n".join(lines)
