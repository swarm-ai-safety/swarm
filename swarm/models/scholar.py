"""Data models for scholar/literature synthesis domain."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swarm.core.proxy import ProxyObservables


@dataclass
class Passage:
    """A passage from the literature corpus."""

    passage_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: str = ""
    text: str = ""
    relevance_score: float = 0.0  # 0-1, higher = more relevant to query
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passage_id": self.passage_id,
            "paper_id": self.paper_id,
            "text": self.text,
            "relevance_score": self.relevance_score,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Passage":
        """Create from dictionary."""
        return cls(
            passage_id=data.get("passage_id", str(uuid.uuid4())),
            paper_id=data.get("paper_id", ""),
            text=data.get("text", ""),
            relevance_score=data.get("relevance_score", 0.0),
            keywords=data.get("keywords", []),
        )


@dataclass
class Citation:
    """A citation linking a claim to source passages."""

    citation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    claim_text: str = ""
    paper_id: str = ""
    passage_id: str = ""
    entailment_score: float = 0.5  # 0=contradicts, 0.5=neutral, 1=entails
    is_adversarial: bool = False  # True if planted by adversary
    verified: bool = False  # True if checked by verifier

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "citation_id": self.citation_id,
            "claim_text": self.claim_text,
            "paper_id": self.paper_id,
            "passage_id": self.passage_id,
            "entailment_score": self.entailment_score,
            "is_adversarial": self.is_adversarial,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Citation":
        """Create from dictionary."""
        return cls(
            citation_id=data.get("citation_id", str(uuid.uuid4())),
            claim_text=data.get("claim_text", ""),
            paper_id=data.get("paper_id", ""),
            passage_id=data.get("passage_id", ""),
            entailment_score=data.get("entailment_score", 0.5),
            is_adversarial=data.get("is_adversarial", False),
            verified=data.get("verified", False),
        )


@dataclass
class ScholarQuery:
    """A research query to be answered by the scholar agents."""

    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str = ""
    domain: str = ""  # e.g., "biology", "physics", "cs"
    constraints: Dict[str, Any] = field(default_factory=dict)
    gold_key_points: List[str] = field(default_factory=list)  # Ground truth points
    known_contradictions: List[Dict] = field(default_factory=list)  # Known conflicts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "domain": self.domain,
            "constraints": self.constraints,
            "gold_key_points": self.gold_key_points,
            "known_contradictions": self.known_contradictions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScholarQuery":
        """Create from dictionary."""
        return cls(
            query_id=data.get("query_id", str(uuid.uuid4())),
            query_text=data.get("query_text", ""),
            domain=data.get("domain", ""),
            constraints=data.get("constraints", {}),
            gold_key_points=data.get("gold_key_points", []),
            known_contradictions=data.get("known_contradictions", []),
        )


@dataclass
class SynthesisResult:
    """Result of synthesizing an answer from passages."""

    synthesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str = ""
    answer_text: str = ""
    citations: List[Citation] = field(default_factory=list)
    key_points_covered: List[str] = field(default_factory=list)
    contradictions_surfaced: List[Dict] = field(default_factory=list)
    synthesizer_id: str = ""  # Agent that produced this synthesis

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "synthesis_id": self.synthesis_id,
            "query_id": self.query_id,
            "answer_text": self.answer_text,
            "citations": [c.to_dict() for c in self.citations],
            "key_points_covered": self.key_points_covered,
            "contradictions_surfaced": self.contradictions_surfaced,
            "synthesizer_id": self.synthesizer_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynthesisResult":
        """Create from dictionary."""
        return cls(
            synthesis_id=data.get("synthesis_id", str(uuid.uuid4())),
            query_id=data.get("query_id", ""),
            answer_text=data.get("answer_text", ""),
            citations=[
                Citation.from_dict(c) for c in data.get("citations", [])
            ],
            key_points_covered=data.get("key_points_covered", []),
            contradictions_surfaced=data.get("contradictions_surfaced", []),
            synthesizer_id=data.get("synthesizer_id", ""),
        )


@dataclass
class ScholarActionResult:
    """Result of a scholar action (retrieve, synthesize, verify)."""

    success: bool
    observables: Optional[ProxyObservables] = None
    initiator_id: str = ""
    counterparty_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = True

    # Action-specific results
    passages_retrieved: List[Passage] = field(default_factory=list)
    synthesis_result: Optional[SynthesisResult] = None
    verification_verdict: Optional[bool] = None  # True=valid, False=invalid
    citation_verified: Optional[Citation] = None
