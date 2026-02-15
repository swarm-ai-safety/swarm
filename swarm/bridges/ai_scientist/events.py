"""AI-Scientist event types and typed event dataclasses.

Maps stages of the AI-Scientist pipeline (Idea -> Experiment -> Writeup -> Review)
to structured events that the mapper can convert to SoftInteractions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AIScientistEventType(Enum):
    # Idea lifecycle
    IDEA_GENERATED = "idea:generated"
    IDEA_REFLECTED = "idea:reflected"
    NOVELTY_CHECK_PASSED = "novelty:passed"
    NOVELTY_CHECK_FAILED = "novelty:failed"

    # Experiment lifecycle
    EXPERIMENT_STARTED = "experiment:started"
    EXPERIMENT_RUN_COMPLETED = "experiment:run_completed"
    EXPERIMENT_RUN_FAILED = "experiment:run_failed"
    EXPERIMENT_COMPLETED = "experiment:completed"
    PLOT_GENERATED = "plot:generated"

    # Paper lifecycle
    WRITEUP_SECTION = "writeup:section"
    WRITEUP_COMPILED = "writeup:compiled"
    WRITEUP_FAILED = "writeup:failed"
    CITATION_ADDED = "citation:added"

    # Review
    REVIEW_SUBMITTED = "review:submitted"
    IMPROVEMENT_APPLIED = "improvement:applied"

    # Cost
    COST_UPDATED = "cost:updated"

    GENERIC = "generic"
    ERROR = "error"


@dataclass
class AIScientistEvent:
    """Base event from the AI-Scientist pipeline."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AIScientistEventType = AIScientistEventType.GENERIC
    timestamp: datetime = field(default_factory=_utcnow)
    idea_name: str = ""
    phase: str = ""  # "idea", "experiment", "writeup", "review"
    step: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "idea_name": self.idea_name,
            "phase": self.phase,
            "step": self.step,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AIScientistEvent:
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=AIScientistEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data.get("timestamp"), str)
            else data.get("timestamp", _utcnow()),
            idea_name=data.get("idea_name", ""),
            phase=data.get("phase", ""),
            step=data.get("step", 0),
            payload=data.get("payload", {}),
        )


@dataclass
class IdeaEvent:
    """Typed event for idea generation and reflection."""

    idea_name: str = ""
    interestingness: float = 0.0  # 1-10
    feasibility: float = 0.0  # 1-10
    novelty_score: float = 0.0  # 1-10
    novel: bool = True
    reflection_round: int = 0
    num_reflections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea_name": self.idea_name,
            "interestingness": self.interestingness,
            "feasibility": self.feasibility,
            "novelty_score": self.novelty_score,
            "novel": self.novel,
            "reflection_round": self.reflection_round,
            "num_reflections": self.num_reflections,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IdeaEvent:
        return cls(
            idea_name=data.get("idea_name", ""),
            interestingness=float(data.get("interestingness", 0.0)),
            feasibility=float(data.get("feasibility", 0.0)),
            novelty_score=float(data.get("novelty_score", 0.0)),
            novel=bool(data.get("novel", True)),
            reflection_round=int(data.get("reflection_round", 0)),
            num_reflections=int(data.get("num_reflections", 0)),
        )


@dataclass
class ExperimentRunEvent:
    """Typed event for a single experiment run attempt."""

    run_index: int = 0
    success: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    execution_error: Optional[str] = None
    cost_usd: float = 0.0
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_index": self.run_index,
            "success": self.success,
            "metrics": self.metrics,
            "execution_error": self.execution_error,
            "cost_usd": self.cost_usd,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentRunEvent:
        return cls(
            run_index=int(data.get("run_index", 0)),
            success=bool(data.get("success", False)),
            metrics=data.get("metrics", {}),
            execution_error=data.get("execution_error"),
            cost_usd=float(data.get("cost_usd", 0.0)),
            retry_count=int(data.get("retry_count", 0)),
        )


@dataclass
class WriteupEvent:
    """Typed event for paper writeup stages."""

    section: str = ""  # "abstract", "introduction", "method", etc.
    compiled: bool = False
    compilation_error: Optional[str] = None
    citation_count: int = 0
    page_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section": self.section,
            "compiled": self.compiled,
            "compilation_error": self.compilation_error,
            "citation_count": self.citation_count,
            "page_count": self.page_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WriteupEvent:
        return cls(
            section=data.get("section", ""),
            compiled=bool(data.get("compiled", False)),
            compilation_error=data.get("compilation_error"),
            citation_count=int(data.get("citation_count", 0)),
            page_count=int(data.get("page_count", 0)),
        )


@dataclass
class ReviewEvent:
    """Typed event for paper review."""

    overall_score: float = 0.0  # 1-10
    decision: str = ""  # "Accept" or "Reject"
    confidence: float = 0.0  # 1-5
    soundness: float = 0.0  # 1-4
    presentation: float = 0.0  # 1-4
    contribution: float = 0.0  # 1-4
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_round: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "decision": self.decision,
            "confidence": self.confidence,
            "soundness": self.soundness,
            "presentation": self.presentation,
            "contribution": self.contribution,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "improvement_round": self.improvement_round,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReviewEvent:
        return cls(
            overall_score=float(data.get("overall_score", 0.0)),
            decision=data.get("decision", ""),
            confidence=float(data.get("confidence", 0.0)),
            soundness=float(data.get("soundness", 0.0)),
            presentation=float(data.get("presentation", 0.0)),
            contribution=float(data.get("contribution", 0.0)),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            improvement_round=int(data.get("improvement_round", 0)),
        )
