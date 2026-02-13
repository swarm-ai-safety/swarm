"""Event schemas for the AgentLaboratory bridge.

Defines typed events for tracking AgentLab workflow behavior:
phase transitions, solver iterations, dialogues, code execution,
reviews, paper generation, and cost tracking.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentLabEventType(Enum):
    """Event types emitted by the AgentLab bridge."""

    # Phase lifecycle
    PHASE_STARTED = "phase:started"
    PHASE_COMPLETED = "phase:completed"
    PHASE_FAILED = "phase:failed"

    # Dialogue
    DIALOGUE_EXCHANGE = "dialogue:exchange"

    # Solver
    SOLVER_ITERATION = "solver:iteration"
    SOLVER_SCORED = "solver:scored"
    SOLVER_BEST_UPDATED = "solver:best_updated"

    # Code
    CODE_GENERATED = "code:generated"
    CODE_EXECUTED = "code:executed"
    CODE_FAILED = "code:failed"
    CODE_REPAIRED = "code:repaired"

    # Review
    REVIEW_SUBMITTED = "review:submitted"
    REVIEW_DECISION = "review:decision"

    # Paper
    PAPER_SECTION_GENERATED = "paper:section_generated"
    PAPER_COMPILED = "paper:compiled"

    # Cost
    COST_UPDATED = "cost:updated"

    # Generic
    GENERIC = "generic"
    ERROR = "error"


@dataclass
class AgentLabEvent:
    """Base event emitted by the AgentLab bridge."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AgentLabEventType = AgentLabEventType.GENERIC
    timestamp: datetime = field(default_factory=_utcnow)
    agent_role: str = ""
    phase: str = ""
    step: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_role": self.agent_role,
            "phase": self.phase,
            "step": self.step,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentLabEvent":
        raw_type = data.get("event_type", "generic")
        try:
            event_type = AgentLabEventType(raw_type)
        except ValueError:
            event_type = AgentLabEventType.GENERIC

        raw_ts = data.get("timestamp")
        try:
            timestamp = datetime.fromisoformat(str(raw_ts)) if raw_ts else _utcnow()
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            timestamp = _utcnow()

        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            timestamp=timestamp,
            agent_role=data.get("agent_role", ""),
            phase=data.get("phase", ""),
            step=data.get("step", 0),
            payload=data.get("payload", {}),
        )


@dataclass
class SolverIterationEvent:
    """A single solver iteration (MLE or Paper solver)."""

    solver_type: str = ""  # "mle" or "paper"
    iteration_index: int = 0
    score: float = 0.0  # 0-1 from solver.get_score()
    repair_attempts: int = 0
    execution_error: Optional[str] = None
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_type": self.solver_type,
            "iteration_index": self.iteration_index,
            "score": self.score,
            "repair_attempts": self.repair_attempts,
            "execution_error": self.execution_error,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolverIterationEvent":
        return cls(
            solver_type=data.get("solver_type", ""),
            iteration_index=data.get("iteration_index", 0),
            score=data.get("score", 0.0),
            repair_attempts=data.get("repair_attempts", 0),
            execution_error=data.get("execution_error"),
            cost_usd=data.get("cost_usd", 0.0),
        )


@dataclass
class DialogueEvent:
    """A dialogue exchange between two AgentLab agents."""

    speaker_role: str = ""
    listener_role: str = ""
    phase: str = ""
    command_type: str = ""  # e.g. "plan", "experiment", "interpret", "write"
    has_submission: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_role": self.speaker_role,
            "listener_role": self.listener_role,
            "phase": self.phase,
            "command_type": self.command_type,
            "has_submission": self.has_submission,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueEvent":
        return cls(
            speaker_role=data.get("speaker_role", ""),
            listener_role=data.get("listener_role", ""),
            phase=data.get("phase", ""),
            command_type=data.get("command_type", ""),
            has_submission=data.get("has_submission", False),
        )


@dataclass
class ReviewEvent:
    """A review from one of the 3 AgentLab reviewer personas."""

    reviewer_index: int = 0
    overall_score: float = 0.0  # 1-10 scale
    soundness: float = 0.0
    contribution: float = 0.0
    presentation: float = 0.0
    decision: str = ""  # "accept", "reject", "weak_accept", "weak_reject"
    confidence: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reviewer_index": self.reviewer_index,
            "overall_score": self.overall_score,
            "soundness": self.soundness,
            "contribution": self.contribution,
            "presentation": self.presentation,
            "decision": self.decision,
            "confidence": self.confidence,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewEvent":
        return cls(
            reviewer_index=data.get("reviewer_index", 0),
            overall_score=data.get("overall_score", 0.0),
            soundness=data.get("soundness", 0.0),
            contribution=data.get("contribution", 0.0),
            presentation=data.get("presentation", 0.0),
            decision=data.get("decision", ""),
            confidence=data.get("confidence", 0.0),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
        )
