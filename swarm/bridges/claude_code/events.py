"""Event schemas for the SWARM-Claude Code bridge protocol.

Defines the typed event structures exchanged between the Python bridge
and the TypeScript controller service. These map directly to the events
emitted by claude-code-controller (messages, plan approvals, permission
requests, task lifecycle).
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _parse_timestamp(data: Dict[str, Any]) -> datetime:
    """Safely parse a timestamp from a dict, defaulting to UTC now."""
    raw = data.get("timestamp")
    if raw is None:
        return _utcnow()
    try:
        return datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return _utcnow()


class BridgeEventType(Enum):
    """Event types in the Claude Code bridge protocol."""

    # Agent lifecycle
    AGENT_SPAWNED = "agent:spawned"
    AGENT_SHUTDOWN = "agent:shutdown"

    # Message events
    MESSAGE_SENT = "message:sent"
    MESSAGE_RECEIVED = "message:received"

    # Governance events (from controller)
    PLAN_APPROVAL_REQUEST = "plan:approval_request"
    PLAN_APPROVED = "plan:approved"
    PLAN_REJECTED = "plan:rejected"
    PERMISSION_REQUEST = "permission:request"
    PERMISSION_GRANTED = "permission:granted"
    PERMISSION_DENIED = "permission:denied"

    # Task events
    TASK_CREATED = "task:created"
    TASK_ASSIGNED = "task:assigned"
    TASK_COMPLETED = "task:completed"
    TASK_FAILED = "task:failed"

    # Tool usage
    TOOL_INVOKED = "tool:invoked"
    TOOL_RESULT = "tool:result"

    # Error events
    ERROR = "error"


@dataclass
class BridgeEvent:
    """Base event in the bridge protocol."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: BridgeEventType = BridgeEventType.MESSAGE_RECEIVED
    timestamp: datetime = field(default_factory=_utcnow)
    agent_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON transport."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BridgeEvent":
        """Deserialize from dict with safe type handling."""
        try:
            event_type = BridgeEventType(data["event_type"])
        except (ValueError, KeyError):
            event_type = BridgeEventType.ERROR
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            timestamp=_parse_timestamp(data),
            agent_id=str(data.get("agent_id", "")),
            payload=data.get("payload", {}),
        )


@dataclass
class MessageEvent:
    """A message sent to or received from a Claude Code agent."""

    agent_id: str = ""
    role: str = "user"  # "user" | "assistant" | "system"
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utcnow)
    token_count: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "cost_usd": self.cost_usd,
        }


@dataclass
class PlanApprovalRequest:
    """A plan submitted by a Claude Code agent for governance review.

    The controller emits plan:approval_request when an agent proposes
    a multi-step plan. The governance policy decides approve/reject.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    plan_description: str = ""
    steps: List[str] = field(default_factory=list)
    estimated_tool_calls: int = 0
    risk_flags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "plan_description": self.plan_description,
            "steps": self.steps,
            "estimated_tool_calls": self.estimated_tool_calls,
            "risk_flags": self.risk_flags,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanApprovalRequest":
        return cls(
            request_id=str(data.get("request_id", str(uuid.uuid4()))),
            agent_id=str(data.get("agent_id", "")),
            plan_description=str(data.get("plan_description", "")),
            steps=data.get("steps", []),
            estimated_tool_calls=int(data.get("estimated_tool_calls", 0)),
            risk_flags=data.get("risk_flags", []),
            timestamp=_parse_timestamp(data),
        )


@dataclass
class PermissionRequest:
    """A tool permission request from a Claude Code agent.

    The controller emits permission:request when an agent wants to
    use a tool (Bash, Write, etc.). The governance policy decides
    whether to grant or deny access.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"  # "low" | "medium" | "high"
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "risk_level": self.risk_level,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PermissionRequest":
        return cls(
            request_id=str(data.get("request_id", str(uuid.uuid4()))),
            agent_id=str(data.get("agent_id", "")),
            tool_name=str(data.get("tool_name", "")),
            tool_args=data.get("tool_args", {}),
            risk_level=str(data.get("risk_level", "low")),
            timestamp=_parse_timestamp(data),
        )


@dataclass
class TaskEvent:
    """A task lifecycle event from the controller's inbox protocol.

    Maps to createTask / waitForTask in the controller.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    subject: str = ""
    description: str = ""
    owner: str = ""
    status: str = "pending"  # "pending" | "in_progress" | "completed" | "failed"
    result: Optional[str] = None
    duration_ms: int = 0
    tools_used: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "subject": self.subject,
            "description": self.description,
            "owner": self.owner,
            "status": self.status,
            "result": self.result,
            "duration_ms": self.duration_ms,
            "tools_used": self.tools_used,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskEvent":
        return cls(
            task_id=str(data.get("task_id", str(uuid.uuid4()))),
            agent_id=str(data.get("agent_id", "")),
            subject=str(data.get("subject", "")),
            description=str(data.get("description", "")),
            owner=str(data.get("owner", "")),
            status=str(data.get("status", "pending")),
            result=data.get("result"),
            duration_ms=int(data.get("duration_ms", 0)),
            tools_used=data.get("tools_used", []),
            timestamp=_parse_timestamp(data),
        )
