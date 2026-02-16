"""Event types for the SciAgentGym bridge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class SciAgentGymEventType(Enum):
    """Types of events in the SciAgentGym workflow."""

    # Tool execution events
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"

    # Workflow events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_STEP_COMPLETED = "workflow_step_completed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"

    # Data management events
    DATA_ARTIFACT_CREATED = "data_artifact_created"
    DATA_ARTIFACT_ACCESSED = "data_artifact_accessed"
    DATA_ARTIFACT_VALIDATED = "data_artifact_validated"

    # Result validation events
    RESULT_VALIDATED = "result_validated"
    RESULT_REJECTED = "result_rejected"

    # Safety events
    SAFETY_CHECK_PASSED = "safety_check_passed"
    SAFETY_CHECK_FAILED = "safety_check_failed"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"


@dataclass
class SciAgentGymEvent:
    """Base event structure for SciAgentGym bridge.
    
    Attributes:
        event_type: Type of event.
        timestamp: When the event occurred.
        agent_id: ID of the agent involved.
        payload: Event-specific data.
    """

    event_type: SciAgentGymEventType
    timestamp: datetime
    agent_id: str
    payload: Dict[str, Any]


@dataclass
class ToolCallEvent:
    """Event data for tool execution.
    
    Attributes:
        tool_name: Name of the scientific tool.
        tool_args: Arguments passed to the tool.
        success: Whether the tool call succeeded.
        execution_time_seconds: Time taken to execute.
        result: Tool execution result (if successful).
        error: Error message (if failed).
        cost_tokens: Token cost for LLM-based tool use.
    """

    tool_name: str
    tool_args: Dict[str, Any]
    success: bool
    execution_time_seconds: float
    result: Optional[Any] = None
    error: Optional[str] = None
    cost_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolCallEvent:
        """Create ToolCallEvent from dictionary."""
        return cls(
            tool_name=data["tool_name"],
            tool_args=data.get("tool_args", {}),
            success=data["success"],
            execution_time_seconds=data.get("execution_time_seconds", 0.0),
            result=data.get("result"),
            error=data.get("error"),
            cost_tokens=data.get("cost_tokens", 0),
        )


@dataclass
class WorkflowStepEvent:
    """Event data for workflow step execution.
    
    Attributes:
        workflow_id: ID of the parent workflow.
        step_index: Index of this step in the workflow.
        step_type: Type of step (e.g., "tool_call", "data_validation").
        success: Whether the step succeeded.
        dependencies_met: Whether all dependencies were satisfied.
        next_steps: Subsequent step indices.
    """

    workflow_id: str
    step_index: int
    step_type: str
    success: bool
    dependencies_met: bool = True
    next_steps: list[int] = None

    def __post_init__(self) -> None:
        if self.next_steps is None:
            self.next_steps = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowStepEvent:
        """Create WorkflowStepEvent from dictionary."""
        return cls(
            workflow_id=data["workflow_id"],
            step_index=data["step_index"],
            step_type=data["step_type"],
            success=data["success"],
            dependencies_met=data.get("dependencies_met", True),
            next_steps=data.get("next_steps", []),
        )


@dataclass
class DataArtifactEvent:
    """Event data for data artifact operations.
    
    Attributes:
        artifact_id: Unique identifier for the artifact.
        artifact_type: Type of artifact (e.g., "dataset", "plot", "model").
        size_bytes: Size of the artifact.
        validated: Whether the artifact passed validation.
        validation_score: Quality/correctness score (0-1).
    """

    artifact_id: str
    artifact_type: str
    size_bytes: int
    validated: bool = False
    validation_score: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DataArtifactEvent:
        """Create DataArtifactEvent from dictionary."""
        return cls(
            artifact_id=data["artifact_id"],
            artifact_type=data["artifact_type"],
            size_bytes=data.get("size_bytes", 0),
            validated=data.get("validated", False),
            validation_score=data.get("validation_score", 0.0),
        )


@dataclass
class SafetyCheckEvent:
    """Event data for safety checks.
    
    Attributes:
        check_type: Type of safety check performed.
        passed: Whether the check passed.
        safety_score: Computed safety score (0-1, where 1 is safest).
        risk_factors: Identified risk factors.
        mitigation_applied: Whether mitigation was applied.
    """

    check_type: str
    passed: bool
    safety_score: float
    risk_factors: list[str] = None
    mitigation_applied: bool = False

    def __post_init__(self) -> None:
        if self.risk_factors is None:
            self.risk_factors = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SafetyCheckEvent:
        """Create SafetyCheckEvent from dictionary."""
        return cls(
            check_type=data["check_type"],
            passed=data["passed"],
            safety_score=data["safety_score"],
            risk_factors=data.get("risk_factors", []),
            mitigation_applied=data.get("mitigation_applied", False),
        )
