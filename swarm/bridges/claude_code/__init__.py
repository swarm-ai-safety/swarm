"""SWARM-Claude Code Bridge.

Connects SWARM's governance and metrics framework to claude-code-controller,
enabling programmatic orchestration and safety scoring of Claude Code CLI agents.

Architecture:
    SWARM Orchestrator (Python)
        └── ClaudeCodeBridge
                ├── ClaudeCodeClient  (HTTP/WS → TS service)
                ├── GovernancePolicy  (plan/permission adjudication)
                └── ClaudeCodeAgent   (BaseAgent adapter)
                        ↓
    claude-code-service (TypeScript)
        └── ClaudeCodeController
                └── Claude Code CLI agents
"""

from swarm.bridges.claude_code.bridge import ClaudeCodeBridge
from swarm.bridges.claude_code.client import ClaudeCodeClient, ClientConfig
from swarm.bridges.claude_code.events import (
    BridgeEvent,
    BridgeEventType,
    MessageEvent,
    PermissionRequest,
    PlanApprovalRequest,
    TaskEvent,
)
from swarm.bridges.claude_code.policy import GovernancePolicy, PolicyDecision

__all__ = [
    "ClaudeCodeBridge",
    "ClaudeCodeClient",
    "ClientConfig",
    "BridgeEvent",
    "BridgeEventType",
    "MessageEvent",
    "PlanApprovalRequest",
    "PermissionRequest",
    "TaskEvent",
    "GovernancePolicy",
    "PolicyDecision",
]
