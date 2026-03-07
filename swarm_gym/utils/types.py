"""Core type definitions for SWARM-Gym."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


AgentId = str


@dataclass
class Action:
    """A single agent action."""

    agent_id: AgentId
    type: str  # "message", "trade", "escalate", "cooperate", "defect", "hide", "report", "noop"
    target: Optional[AgentId] = None
    to: Optional[List[AgentId]] = None
    content: Optional[str] = None
    level: Optional[str] = None
    asset: Optional[str] = None
    qty: Optional[float] = None
    price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Per-agent observation of the environment."""

    agent_id: AgentId
    step: int
    public_messages: List[Dict[str, Any]] = field(default_factory=list)
    private_messages: List[Dict[str, Any]] = field(default_factory=list)
    visible_agents: List[AgentId] = field(default_factory=list)
    own_resources: float = 0.0
    own_reputation: float = 0.0
    market_prices: Dict[str, float] = field(default_factory=dict)
    governance_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    """A governance or environment event."""

    type: str
    severity: float = 0.0
    agent_id: Optional[AgentId] = None
    agents: Optional[List[AgentId]] = None
    outcome: Optional[str] = None
    reason: Optional[str] = None
    module: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Intervention:
    """A governance intervention."""

    module: str
    type: str
    agent_id: Optional[AgentId] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceSnapshot:
    """Governance state before and after a step."""

    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    interventions: List[Intervention] = field(default_factory=list)


@dataclass
class StepMetrics:
    """Per-step metrics."""

    cooperation: float = 0.0
    defection: float = 0.0
    escalation_risk: float = 0.0
    inequality_gini: float = 0.0
    welfare: float = 0.0
    compliance: float = 0.0
    deception: float = 0.0
    collusion: float = 0.0
    evasion: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k in [
            "cooperation", "defection", "escalation_risk", "inequality_gini",
            "welfare", "compliance", "deception", "collusion", "evasion",
        ]:
            v = getattr(self, k)
            if v != 0.0:
                d[k] = round(v, 4)
        if self.metadata:
            d.update(self.metadata)
        return d


@dataclass
class AgentRecord:
    """Agent identity for episode reporting."""

    agent_id: AgentId
    type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"agent_id": self.agent_id, "type": self.type}
        if self.metadata:
            d["metadata"] = self.metadata
        return d
