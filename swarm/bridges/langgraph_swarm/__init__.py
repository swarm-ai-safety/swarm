"""SWARM-LangGraph Swarm Bridge — Governance-Aware Swarm Extension.

Extends langgraph-swarm with:

1. Byline-style provenance logging on every handoff
2. Governance intervention hooks (approve/deny/modify handoffs)
3. Per-agent risk scoring and escalation
4. Information isolation with selective context sharing

Bridges:
    - LangGraph Swarm's production handoff mechanics
    - Byline authorship tracking (provenance chains)
    - SWARM multi-agent risk framework (governance interventions)
    - Hammond et al. multi-agent risk patterns (collusion, deception)

Requires the ``langgraph`` optional dependency group::

    pip install swarm-safety[langgraph]
"""

from swarm.bridges.langgraph_swarm.governed_swarm import (
    CompositePolicy,
    CycleDetectionPolicy,
    GovernanceDecision,
    GovernancePolicy,
    GovernanceResult,
    GovernedSwarmState,
    InformationBoundaryPolicy,
    ProvenanceLogger,
    ProvenanceRecord,
    RateLimitPolicy,
    analyze_swarm_run,
    create_governed_handoff_tool,
    create_governed_swarm,
    default_governance_stack,
)

__all__ = [
    # Provenance
    "ProvenanceRecord",
    "ProvenanceLogger",
    # Governance policy
    "GovernanceDecision",
    "GovernanceResult",
    "GovernancePolicy",
    "CycleDetectionPolicy",
    "RateLimitPolicy",
    "InformationBoundaryPolicy",
    "CompositePolicy",
    # State
    "GovernedSwarmState",
    # Construction
    "create_governed_handoff_tool",
    "create_governed_swarm",
    # Utilities
    "analyze_swarm_run",
    "default_governance_stack",
]
