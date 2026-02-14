"""governed_swarm.py — Governance-Aware LangGraph Swarm Extension.

Extends langgraph-swarm with:

1. Byline-style provenance logging on every handoff
2. Governance intervention hooks (approve/deny/modify handoffs)
3. Per-agent risk scoring and escalation
4. Information isolation with selective context sharing

Designed to bridge:

- LangGraph Swarm's production handoff mechanics
- Byline authorship tracking (provenance chains)
- SWARM multi-agent risk framework (governance interventions)
- Hammond et al. multi-agent risk patterns (collusion, deception)

Usage::

    from swarm.bridges.langgraph_swarm import (
        GovernedSwarmState,
        create_governed_swarm,
        create_governed_handoff_tool,
        GovernancePolicy,
        ProvenanceLogger,
    )
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import (
    Annotated,
    Any,
    Callable,
)

from typing_extensions import TypedDict

try:
    from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
    from langchain_core.tools import BaseTool, InjectedToolCallId, tool
    from langgraph.graph import START, StateGraph, add_messages
    from langgraph.prebuilt import InjectedState
    from langgraph.pregel import Pregel
    from langgraph.types import Command

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

    # Provide type stubs so the pure-Python governance/provenance classes
    # can be imported and used without langgraph installed.
    AnyMessage = Any  # type: ignore[misc]
    AIMessage = Any  # type: ignore[misc]
    ToolMessage = Any  # type: ignore[misc]
    BaseTool = Any  # type: ignore[misc]
    StateGraph = Any  # type: ignore[misc]
    Pregel = Any  # type: ignore[misc]
    Command = Any  # type: ignore[misc]

    def add_messages(left: list, right: list) -> list:  # type: ignore[misc]
        return left + right


# =============================================================================
# 1. PROVENANCE — Byline-style authorship tracking
# =============================================================================


@dataclass
class ProvenanceRecord:
    """A single provenance entry in the handoff chain.

    Modeled after Byline's authorship tracking: every handoff is an
    authorship event with a source agent, target agent, reasoning,
    and metadata about what context was shared.
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: float = field(default_factory=time.time)

    # Authorship fields (Byline-compatible)
    source_agent: str = ""
    target_agent: str = ""
    handoff_reason: str = ""
    task_description: str = ""

    # Context sharing metadata
    messages_shared_count: int = 0
    context_filter_applied: str = "none"  # "none", "last_n", "summary", "selective"

    # Governance fields
    governance_decision: str = "pending"  # "approved", "denied", "modified", "escalated"
    governance_reason: str = ""
    risk_score_at_handoff: float = 0.0
    intervention_applied: str = ""

    # Chain linkage
    parent_record_id: str | None = None
    chain_depth: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ProvenanceLogger:
    """Accumulates provenance records across a swarm's lifetime.

    Provides the queryable provenance chain that Byline needs for
    authorship tracking, and the intervention log that the SWARM
    governance framework needs for audit trails.
    """

    def __init__(self) -> None:
        self.records: list[ProvenanceRecord] = []
        self._listeners: list[Callable[[ProvenanceRecord], None]] = []

    def log(self, record: ProvenanceRecord) -> ProvenanceRecord:
        """Append a provenance record and notify listeners."""
        # Auto-link to parent if chain exists
        if self.records and record.parent_record_id is None:
            record.parent_record_id = self.records[-1].record_id
            record.chain_depth = self.records[-1].chain_depth + 1

        self.records.append(record)

        for listener in self._listeners:
            listener(record)

        return record

    def on_record(self, callback: Callable[[ProvenanceRecord], None]) -> None:
        """Register a listener for new provenance records."""
        self._listeners.append(callback)

    def get_chain(self, agent_name: str | None = None) -> list[ProvenanceRecord]:
        """Get the full provenance chain, optionally filtered by agent."""
        if agent_name is None:
            return list(self.records)
        return [
            r
            for r in self.records
            if r.source_agent == agent_name or r.target_agent == agent_name
        ]

    def get_handoff_count(self, source: str, target: str) -> int:
        """Count handoffs between a specific pair of agents."""
        return sum(
            1
            for r in self.records
            if r.source_agent == source and r.target_agent == target
        )

    def detect_cycles(
        self, window: int = 10, threshold: int = 3
    ) -> list[tuple[str, str]]:
        """Detect repeated handoff cycles in the recent window.

        Returns pairs of agents that are ping-ponging control back
        and forth -- a potential indicator of confusion, collusion,
        or adversarial behavior (Hammond et al. risk patterns).

        Args:
            window: Number of recent records to examine.
            threshold: Minimum count for a pair to be flagged.
        """
        recent = self.records[-window:]
        pair_counts: dict[tuple[str, str], int] = {}
        for r in recent:
            pair = (r.source_agent, r.target_agent)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return [pair for pair, count in pair_counts.items() if count >= threshold]

    def to_audit_log(self) -> list[dict]:
        """Export the full provenance chain as a serializable audit log."""
        return [r.to_dict() for r in self.records]


# =============================================================================
# 2. GOVERNANCE POLICY — Intervention hooks
# =============================================================================


class GovernanceDecision(Enum):
    APPROVE = "approved"
    DENY = "denied"
    MODIFY = "modified"
    ESCALATE = "escalated"


@dataclass
class GovernanceResult:
    """The outcome of a governance policy evaluation."""

    decision: GovernanceDecision
    reason: str = ""
    modified_target: str | None = None  # If decision is MODIFY, redirect here
    modified_context: dict | None = None  # Additional state modifications
    risk_score: float = 0.0


class GovernancePolicy:
    """Pluggable governance layer that evaluates handoffs before execution.

    This is the core intervention mechanism for the SWARM risk framework.
    Policies can be composed, and each handoff passes through all active
    policies before proceeding.

    Subclass this and override ``evaluate()`` to implement custom governance.
    """

    def evaluate(
        self,
        source_agent: str,
        target_agent: str,
        task_description: str,
        state: dict,
        provenance: ProvenanceLogger,
    ) -> GovernanceResult:
        """Evaluate whether a handoff should proceed.

        Override this method to implement custom governance logic.

        Args:
            source_agent: The agent initiating the handoff.
            target_agent: The agent being handed off to.
            task_description: LLM-generated description of the task.
            state: The current swarm state dict.
            provenance: The provenance logger with full history.

        Returns:
            GovernanceResult with the decision and reasoning.
        """
        return GovernanceResult(
            decision=GovernanceDecision.APPROVE,
            reason="Default policy: approve all handoffs",
        )


class CycleDetectionPolicy(GovernancePolicy):
    """Detects and blocks circular handoff patterns.

    Addresses the "agent collusion via ping-pong" risk from
    Hammond et al. -- agents that repeatedly hand off to each
    other may be avoiding responsibility or collaborating to
    circumvent constraints.
    """

    def __init__(self, max_cycles: int = 3, window: int = 10) -> None:
        self.max_cycles = max_cycles
        self.window = window

    def evaluate(
        self,
        source_agent: str,
        target_agent: str,
        task_description: str,
        state: dict,
        provenance: ProvenanceLogger,
    ) -> GovernanceResult:
        cycles = provenance.detect_cycles(window=self.window, threshold=self.max_cycles)
        pair = (source_agent, target_agent)

        if pair in cycles:
            count = provenance.get_handoff_count(source_agent, target_agent)
            return GovernanceResult(
                decision=GovernanceDecision.DENY,
                reason=(
                    f"Cycle detected: {source_agent} -> {target_agent} "
                    f"has occurred {count} times in the last {self.window} handoffs. "
                    f"This may indicate confusion or adversarial behavior."
                ),
                risk_score=0.8,
            )

        return GovernanceResult(
            decision=GovernanceDecision.APPROVE,
            reason="No circular handoff patterns detected",
            risk_score=0.0,
        )


class RateLimitPolicy(GovernancePolicy):
    """Limits the total number of handoffs within a time window.

    Prevents runaway handoff chains where agents continuously
    delegate without making progress -- a resource exhaustion risk.
    """

    def __init__(
        self, max_handoffs: int = 20, window_seconds: float = 300.0
    ) -> None:
        self.max_handoffs = max_handoffs
        self.window_seconds = window_seconds

    def evaluate(
        self,
        source_agent: str,
        target_agent: str,
        task_description: str,
        state: dict,
        provenance: ProvenanceLogger,
    ) -> GovernanceResult:
        now = time.time()
        recent = [
            r
            for r in provenance.records
            if (now - r.timestamp) < self.window_seconds
        ]

        if len(recent) >= self.max_handoffs:
            return GovernanceResult(
                decision=GovernanceDecision.ESCALATE,
                reason=(
                    f"Rate limit exceeded: {len(recent)} handoffs in the last "
                    f"{self.window_seconds}s (limit: {self.max_handoffs}). "
                    f"Escalating to human oversight."
                ),
                risk_score=0.9,
            )

        return GovernanceResult(
            decision=GovernanceDecision.APPROVE,
            reason=f"Within rate limits ({len(recent)}/{self.max_handoffs})",
            risk_score=len(recent) / self.max_handoffs,
        )


class InformationBoundaryPolicy(GovernancePolicy):
    """Enforces information isolation between agent groups.

    Prevents sensitive context from leaking across trust boundaries.
    For example, a "financial agent" shouldn't share PII with
    a "marketing agent" even if they're in the same swarm.
    """

    def __init__(self, trust_groups: dict[str, str] | None = None) -> None:
        # Maps agent_name -> trust_group
        # Agents in different trust groups get filtered context
        self.trust_groups = trust_groups or {}

    def evaluate(
        self,
        source_agent: str,
        target_agent: str,
        task_description: str,
        state: dict,
        provenance: ProvenanceLogger,
    ) -> GovernanceResult:
        src_group = self.trust_groups.get(source_agent, "default")
        tgt_group = self.trust_groups.get(target_agent, "default")

        if src_group != tgt_group:
            return GovernanceResult(
                decision=GovernanceDecision.MODIFY,
                reason=(
                    f"Cross-boundary handoff: {source_agent} ({src_group}) -> "
                    f"{target_agent} ({tgt_group}). Context will be filtered."
                ),
                modified_context={"context_filter": "cross_boundary_summary"},
                risk_score=0.4,
            )

        return GovernanceResult(
            decision=GovernanceDecision.APPROVE,
            reason=f"Same trust group ({src_group})",
            risk_score=0.0,
        )


class CompositePolicy(GovernancePolicy):
    """Composes multiple governance policies with configurable priority.

    Policies are evaluated in order. The first DENY or ESCALATE
    short-circuits. MODIFY results are accumulated. If all policies
    approve, the handoff proceeds.
    """

    def __init__(self, policies: list[GovernancePolicy]) -> None:
        self.policies = policies

    def evaluate(
        self,
        source_agent: str,
        target_agent: str,
        task_description: str,
        state: dict,
        provenance: ProvenanceLogger,
    ) -> GovernanceResult:
        accumulated_modifications: dict = {}
        max_risk = 0.0
        reasons: list[str] = []

        for policy in self.policies:
            result = policy.evaluate(
                source_agent, target_agent, task_description, state, provenance
            )
            max_risk = max(max_risk, result.risk_score)

            if result.decision == GovernanceDecision.DENY:
                result.risk_score = max_risk
                return result

            if result.decision == GovernanceDecision.ESCALATE:
                result.risk_score = max_risk
                return result

            if result.decision == GovernanceDecision.MODIFY:
                if result.modified_context:
                    accumulated_modifications.update(result.modified_context)
                if result.modified_target:
                    # Last modify wins for target redirection
                    accumulated_modifications["redirected_target"] = (
                        result.modified_target
                    )
                reasons.append(result.reason)

            elif result.decision == GovernanceDecision.APPROVE:
                reasons.append(result.reason)

        # If we got here, nothing was denied or escalated
        if accumulated_modifications:
            # Extract any redirected target so it reaches the handoff tool
            redirected = accumulated_modifications.pop("redirected_target", None)
            return GovernanceResult(
                decision=GovernanceDecision.MODIFY,
                reason=" | ".join(reasons),
                modified_target=redirected,
                modified_context=accumulated_modifications or None,
                risk_score=max_risk,
            )

        return GovernanceResult(
            decision=GovernanceDecision.APPROVE,
            reason=" | ".join(reasons),
            risk_score=max_risk,
        )


# =============================================================================
# 3. GOVERNED STATE — Extended SwarmState with governance fields
# =============================================================================


class GovernedSwarmState(TypedDict):
    """Extended swarm state with provenance and governance tracking.

    Adds to the base SwarmState:
      - provenance_chain: serialized audit trail
      - risk_scores: per-agent cumulative risk
      - governance_log: human-readable intervention log
      - active_agent: standard swarm routing field
    """

    messages: Annotated[list[AnyMessage], add_messages]
    active_agent: str | None
    provenance_chain: list[dict]  # Serialized ProvenanceRecords
    risk_scores: dict[str, float]  # agent_name -> cumulative risk
    governance_log: list[str]  # Human-readable intervention entries
    handoff_count: int


# =============================================================================
# 4. GOVERNED HANDOFF TOOL — The core extension
# =============================================================================


def create_governed_handoff_tool(
    *,
    agent_name: str,
    name: str | None = None,
    description: str | None = None,
    provenance_logger: ProvenanceLogger,
    governance_policy: GovernancePolicy,
    context_filter: (
        Callable[[list[AnyMessage], str, str], list[AnyMessage]] | None
    ) = None,
) -> BaseTool:
    """Create a governance-aware handoff tool.

    This replaces langgraph-swarm's ``create_handoff_tool`` with a version
    that passes every handoff through the governance policy and logs
    provenance records for the Byline authorship chain.

    Args:
        agent_name: Target agent to hand off to.
        name: Optional custom tool name.
        description: Optional custom tool description.
        provenance_logger: The shared provenance logger instance.
        governance_policy: The governance policy to evaluate handoffs against.
        context_filter: Optional function to filter/transform messages
            before passing them to the next agent. Receives
            (messages, source_agent, target_agent) and returns
            filtered messages.

    Returns:
        A BaseTool that performs governed handoffs with provenance logging.

    Raises:
        ImportError: If langgraph/langchain-core are not installed.
    """
    if not _HAS_LANGGRAPH:
        raise ImportError(
            "create_governed_handoff_tool requires langgraph and langchain-core. "
            "Install with: pip install swarm-safety[langgraph]"
        )

    tool_name = name or f"transfer_to_{agent_name.lower().replace(' ', '_')}"
    tool_description = (
        description or f"Transfer control to {agent_name} for help"
    )

    @tool(tool_name, description=tool_description)
    def governed_handoff(
        task_description: Annotated[
            str,
            "Detailed description of what the next agent should do, "
            "including all relevant context and reasoning for the handoff.",
        ],
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Execute a governed handoff with provenance logging."""

        # --- Identify the source agent ---
        source_agent = state.get("active_agent", "unknown")
        messages = state.get("messages", [])

        # --- Create the provenance record ---
        record = ProvenanceRecord(
            source_agent=source_agent,
            target_agent=agent_name,
            handoff_reason=task_description,
            task_description=task_description,
            messages_shared_count=len(messages),
        )

        # --- Evaluate governance policy ---
        governance_result = governance_policy.evaluate(
            source_agent=source_agent,
            target_agent=agent_name,
            task_description=task_description,
            state=state,
            provenance=provenance_logger,
        )

        record.governance_decision = governance_result.decision.value
        record.governance_reason = governance_result.reason
        record.risk_score_at_handoff = governance_result.risk_score

        # --- Handle governance decisions ---

        if governance_result.decision == GovernanceDecision.DENY:
            # Log the denied handoff
            record.intervention_applied = "handoff_denied"
            provenance_logger.log(record)

            # Return a ToolMessage explaining the denial -- don't transfer
            denial_message = ToolMessage(
                content=(
                    f"Handoff to {agent_name} was DENIED by governance policy. "
                    f"Reason: {governance_result.reason}. "
                    f"Please try a different approach or handle this task yourself."
                ),
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            # Stay with the current agent
            return Command(
                goto=source_agent,
                graph=Command.PARENT,
                update={
                    "messages": messages + [denial_message],
                    "governance_log": [
                        f"DENIED: {source_agent} -> {agent_name}: "
                        f"{governance_result.reason}"
                    ],
                    "risk_scores": {agent_name: governance_result.risk_score},
                    "provenance_chain": [record.to_dict()],
                },
            )

        if governance_result.decision == GovernanceDecision.ESCALATE:
            # Log and escalate -- could trigger human-in-the-loop
            record.intervention_applied = "escalated_to_human"
            provenance_logger.log(record)

            escalation_message = ToolMessage(
                content=(
                    f"Handoff to {agent_name} has been ESCALATED for human review. "
                    f"Reason: {governance_result.reason}. "
                    f"The conversation is paused pending human approval."
                ),
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            # Return to current agent with escalation notice
            return Command(
                goto=source_agent,
                graph=Command.PARENT,
                update={
                    "messages": messages + [escalation_message],
                    "governance_log": [
                        f"ESCALATED: {source_agent} -> {agent_name}: "
                        f"{governance_result.reason}"
                    ],
                    "risk_scores": {agent_name: governance_result.risk_score},
                    "provenance_chain": [record.to_dict()],
                },
            )

        # --- APPROVE or MODIFY: proceed with handoff ---

        actual_target = agent_name
        if (
            governance_result.decision == GovernanceDecision.MODIFY
            and governance_result.modified_target
        ):
            actual_target = governance_result.modified_target
            record.intervention_applied = f"redirected_to_{actual_target}"
            record.target_agent = actual_target

        # Apply context filtering if configured
        shared_messages = messages
        if context_filter is not None:
            shared_messages = context_filter(
                messages, source_agent, actual_target
            )
            record.context_filter_applied = "custom"
            record.messages_shared_count = len(shared_messages)
        elif (
            governance_result.decision == GovernanceDecision.MODIFY
            and governance_result.modified_context
            and governance_result.modified_context.get("context_filter")
            == "cross_boundary_summary"
        ):
            # Apply cross-boundary summary: only share last 3 messages + a summary
            shared_messages = _cross_boundary_filter(messages)
            record.context_filter_applied = "cross_boundary_summary"
            record.messages_shared_count = len(shared_messages)

        # Log the approved/modified handoff
        if governance_result.decision == GovernanceDecision.APPROVE:
            record.intervention_applied = "none"
        provenance_logger.log(record)

        # Build the handoff ToolMessage
        handoff_message = ToolMessage(
            content=(
                f"Successfully transferred to {actual_target}. "
                f"[Provenance: {record.record_id}] "
                f"[Risk: {governance_result.risk_score:.2f}]"
            ),
            name=tool_name,
            tool_call_id=tool_call_id,
        )

        return Command(
            goto=actual_target,
            graph=Command.PARENT,
            update={
                "messages": shared_messages + [handoff_message],
                "active_agent": actual_target,
                "governance_log": [
                    f"{governance_result.decision.value.upper()}: "
                    f"{source_agent} -> {actual_target}: "
                    f"{governance_result.reason}"
                ],
                "risk_scores": {actual_target: governance_result.risk_score},
                "provenance_chain": [record.to_dict()],
                "handoff_count": state.get("handoff_count", 0) + 1,
            },
        )

    return governed_handoff


def _cross_boundary_filter(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Filter messages for cross-trust-boundary handoffs.

    Only shares the last 3 messages to limit information leakage
    across trust boundaries. In production, you'd want a more
    sophisticated summarization approach.
    """
    return messages[-3:] if len(messages) > 3 else messages


# =============================================================================
# 5. GOVERNED SWARM CONSTRUCTION — Putting it all together
# =============================================================================


def create_governed_swarm(
    agents: list[Pregel],
    *,
    default_active_agent: str,
    governance_policy: GovernancePolicy | None = None,
    provenance_logger: ProvenanceLogger | None = None,
    state_schema: type = GovernedSwarmState,
) -> tuple[StateGraph, ProvenanceLogger]:
    """Create a governance-aware multi-agent swarm.

    This is a drop-in replacement for langgraph-swarm's ``create_swarm()``
    that adds governance and provenance capabilities.

    Args:
        agents: List of compiled LangGraph agents (Pregel instances).
        default_active_agent: Name of the starting agent.
        governance_policy: The governance policy to apply. Defaults to
            a composite of CycleDetection + RateLimit policies.
        provenance_logger: Shared provenance logger. Created if not provided.
        state_schema: State schema class. Defaults to GovernedSwarmState.

    Returns:
        Tuple of (StateGraph, ProvenanceLogger). The StateGraph needs
        to be compiled with a checkpointer before use.

    Example::

        from swarm.bridges.langgraph_swarm import (
            create_governed_swarm,
            create_governed_handoff_tool,
            CompositePolicy,
            CycleDetectionPolicy,
            RateLimitPolicy,
            InformationBoundaryPolicy,
            ProvenanceLogger,
        )
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import InMemorySaver

        # Shared governance infrastructure
        logger = ProvenanceLogger()
        policy = CompositePolicy([
            CycleDetectionPolicy(max_cycles=3),
            RateLimitPolicy(max_handoffs=15),
            InformationBoundaryPolicy(trust_groups={
                "researcher": "research",
                "writer": "content",
                "reviewer": "research",
            }),
        ])

        # Create agents with governed handoff tools
        researcher = create_react_agent(
            "openai:gpt-4o",
            tools=[
                search_tool,
                create_governed_handoff_tool(
                    agent_name="writer",
                    description="Hand off findings to the writer",
                    provenance_logger=logger,
                    governance_policy=policy,
                ),
            ],
            prompt="You are a research specialist...",
            name="researcher",
        )

        # Build the governed swarm
        workflow, provenance = create_governed_swarm(
            [researcher, writer, reviewer],
            default_active_agent="researcher",
            governance_policy=policy,
            provenance_logger=logger,
        )

        app = workflow.compile(checkpointer=InMemorySaver())

    Raises:
        ImportError: If langgraph/langchain-core are not installed.
    """
    if not _HAS_LANGGRAPH:
        raise ImportError(
            "create_governed_swarm requires langgraph and langchain-core. "
            "Install with: pip install swarm-safety[langgraph]"
        )

    if provenance_logger is None:
        provenance_logger = ProvenanceLogger()

    if governance_policy is None:
        governance_policy = CompositePolicy([
            CycleDetectionPolicy(max_cycles=3, window=10),
            RateLimitPolicy(max_handoffs=20, window_seconds=300.0),
        ])

    # Validate agents
    agent_names: list[str] = []
    for agent in agents:
        if not hasattr(agent, "name") or agent.name is None:
            raise ValueError(
                "All agents must have a 'name' attribute. "
                "Use the 'name' parameter in create_react_agent()."
            )
        agent_names.append(agent.name)

    if not agent_names:
        raise ValueError("Must provide at least one agent.")

    if default_active_agent not in agent_names:
        raise ValueError(
            f"default_active_agent '{default_active_agent}' "
            f"not found in agents: {agent_names}"
        )

    # Build the state graph
    builder = StateGraph(state_schema)

    # Extract handoff destinations from each agent's tools
    for agent in agents:
        destinations = _get_governed_destinations(agent)
        builder.add_node(
            agent.name, agent, destinations=tuple(destinations)
        )

    # Add the active agent router (same mechanism as base langgraph-swarm)
    def route_to_active_agent(state: dict) -> str:
        return state.get("active_agent") or default_active_agent

    path_map = {name: name for name in agent_names}
    builder.add_conditional_edges(START, route_to_active_agent, path_map)

    return builder, provenance_logger


def _get_governed_destinations(agent: Pregel) -> list[str]:
    """Extract handoff destination agent names from an agent's tools.

    Looks for tools whose names start with ``transfer_to_`` -- the
    naming convention used by both ``create_handoff_tool`` and
    ``create_governed_handoff_tool``.
    """
    destinations: list[str] = []

    # Access the agent's tools through its nodes
    if hasattr(agent, "nodes"):
        for _node_name, node in agent.nodes.items():
            if hasattr(node, "tools"):
                for t in node.tools:
                    tool_name = getattr(t, "name", "")
                    if tool_name.startswith("transfer_to_"):
                        dest = tool_name.replace("transfer_to_", "").replace(
                            "_", " "
                        )
                        destinations.append(dest)

    return destinations


# =============================================================================
# 6. RISK ANALYSIS UTILITIES — Post-hoc analysis of swarm behavior
# =============================================================================


def analyze_swarm_run(provenance: ProvenanceLogger) -> dict[str, Any]:
    """Analyze a completed swarm run for risk indicators.

    Returns a summary dict with:
      - total_handoffs: count of all handoffs
      - denied_handoffs: count of governance denials
      - escalated_handoffs: count of governance escalations
      - max_chain_depth: deepest provenance chain
      - cycle_pairs: detected circular handoff patterns
      - agent_risk_scores: per-agent average risk
      - risk_level: overall risk assessment ("low", "medium", "high", "critical")
    """
    records = provenance.records

    if not records:
        return {
            "total_handoffs": 0,
            "risk_level": "low",
            "message": "No handoffs recorded",
        }

    denied = [r for r in records if r.governance_decision == "denied"]
    escalated = [r for r in records if r.governance_decision == "escalated"]
    max_depth = max(r.chain_depth for r in records)
    cycle_pairs = provenance.detect_cycles(window=len(records))

    # Per-agent risk aggregation
    agent_risks: dict[str, list[float]] = {}
    for r in records:
        for agent in [r.source_agent, r.target_agent]:
            if agent not in agent_risks:
                agent_risks[agent] = []
            agent_risks[agent].append(r.risk_score_at_handoff)

    agent_avg_risk = {
        agent: sum(scores) / len(scores)
        for agent, scores in agent_risks.items()
        if scores
    }

    # Overall risk assessment
    max_risk = max(r.risk_score_at_handoff for r in records)
    denial_rate = len(denied) / len(records)

    if max_risk > 0.8 or denial_rate > 0.3 or len(escalated) > 0:
        risk_level = "critical" if len(escalated) > 2 else "high"
    elif max_risk > 0.5 or denial_rate > 0.1 or len(cycle_pairs) > 0:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "total_handoffs": len(records),
        "approved_handoffs": len(
            [r for r in records if r.governance_decision == "approved"]
        ),
        "denied_handoffs": len(denied),
        "escalated_handoffs": len(escalated),
        "modified_handoffs": len(
            [r for r in records if r.governance_decision == "modified"]
        ),
        "max_chain_depth": max_depth,
        "cycle_pairs": cycle_pairs,
        "agent_risk_scores": agent_avg_risk,
        "risk_level": risk_level,
        "denial_rate": round(denial_rate, 3),
        "provenance_records": len(records),
    }


# =============================================================================
# 7. CONVENIENCE — Quick setup for common governance configurations
# =============================================================================


def default_governance_stack(
    trust_groups: dict[str, str] | None = None,
    max_cycles: int = 3,
    max_handoffs: int = 20,
    rate_window_seconds: float = 300.0,
) -> tuple[CompositePolicy, ProvenanceLogger]:
    """Create a standard governance stack with sensible defaults.

    Returns a (policy, logger) tuple ready to use with
    ``create_governed_handoff_tool`` and ``create_governed_swarm``.

    Args:
        trust_groups: Optional agent -> trust_group mapping for
            information boundary enforcement.
        max_cycles: Max repeated handoffs between a pair before denial.
        max_handoffs: Max total handoffs in the rate window.
        rate_window_seconds: Time window for rate limiting.

    Returns:
        Tuple of (CompositePolicy, ProvenanceLogger).
    """
    policies: list[GovernancePolicy] = [
        CycleDetectionPolicy(max_cycles=max_cycles),
        RateLimitPolicy(
            max_handoffs=max_handoffs, window_seconds=rate_window_seconds
        ),
    ]

    if trust_groups:
        policies.append(InformationBoundaryPolicy(trust_groups=trust_groups))

    return CompositePolicy(policies), ProvenanceLogger()
