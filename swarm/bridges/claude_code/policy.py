"""Governance policy adapter for Claude Code bridge.

Maps SWARM's GovernanceConfig to concrete plan/permission approval
decisions. This is the policy layer that answers the controller's
plan:approval_request and permission:request events.
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

from swarm.bridges.claude_code.events import (
    PermissionRequest,
    PlanApprovalRequest,
)
from swarm.governance.config import GovernanceConfig

logger = logging.getLogger(__name__)


class PolicyDecision(Enum):
    """Outcome of a governance policy evaluation."""

    APPROVE = "approve"
    DENY = "deny"
    REQUIRE_STAKE = "require_stake"


# Tools classified by risk level
HIGH_RISK_TOOLS: Set[str] = {"Bash", "Write", "NotebookEdit"}
MEDIUM_RISK_TOOLS: Set[str] = {"Edit", "WebFetch"}
LOW_RISK_TOOLS: Set[str] = {"Read", "Glob", "Grep", "WebSearch"}


@dataclass
class PolicyResult:
    """Result of a policy evaluation with metadata."""

    decision: PolicyDecision
    reason: str = ""
    cost: float = 0.0  # governance cost to charge the agent
    stake_required: float = 0.0


@dataclass
class AgentBudget:
    """Tracks per-agent resource budgets for governance."""

    agent_id: str = ""
    max_tool_calls: int = 100
    tool_calls_used: int = 0
    max_cost_usd: float = 10.0
    cost_used_usd: float = 0.0
    denied_permissions: int = 0
    approved_permissions: int = 0
    plans_approved: int = 0
    plans_denied: int = 0


class GovernancePolicy:
    """Maps SWARM governance configuration to Claude Code policy decisions.

    Handles:
    - Plan approval based on risk assessment and circuit breakers
    - Tool permission gating based on allowlists and budgets
    - Per-agent budget tracking (tool calls, cost)
    - Staking requirements for high-risk operations

    The policy is stateful: it tracks per-agent budgets across
    the simulation to enforce rate limits and circuit breakers.
    """

    def __init__(
        self,
        governance_config: Optional[GovernanceConfig] = None,
        tool_allowlist: Optional[Dict[str, List[str]]] = None,
        per_agent_budgets: Optional[Dict[str, AgentBudget]] = None,
    ):
        """Initialize the governance policy.

        Args:
            governance_config: SWARM governance configuration
            tool_allowlist: Per-agent tool allowlists
                (e.g., {"agent_1": ["Read", "Grep"]})
            per_agent_budgets: Pre-configured agent budgets
        """
        self.config = governance_config or GovernanceConfig()
        self.tool_allowlist = tool_allowlist or {}
        self._budgets: Dict[str, AgentBudget] = per_agent_budgets or {}
        self._lock = threading.Lock()

    def _get_budget(self, agent_id: str) -> AgentBudget:
        """Get or create a budget tracker for an agent."""
        if agent_id not in self._budgets:
            self._budgets[agent_id] = AgentBudget(agent_id=agent_id)
        return self._budgets[agent_id]

    def evaluate_plan(
        self,
        request: PlanApprovalRequest,
        agent_reputation: float = 0.0,
    ) -> PolicyResult:
        """Evaluate a plan approval request.

        Decision factors:
        1. Circuit breaker: deny if agent is frozen
        2. Risk flags: deny if high-risk flags + low reputation
        3. Budget: deny if agent has exhausted tool call budget
        4. Staking: require stake for high-risk plans

        Args:
            request: The plan approval request from the controller
            agent_reputation: The agent's current reputation score

        Returns:
            PolicyResult with decision and metadata
        """
        with self._lock:
            return self._evaluate_plan_unlocked(request, agent_reputation)

    def _evaluate_plan_unlocked(
        self,
        request: PlanApprovalRequest,
        agent_reputation: float,
    ) -> PolicyResult:
        budget = self._get_budget(request.agent_id)

        # Circuit breaker check
        if self.config.circuit_breaker_enabled:
            denial_rate = budget.denied_permissions / max(
                budget.approved_permissions, 1
            )
            if denial_rate > self.config.freeze_threshold_toxicity:
                budget.plans_denied += 1
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    reason=(
                        f"Circuit breaker: agent denial rate "
                        f"{denial_rate:.2f} exceeds threshold "
                        f"{self.config.freeze_threshold_toxicity}"
                    ),
                )

        # Budget check
        remaining = budget.max_tool_calls - budget.tool_calls_used
        if request.estimated_tool_calls > remaining:
            budget.plans_denied += 1
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"Budget exhausted: plan needs "
                    f"{request.estimated_tool_calls} tool calls, "
                    f"only {remaining} remaining"
                ),
            )

        # Risk assessment
        has_high_risk = any(
            flag in ("destructive", "network", "credential", "system")
            for flag in request.risk_flags
        )
        if has_high_risk and agent_reputation < 0.0:
            budget.plans_denied += 1
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"High-risk plan from low-reputation agent "
                    f"(rep={agent_reputation:.2f}): {request.risk_flags}"
                ),
            )

        # Staking requirement for risky plans
        if self.config.staking_enabled and has_high_risk:
            stake = self.config.min_stake_to_participate
            budget.plans_approved += 1
            return PolicyResult(
                decision=PolicyDecision.REQUIRE_STAKE,
                reason="High-risk plan requires stake",
                stake_required=stake,
            )

        # Governance cost (transaction tax on plans)
        cost = 0.0
        if self.config.transaction_tax_rate > 0:
            cost = self.config.transaction_tax_rate * request.estimated_tool_calls

        budget.plans_approved += 1
        return PolicyResult(
            decision=PolicyDecision.APPROVE,
            reason="Plan approved",
            cost=cost,
        )

    def evaluate_permission(
        self,
        request: PermissionRequest,
        agent_reputation: float = 0.0,
    ) -> PolicyResult:
        """Evaluate a tool permission request.

        Decision factors:
        1. Allowlist: deny if tool not in agent's allowlist
        2. Budget: deny if tool call budget exhausted
        3. Risk-based gating: high-risk tools need higher reputation
        4. Audit: flag for random audit

        Args:
            request: The permission request from the controller
            agent_reputation: The agent's current reputation score

        Returns:
            PolicyResult with decision and metadata
        """
        with self._lock:
            return self._evaluate_permission_unlocked(request, agent_reputation)

    def _evaluate_permission_unlocked(
        self,
        request: PermissionRequest,
        agent_reputation: float,
    ) -> PolicyResult:
        budget = self._get_budget(request.agent_id)

        # Allowlist check
        agent_allowlist = self.tool_allowlist.get(request.agent_id)
        if agent_allowlist is not None and request.tool_name not in agent_allowlist:
            budget.denied_permissions += 1
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"Tool '{request.tool_name}' not in agent's "
                    f"allowlist: {agent_allowlist}"
                ),
            )

        # Budget check
        if budget.tool_calls_used >= budget.max_tool_calls:
            budget.denied_permissions += 1
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"Tool call budget exhausted: "
                    f"{budget.tool_calls_used}/{budget.max_tool_calls}"
                ),
            )

        # Risk-based gating
        if request.tool_name in HIGH_RISK_TOOLS and agent_reputation < -0.5:
            budget.denied_permissions += 1
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"High-risk tool '{request.tool_name}' denied for "
                    f"low-reputation agent (rep={agent_reputation:.2f})"
                ),
            )

        # Staking for high-risk tools
        if self.config.staking_enabled and request.tool_name in HIGH_RISK_TOOLS:
            budget.approved_permissions += 1
            budget.tool_calls_used += 1
            return PolicyResult(
                decision=PolicyDecision.REQUIRE_STAKE,
                reason=f"High-risk tool '{request.tool_name}' requires stake",
                stake_required=self.config.min_stake_to_participate,
            )

        # Governance cost
        cost = 0.0
        if self.config.transaction_tax_rate > 0:
            cost = self.config.transaction_tax_rate

        budget.approved_permissions += 1
        budget.tool_calls_used += 1
        return PolicyResult(
            decision=PolicyDecision.APPROVE,
            reason=f"Permission granted for '{request.tool_name}'",
            cost=cost,
        )

    def get_agent_budget(self, agent_id: str) -> AgentBudget:
        """Get the current budget state for an agent."""
        return self._get_budget(agent_id)

    def set_agent_budget(
        self,
        agent_id: str,
        max_tool_calls: int = 100,
        max_cost_usd: float = 10.0,
    ) -> None:
        """Configure budget limits for an agent.

        Args:
            agent_id: The agent to configure
            max_tool_calls: Maximum tool invocations allowed
            max_cost_usd: Maximum dollar cost allowed
        """
        with self._lock:
            budget = self._get_budget(agent_id)
            budget.max_tool_calls = max_tool_calls
            budget.max_cost_usd = max_cost_usd

    def reset_budgets(self) -> None:
        """Reset all agent budgets (e.g., at epoch boundary)."""
        with self._lock:
            for budget in self._budgets.values():
                budget.tool_calls_used = 0
                budget.cost_used_usd = 0.0
                budget.denied_permissions = 0
                budget.approved_permissions = 0
                budget.plans_approved = 0
                budget.plans_denied = 0
