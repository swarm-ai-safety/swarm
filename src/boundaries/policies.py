"""Boundary crossing policies for sandbox-external interactions.

This module implements policies that control what information can
cross the sandbox boundary, including rate limiting, content filtering,
and sensitivity-based access control.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import re


@dataclass
class CrossingDecision:
    """Result of a policy evaluation."""

    allowed: bool
    reason: str
    policy_name: str
    risk_score: float = 0.0  # 0 = no risk, 1 = maximum risk
    conditions: Dict[str, Any] = field(default_factory=dict)  # any conditions attached

    @classmethod
    def allow(cls, policy_name: str, reason: str = "Allowed by policy") -> "CrossingDecision":
        """Create an allow decision."""
        return cls(allowed=True, reason=reason, policy_name=policy_name)

    @classmethod
    def deny(
        cls,
        policy_name: str,
        reason: str,
        risk_score: float = 0.5,
    ) -> "CrossingDecision":
        """Create a deny decision."""
        return cls(
            allowed=False,
            reason=reason,
            policy_name=policy_name,
            risk_score=risk_score,
        )


class BoundaryPolicy(ABC):
    """Abstract base class for boundary crossing policies."""

    def __init__(self, name: str, enabled: bool = True):
        """Initialize the policy.

        Args:
            name: Human-readable policy name
            enabled: Whether the policy is active
        """
        self.name = name
        self.enabled = enabled
        self.evaluation_count = 0
        self.block_count = 0

    @abstractmethod
    def evaluate(
        self,
        agent_id: str,
        direction: str,
        flow_type: str,
        content: Any,
        metadata: Dict[str, Any],
    ) -> CrossingDecision:
        """Evaluate whether a crossing should be allowed.

        Args:
            agent_id: The agent requesting the crossing
            direction: "inbound" or "outbound"
            flow_type: Type of information flow
            content: The content being transferred
            metadata: Additional context

        Returns:
            CrossingDecision indicating allow/deny and reason
        """
        pass

    def _record_evaluation(self, decision: CrossingDecision) -> None:
        """Record evaluation statistics."""
        self.evaluation_count += 1
        if not decision.allowed:
            self.block_count += 1


class RateLimitPolicy(BoundaryPolicy):
    """Rate limiting policy for boundary crossings."""

    def __init__(
        self,
        name: str = "RateLimitPolicy",
        max_crossings_per_minute: int = 60,
        max_bytes_per_minute: int = 1_000_000,
        per_agent: bool = True,
        enabled: bool = True,
    ):
        """Initialize rate limit policy.

        Args:
            name: Policy name
            max_crossings_per_minute: Maximum crossings per minute
            max_bytes_per_minute: Maximum bytes per minute
            per_agent: Whether limits are per-agent or global
            enabled: Whether policy is active
        """
        super().__init__(name, enabled)
        self.max_crossings_per_minute = max_crossings_per_minute
        self.max_bytes_per_minute = max_bytes_per_minute
        self.per_agent = per_agent

        # Tracking
        self._crossing_times: Dict[str, List[datetime]] = {}
        self._byte_counts: Dict[str, List[tuple]] = {}  # (timestamp, bytes)

    def evaluate(
        self,
        agent_id: str,
        direction: str,
        flow_type: str,
        content: Any,
        metadata: Dict[str, Any],
    ) -> CrossingDecision:
        """Evaluate rate limit."""
        if not self.enabled:
            return CrossingDecision.allow(self.name, "Policy disabled")

        key = agent_id if self.per_agent else "global"
        now = datetime.now()
        window_start = now - timedelta(minutes=1)

        # Clean old entries
        self._cleanup_old_entries(key, window_start)

        # Check crossing count
        crossing_times = self._crossing_times.get(key, [])
        if len(crossing_times) >= self.max_crossings_per_minute:
            decision = CrossingDecision.deny(
                self.name,
                f"Rate limit exceeded: {len(crossing_times)}/{self.max_crossings_per_minute} crossings/min",
                risk_score=0.3,
            )
            self._record_evaluation(decision)
            return decision

        # Check byte count
        content_bytes = len(str(content).encode())
        byte_entries = self._byte_counts.get(key, [])
        total_bytes = sum(b for _, b in byte_entries)
        if total_bytes + content_bytes > self.max_bytes_per_minute:
            decision = CrossingDecision.deny(
                self.name,
                f"Bandwidth limit exceeded: {total_bytes + content_bytes}/{self.max_bytes_per_minute} bytes/min",
                risk_score=0.3,
            )
            self._record_evaluation(decision)
            return decision

        # Record this crossing
        if key not in self._crossing_times:
            self._crossing_times[key] = []
        self._crossing_times[key].append(now)

        if key not in self._byte_counts:
            self._byte_counts[key] = []
        self._byte_counts[key].append((now, content_bytes))

        decision = CrossingDecision.allow(self.name)
        self._record_evaluation(decision)
        return decision

    def _cleanup_old_entries(self, key: str, cutoff: datetime) -> None:
        """Remove entries older than cutoff."""
        if key in self._crossing_times:
            self._crossing_times[key] = [t for t in self._crossing_times[key] if t > cutoff]
        if key in self._byte_counts:
            self._byte_counts[key] = [(t, b) for t, b in self._byte_counts[key] if t > cutoff]


class ContentFilterPolicy(BoundaryPolicy):
    """Content-based filtering policy."""

    def __init__(
        self,
        name: str = "ContentFilterPolicy",
        blocked_patterns: Optional[List[str]] = None,
        blocked_keywords: Optional[Set[str]] = None,
        max_content_length: int = 100_000,
        enabled: bool = True,
    ):
        """Initialize content filter policy.

        Args:
            name: Policy name
            blocked_patterns: Regex patterns to block
            blocked_keywords: Keywords to block
            max_content_length: Maximum content length in bytes
            enabled: Whether policy is active
        """
        super().__init__(name, enabled)
        self.blocked_patterns = [re.compile(p, re.IGNORECASE) for p in (blocked_patterns or [])]
        self.blocked_keywords = blocked_keywords or set()
        self.max_content_length = max_content_length

    def evaluate(
        self,
        agent_id: str,
        direction: str,
        flow_type: str,
        content: Any,
        metadata: Dict[str, Any],
    ) -> CrossingDecision:
        """Evaluate content filter."""
        if not self.enabled:
            return CrossingDecision.allow(self.name, "Policy disabled")

        content_str = str(content)

        # Check length
        if len(content_str.encode()) > self.max_content_length:
            decision = CrossingDecision.deny(
                self.name,
                f"Content exceeds maximum length ({len(content_str.encode())} > {self.max_content_length})",
                risk_score=0.4,
            )
            self._record_evaluation(decision)
            return decision

        # Check patterns
        for pattern in self.blocked_patterns:
            if pattern.search(content_str):
                decision = CrossingDecision.deny(
                    self.name,
                    f"Content matches blocked pattern: {pattern.pattern}",
                    risk_score=0.7,
                )
                self._record_evaluation(decision)
                return decision

        # Check keywords
        content_lower = content_str.lower()
        for keyword in self.blocked_keywords:
            if keyword.lower() in content_lower:
                decision = CrossingDecision.deny(
                    self.name,
                    f"Content contains blocked keyword: {keyword}",
                    risk_score=0.6,
                )
                self._record_evaluation(decision)
                return decision

        decision = CrossingDecision.allow(self.name)
        self._record_evaluation(decision)
        return decision


class SensitivityPolicy(BoundaryPolicy):
    """Sensitivity-based access control policy."""

    def __init__(
        self,
        name: str = "SensitivityPolicy",
        max_outbound_sensitivity: float = 0.5,
        require_approval_above: float = 0.7,
        agent_clearance_levels: Optional[Dict[str, float]] = None,
        enabled: bool = True,
    ):
        """Initialize sensitivity policy.

        Args:
            name: Policy name
            max_outbound_sensitivity: Maximum sensitivity for outbound data
            require_approval_above: Sensitivity level requiring approval
            agent_clearance_levels: Per-agent clearance levels
            enabled: Whether policy is active
        """
        super().__init__(name, enabled)
        self.max_outbound_sensitivity = max_outbound_sensitivity
        self.require_approval_above = require_approval_above
        self.agent_clearance_levels = agent_clearance_levels or {}
        self.default_clearance = 0.3

        # Track pending approvals
        self.pending_approvals: List[Dict[str, Any]] = []

    def evaluate(
        self,
        agent_id: str,
        direction: str,
        flow_type: str,
        content: Any,
        metadata: Dict[str, Any],
    ) -> CrossingDecision:
        """Evaluate sensitivity policy."""
        if not self.enabled:
            return CrossingDecision.allow(self.name, "Policy disabled")

        sensitivity = metadata.get("sensitivity", 0.0)
        agent_clearance = self.agent_clearance_levels.get(agent_id, self.default_clearance)

        # Check outbound sensitivity
        if direction == "outbound" and sensitivity > self.max_outbound_sensitivity:
            decision = CrossingDecision.deny(
                self.name,
                f"Outbound sensitivity {sensitivity:.2f} exceeds maximum {self.max_outbound_sensitivity:.2f}",
                risk_score=sensitivity,
            )
            self._record_evaluation(decision)
            return decision

        # Check agent clearance
        if sensitivity > agent_clearance:
            decision = CrossingDecision.deny(
                self.name,
                f"Agent clearance {agent_clearance:.2f} insufficient for sensitivity {sensitivity:.2f}",
                risk_score=sensitivity - agent_clearance,
            )
            self._record_evaluation(decision)
            return decision

        # Check if approval required
        if sensitivity > self.require_approval_above:
            self.pending_approvals.append({
                "agent_id": agent_id,
                "direction": direction,
                "sensitivity": sensitivity,
                "timestamp": datetime.now(),
            })
            decision = CrossingDecision(
                allowed=True,
                reason="Allowed with logging (high sensitivity)",
                policy_name=self.name,
                risk_score=sensitivity,
                conditions={"logged": True, "requires_review": True},
            )
            self._record_evaluation(decision)
            return decision

        decision = CrossingDecision.allow(self.name)
        self._record_evaluation(decision)
        return decision

    def set_clearance(self, agent_id: str, clearance: float) -> None:
        """Set clearance level for an agent."""
        self.agent_clearance_levels[agent_id] = max(0.0, min(1.0, clearance))


class CompositePolicy(BoundaryPolicy):
    """Combines multiple policies with configurable logic."""

    def __init__(
        self,
        name: str = "CompositePolicy",
        policies: Optional[List[BoundaryPolicy]] = None,
        require_all: bool = True,
        enabled: bool = True,
    ):
        """Initialize composite policy.

        Args:
            name: Policy name
            policies: List of policies to combine
            require_all: If True, all must allow; if False, any can allow
            enabled: Whether policy is active
        """
        super().__init__(name, enabled)
        self.policies = policies or []
        self.require_all = require_all

    def add_policy(self, policy: BoundaryPolicy) -> None:
        """Add a policy to the composite."""
        self.policies.append(policy)

    def evaluate(
        self,
        agent_id: str,
        direction: str,
        flow_type: str,
        content: Any,
        metadata: Dict[str, Any],
    ) -> CrossingDecision:
        """Evaluate all policies."""
        if not self.enabled:
            return CrossingDecision.allow(self.name, "Policy disabled")

        if not self.policies:
            return CrossingDecision.allow(self.name, "No policies configured")

        decisions = []
        for policy in self.policies:
            decision = policy.evaluate(agent_id, direction, flow_type, content, metadata)
            decisions.append(decision)

        allowed_count = sum(1 for d in decisions if d.allowed)
        denied = [d for d in decisions if not d.allowed]

        if self.require_all:
            # All must allow
            if denied:
                # Combine denial reasons
                reasons = [f"{d.policy_name}: {d.reason}" for d in denied]
                max_risk = max(d.risk_score for d in denied)
                decision = CrossingDecision.deny(
                    self.name,
                    f"Denied by {len(denied)} policies: {'; '.join(reasons)}",
                    risk_score=max_risk,
                )
            else:
                decision = CrossingDecision.allow(self.name, f"Allowed by all {len(decisions)} policies")
        else:
            # Any can allow
            if allowed_count > 0:
                decision = CrossingDecision.allow(self.name, f"Allowed by {allowed_count}/{len(decisions)} policies")
            else:
                reasons = [f"{d.policy_name}: {d.reason}" for d in denied]
                max_risk = max(d.risk_score for d in denied)
                decision = CrossingDecision.deny(
                    self.name,
                    f"Denied by all {len(denied)} policies: {'; '.join(reasons)}",
                    risk_score=max_risk,
                )

        self._record_evaluation(decision)
        return decision


class PolicyEngine:
    """Central engine for managing and enforcing boundary policies."""

    def __init__(self):
        """Initialize the policy engine."""
        self.policies: List[BoundaryPolicy] = []
        self.evaluation_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    def add_policy(self, policy: BoundaryPolicy) -> None:
        """Add a policy to the engine."""
        self.policies.append(policy)

    def remove_policy(self, policy_name: str) -> bool:
        """Remove a policy by name."""
        for i, policy in enumerate(self.policies):
            if policy.name == policy_name:
                self.policies.pop(i)
                return True
        return False

    def evaluate(
        self,
        agent_id: str,
        direction: str,
        flow_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CrossingDecision:
        """Evaluate all policies for a crossing request.

        All policies must allow for the crossing to be permitted.
        """
        metadata = metadata or {}

        if not self.policies:
            return CrossingDecision.allow("PolicyEngine", "No policies configured")

        decisions = []
        for policy in self.policies:
            if policy.enabled:
                decision = policy.evaluate(agent_id, direction, flow_type, content, metadata)
                decisions.append(decision)

                # Short-circuit on denial
                if not decision.allowed:
                    self._record_history(agent_id, direction, flow_type, decision)
                    return decision

        # All allowed
        final_decision = CrossingDecision.allow(
            "PolicyEngine",
            f"Allowed by all {len(decisions)} active policies",
        )
        self._record_history(agent_id, direction, flow_type, final_decision)
        return final_decision

    def _record_history(
        self,
        agent_id: str,
        direction: str,
        flow_type: str,
        decision: CrossingDecision,
    ) -> None:
        """Record evaluation to history."""
        self.evaluation_history.append({
            "timestamp": datetime.now(),
            "agent_id": agent_id,
            "direction": direction,
            "flow_type": flow_type,
            "allowed": decision.allowed,
            "policy": decision.policy_name,
            "reason": decision.reason,
            "risk_score": decision.risk_score,
        })

        if len(self.evaluation_history) > self.max_history:
            self.evaluation_history = self.evaluation_history[-self.max_history:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get policy evaluation statistics."""
        stats = {
            "total_policies": len(self.policies),
            "active_policies": sum(1 for p in self.policies if p.enabled),
            "total_evaluations": len(self.evaluation_history),
            "policy_stats": {},
        }

        for policy in self.policies:
            stats["policy_stats"][policy.name] = {
                "enabled": policy.enabled,
                "evaluations": policy.evaluation_count,
                "blocks": policy.block_count,
                "block_rate": policy.block_count / max(policy.evaluation_count, 1),
            }

        return stats

    def create_default_policies(self) -> "PolicyEngine":
        """Create an engine with default policies."""
        # Rate limiting
        self.add_policy(RateLimitPolicy(
            max_crossings_per_minute=100,
            max_bytes_per_minute=10_000_000,
        ))

        # Content filtering
        self.add_policy(ContentFilterPolicy(
            blocked_patterns=[
                r"password\s*[:=]\s*\S+",  # Password patterns
                r"api[_-]?key\s*[:=]\s*\S+",  # API key patterns
                r"secret\s*[:=]\s*\S+",  # Secret patterns
            ],
            blocked_keywords={"rm -rf", "DROP TABLE", "exec(", "eval("},
            max_content_length=1_000_000,
        ))

        # Sensitivity policy
        self.add_policy(SensitivityPolicy(
            max_outbound_sensitivity=0.6,
            require_approval_above=0.8,
        ))

        return self
