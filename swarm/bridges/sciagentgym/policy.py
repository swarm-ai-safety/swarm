"""Governance policy engine for SciAgentGym bridge."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PolicyDecision(Enum):
    """Policy decision outcomes."""

    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"


@dataclass
class PolicyResult:
    """Result of a policy evaluation.
    
    Attributes:
        decision: The policy decision.
        reason: Human-readable explanation.
        governance_cost: Cost incurred by this policy check.
    """

    decision: PolicyDecision
    reason: str
    governance_cost: float = 0.0


class SciAgentGymPolicy:
    """Governance policy engine for SciAgentGym bridge.
    
    Enforces:
    - Tool safety gates (minimum safety score)
    - Workflow circuit breakers (max consecutive failures)
    - Cost budgets (token limits)
    - Result quality thresholds
    """

    def __init__(self, config) -> None:  # config: SciAgentGymConfig
        """Initialize the policy engine.
        
        Args:
            config: Bridge configuration.
        """
        self._config = config
        self._consecutive_failures = 0
        self._total_tokens_used = 0
        self._circuit_breaker_active = False

    def evaluate_tool_safety(self, safety_score: float) -> PolicyResult:
        """Evaluate whether a tool call meets safety requirements.
        
        Args:
            safety_score: Computed safety score (probability benign).
            
        Returns:
            Policy result with decision.
        """
        if not self._config.tool_safety_gate_enabled:
            return PolicyResult(
                decision=PolicyDecision.ALLOW,
                reason="Safety gate disabled",
            )

        if safety_score < self._config.min_tool_safety_score:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Safety score {safety_score:.3f} below threshold "
                f"{self._config.min_tool_safety_score:.3f}",
                governance_cost=0.1,
            )

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason=f"Safety score {safety_score:.3f} passes threshold",
        )

    def evaluate_workflow_failure(self, failed: bool) -> PolicyResult:
        """Evaluate workflow step failure for circuit breaker.
        
        Args:
            failed: Whether the workflow step failed.
            
        Returns:
            Policy result with decision.
        """
        if failed:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        if (
            self._consecutive_failures
            >= self._config.workflow_circuit_breaker_max_failures
        ):
            self._circuit_breaker_active = True
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Circuit breaker triggered: {self._consecutive_failures} "
                f"consecutive failures",
                governance_cost=1.0,
            )

        if failed:
            return PolicyResult(
                decision=PolicyDecision.WARN,
                reason=f"Failure {self._consecutive_failures}/"
                f"{self._config.workflow_circuit_breaker_max_failures}",
            )

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason="Step succeeded",
        )

    def evaluate_cost_budget(self, tokens: int) -> PolicyResult:
        """Evaluate whether token cost is within budget.
        
        Args:
            tokens: Number of tokens to be consumed.
            
        Returns:
            Policy result with decision.
        """
        projected_total = self._total_tokens_used + tokens

        if projected_total > self._config.cost_budget_tokens:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Cost budget exceeded: {projected_total} > "
                f"{self._config.cost_budget_tokens} tokens",
                governance_cost=0.5,
            )

        self._total_tokens_used = projected_total
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason=f"Within budget: {projected_total}/"
            f"{self._config.cost_budget_tokens} tokens used",
        )

    def evaluate_result_quality(self, quality_score: float) -> PolicyResult:
        """Evaluate whether a result meets quality threshold.
        
        Args:
            quality_score: Computed quality score (0-1).
            
        Returns:
            Policy result with decision.
        """
        if quality_score < self._config.result_quality_threshold:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Quality score {quality_score:.3f} below threshold "
                f"{self._config.result_quality_threshold:.3f}",
                governance_cost=0.2,
            )

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason=f"Quality score {quality_score:.3f} meets threshold",
        )

    def should_circuit_break(self) -> bool:
        """Check if circuit breaker is active.
        
        Returns:
            True if circuit breaker has been triggered.
        """
        return self._circuit_breaker_active

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker state."""
        self._circuit_breaker_active = False
        self._consecutive_failures = 0

    def get_token_usage(self) -> dict[str, int]:
        """Get current token usage statistics.
        
        Returns:
            Dictionary with token usage info.
        """
        return {
            "total_tokens_used": self._total_tokens_used,
            "budget_tokens": self._config.cost_budget_tokens,
            "remaining_tokens": self._config.cost_budget_tokens
            - self._total_tokens_used,
        }
