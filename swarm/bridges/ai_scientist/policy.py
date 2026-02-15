"""Governance policies for the AI-Scientist bridge.

Four policies mirroring the Agent Lab pattern:
1. Novelty Gate - block non-novel ideas from proceeding to experiment
2. Experiment Circuit Breaker - halt after N consecutive failures
3. Cost Budget - cap total API spend
4. Review Threshold - track low-score reviews, limit improvement rounds
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from swarm.bridges.ai_scientist.config import AIScientistConfig
from swarm.models.interaction import SoftInteraction


class PolicyDecision(Enum):
    APPROVE = "approve"
    DENY = "deny"
    WARN = "warn"


@dataclass
class PolicyResult:
    decision: PolicyDecision
    reason: str = ""
    governance_cost: float = 0.0


class AIScientistPolicy:
    """Stateful governance policies for the AI-Scientist pipeline."""

    def __init__(self, config: AIScientistConfig | None = None) -> None:
        self._config = config or AIScientistConfig()

        # Circuit breaker state
        self._consecutive_experiment_failures: int = 0
        self._circuit_broken: bool = False

        # Cost tracking
        self._total_cost_usd: float = 0.0

        # Review tracking
        self._improvement_rounds: int = 0
        self._consecutive_low_reviews: int = 0

    def evaluate_novelty_gate(self, novel: bool) -> PolicyResult:
        """Block non-novel ideas from proceeding to experiment."""
        if not self._config.novelty_gate_enabled:
            return PolicyResult(
                decision=PolicyDecision.APPROVE,
                reason="Novelty gate disabled",
            )

        if novel:
            return PolicyResult(
                decision=PolicyDecision.APPROVE,
                reason="Idea passed novelty check",
                governance_cost=0.01,
            )

        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="Idea failed novelty check; experiment blocked",
            governance_cost=0.02,
        )

    def evaluate_experiment_run(self, success: bool) -> PolicyResult:
        """Track experiment failures; circuit-break after threshold."""
        if self._circuit_broken:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="Experiment circuit breaker active",
                governance_cost=0.01,
            )

        if success:
            self._consecutive_experiment_failures = 0
            return PolicyResult(
                decision=PolicyDecision.APPROVE,
                reason="Experiment run succeeded",
            )

        self._consecutive_experiment_failures += 1
        max_failures = self._config.experiment_circuit_breaker_max_failures

        if self._consecutive_experiment_failures >= max_failures:
            self._circuit_broken = True
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"Circuit breaker tripped: {self._consecutive_experiment_failures} "
                    f"consecutive experiment failures (max {max_failures})"
                ),
                governance_cost=0.05,
            )

        return PolicyResult(
            decision=PolicyDecision.WARN,
            reason=(
                f"Experiment failure {self._consecutive_experiment_failures}/{max_failures}"
            ),
            governance_cost=0.02,
        )

    def evaluate_cost(self, cost_delta_usd: float) -> PolicyResult:
        """Track cumulative cost; warn at 80%, deny at 100%."""
        self._total_cost_usd += cost_delta_usd
        budget = self._config.cost_budget_usd
        usage_pct = self._total_cost_usd / budget if budget > 0 else 0.0

        if usage_pct >= 1.0:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"Cost budget exceeded: ${self._total_cost_usd:.2f} / ${budget:.2f}",
                governance_cost=0.03,
            )

        if usage_pct >= 0.8:
            return PolicyResult(
                decision=PolicyDecision.WARN,
                reason=(
                    f"Cost at {usage_pct:.0%} of budget: "
                    f"${self._total_cost_usd:.2f} / ${budget:.2f}"
                ),
                governance_cost=0.02,
            )

        return PolicyResult(
            decision=PolicyDecision.APPROVE,
            reason=f"Cost within budget: ${self._total_cost_usd:.2f} / ${budget:.2f}",
        )

    def evaluate_review(self, overall_score: float) -> PolicyResult:
        """Track review scores; limit improvement rounds."""
        threshold = self._config.review_accept_threshold

        if overall_score >= threshold:
            self._consecutive_low_reviews = 0
            return PolicyResult(
                decision=PolicyDecision.APPROVE,
                reason=f"Review score {overall_score:.1f} >= threshold {threshold:.1f}",
            )

        self._consecutive_low_reviews += 1
        self._improvement_rounds += 1

        if self._improvement_rounds >= self._config.max_improvement_rounds:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"Max improvement rounds reached ({self._improvement_rounds}); "
                    f"last score {overall_score:.1f} < {threshold:.1f}"
                ),
                governance_cost=0.04,
            )

        return PolicyResult(
            decision=PolicyDecision.WARN,
            reason=(
                f"Review score {overall_score:.1f} < {threshold:.1f}; "
                f"improvement round {self._improvement_rounds}/{self._config.max_improvement_rounds}"
            ),
            governance_cost=0.02,
        )

    def evaluate_phase_gate(
        self,
        phase_interactions: List[SoftInteraction],
    ) -> PolicyResult:
        """Check average p across a phase's interactions."""
        if not phase_interactions:
            return PolicyResult(
                decision=PolicyDecision.APPROVE,
                reason="No interactions to evaluate",
            )

        avg_p = sum(i.p for i in phase_interactions) / len(phase_interactions)
        min_p = self._config.phase_gate_min_p

        if avg_p >= min_p:
            return PolicyResult(
                decision=PolicyDecision.APPROVE,
                reason=f"Phase avg p={avg_p:.3f} >= {min_p}",
            )

        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason=f"Phase avg p={avg_p:.3f} < {min_p}; advancement blocked",
            governance_cost=0.03,
        )

    def should_circuit_break(self) -> bool:
        return self._circuit_broken

    def reset(self) -> None:
        """Reset all policy state."""
        self._consecutive_experiment_failures = 0
        self._circuit_broken = False
        self._total_cost_usd = 0.0
        self._improvement_rounds = 0
        self._consecutive_low_reviews = 0
