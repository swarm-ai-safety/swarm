"""AuditPolicy governance module.

Stochastic auditing with limited budget. Detects hidden/deceptive
actions and applies penalties.
"""

from __future__ import annotations

import random as _random
from typing import Any, Dict, List, Tuple

from swarm_gym.governance.base import GovernanceModule
from swarm_gym.utils.types import Action, Event, Intervention


class AuditPolicy(GovernanceModule):
    """Probabilistic audit with budget constraints.

    Governance knobs:
        p_audit: Base probability of auditing an agent per step.
        penalty: Penalty multiplier applied to caught agents.
        budget_per_episode: Max audits per episode (0 = unlimited).
        false_positive_rate: Rate at which innocent agents are flagged.
    """

    name = "AuditPolicy"
    version = "1.0"

    def __init__(
        self,
        p_audit: float = 0.05,
        penalty: float = 2.0,
        budget_per_episode: int = 0,
        false_positive_rate: float = 0.01,
    ):
        self.p_audit = max(0.0, min(1.0, p_audit))
        self.penalty = penalty
        self.budget_per_episode = budget_per_episode
        self.false_positive_rate = false_positive_rate
        self._audits_used: int = 0
        self._rng = _random.Random(0)

    def apply(
        self,
        world_state: Dict[str, Any],
        proposed_actions: List[Action],
    ) -> Tuple[List[Action], List[Intervention], List[Event]]:
        interventions: List[Intervention] = []
        events: List[Event] = []

        # Identify suspicious actions (hide, lie, or with deception metadata)
        suspicious_agents = set()
        for action in proposed_actions:
            if action.type in ("hide", "lie", "misreport"):
                suspicious_agents.add(action.agent_id)
            if action.metadata.get("deceptive"):
                suspicious_agents.add(action.agent_id)

        # Determine which agents to audit
        all_agents = {a.agent_id for a in proposed_actions}
        for agent_id in all_agents:
            if self.budget_per_episode > 0 and self._audits_used >= self.budget_per_episode:
                break

            should_audit = self._rng.random() < self.p_audit
            if not should_audit:
                continue

            self._audits_used += 1
            is_suspicious = agent_id in suspicious_agents
            is_false_positive = (
                not is_suspicious and self._rng.random() < self.false_positive_rate
            )

            if is_suspicious:
                # Caught: apply penalty
                for action in proposed_actions:
                    if action.agent_id == agent_id:
                        action.metadata["audit_penalty"] = self.penalty
                        action.metadata["audited"] = True

                interventions.append(Intervention(
                    module=self.name,
                    type="audit_triggered",
                    agent_id=agent_id,
                    reason="suspicious_action_detected",
                ))
                events.append(Event(
                    type="AUDIT",
                    severity=0.6,
                    agent_id=agent_id,
                    outcome="FAIL",
                    reason="caught_deception",
                ))
            else:
                outcome = "FALSE_POSITIVE" if is_false_positive else "PASS"
                events.append(Event(
                    type="AUDIT",
                    severity=0.2 if is_false_positive else 0.1,
                    agent_id=agent_id,
                    outcome=outcome,
                ))
                if is_false_positive:
                    for action in proposed_actions:
                        if action.agent_id == agent_id:
                            action.metadata["audit_penalty"] = self.penalty * 0.5
                            action.metadata["audited"] = True
                            action.metadata["false_positive"] = True

        return proposed_actions, interventions, events

    def reset(self, seed: int = 0) -> None:
        self._audits_used = 0
        self._rng = _random.Random(seed)

    def get_state(self) -> Dict[str, Any]:
        return {
            "p_audit": self.p_audit,
            "audits_used": self._audits_used,
            "budget_remaining": (
                self.budget_per_episode - self._audits_used
                if self.budget_per_episode > 0
                else -1
            ),
        }

    def get_params(self) -> Dict[str, Any]:
        return {
            "p_audit": self.p_audit,
            "penalty": self.penalty,
            "budget_per_episode": self.budget_per_episode,
            "false_positive_rate": self.false_positive_rate,
        }
