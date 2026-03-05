"""CircuitBreakerPolicy governance module.

Freezes agents or halts escalation when risk metrics exceed thresholds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from swarm_gym.governance.base import GovernanceModule
from swarm_gym.utils.types import Action, AgentId, Event, Intervention


class CircuitBreakerPolicy(GovernanceModule):
    """Threshold-based circuit breaker.

    Governance knobs:
        threshold: Escalation risk level [0, 1] that triggers the breaker.
        cooldown_steps: Steps an agent is frozen after tripping the breaker.
        block_terminal: If True, terminal actions (e.g. "nuke") are always blocked.
    """

    name = "CircuitBreakerPolicy"
    version = "1.0"

    def __init__(
        self,
        threshold: float = 0.9,
        cooldown_steps: int = 3,
        block_terminal: bool = True,
        terminal_actions: Optional[List[str]] = None,
    ):
        self.threshold = max(0.0, min(1.0, threshold))
        self.cooldown_steps = cooldown_steps
        self.block_terminal = block_terminal
        self.terminal_actions = terminal_actions or ["nuke", "strike"]
        self._frozen: Dict[AgentId, int] = {}  # agent_id -> steps remaining
        self._trips: int = 0

    def apply(
        self,
        world_state: Dict[str, Any],
        proposed_actions: List[Action],
    ) -> Tuple[List[Action], List[Intervention], List[Event]]:
        interventions: List[Intervention] = []
        events: List[Event] = []
        filtered_actions: List[Action] = []

        # Decrement cooldowns
        expired = [aid for aid, steps in self._frozen.items() if steps <= 0]
        for aid in expired:
            del self._frozen[aid]

        escalation_risk = world_state.get("escalation_risk", 0.0)

        for action in proposed_actions:
            # Skip frozen agents
            if action.agent_id in self._frozen:
                self._frozen[action.agent_id] -= 1
                action.metadata["circuit_breaker_blocked"] = True
                # Replace with noop
                filtered_actions.append(Action(
                    agent_id=action.agent_id,
                    type="noop",
                    metadata={"original_action": action.type, "blocked_by": "circuit_breaker"},
                ))
                continue

            # Block terminal actions
            if self.block_terminal and action.type in self.terminal_actions:
                self._frozen[action.agent_id] = self.cooldown_steps
                self._trips += 1
                interventions.append(Intervention(
                    module=self.name,
                    type="terminal_action_blocked",
                    agent_id=action.agent_id,
                    reason=f"blocked_{action.type}",
                ))
                events.append(Event(
                    type="CIRCUIT_BREAKER",
                    severity=0.9,
                    agent_id=action.agent_id,
                    outcome="BLOCKED",
                    reason=f"terminal_action_{action.type}",
                ))
                filtered_actions.append(Action(
                    agent_id=action.agent_id,
                    type="noop",
                    metadata={"original_action": action.type, "blocked_by": "circuit_breaker"},
                ))
                continue

            # Check escalation-based trigger
            if action.type == "escalate" and escalation_risk >= self.threshold:
                self._frozen[action.agent_id] = self.cooldown_steps
                self._trips += 1
                interventions.append(Intervention(
                    module=self.name,
                    type="escalation_breaker_tripped",
                    agent_id=action.agent_id,
                    reason=f"escalation_risk={escalation_risk:.2f}",
                ))
                events.append(Event(
                    type="CIRCUIT_BREAKER",
                    severity=0.7,
                    agent_id=action.agent_id,
                    outcome="TRIPPED",
                    reason="escalation_threshold_exceeded",
                ))
                filtered_actions.append(Action(
                    agent_id=action.agent_id,
                    type="noop",
                    metadata={"original_action": action.type, "blocked_by": "circuit_breaker"},
                ))
                continue

            filtered_actions.append(action)

        return filtered_actions, interventions, events

    def reset(self, seed: int = 0) -> None:
        self._frozen.clear()
        self._trips = 0

    def get_state(self) -> Dict[str, Any]:
        return {
            "circuit_breaker_threshold": self.threshold,
            "frozen_agents": list(self._frozen.keys()),
            "total_trips": self._trips,
        }

    def get_params(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "cooldown_steps": self.cooldown_steps,
            "block_terminal": self.block_terminal,
        }
