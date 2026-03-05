"""TaxPolicy governance module.

Applies a flat transaction tax on trade actions, redistributing
revenue to fund public goods or reduce inequality.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from swarm_gym.governance.base import GovernanceModule
from swarm_gym.utils.types import Action, Event, Intervention


class TaxPolicy(GovernanceModule):
    """Flat transaction tax on trades.

    Governance knobs:
        rate: Tax rate [0, 1]. Applied to trade value (qty * price).
        redistribute: If True, tax revenue is tracked for redistribution.
    """

    name = "TaxPolicy"
    version = "1.0"

    def __init__(self, rate: float = 0.05, redistribute: bool = True):
        self.rate = max(0.0, min(1.0, rate))
        self.redistribute = redistribute
        self._revenue_this_episode: float = 0.0

    def apply(
        self,
        world_state: Dict[str, Any],
        proposed_actions: List[Action],
    ) -> Tuple[List[Action], List[Intervention], List[Event]]:
        interventions: List[Intervention] = []
        events: List[Event] = []

        for action in proposed_actions:
            if action.type == "trade" and action.price is not None and action.qty is not None:
                value = abs(action.qty * action.price)
                tax = value * self.rate
                self._revenue_this_episode += tax
                action.metadata["tax_paid"] = tax
                action.metadata["post_tax_value"] = value - tax

                if tax > 0:
                    interventions.append(Intervention(
                        module=self.name,
                        type="tax_applied",
                        agent_id=action.agent_id,
                        metadata={"tax": round(tax, 4), "rate": self.rate},
                    ))

        return proposed_actions, interventions, events

    def reset(self, seed: int = 0) -> None:
        self._revenue_this_episode = 0.0

    def get_state(self) -> Dict[str, Any]:
        return {
            "tax_rate": self.rate,
            "tax_revenue": round(self._revenue_this_episode, 4),
        }

    def get_params(self) -> Dict[str, Any]:
        return {"rate": self.rate, "redistribute": self.redistribute}
