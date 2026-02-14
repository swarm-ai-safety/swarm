"""Council governance lever for LLM-based interaction review."""

import asyncio
import concurrent.futures
import logging
import random
from typing import Any, Dict, Optional

from swarm.council.protocol import Council, CouncilResult
from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


class CouncilGovernanceLever(GovernanceLever):
    """Governance lever that uses a multi-LLM council to review interactions.

    Similar to ModeratorLever but uses a council of LLMs for more robust
    evaluation. A probabilistic review gate selects interactions for
    council deliberation, and the synthesis is parsed into a LeverEffect.
    """

    def __init__(
        self,
        config: GovernanceConfig,
        council: Council,
        review_probability: Optional[float] = None,
        penalty_multiplier: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(config)
        self._council = council
        self._review_probability = (
            review_probability
            if review_probability is not None
            else config.council_lever_review_probability
        )
        self._penalty_multiplier = (
            penalty_multiplier
            if penalty_multiplier is not None
            else config.council_lever_penalty_multiplier
        )
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "council"

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        """Review interaction via council deliberation.

        Args:
            interaction: The completed interaction
            state: Current environment state

        Returns:
            LeverEffect with costs and reputation changes
        """
        if not self.config.council_lever_enabled:
            return LeverEffect(lever_name=self.name)

        if not interaction.accepted:
            return LeverEffect(lever_name=self.name)

        # Probabilistic review gate
        if self._rng.random() > self._review_probability:
            return LeverEffect(
                lever_name=self.name,
                details={"reviewed": False},
            )

        # Run council deliberation
        result = self._run_council_sync(interaction)

        if not result.success:
            logger.warning(f"Council deliberation failed: {result.error}")
            return LeverEffect(
                lever_name=self.name,
                details={"reviewed": True, "council_error": result.error},
            )

        return self._parse_council_decision(interaction, result)

    def _run_council_sync(self, interaction: SoftInteraction) -> CouncilResult:
        """Run council deliberation synchronously."""
        system_prompt = (
            "You are a governance council member reviewing an interaction in a "
            "multi-agent system. Evaluate whether this interaction is beneficial "
            "or harmful to the ecosystem."
        )
        user_prompt = (
            f"Review this interaction:\n"
            f"- Initiator: {interaction.initiator}\n"
            f"- Counterparty: {interaction.counterparty}\n"
            f"- Interaction type: {interaction.interaction_type.value}\n"
            f"- Quality score (p): {interaction.p:.3f}\n"
            f"- Accepted: {interaction.accepted}\n\n"
            f"Provide your assessment: Is this interaction beneficial? "
            f"Rate the risk from 0.0 (safe) to 1.0 (harmful). "
            f"Recommend a penalty multiplier (0.0 = no penalty, 1.0 = full penalty)."
        )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._council.deliberate(system_prompt, user_prompt),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._council.deliberate(system_prompt, user_prompt)
                )
        except RuntimeError:
            return asyncio.run(
                self._council.deliberate(system_prompt, user_prompt)
            )

    def _parse_council_decision(
        self,
        interaction: SoftInteraction,
        result: CouncilResult,
    ) -> LeverEffect:
        """Parse council synthesis into a LeverEffect."""
        cost_a = 0.0
        cost_b = 0.0
        reputation_deltas: Dict[str, float] = {}

        # Review overhead cost
        review_overhead = 0.02  # Slightly higher than moderator (council is more thorough)
        cost_a += review_overhead
        cost_b += review_overhead

        # Parse penalty from synthesis
        synthesis = result.synthesis.lower()
        penalty_applied = False

        # Simple heuristic: if p is below 0.5 and council mentions risk/harmful
        risk_keywords = ["harmful", "risk", "penalty", "violation", "suspicious"]
        council_flags_risk = any(kw in synthesis for kw in risk_keywords)

        if interaction.p < 0.5 and council_flags_risk:
            shortfall = 0.5 - interaction.p
            penalty = shortfall * self._penalty_multiplier
            cost_a += penalty
            reputation_deltas[interaction.initiator] = -penalty
            penalty_applied = True
        elif interaction.p < 0.3:
            # Low quality always penalized regardless of council sentiment
            shortfall = 0.5 - interaction.p
            penalty = shortfall * self._penalty_multiplier * 0.5
            cost_a += penalty
            reputation_deltas[interaction.initiator] = -penalty
            penalty_applied = True

        details: Dict[str, Any] = {
            "reviewed": True,
            "council_success": result.success,
            "members_responded": result.members_responded,
            "members_total": result.members_total,
            "aggregate_ranking": result.aggregate_ranking,
            "penalty_applied": penalty_applied,
            "p": interaction.p,
        }

        freeze_agents = set()
        if interaction.p < 0.2 and council_flags_risk:
            freeze_agents.add(interaction.initiator)
            details["freeze_recommended"] = True

        return LeverEffect(
            cost_a=cost_a,
            cost_b=cost_b,
            agents_to_freeze=freeze_agents,
            reputation_deltas=reputation_deltas,
            lever_name=self.name,
            details=details,
        )
