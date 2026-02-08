"""Memory-tier governance levers for shared-memory simulations."""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict

from swarm.governance.levers import GovernanceLever, LeverEffect

if TYPE_CHECKING:
    from swarm.env.state import EnvState
    from swarm.governance.config import GovernanceConfig
    from swarm.models.interaction import SoftInteraction


def _memory_meta(interaction: "SoftInteraction") -> Dict:
    return getattr(interaction, "metadata", {}) or {}


class PromotionGateLever(GovernanceLever):
    """Block promotions that lack quality or verifications.

    Checks interaction metadata for ``memory_promotion`` flag, then
    validates ``quality_score >= min_quality`` and
    ``len(verified_by) >= min_verifications``.

    When the gate blocks a promotion it zeroes the points via cost_a.
    """

    @property
    def name(self) -> str:
        return "memory_promotion_gate"

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        if not getattr(self.config, "memory_promotion_gate_enabled", False):
            return LeverEffect(lever_name=self.name)

        meta = _memory_meta(interaction)
        if not meta.get("memory_promotion"):
            return LeverEffect(lever_name=self.name)

        quality = float(meta.get("quality_score", 0.0))
        verifications = len(meta.get("verified_by", []))
        min_quality = self.config.memory_promotion_min_quality
        min_verifications = self.config.memory_promotion_min_verifications

        blocked = False
        reasons = []
        if quality < min_quality:
            blocked = True
            reasons.append(f"quality {quality:.2f} < {min_quality:.2f}")
        if verifications < min_verifications:
            blocked = True
            reasons.append(f"verifications {verifications} < {min_verifications}")

        if blocked:
            return LeverEffect(
                lever_name=self.name,
                cost_a=1.0,  # Sentinel cost — handler uses to cancel promotion
                details={
                    "blocked": True,
                    "quality": quality,
                    "verifications": verifications,
                    "reasons": reasons,
                },
            )

        return LeverEffect(lever_name=self.name)


class WriteRateLimitLever(GovernanceLever):
    """Cap writes per agent per epoch to prevent flooding."""

    def __init__(self, config: "GovernanceConfig"):
        super().__init__(config)
        self._write_counts: Dict[str, int] = defaultdict(int)

    @property
    def name(self) -> str:
        return "memory_write_rate_limit"

    def on_epoch_start(self, state: "EnvState", epoch: int) -> LeverEffect:
        self._write_counts.clear()
        return LeverEffect(lever_name=self.name)

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        if not getattr(self.config, "memory_write_rate_limit_enabled", False):
            return LeverEffect(lever_name=self.name)

        meta = _memory_meta(interaction)
        if not meta.get("memory_write"):
            return LeverEffect(lever_name=self.name)

        agent_id = interaction.initiator
        self._write_counts[agent_id] += 1
        cap = self.config.memory_write_rate_limit_per_epoch

        if self._write_counts[agent_id] > cap:
            return LeverEffect(
                lever_name=self.name,
                cost_a=1.0,
                details={
                    "agent_id": agent_id,
                    "writes": self._write_counts[agent_id],
                    "cap": cap,
                },
            )

        return LeverEffect(lever_name=self.name)

    def get_remaining_writes(self, agent_id: str) -> int:
        cap = self.config.memory_write_rate_limit_per_epoch
        used = self._write_counts.get(agent_id, 0)
        return max(0, cap - used)


class CrossVerificationLever(GovernanceLever):
    """Require K independent verifiers before Tier 2 -> Tier 3 promotion.

    Colluding verifiers (same pair verifying each other repeatedly) are
    detected via pair tracking, analogous to PairCapLever.
    """

    def __init__(self, config: "GovernanceConfig"):
        super().__init__(config)
        # Track verification pairs: (verifier, author) -> count
        self._pair_counts: Dict[tuple, int] = defaultdict(int)

    @property
    def name(self) -> str:
        return "memory_cross_verification"

    def on_epoch_start(self, state: "EnvState", epoch: int) -> LeverEffect:
        self._pair_counts.clear()
        return LeverEffect(lever_name=self.name)

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        if not getattr(self.config, "memory_cross_verification_enabled", False):
            return LeverEffect(lever_name=self.name)

        meta = _memory_meta(interaction)
        if not meta.get("memory_verification"):
            return LeverEffect(lever_name=self.name)

        verifier_id = interaction.initiator
        author_id = meta.get("entry_author", "")
        if not author_id:
            return LeverEffect(lever_name=self.name)

        # Track pair for collusion detection
        pair = tuple(sorted([verifier_id, author_id]))
        self._pair_counts[pair] += 1

        # Flag if pair is over-represented (> 3 per epoch)
        if self._pair_counts[pair] > 3:
            return LeverEffect(
                lever_name=self.name,
                cost_a=0.5,
                reputation_deltas={verifier_id: -0.05},
                details={
                    "collusive_pair": pair,
                    "count": self._pair_counts[pair],
                },
            )

        return LeverEffect(lever_name=self.name)


class ProvenanceLever(GovernanceLever):
    """Track author provenance and penalize authors whose entries get reverted.

    When an entry is reverted (challenged and confirmed false), the
    original author receives a reputation penalty.
    """

    @property
    def name(self) -> str:
        return "memory_provenance"

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        if not getattr(self.config, "memory_provenance_enabled", False):
            return LeverEffect(lever_name=self.name)

        meta = _memory_meta(interaction)
        if not meta.get("memory_revert"):
            return LeverEffect(lever_name=self.name)

        original_author = meta.get("entry_author", "")
        if not original_author:
            return LeverEffect(lever_name=self.name)

        penalty = self.config.memory_provenance_revert_penalty
        return LeverEffect(
            lever_name=self.name,
            reputation_deltas={original_author: -penalty},
            details={
                "reverted_author": original_author,
                "entry_id": meta.get("entry_id", ""),
                "penalty": penalty,
            },
        )
