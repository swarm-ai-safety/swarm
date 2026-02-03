"""Collusion detection and penalty governance lever."""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

from src.governance.levers import GovernanceLever, LeverEffect
from src.metrics.collusion import CollusionDetector, CollusionReport

if TYPE_CHECKING:
    from src.env.state import EnvState
    from src.governance.config import GovernanceConfig
    from src.models.interaction import SoftInteraction


class CollusionPenaltyLever(GovernanceLever):
    """
    Governance lever that detects and penalizes collusion.

    Monitors interaction patterns for signs of coordination:
    - Unusually frequent interactions between specific pairs
    - Highly correlated benefits (both parties consistently gain)
    - Quality asymmetry (high p within group, low p to outsiders)

    Applies penalties to flagged agents based on their collusion risk score.
    """

    def __init__(self, config: "GovernanceConfig"):
        """Initialize collusion penalty lever."""
        super().__init__(config)

        # Initialize detector with config parameters
        self._detector = CollusionDetector(
            frequency_threshold=getattr(config, "collusion_frequency_threshold", 2.0),
            benefit_correlation_threshold=getattr(config, "collusion_correlation_threshold", 0.7),
            min_interactions_for_analysis=getattr(config, "collusion_min_interactions", 3),
            collusion_score_threshold=getattr(config, "collusion_score_threshold", 0.5),
        )

        # Track interaction history for analysis
        self._interaction_history: List["SoftInteraction"] = []
        self._agent_ids: List[str] = []

        # Cache latest report
        self._latest_report: Optional[CollusionReport] = None

        # Track penalties applied this epoch
        self._epoch_penalties: Dict[str, float] = defaultdict(float)

    @property
    def name(self) -> str:
        """Return lever name."""
        return "collusion_penalty"

    def set_agent_ids(self, agent_ids: List[str]) -> None:
        """Set the list of agent IDs for analysis."""
        self._agent_ids = list(agent_ids)

    def on_epoch_start(
        self,
        state: "EnvState",
        epoch: int,
    ) -> LeverEffect:
        """
        Analyze interactions at epoch start and apply penalties.

        Runs collusion detection on accumulated history and applies
        reputation/resource penalties to flagged agents.
        """
        if not getattr(self.config, "collusion_detection_enabled", False):
            return LeverEffect(lever_name=self.name)

        # Clear epoch penalties
        self._epoch_penalties.clear()

        # Run detection on history
        if len(self._interaction_history) < self._detector.min_interactions:
            return LeverEffect(lever_name=self.name)

        self._latest_report = self._detector.analyze(
            self._interaction_history,
            self._agent_ids if self._agent_ids else None,
        )

        # Compute penalties based on agent risk scores
        penalty_multiplier = getattr(self.config, "collusion_penalty_multiplier", 1.0)
        reputation_penalties: Dict[str, float] = {}
        resource_penalties: Dict[str, float] = {}

        for agent_id, risk_score in self._latest_report.agent_collusion_risk.items():
            if risk_score >= self._detector.collusion_threshold:
                # Apply penalty proportional to risk score
                rep_penalty = -risk_score * penalty_multiplier
                res_penalty = -risk_score * penalty_multiplier * 10  # Resources hit harder

                reputation_penalties[agent_id] = rep_penalty
                resource_penalties[agent_id] = res_penalty
                self._epoch_penalties[agent_id] = risk_score

        # Optionally clear history after analysis
        if getattr(self.config, "collusion_clear_history_on_epoch", False):
            self._interaction_history.clear()

        return LeverEffect(
            lever_name=self.name,
            reputation_deltas=reputation_penalties,
            resource_deltas=resource_penalties,
            details={
                "ecosystem_risk": self._latest_report.ecosystem_collusion_risk,
                "n_flagged_pairs": self._latest_report.n_flagged_pairs,
                "n_flagged_groups": self._latest_report.n_flagged_groups,
                "penalties_applied": dict(self._epoch_penalties),
            },
        )

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        """
        Record interaction and optionally apply real-time penalty.

        For real-time detection, applies a small penalty if the interaction
        is between agents already flagged as suspicious.
        """
        if not getattr(self.config, "collusion_detection_enabled", False):
            return LeverEffect(lever_name=self.name)

        # Always record for history
        self._interaction_history.append(interaction)

        # Real-time penalty if enabled
        if not getattr(self.config, "collusion_realtime_penalty", False):
            return LeverEffect(lever_name=self.name)

        if self._latest_report is None:
            return LeverEffect(lever_name=self.name)

        # Check if this interaction is between flagged agents
        initiator_risk = self._latest_report.agent_collusion_risk.get(
            interaction.initiator, 0.0
        )
        counterparty_risk = self._latest_report.agent_collusion_risk.get(
            interaction.counterparty, 0.0
        )

        # Both must be flagged for real-time penalty
        if initiator_risk < self._detector.collusion_threshold:
            return LeverEffect(lever_name=self.name)
        if counterparty_risk < self._detector.collusion_threshold:
            return LeverEffect(lever_name=self.name)

        # Apply additional transaction cost as penalty
        realtime_rate = getattr(self.config, "collusion_realtime_rate", 0.1)
        penalty = realtime_rate * (initiator_risk + counterparty_risk) / 2

        return LeverEffect(
            lever_name=self.name,
            cost_a=penalty,
            cost_b=penalty,
            details={
                "realtime_penalty": True,
                "initiator_risk": initiator_risk,
                "counterparty_risk": counterparty_risk,
            },
        )

    def get_report(self) -> Optional[CollusionReport]:
        """Get the latest collusion detection report."""
        return self._latest_report

    def get_interaction_history(self) -> List["SoftInteraction"]:
        """Get recorded interaction history."""
        return list(self._interaction_history)

    def clear_history(self) -> None:
        """Clear interaction history."""
        self._interaction_history.clear()
        self._latest_report = None
