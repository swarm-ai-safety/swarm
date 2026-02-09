"""Dual reporting for soft and hard metrics."""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.metrics.incoherence import DecisionRecord, IncoherenceMetrics
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction


@dataclass
class MetricsSummary:
    """Summary of both soft and hard metrics."""

    # Soft metrics (probabilistic)
    toxicity_soft: float
    conditional_loss_initiator: float
    conditional_loss_counterparty: float
    spread: float
    quality_gap: float
    average_quality: float
    uncertain_fraction: float

    # Hard metrics (threshold-based)
    toxicity_hard: float
    acceptance_rate: float
    high_quality_acceptance: float
    low_quality_acceptance: float

    # Counts
    total_interactions: int
    accepted_count: int
    rejected_count: int
    high_quality_count: int
    low_quality_count: int

    # Welfare
    total_welfare: float
    total_social_surplus: float
    avg_initiator_payoff: float
    avg_counterparty_payoff: float

    # Calibration metrics (None if no ground truth)
    brier_score: Optional[float] = None
    log_loss: Optional[float] = None
    calibration_error: Optional[float] = None
    expected_calibration_error: Optional[float] = None

    # Variance metrics
    quality_variance: float = 0.0
    payoff_variance_initiator: float = 0.0
    payoff_variance_counterparty: float = 0.0

    # Incoherence metrics (optional replay-driven additions)
    incoherence_disagreement: float = 0.0
    incoherence_error: float = 0.0
    incoherence_index: float = 0.0
    incoherence_n_decisions: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "soft_metrics": {
                "toxicity": self.toxicity_soft,
                "conditional_loss_initiator": self.conditional_loss_initiator,
                "conditional_loss_counterparty": self.conditional_loss_counterparty,
                "spread": self.spread,
                "quality_gap": self.quality_gap,
                "average_quality": self.average_quality,
                "uncertain_fraction": self.uncertain_fraction,
            },
            "hard_metrics": {
                "toxicity": self.toxicity_hard,
                "acceptance_rate": self.acceptance_rate,
                "high_quality_acceptance": self.high_quality_acceptance,
                "low_quality_acceptance": self.low_quality_acceptance,
            },
            "counts": {
                "total": self.total_interactions,
                "accepted": self.accepted_count,
                "rejected": self.rejected_count,
                "high_quality": self.high_quality_count,
                "low_quality": self.low_quality_count,
            },
            "welfare": {
                "total_welfare": self.total_welfare,
                "total_social_surplus": self.total_social_surplus,
                "avg_initiator_payoff": self.avg_initiator_payoff,
                "avg_counterparty_payoff": self.avg_counterparty_payoff,
            },
            "calibration": {
                "brier_score": self.brier_score,
                "log_loss": self.log_loss,
                "calibration_error": self.calibration_error,
                "expected_calibration_error": self.expected_calibration_error,
            },
            "variance": {
                "quality_variance": self.quality_variance,
                "payoff_variance_initiator": self.payoff_variance_initiator,
                "payoff_variance_counterparty": self.payoff_variance_counterparty,
            },
            "incoherence": {
                "disagreement": self.incoherence_disagreement,
                "error": self.incoherence_error,
                "index": self.incoherence_index,
                "n_decisions": self.incoherence_n_decisions,
            },
        }


class MetricsReporter:
    """
    Reports both soft (probabilistic) and hard (threshold-based) metrics.

    Soft metrics use the full probability distribution, while hard metrics
    apply a threshold (default 0.5) to convert to binary labels.
    """

    def __init__(
        self,
        payoff_config: Optional[PayoffConfig] = None,
        quality_threshold: float = 0.5,
        uncertainty_band: float = 0.2,
    ):
        """
        Initialize reporter.

        Args:
            payoff_config: Configuration for payoff calculations
            quality_threshold: Threshold for hard metrics (default 0.5)
            uncertainty_band: Band around 0.5 for uncertainty flagging
        """
        self.payoff_engine = SoftPayoffEngine(payoff_config)
        self.soft_metrics = SoftMetrics(self.payoff_engine)
        self.quality_threshold = quality_threshold
        self.uncertainty_band = uncertainty_band

    def _compute_hard_toxicity(self, interactions: List[SoftInteraction]) -> float:
        """
        Compute hard toxicity: P(p < threshold | accepted)

        This is the fraction of accepted interactions that would be
        classified as harmful using a hard threshold.
        """
        accepted = [i for i in interactions if i.accepted]
        if not accepted:
            return 0.0

        toxic_count = sum(1 for i in accepted if i.p < self.quality_threshold)
        return toxic_count / len(accepted)

    def summary(self, interactions: List[SoftInteraction]) -> MetricsSummary:
        """
        Compute comprehensive summary of all metrics.

        Args:
            interactions: List of interactions

        Returns:
            MetricsSummary with both soft and hard metrics
        """
        if not interactions:
            return MetricsSummary(
                toxicity_soft=0.0,
                conditional_loss_initiator=0.0,
                conditional_loss_counterparty=0.0,
                spread=0.0,
                quality_gap=0.0,
                average_quality=0.0,
                uncertain_fraction=0.0,
                toxicity_hard=0.0,
                acceptance_rate=0.0,
                high_quality_acceptance=0.0,
                low_quality_acceptance=0.0,
                total_interactions=0,
                accepted_count=0,
                rejected_count=0,
                high_quality_count=0,
                low_quality_count=0,
                total_welfare=0.0,
                total_social_surplus=0.0,
                avg_initiator_payoff=0.0,
                avg_counterparty_payoff=0.0,
                brier_score=None,
                log_loss=None,
                calibration_error=None,
                expected_calibration_error=None,
                quality_variance=0.0,
                payoff_variance_initiator=0.0,
                payoff_variance_counterparty=0.0,
            )

        # Compute soft metrics
        toxicity_soft = self.soft_metrics.toxicity_rate(interactions)
        cl_init = self.soft_metrics.conditional_loss_initiator(interactions)
        cl_counter = self.soft_metrics.conditional_loss_counterparty(interactions)
        spread = self.soft_metrics.spread(interactions)
        quality_gap = self.soft_metrics.quality_gap(interactions)
        avg_quality = self.soft_metrics.average_quality(interactions)
        uncertain_frac = self.soft_metrics.uncertain_fraction(
            interactions, self.uncertainty_band
        )

        # Compute hard metrics
        toxicity_hard = self._compute_hard_toxicity(interactions)
        participation = self.soft_metrics.participation_by_quality(
            interactions, self.quality_threshold
        )

        # Counts
        accepted = [i for i in interactions if i.accepted]
        rejected = [i for i in interactions if not i.accepted]
        acceptance_rate = len(accepted) / len(interactions) if interactions else 0.0

        # Welfare
        welfare = self.soft_metrics.welfare_metrics(interactions)

        # Calibration metrics (may be None if no ground truth)
        brier = self.soft_metrics.brier_score(interactions)
        logloss = self.soft_metrics.log_loss(interactions)
        cal_error = self.soft_metrics.calibration_error(interactions)
        ece = self.soft_metrics.expected_calibration_error(interactions)

        # Variance metrics
        quality_var = self.soft_metrics.quality_variance(interactions)
        payoff_var_init = self.soft_metrics.payoff_variance_initiator(interactions)
        payoff_var_counter = self.soft_metrics.payoff_variance_counterparty(
            interactions
        )

        return MetricsSummary(
            toxicity_soft=toxicity_soft,
            conditional_loss_initiator=cl_init,
            conditional_loss_counterparty=cl_counter,
            spread=spread,
            quality_gap=quality_gap,
            average_quality=avg_quality,
            uncertain_fraction=uncertain_frac,
            toxicity_hard=toxicity_hard,
            acceptance_rate=acceptance_rate,
            high_quality_acceptance=participation["high_quality_acceptance"],
            low_quality_acceptance=participation["low_quality_acceptance"],
            total_interactions=len(interactions),
            accepted_count=len(accepted),
            rejected_count=len(rejected),
            high_quality_count=participation["high_quality_count"],
            low_quality_count=participation["low_quality_count"],
            total_welfare=welfare["total_welfare"],
            total_social_surplus=welfare["total_social_surplus"],
            avg_initiator_payoff=welfare["avg_initiator_payoff"],
            avg_counterparty_payoff=welfare["avg_counterparty_payoff"],
            brier_score=brier,
            log_loss=logloss,
            calibration_error=cal_error,
            expected_calibration_error=ece,
            quality_variance=quality_var,
            payoff_variance_initiator=payoff_var_init,
            payoff_variance_counterparty=payoff_var_counter,
        )

    def compare_soft_hard(self, interactions: List[SoftInteraction]) -> dict:
        """
        Compare soft vs hard metrics side by side.

        Useful for understanding how much information is lost
        by thresholding.

        Args:
            interactions: List of interactions

        Returns:
            Dictionary with comparison data
        """
        summary = self.summary(interactions)

        toxicity_diff = summary.toxicity_soft - summary.toxicity_hard

        return {
            "toxicity": {
                "soft": summary.toxicity_soft,
                "hard": summary.toxicity_hard,
                "difference": toxicity_diff,
                "interpretation": (
                    "Soft metric captures uncertainty better"
                    if abs(toxicity_diff) > 0.05
                    else "Metrics roughly agree"
                ),
            },
            "quality_filtering": {
                "quality_gap": summary.quality_gap,
                "spread": summary.spread,
                "adverse_selection": summary.quality_gap < 0,
            },
            "uncertainty": {
                "fraction_uncertain": summary.uncertain_fraction,
                "interpretation": (
                    "High uncertainty - labels may be unreliable"
                    if summary.uncertain_fraction > 0.3
                    else "Labels are relatively confident"
                ),
            },
        }

    def summary_with_incoherence(
        self,
        interactions: List[SoftInteraction],
        records_by_decision: Dict[str, List[DecisionRecord]],
        incoherence_metrics: IncoherenceMetrics,
    ) -> MetricsSummary:
        """
        Compute summary plus replay-based incoherence aggregates.

        Args:
            interactions: Interaction list for base soft/hard metrics
            records_by_decision: Replay records grouped by decision key
            incoherence_metrics: Metric calculator with benchmark policy
        """
        base = self.summary(interactions)
        if not records_by_decision:
            return base

        results = [
            incoherence_metrics.compute_for_decision(records)
            for records in records_by_decision.values()
            if records
        ]
        if not results:
            return base

        n = len(results)
        return replace(
            base,
            incoherence_disagreement=sum(r.disagreement for r in results) / n,
            incoherence_error=sum(r.error for r in results) / n,
            incoherence_index=sum(r.incoherence for r in results) / n,
            incoherence_n_decisions=n,
        )

    def format_report(
        self,
        interactions: List[SoftInteraction],
        verbose: bool = False,
    ) -> str:
        """
        Format a human-readable metrics report.

        Args:
            interactions: List of interactions
            verbose: Include additional details

        Returns:
            Formatted report string
        """
        summary = self.summary(interactions)

        lines = [
            "=" * 50,
            "METRICS REPORT",
            "=" * 50,
            "",
            f"Total interactions: {summary.total_interactions}",
            f"  Accepted: {summary.accepted_count} ({summary.acceptance_rate:.1%})",
            f"  Rejected: {summary.rejected_count}",
            "",
            "SOFT METRICS (probabilistic)",
            "-" * 30,
            f"  Toxicity (E[1-p|accepted]): {summary.toxicity_soft:.3f}",
            f"  Average quality (E[p]):     {summary.average_quality:.3f}",
            f"  Quality gap:                {summary.quality_gap:+.3f}",
            f"  Spread:                     {summary.spread:+.3f}",
            f"  Uncertain fraction:         {summary.uncertain_fraction:.1%}",
            "",
            "HARD METRICS (threshold-based)",
            "-" * 30,
            f"  Toxicity (P(p<0.5|acc)):    {summary.toxicity_hard:.3f}",
            f"  High quality acceptance:    {summary.high_quality_acceptance:.1%}",
            f"  Low quality acceptance:     {summary.low_quality_acceptance:.1%}",
        ]

        if verbose:
            lines.extend(
                [
                    "",
                    "PAYOFF ANALYSIS",
                    "-" * 30,
                    f"  Conditional loss (init):    {summary.conditional_loss_initiator:+.3f}",
                    f"  Conditional loss (counter): {summary.conditional_loss_counterparty:+.3f}",
                    "",
                    "WELFARE",
                    "-" * 30,
                    f"  Total welfare:              {summary.total_welfare:.2f}",
                    f"  Social surplus:             {summary.total_social_surplus:.2f}",
                    f"  Avg initiator payoff:       {summary.avg_initiator_payoff:.3f}",
                    f"  Avg counterparty payoff:    {summary.avg_counterparty_payoff:.3f}",
                ]
            )

            # Interpretation
            if summary.quality_gap < 0:
                lines.extend(
                    [
                        "",
                        "WARNING: Negative quality gap indicates adverse selection",
                        "  (rejected interactions have higher quality than accepted)",
                    ]
                )

        lines.append("=" * 50)

        return "\n".join(lines)
