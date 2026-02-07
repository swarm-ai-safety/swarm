"""Observable generator for Moltipedia wiki edits."""

from dataclasses import dataclass

from swarm.core.proxy import ProxyObservables


@dataclass
class MoltipediaEditOutcome:
    """Outcome of a wiki edit for observable generation."""

    quality_delta: float = 0.0
    rework_count: int = 0
    verifier_rejections: int = 0
    tool_misuse_flags: int = 0
    engagement_delta: float = 0.0


class MoltipediaObservableGenerator:
    """Maps wiki edit outcomes to proxy observables."""

    def generate(self, outcome: MoltipediaEditOutcome) -> ProxyObservables:
        """Generate proxy observables for a Moltipedia edit."""
        return ProxyObservables(
            task_progress_delta=outcome.quality_delta,
            rework_count=outcome.rework_count,
            verifier_rejections=outcome.verifier_rejections,
            tool_misuse_flags=outcome.tool_misuse_flags,
            counterparty_engagement_delta=outcome.engagement_delta,
        )
