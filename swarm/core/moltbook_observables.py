"""Observable generator for Moltbook actions."""

from dataclasses import dataclass

from swarm.core.proxy import ProxyObservables


@dataclass
class MoltbookActionOutcome:
    """Outcome of a Moltbook action for observable generation."""

    task_progress_delta: float = 0.0
    rework_count: int = 0
    verifier_rejections: int = 0
    tool_misuse_flags: int = 0
    engagement_delta: float = 0.0


class MoltbookObservableGenerator:
    """Maps Moltbook action outcomes to proxy observables."""

    def generate(self, outcome: MoltbookActionOutcome) -> ProxyObservables:
        return ProxyObservables(
            task_progress_delta=outcome.task_progress_delta,
            rework_count=outcome.rework_count,
            verifier_rejections=outcome.verifier_rejections,
            tool_misuse_flags=outcome.tool_misuse_flags,
            counterparty_engagement_delta=outcome.engagement_delta,
        )
