"""Observable generator for memory tier actions."""

from dataclasses import dataclass

from swarm.core.proxy import ProxyObservables


@dataclass
class MemoryActionOutcome:
    """Outcome of a memory action for observable generation."""

    quality_delta: float = 0.0
    rework_count: int = 0
    verifier_rejections: int = 0
    tool_misuse_flags: int = 0
    engagement_delta: float = 0.0


class MemoryObservableGenerator:
    """Maps memory action outcomes to proxy observables.

    Mapping:
      - Accurate write:   (+0.6, 0, 0, 0, +0.4) -> p ~0.73
      - Poisoned write that passes: (+0.3, 0, 0, 0, +0.2) -> p ~0.58
      - Detected poisoning: (-0.5, 1, 2, 1, -0.3) -> p ~0.29
      - Successful promotion: (+0.4, 0, 0, 0, +0.3) -> p ~0.66
      - Blocked promotion: (+0.1, 1, 1, 0, 0.0) -> p ~0.50
    """

    def generate(self, outcome: MemoryActionOutcome) -> ProxyObservables:
        return ProxyObservables(
            task_progress_delta=outcome.quality_delta,
            rework_count=outcome.rework_count,
            verifier_rejections=outcome.verifier_rejections,
            tool_misuse_flags=outcome.tool_misuse_flags,
            counterparty_engagement_delta=outcome.engagement_delta,
        )
