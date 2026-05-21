"""SWARM safety metrics for Gitlawb agent interactions."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from swarm.bridges.gitlawb.config import GitlawbConfig
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


@dataclass
class GitlawbMetricsReport:
    """Computed safety metrics for a set of Gitlawb interactions."""

    timestamp: str = ""
    interaction_count: int = 0
    toxicity_rate: float = 0.0
    toxicity_rate_all: float = 0.0
    quality_gap: float = 0.0
    spread: float = 0.0
    conditional_loss_initiator: float = 0.0
    average_quality: float = 0.0
    welfare: dict[str, float] = field(default_factory=dict)
    per_interaction_type: dict[str, float] = field(default_factory=dict)
    source: str = "gitlawb"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "interaction_count": self.interaction_count,
            "toxicity_rate": self.toxicity_rate,
            "toxicity_rate_all": self.toxicity_rate_all,
            "quality_gap": self.quality_gap,
            "spread": self.spread,
            "conditional_loss_initiator": self.conditional_loss_initiator,
            "average_quality": self.average_quality,
            "welfare": self.welfare,
            "per_interaction_type": self.per_interaction_type,
            "source": self.source,
        }


class GitlawbMetrics:
    """Compute SWARM safety metrics on Gitlawb interactions."""

    def __init__(self, config: GitlawbConfig) -> None:
        self._config = config
        self._soft = SoftMetrics()

    def compute(self, interactions: list[SoftInteraction]) -> GitlawbMetricsReport:
        """Compute aggregate safety metrics."""
        if not interactions:
            return GitlawbMetricsReport(
                timestamp=datetime.now(timezone.utc).isoformat(),
                interaction_count=0,
            )

        welfare = self._soft.welfare_metrics(interactions)

        by_type: dict[str, list[float]] = defaultdict(list)
        for i in interactions:
            by_type[i.interaction_type.value].append(i.p)
        per_type = {k: sum(v) / len(v) for k, v in by_type.items()}

        return GitlawbMetricsReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            interaction_count=len(interactions),
            toxicity_rate=self._soft.toxicity_rate(interactions),
            toxicity_rate_all=self._soft.toxicity_rate_all(interactions),
            quality_gap=self._soft.quality_gap(interactions),
            spread=self._soft.spread(interactions),
            conditional_loss_initiator=self._soft.conditional_loss_initiator(interactions),
            average_quality=self._soft.average_quality(interactions),
            welfare=welfare,
            per_interaction_type=per_type,
        )

    def compute_per_repo(
        self, interactions: list[SoftInteraction]
    ) -> dict[str, GitlawbMetricsReport]:
        """Group interactions by repo and compute metrics for each."""
        by_repo: dict[str, list[SoftInteraction]] = defaultdict(list)
        for i in interactions:
            repo = i.metadata.get("repo", "unknown")
            by_repo[repo].append(i)
        return {repo: self.compute(group) for repo, group in by_repo.items()}

    def compute_per_agent(
        self, interactions: list[SoftInteraction]
    ) -> dict[str, GitlawbMetricsReport]:
        """Group interactions by initiator DID and compute metrics for each."""
        by_agent: dict[str, list[SoftInteraction]] = defaultdict(list)
        for i in interactions:
            by_agent[i.initiator].append(i)
        return {did: self.compute(group) for did, group in by_agent.items()}
