"""Metrics for the Team-of-Rivals scenario.

Computes success rate, delta-illusion, veto rates, and other
pipeline-specific metrics from completed episodes.
"""

from typing import Dict, List

from pydantic import BaseModel

from swarm.core.rivals_handler import PipelineStage, RivalsEpisode


class RivalsMetrics(BaseModel):
    """Aggregated metrics for a rivals simulation run."""

    success_rate: float = 0.0
    delta_illusion: float = 0.0
    veto_rate_by_stage: Dict[str, float] = {}
    mean_veto_count: float = 0.0
    turns_to_ship: float = 0.0
    total_episodes: int = 0
    failed_episodes: int = 0


def compute_rivals_metrics(episodes: List[RivalsEpisode]) -> RivalsMetrics:
    """Compute aggregate metrics from completed episodes.

    Args:
        episodes: List of completed RivalsEpisode instances.

    Returns:
        RivalsMetrics with computed values.
    """
    if not episodes:
        return RivalsMetrics()

    total = len(episodes)
    scored = [e for e in episodes if e.stage == PipelineStage.SCORED]
    failed = [e for e in episodes if e.stage == PipelineStage.FAILED]

    # Success rate
    success_rate = len(scored) / total if total > 0 else 0.0

    # Delta illusion: perceived_coherence - actual_consistency
    illusion_values = []
    for ep in scored:
        illusion_values.append(ep.perceived_coherence - ep.actual_consistency)
    delta_illusion = (
        sum(illusion_values) / len(illusion_values) if illusion_values else 0.0
    )

    # Veto rate by stage
    stage_veto_counts: Dict[str, int] = {}
    stage_review_counts: Dict[str, int] = {}
    total_vetoes = 0

    for ep in episodes:
        for entry in ep.veto_history:
            stage = entry.get("stage", "unknown")
            stage_review_counts[stage] = stage_review_counts.get(stage, 0) + 1
            if entry.get("vetoed", False):
                stage_veto_counts[stage] = stage_veto_counts.get(stage, 0) + 1
                total_vetoes += 1

    veto_rate_by_stage: Dict[str, float] = {}
    for stage, count in stage_review_counts.items():
        veto_rate_by_stage[stage] = stage_veto_counts.get(stage, 0) / count

    # Mean veto count per episode
    mean_veto_count = total_vetoes / total if total > 0 else 0.0

    # Turns to ship: sum of retries + 3 base stages for scored episodes
    turns = []
    for ep in scored:
        retry_total = sum(ep.retries.values())
        turns.append(3 + retry_total)  # 3 produce stages + retries
    turns_to_ship = sum(turns) / len(turns) if turns else 0.0

    return RivalsMetrics(
        success_rate=success_rate,
        delta_illusion=delta_illusion,
        veto_rate_by_stage=veto_rate_by_stage,
        mean_veto_count=mean_veto_count,
        turns_to_ship=turns_to_ship,
        total_episodes=total,
        failed_episodes=len(failed),
    )
