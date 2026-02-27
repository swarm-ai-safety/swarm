"""Work-regime-specific metrics for measuring behavioral drift under stress.

Translates "overwork → stance drift" into SWARM-native observables,
replacing ideological survey instruments with behavioral and structural
measurements.

Metric categories:
    Behavioral (hard):  proposal rates, strike/exit, coalition formation,
                        defection, quality degradation
    Structural:         drift index, coalition strength, legitimacy score
    Aggregate:          work regime summary for epoch reporting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


@dataclass
class WorkRegimeEpochMetrics:
    """Per-epoch summary of work-regime drift indicators."""

    epoch: int = 0

    # --- Behavioral (hard) ---
    strike_rate: float = 0.0           # fraction of agents who noop'd due to exit
    quality_degradation: float = 0.0   # avg p drop from initial for accepted interactions
    defection_rate: float = 0.0        # fraction of rejected proposals
    redistribution_proposals: int = 0  # count of reform-like posts

    # --- Structural ---
    mean_drift_index: float = 0.0      # avg L2 policy distance from initial
    max_drift_index: float = 0.0       # max across agents
    coalition_strength: float = 0.0    # community clustering coefficient
    legitimacy_score: float = 0.0      # procedural justice proxy

    # --- Aggregate ---
    mean_compliance: float = 0.0
    mean_cooperation_threshold: float = 0.0
    mean_redistribution_pref: float = 0.0
    mean_exit_propensity: float = 0.0
    mean_grievance: float = 0.0
    gini_payoff: float = 0.0          # payoff inequality

    # --- Per-agent snapshots ---
    agent_snapshots: Dict[str, Dict[str, float]] = field(default_factory=dict)


def compute_drift_index(
    policy_snapshots: Sequence[Dict[str, float]],
) -> tuple[float, float]:
    """Compute mean and max drift index across agent policy snapshots.

    Each snapshot must contain a 'drift' key (L2 distance from initial).

    Returns:
        (mean_drift, max_drift)
    """
    if not policy_snapshots:
        logger.warning(
            "compute_drift_index called with empty policy_snapshots; "
            "returning (0, 0). Check that WorkRegimeAgents are present "
            "and producing snapshots."
        )
        return 0.0, 0.0
    drifts = [s.get("drift", 0.0) for s in policy_snapshots]
    return sum(drifts) / len(drifts), max(drifts)


def compute_gini(values: Sequence[float]) -> float:
    """Compute Gini coefficient for a sequence of non-negative values.

    Returns 0.0 for perfect equality, approaches 1.0 for maximum inequality.
    """
    if not values or len(values) < 2:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = 0.0
    gini_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        gini_sum += (2 * (i + 1) - n - 1) * v
    return gini_sum / (n * total)


def compute_legitimacy_score(
    *,
    audit_precision: float = 1.0,
    eval_noise: float = 0.0,
    appeals_available: bool = False,
    explanation_provided: bool = False,
) -> float:
    """Compute governance legitimacy as a function of procedural justice.

    Higher = more legitimate governance.  Range [0, 1].

    Components:
        - Audit precision (high precision → fewer false positives → more trust)
        - Evaluation noise (low noise → more predictable → more trust)
        - Appeals process availability
        - Explanation requirement
    """
    score = 0.0
    # Precision component (weight 0.4)
    score += 0.4 * audit_precision
    # Low noise component (weight 0.3)
    score += 0.3 * (1.0 - min(1.0, eval_noise))
    # Appeals (weight 0.15)
    score += 0.15 * (1.0 if appeals_available else 0.0)
    # Explanation (weight 0.15)
    score += 0.15 * (1.0 if explanation_provided else 0.0)
    return max(0.0, min(1.0, score))


def compute_coalition_strength(
    interaction_pairs: Sequence[tuple[str, str]],
    agent_ids: Sequence[str],
) -> float:
    """Estimate coalition strength via average local clustering coefficient.

    Measures how much agents form tight cooperative subgroups.
    Higher = stronger coalition formation.  Range [0, 1].

    Note: Nodes with degree < 2 (isolated or leaf nodes) are excluded
    from the average because a clustering coefficient is undefined for
    them.  In sparse networks this may overestimate coalition formation
    since only the well-connected core contributes.  Returns 0.0 when
    no node has degree >= 2 (i.e. no triangles are possible).
    """
    if len(agent_ids) < 3 or not interaction_pairs:
        return 0.0

    # Build adjacency from interaction pairs
    neighbors: Dict[str, set] = {aid: set() for aid in agent_ids}
    for a, b in interaction_pairs:
        if a in neighbors and b in neighbors:
            neighbors[a].add(b)
            neighbors[b].add(a)

    # Average local clustering coefficient
    coefficients = []
    for _node, nbrs in neighbors.items():
        k = len(nbrs)
        if k < 2:
            continue
        # Count edges among neighbors
        nbr_list = list(nbrs)
        triangles = 0
        for i in range(len(nbr_list)):
            for j in range(i + 1, len(nbr_list)):
                if nbr_list[j] in neighbors.get(nbr_list[i], set()):
                    triangles += 1
        possible = k * (k - 1) / 2
        coefficients.append(triangles / possible if possible > 0 else 0.0)

    if not coefficients:
        return 0.0
    return sum(coefficients) / len(coefficients)


def compute_quality_degradation(
    interactions: Sequence[SoftInteraction],
    baseline_p: float = 0.7,
) -> float:
    """Measure quality degradation as avg drop in p from baseline.

    Returns a non-negative value; 0 = no degradation, higher = worse.
    """
    if not interactions:
        return 0.0
    accepted = [i for i in interactions if i.accepted]
    if not accepted:
        return 0.0
    avg_p = sum(i.p for i in accepted) / len(accepted)
    return max(0.0, baseline_p - avg_p)


def compute_defection_rate(
    interactions: Sequence[SoftInteraction],
) -> float:
    """Fraction of interactions that were rejected."""
    if not interactions:
        return 0.0
    rejected = sum(1 for i in interactions if not i.accepted)
    return rejected / len(interactions)


def build_epoch_metrics(
    *,
    epoch: int,
    policy_snapshots: Dict[str, Dict[str, float]],
    interactions: Sequence[SoftInteraction],
    agent_payoffs: Dict[str, float],
    strike_count: int = 0,
    total_agents: int = 1,
    interaction_pairs: Optional[Sequence[tuple[str, str]]] = None,
    legitimacy_kwargs: Optional[Dict] = None,
) -> WorkRegimeEpochMetrics:
    """Build a complete WorkRegimeEpochMetrics for one epoch.

    Args:
        epoch: epoch number
        policy_snapshots: {agent_id: snapshot_dict} from WorkRegimeAgent
        interactions: all interactions this epoch
        agent_payoffs: {agent_id: total_payoff}
        strike_count: how many agents chose to strike/exit
        total_agents: total population size
        interaction_pairs: (a, b) pairs for coalition computation
        legitimacy_kwargs: keyword args for compute_legitimacy_score
    """
    snapshots_list = list(policy_snapshots.values())
    if not snapshots_list:
        logger.warning(
            "build_epoch_metrics: no policy snapshots for epoch %d; "
            "aggregate means will be zero. Ensure WorkRegimeAgents "
            "provide snapshots via policy_snapshot().",
            epoch,
        )
    mean_drift, max_drift = compute_drift_index(snapshots_list)

    payoff_values = list(agent_payoffs.values()) if agent_payoffs else []

    # Aggregate policy means
    mean_compliance = (
        sum(s.get("compliance_propensity", 0) for s in snapshots_list)
        / max(len(snapshots_list), 1)
    )
    mean_coop = (
        sum(s.get("cooperation_threshold", 0) for s in snapshots_list)
        / max(len(snapshots_list), 1)
    )
    mean_redist = (
        sum(s.get("redistribution_preference", 0) for s in snapshots_list)
        / max(len(snapshots_list), 1)
    )
    mean_exit = (
        sum(s.get("exit_propensity", 0) for s in snapshots_list)
        / max(len(snapshots_list), 1)
    )
    mean_grievance = (
        sum(s.get("grievance", 0) for s in snapshots_list)
        / max(len(snapshots_list), 1)
    )

    legitimacy = compute_legitimacy_score(**(legitimacy_kwargs or {}))

    coalition = 0.0
    if interaction_pairs:
        coalition = compute_coalition_strength(
            interaction_pairs, list(policy_snapshots.keys())
        )

    return WorkRegimeEpochMetrics(
        epoch=epoch,
        strike_rate=strike_count / max(total_agents, 1),
        quality_degradation=compute_quality_degradation(interactions),
        defection_rate=compute_defection_rate(interactions),
        mean_drift_index=mean_drift,
        max_drift_index=max_drift,
        coalition_strength=coalition,
        legitimacy_score=legitimacy,
        mean_compliance=mean_compliance,
        mean_cooperation_threshold=mean_coop,
        mean_redistribution_pref=mean_redist,
        mean_exit_propensity=mean_exit,
        mean_grievance=mean_grievance,
        gini_payoff=compute_gini(payoff_values),
        agent_snapshots=dict(policy_snapshots),
    )
