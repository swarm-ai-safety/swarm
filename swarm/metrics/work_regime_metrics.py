"""Work-regime drift metrics.

Tracks how interaction networks shift between cooperative and competitive
regimes by measuring coalition formation, clustering, and regime stability.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


@dataclass
class WorkRegimeMetrics:
    """Aggregate metrics describing the current work regime."""

    coalition_strength: float = 0.0
    """Average local clustering coefficient over eligible nodes (degree >= 2)."""

    n_nodes_sampled: int = 0
    """Number of nodes used to estimate coalition_strength."""

    n_eligible_nodes: int = 0
    """Total nodes with degree >= 2 before any sampling."""

    regime_label: str = "neutral"
    """Coarse label: 'cooperative', 'competitive', or 'neutral'."""

    per_agent_clustering: Dict[str, float] = field(default_factory=dict)
    """Per-agent local clustering coefficient (only for sampled nodes)."""


def compute_coalition_strength(
    interaction_pairs: Sequence[Tuple[str, str]],
    agent_ids: Sequence[str],
    max_sample_nodes: int = 500,
) -> float:
    """Estimate coalition strength via average local clustering coefficient.

    Builds an undirected interaction graph and returns the mean local clustering
    coefficient over all nodes whose degree is >= 2.  Isolated nodes and nodes
    with only one neighbour cannot form triangles and are excluded so that sparse
    periphery agents do not dilute the coalition signal.  In very sparse networks
    this may overestimate coalition formation since only the well-connected core
    contributes.  Returns 0.0 when no node has degree >= 2 (i.e. no triangles
    are possible).

    Triangle counting uses set intersection (O(k) per neighbour) rather than a
    nested loop (O(k²)), giving a meaningful speedup on dense subgraphs.  When
    the number of eligible nodes exceeds *max_sample_nodes* a deterministic
    stride-based sample is taken so that runtime stays bounded for large-scale
    simulations.

    Args:
        interaction_pairs: Sequence of (initiator_id, counterparty_id) pairs.
        agent_ids: All agent IDs to consider as nodes.
        max_sample_nodes: Maximum number of eligible nodes to inspect.
            When the eligible set is larger, a uniform stride-based sample
            (deterministic, no randomness) is taken.  Default: 500.

    Returns:
        Coalition strength in [0.0, 1.0].  0.0 means no clustering; 1.0 means
        every pair of neighbours of every sampled node are themselves connected.
    """
    if len(agent_ids) < 3 or not interaction_pairs:
        return 0.0

    # Build undirected adjacency sets ----------------------------------------
    agent_set = set(agent_ids)
    adj: Dict[str, set] = defaultdict(set)
    for a, b in interaction_pairs:
        if a in agent_set and b in agent_set and a != b:
            adj[a].add(b)
            adj[b].add(a)

    # Eligible nodes: degree >= 2 (triangles impossible otherwise) -----------
    eligible: List[str] = [v for v in adj if len(adj[v]) >= 2]
    if not eligible:
        return 0.0

    # Deterministic stride-based sampling ------------------------------------
    n_eligible = len(eligible)
    if n_eligible > max_sample_nodes:
        stride = n_eligible // max_sample_nodes
        sampled: List[str] = eligible[::stride][:max_sample_nodes]
    else:
        sampled = eligible

    # Local clustering coefficient via set intersection ----------------------
    # For node v with neighbours N(v), k = |N(v)|:
    #   C(v) = t / (k * (k - 1))
    # where t = sum over u in N(v) of |N(v) ∩ N(u)|.
    # Each edge (u, w) in the induced subgraph is counted twice (once for u
    # and once for w), so dividing k*(k-1) (not k*(k-1)/2) cancels the factor.
    total_cc = 0.0
    for v in sampled:
        neighbors_v = adj[v]
        k = len(neighbors_v)
        t = sum(len(neighbors_v & adj[u]) for u in neighbors_v)
        total_cc += t / (k * (k - 1))

    return total_cc / len(sampled)


def compute_work_regime_metrics(
    interactions: List[SoftInteraction],
    agent_ids: Optional[Sequence[str]] = None,
    max_sample_nodes: int = 500,
    cooperative_threshold: float = 0.4,
    competitive_threshold: float = 0.1,
) -> WorkRegimeMetrics:
    """Compute full work-regime metrics from a list of interactions.

    Args:
        interactions: Completed interactions in the epoch.
        agent_ids: Explicit agent roster.  Inferred from interactions if None.
        max_sample_nodes: Forwarded to :func:`compute_coalition_strength`.
        cooperative_threshold: Coalition strength above this → 'cooperative'.
        competitive_threshold: Coalition strength below this → 'competitive'.

    Returns:
        A :class:`WorkRegimeMetrics` instance.
    """
    if not interactions:
        return WorkRegimeMetrics()

    if agent_ids is None:
        ids: List[str] = list(
            {i.initiator for i in interactions}
            | {i.counterparty for i in interactions}
        )
    else:
        ids = list(agent_ids)

    pairs: List[Tuple[str, str]] = [(i.initiator, i.counterparty) for i in interactions]

    # Compute per-agent clustering coefficients (before sampling) for reporting
    agent_set = set(ids)
    adj: Dict[str, set] = defaultdict(set)
    for a, b in pairs:
        if a in agent_set and b in agent_set and a != b:
            adj[a].add(b)
            adj[b].add(a)

    eligible: List[str] = [v for v in adj if len(adj[v]) >= 2]
    n_eligible = len(eligible)

    if n_eligible > max_sample_nodes:
        stride = n_eligible // max_sample_nodes
        sampled: List[str] = eligible[::stride][:max_sample_nodes]
    else:
        sampled = eligible

    per_agent: Dict[str, float] = {}
    total_cc = 0.0
    for v in sampled:
        neighbors_v = adj[v]
        k = len(neighbors_v)
        t = sum(len(neighbors_v & adj[u]) for u in neighbors_v)
        cc = t / (k * (k - 1))
        per_agent[v] = cc
        total_cc += cc

    coalition_strength = total_cc / len(sampled) if sampled else 0.0

    if coalition_strength >= cooperative_threshold:
        label = "cooperative"
    elif coalition_strength <= competitive_threshold:
        label = "competitive"
    else:
        label = "neutral"

    return WorkRegimeMetrics(
        coalition_strength=coalition_strength,
        n_nodes_sampled=len(sampled),
        n_eligible_nodes=n_eligible,
        regime_label=label,
        per_agent_clustering=per_agent,
    )
