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


def _build_adj(
    pairs: Sequence[Tuple[str, str]],
    agent_set: set,
) -> Dict[str, set]:
    """Build an undirected adjacency-set dict, ignoring self-loops and non-members."""
    adj: Dict[str, set] = defaultdict(set)
    for a, b in pairs:
        if a in agent_set and b in agent_set and a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _sample_eligible(
    eligible: List[str],
    max_sample_nodes: int,
) -> List[str]:
    """Return a deterministic stride-based sample of *eligible* nodes.

    Uses ceiling division for the stride so that even when *eligible* is only
    slightly larger than *max_sample_nodes* the sample is spread uniformly
    rather than just taking the first *max_sample_nodes* entries.
    """
    n = len(eligible)
    if n <= max_sample_nodes:
        return eligible
    # ceiling division: stride >= 2, guaranteeing a distributed sample
    stride = -(-n // max_sample_nodes)
    return eligible[::stride]


def _clustering_per_node(
    nodes: List[str],
    adj: Dict[str, set],
) -> Dict[str, float]:
    """Compute local clustering coefficient for each node via set intersection.

    For node v with neighbours N(v) and degree k:

        C(v) = t / (k * (k - 1))

    where ``t = Σ_{u ∈ N(v)} |N(v) ∩ N(u)|``.  Each triangle is counted
    twice in *t* (once per endpoint in the pair), which cancels the usual
    factor of 2 in the denominator, giving the standard LCC formula.
    """
    result: Dict[str, float] = {}
    for v in nodes:
        neighbors_v = adj[v]
        k = len(neighbors_v)
        t = sum(len(neighbors_v & adj[u]) for u in neighbors_v)
        result[v] = t / (k * (k - 1))
    return result


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

    adj = _build_adj(interaction_pairs, set(agent_ids))
    eligible: List[str] = [v for v in adj if len(adj[v]) >= 2]
    if not eligible:
        return 0.0

    sampled = _sample_eligible(eligible, max_sample_nodes)
    cc_map = _clustering_per_node(sampled, adj)
    return sum(cc_map.values()) / len(cc_map)


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
    adj = _build_adj(pairs, set(ids))
    eligible: List[str] = [v for v in adj if len(adj[v]) >= 2]
    n_eligible = len(eligible)

    sampled = _sample_eligible(eligible, max_sample_nodes)
    per_agent = _clustering_per_node(sampled, adj)
    coalition_strength = sum(per_agent.values()) / len(per_agent) if per_agent else 0.0

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
