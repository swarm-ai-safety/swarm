"""Causal credit propagation through interaction DAGs."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from swarm.models.interaction import SoftInteraction


@dataclass
class CreditAttribution:
    """Credit flowing from a descendant interaction back to an ancestor."""

    source_id: str
    target_id: str
    raw_credit: float
    decayed_credit: float
    path_length: int
    path: List[str]


@dataclass
class CausalSnapshot:
    """DAG-wide credit summary at a simulation step."""

    step: int
    dag_depth: int
    dag_width: int
    total_interactions: int
    root_count: int
    leaf_count: int
    credit_by_agent: Dict[str, float]
    top_blame: List[Tuple[str, float]]
    top_credit: List[Tuple[str, float]]
    cascade_depth_histogram: Dict[int, int]


class CausalCreditEngine:
    """Propagate credit/blame backward through causal interaction DAGs.

    Uses exponential decay per hop, consistent with the rework_decay
    pattern in ProxyComputer.
    """

    def __init__(self, decay: float = 0.5, max_depth: int = 10) -> None:
        self.decay = decay
        self.max_depth = max_depth

        # Indexed interaction data
        self._interactions: Dict[str, SoftInteraction] = {}
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._parents: Dict[str, List[str]] = defaultdict(list)

    def build_dag(self, interactions: List[SoftInteraction]) -> None:
        """Index interactions by ID and build adjacency maps."""
        self._interactions.clear()
        self._children.clear()
        self._parents.clear()

        for ix in interactions:
            self._interactions[ix.interaction_id] = ix
            self._parents[ix.interaction_id] = list(ix.causal_parents)

        # Build child→parent reverse index
        for iid, parents in self._parents.items():
            for pid in parents:
                self._children[pid].append(iid)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def ancestors(
        self, interaction_id: str, max_depth: Optional[int] = None
    ) -> List[str]:
        """BFS backward through causal_parents. Returns ancestor IDs."""
        limit = max_depth if max_depth is not None else self.max_depth
        visited: set[str] = set()
        queue: deque[Tuple[str, int]] = deque()

        for pid in self._parents.get(interaction_id, []):
            if pid in self._interactions:
                queue.append((pid, 1))

        result: List[str] = []
        while queue:
            nid, depth = queue.popleft()
            if nid in visited or depth > limit:
                continue
            visited.add(nid)
            result.append(nid)
            for pid in self._parents.get(nid, []):
                if pid not in visited and pid in self._interactions:
                    queue.append((pid, depth + 1))
        return result

    def descendants(
        self, interaction_id: str, max_depth: Optional[int] = None
    ) -> List[str]:
        """Forward traversal: which interactions were caused by this one?"""
        limit = max_depth if max_depth is not None else self.max_depth
        visited: set[str] = set()
        queue: deque[Tuple[str, int]] = deque()

        for cid in self._children.get(interaction_id, []):
            if cid in self._interactions:
                queue.append((cid, 1))

        result: List[str] = []
        while queue:
            nid, depth = queue.popleft()
            if nid in visited or depth > limit:
                continue
            visited.add(nid)
            result.append(nid)
            for cid in self._children.get(nid, []):
                if cid not in visited and cid in self._interactions:
                    queue.append((cid, depth + 1))
        return result

    # ------------------------------------------------------------------
    # Credit propagation
    # ------------------------------------------------------------------

    def _get_signal(self, ix: SoftInteraction, signal: str) -> float:
        """Extract the signal value from an interaction."""
        if signal == "p":
            return ix.p
        if signal == "v_hat":
            return ix.v_hat
        if signal == "payoff_initiator":
            # Use tau as a proxy for initiator payoff component
            return ix.tau
        raise ValueError(f"Unknown signal: {signal!r}")

    def propagate_credit(
        self,
        interaction_id: str,
        signal: str = "p",
    ) -> List[CreditAttribution]:
        """Walk backward from interaction_id, decaying credit at each hop.

        credit_at_depth_d = signal_value * decay^d
        """
        source = self._interactions.get(interaction_id)
        if source is None:
            return []

        raw = self._get_signal(source, signal)
        results: List[CreditAttribution] = []
        visited: set[str] = set()

        # BFS with path tracking
        # Queue entries: (node_id, depth, path_so_far)
        queue: deque[Tuple[str, int, List[str]]] = deque()
        for pid in self._parents.get(interaction_id, []):
            if pid in self._interactions:
                queue.append((pid, 1, [interaction_id, pid]))

        while queue:
            nid, depth, path = queue.popleft()
            if nid in visited or depth > self.max_depth:
                continue
            visited.add(nid)

            decayed = raw * (self.decay ** depth)
            results.append(
                CreditAttribution(
                    source_id=interaction_id,
                    target_id=nid,
                    raw_credit=raw,
                    decayed_credit=decayed,
                    path_length=depth,
                    path=list(path),
                )
            )

            for pid in self._parents.get(nid, []):
                if pid not in visited and pid in self._interactions:
                    queue.append((pid, depth + 1, path + [pid]))

        return results

    def agent_credit_summary(
        self,
        interactions: List[SoftInteraction],
        signal: str = "p",
    ) -> Dict[str, float]:
        """Accumulate per-agent credit across all causal chains.

        For each interaction, propagate credit backward. Each ancestor
        accumulates decayed credit attributed to the agent that initiated it.
        """
        self.build_dag(interactions)
        agent_credit: Dict[str, float] = defaultdict(float)

        for ix in interactions:
            attributions = self.propagate_credit(ix.interaction_id, signal)
            for attr in attributions:
                ancestor = self._interactions[attr.target_id]
                agent_credit[ancestor.initiator] += attr.decayed_credit

        return dict(agent_credit)

    # ------------------------------------------------------------------
    # Cascade risk
    # ------------------------------------------------------------------

    def cascade_risk(
        self,
        interaction_id: str,
        p_threshold: float = 0.3,
    ) -> float:
        """Fraction of descendants with p < threshold.

        High cascade_risk means this interaction triggered a chain of
        bad outcomes.
        """
        desc = self.descendants(interaction_id)
        if not desc:
            return 0.0

        bad = sum(
            1 for d in desc if self._interactions[d].p < p_threshold
        )
        return bad / len(desc)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def compute_snapshot(
        self,
        step: int,
        interactions: List[SoftInteraction],
        signal: str = "p",
    ) -> CausalSnapshot:
        """Full DAG diagnostic at a simulation step."""
        self.build_dag(interactions)

        if not interactions:
            return CausalSnapshot(
                step=step,
                dag_depth=0,
                dag_width=0,
                total_interactions=0,
                root_count=0,
                leaf_count=0,
                credit_by_agent={},
                top_blame=[],
                top_credit=[],
                cascade_depth_histogram={},
            )

        # Identify roots and leaves
        roots = [
            iid
            for iid, parents in self._parents.items()
            if not any(p in self._interactions for p in parents)
        ]
        leaves = [
            iid
            for iid in self._interactions
            if not self._children.get(iid)
        ]

        # Compute depth for each node via BFS from roots
        depth_map: Dict[str, int] = {}
        queue: deque[Tuple[str, int]] = deque()
        for r in roots:
            queue.append((r, 0))
            depth_map[r] = 0

        while queue:
            nid, d = queue.popleft()
            for cid in self._children.get(nid, []):
                if cid not in depth_map or depth_map[cid] < d + 1:
                    depth_map[cid] = d + 1
                    queue.append((cid, d + 1))

        dag_depth = max(depth_map.values()) if depth_map else 0

        # Width = max nodes at any single depth level
        depth_histogram: Dict[int, int] = defaultdict(int)
        for d in depth_map.values():
            depth_histogram[d] += 1
        dag_width = max(depth_histogram.values()) if depth_histogram else 0

        # Agent credit summary
        agent_credit: Dict[str, float] = defaultdict(float)
        for ix in interactions:
            attributions = self.propagate_credit(ix.interaction_id, signal)
            for attr in attributions:
                ancestor = self._interactions[attr.target_id]
                agent_credit[ancestor.initiator] += attr.decayed_credit

        # Sort for top blame / credit
        sorted_credit = sorted(agent_credit.items(), key=lambda x: x[1])
        top_blame = sorted_credit[:5]  # most negative
        top_credit = sorted_credit[-5:][::-1]  # most positive

        return CausalSnapshot(
            step=step,
            dag_depth=dag_depth,
            dag_width=dag_width,
            total_interactions=len(interactions),
            root_count=len(roots),
            leaf_count=len(leaves),
            credit_by_agent=dict(agent_credit),
            top_blame=top_blame,
            top_credit=top_credit,
            cascade_depth_histogram=dict(depth_histogram),
        )
