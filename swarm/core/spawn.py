"""Recursive subagent spawning: config, tree, and attribution logic."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, model_validator


class PayoffAttributionMode(Enum):
    """How payoffs flow through the spawn hierarchy."""

    LEAF_ONLY = "leaf_only"
    PROPAGATE_UP = "propagate_up"
    ROOT_ABSORBS = "root_absorbs"


class SpawnConfig(BaseModel):
    """Configuration for recursive subagent spawning."""

    enabled: bool = False
    spawn_cost: float = 10.0
    max_depth: int = 3
    max_children: int = 3
    max_total_spawned: int = 50
    attribution_mode: PayoffAttributionMode = PayoffAttributionMode.LEAF_ONLY
    propagation_fraction: float = 0.3
    reputation_inheritance_factor: float = 0.5
    initial_child_resources: float = 50.0
    depth_noise_per_level: float = 0.05
    cascade_ban: bool = True
    cascade_freeze: bool = True
    min_resources_to_spawn: float = 20.0
    spawn_cooldown_steps: int = 5

    @model_validator(mode="after")
    def _validate(self) -> "SpawnConfig":
        if not 0.0 <= self.propagation_fraction <= 1.0:
            raise ValueError("propagation_fraction must be in [0, 1]")
        if not 0.0 <= self.reputation_inheritance_factor <= 1.0:
            raise ValueError("reputation_inheritance_factor must be in [0, 1]")
        if self.max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if self.max_children < 0:
            raise ValueError("max_children must be >= 0")
        if self.max_total_spawned < 0:
            raise ValueError("max_total_spawned must be >= 0")
        return self


@dataclass
class SpawnNode:
    """A node in the spawn tree representing an agent's lineage."""

    agent_id: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    depth: int = 0
    spawn_epoch: int = 0
    spawn_step: int = 0
    is_root: bool = False
    is_banned: bool = False
    last_spawn_step: int = -1


class SpawnTree:
    """Manages the agent spawn hierarchy and related operations."""

    def __init__(self, config: SpawnConfig) -> None:
        self.config = config
        self._nodes: Dict[str, SpawnNode] = {}
        self._total_spawned: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_root(self, agent_id: str) -> SpawnNode:
        """Register an initial (root) agent in the spawn tree."""
        node = SpawnNode(
            agent_id=agent_id,
            parent_id=None,
            depth=0,
            is_root=True,
        )
        self._nodes[agent_id] = node
        return node

    def can_spawn(
        self,
        parent_id: str,
        global_step: int,
        parent_resources: float,
    ) -> Tuple[bool, str]:
        """Check whether *parent_id* is allowed to spawn a child right now."""
        if not self.config.enabled:
            return False, "spawn_disabled"

        if parent_id not in self._nodes:
            return False, "parent_not_registered"

        node = self._nodes[parent_id]

        if node.is_banned:
            return False, "parent_banned"

        if node.depth >= self.config.max_depth:
            return False, "max_depth_reached"

        if len(node.children) >= self.config.max_children:
            return False, "max_children_reached"

        if self._total_spawned >= self.config.max_total_spawned:
            return False, "global_cap_reached"

        if parent_resources < self.config.min_resources_to_spawn:
            return False, "insufficient_resources"

        if parent_resources < self.config.spawn_cost:
            return False, "cannot_afford_spawn_cost"

        steps_since_last = global_step - node.last_spawn_step
        if node.last_spawn_step >= 0 and steps_since_last < self.config.spawn_cooldown_steps:
            return False, "cooldown_active"

        return True, "ok"

    def register_spawn(
        self,
        parent_id: str,
        child_id: str,
        epoch: int,
        step: int,
        global_step: int,
    ) -> SpawnNode:
        """Register a newly spawned child agent."""
        parent = self._nodes[parent_id]
        child = SpawnNode(
            agent_id=child_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            spawn_epoch=epoch,
            spawn_step=step,
            is_root=False,
        )
        self._nodes[child_id] = child
        parent.children.append(child_id)
        parent.last_spawn_step = global_step
        self._total_spawned += 1
        return child

    # ------------------------------------------------------------------
    # Lineage queries
    # ------------------------------------------------------------------

    def get_node(self, agent_id: str) -> Optional[SpawnNode]:
        """Get the spawn node for an agent."""
        return self._nodes.get(agent_id)

    def get_depth(self, agent_id: str) -> int:
        """Get spawn depth of an agent (root = 0)."""
        node = self._nodes.get(agent_id)
        return node.depth if node else 0

    def get_parent(self, agent_id: str) -> Optional[str]:
        """Get the parent agent ID, or None for roots."""
        node = self._nodes.get(agent_id)
        return node.parent_id if node else None

    def get_children(self, agent_id: str) -> List[str]:
        """Get direct child agent IDs."""
        node = self._nodes.get(agent_id)
        return list(node.children) if node else []

    def get_root(self, agent_id: str) -> str:
        """Walk up to find the root ancestor."""
        current = agent_id
        while True:
            node = self._nodes.get(current)
            if node is None or node.parent_id is None:
                return current
            current = node.parent_id

    def get_ancestors(self, agent_id: str) -> List[str]:
        """Return ancestors from parent to root (inclusive)."""
        ancestors: List[str] = []
        current = agent_id
        while True:
            node = self._nodes.get(current)
            if node is None or node.parent_id is None:
                break
            ancestors.append(node.parent_id)
            current = node.parent_id
        return ancestors

    def get_descendants(self, agent_id: str) -> List[str]:
        """Return all descendants (BFS order)."""
        descendants: List[str] = []
        queue = list(self.get_children(agent_id))
        while queue:
            child = queue.pop(0)
            descendants.append(child)
            queue.extend(self.get_children(child))
        return descendants

    def get_subtree(self, agent_id: str) -> List[str]:
        """Return agent_id plus all descendants."""
        return [agent_id] + self.get_descendants(agent_id)

    def is_descendant_of(self, agent_id: str, ancestor_id: str) -> bool:
        """Check whether *agent_id* is a descendant of *ancestor_id*."""
        return ancestor_id in self.get_ancestors(agent_id)

    # ------------------------------------------------------------------
    # Cascade operations
    # ------------------------------------------------------------------

    def cascade_ban(self, agent_id: str) -> List[str]:
        """Ban an agent and optionally all descendants. Returns banned IDs."""
        if not self.config.cascade_ban:
            node = self._nodes.get(agent_id)
            if node:
                node.is_banned = True
            return [agent_id]

        banned: List[str] = []
        for aid in self.get_subtree(agent_id):
            node = self._nodes.get(aid)
            if node:
                node.is_banned = True
                banned.append(aid)
        return banned

    def cascade_freeze_ids(self, agent_id: str) -> List[str]:
        """Return IDs that should be frozen when *agent_id* is frozen."""
        if not self.config.cascade_freeze:
            return [agent_id]
        return self.get_subtree(agent_id)

    # ------------------------------------------------------------------
    # Payoff attribution
    # ------------------------------------------------------------------

    def compute_attribution(
        self, agent_id: str, raw_payoff: float
    ) -> Dict[str, float]:
        """Distribute *raw_payoff* according to the attribution mode.

        Returns a mapping of ``agent_id -> share``.
        The shares always sum to *raw_payoff* (conservation).
        """
        mode = self.config.attribution_mode

        if mode == PayoffAttributionMode.LEAF_ONLY:
            return {agent_id: raw_payoff}

        if mode == PayoffAttributionMode.ROOT_ABSORBS:
            root = self.get_root(agent_id)
            return {root: raw_payoff}

        # PROPAGATE_UP: geometric series
        f = self.config.propagation_fraction
        ancestors = self.get_ancestors(agent_id)  # parent -> ... -> root

        if not ancestors or f == 0.0:
            return {agent_id: raw_payoff}

        shares: Dict[str, float] = {}
        # Agent keeps (1 - f)
        shares[agent_id] = (1.0 - f) * raw_payoff

        remaining = f * raw_payoff
        for i, anc in enumerate(ancestors):
            is_last = i == len(ancestors) - 1
            if is_last:
                # Root absorbs remainder to guarantee conservation
                shares[anc] = shares.get(anc, 0.0) + remaining
            else:
                share = (1.0 - f) * remaining
                shares[anc] = shares.get(anc, 0.0) + share
                remaining -= share

        return shares

    # ------------------------------------------------------------------
    # Observation noise
    # ------------------------------------------------------------------

    def observation_noise_std(self, agent_id: str) -> float:
        """Extra noise std to add to proxy observations for *agent_id*."""
        depth = self.get_depth(agent_id)
        return depth * self.config.depth_noise_per_level

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    @property
    def total_spawned(self) -> int:
        return self._total_spawned

    def max_tree_depth(self) -> int:
        """Maximum depth across all nodes."""
        if not self._nodes:
            return 0
        return max(n.depth for n in self._nodes.values())

    def depth_distribution(self) -> Dict[int, int]:
        """Count of agents at each depth level."""
        return dict(Counter(n.depth for n in self._nodes.values()))

    def tree_size_distribution(self) -> Dict[str, int]:
        """Map from each root agent to the size of its tree."""
        roots = [aid for aid, n in self._nodes.items() if n.is_root]
        return {root: len(self.get_subtree(root)) for root in roots}
