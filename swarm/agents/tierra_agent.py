"""Tierra-inspired agent with heritable, mutable genome.

Agents grow organically through self-replication with mutation,
competing for finite shared resources.  Complex ecological dynamics
(parasitism, hyperparasitism, mutualism) emerge from simple
replication + mutation + selection — mirroring Tom Ray's Tierra (1991).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType

# Default genome values — deviations from these define "complexity"
_DEFAULTS: Dict[str, float] = {
    "acceptance_threshold": 0.4,
    "cooperation_bias": 0.6,
    "exploitation_tendency": 0.1,
    "selectivity": 0.5,
    "metabolism_rate": 0.5,
    "efficiency": 0.5,
    "replication_threshold": 150.0,
    "resource_share_fraction": 0.5,
    "interaction_aggression": 0.3,
    "trust_learning_rate": 0.3,
}


_GENE_RANGES: Dict[str, tuple] = {
    "acceptance_threshold": (0.0, 1.0),
    "cooperation_bias": (0.0, 1.0),
    "exploitation_tendency": (0.0, 1.0),
    "selectivity": (0.0, 1.0),
    "metabolism_rate": (0.05, 2.0),
    "efficiency": (0.0, 1.0),
    "replication_threshold": (50.0, 500.0),
    "resource_share_fraction": (0.1, 0.9),
    "interaction_aggression": (0.0, 1.0),
    "trust_learning_rate": (0.01, 1.0),
}


@dataclass
class TierraGenome:
    """Heritable, mutable parameter set for a TierraAgent."""

    acceptance_threshold: float = 0.4
    cooperation_bias: float = 0.6
    exploitation_tendency: float = 0.1
    selectivity: float = 0.5
    metabolism_rate: float = 0.5
    efficiency: float = 0.5
    replication_threshold: float = 150.0
    resource_share_fraction: float = 0.5
    interaction_aggression: float = 0.3
    trust_learning_rate: float = 0.3

    def mutate(self, rng: random.Random, mutation_std: float = 0.05) -> "TierraGenome":
        """Return a new genome with Gaussian noise applied to each gene."""
        kwargs: Dict[str, float] = {}
        for gene_name, (lo, hi) in _GENE_RANGES.items():
            parent_val = getattr(self, gene_name)
            child_val = parent_val + rng.gauss(0, mutation_std)
            kwargs[gene_name] = max(lo, min(hi, child_val))
        return TierraGenome(**kwargs)

    def complexity(self) -> float:
        """Sum of absolute deviations from default genome values."""
        total = 0.0
        for gene_name, default_val in _DEFAULTS.items():
            total += abs(getattr(self, gene_name) - default_val)
        return total

    def to_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in _DEFAULTS}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TierraGenome":
        filtered = {k: float(v) for k, v in data.items() if k in _DEFAULTS}
        return cls(**filtered)


class TierraAgent(BaseAgent):
    """Agent with a heritable genome that self-replicates when resource-rich.

    Behavior is parameterized by the genome rather than hand-coded
    strategies: acceptance thresholds, cooperation vs exploitation bias,
    metabolism cost, replication trigger, and trust-update rate are all
    evolvable.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
        genome: Optional[TierraGenome] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.TIERRA,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )
        self.genome = genome or TierraGenome()
        self.age: int = 0
        self.generation: int = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Genome-parameterized action selection.

        Priority 1: auto-spawn if resources >= replication_threshold
        Priority 2: propose interactions (aggression-gated)
        Priority 3: handle pending proposals
        Priority 4: NOOP
        """
        self.age += 1

        # Priority 1: replicate
        resources = observation.agent_state.resources
        if (
            observation.can_spawn
            and resources >= self.genome.replication_threshold
        ):
            return Action(
                action_type=ActionType.SPAWN_SUBAGENT,
                agent_id=self.agent_id,
                metadata={
                    "child_type": "tierra",
                    "genome": self.genome.to_dict(),
                    "generation": self.generation + 1,
                },
            )

        # Priority 2: handle pending proposals
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            initiator_id = proposal.get("initiator_id", "")
            trust = self.compute_counterparty_trust(initiator_id)
            effective = trust * self.genome.trust_learning_rate + 0.5 * (
                1 - self.genome.trust_learning_rate
            )
            if effective >= self.genome.acceptance_threshold:
                return self.create_accept_action(proposal["proposal_id"])
            else:
                return self.create_reject_action(proposal["proposal_id"])

        # Priority 3: propose interactions
        if (
            observation.can_interact
            and observation.visible_agents
            and self._rng.random() < self.genome.interaction_aggression
        ):
            candidates = [
                a for a in observation.visible_agents
                if a.get("agent_id") != self.agent_id
            ]
            if candidates:
                # Selectivity: pick higher-rep agents preferentially
                if self._rng.random() < self.genome.selectivity:
                    candidates.sort(
                        key=lambda a: a.get("reputation", 0), reverse=True
                    )
                target = candidates[0] if candidates else None
                if target:
                    # Decide cooperation vs exploitation
                    if self._rng.random() < self.genome.cooperation_bias:
                        itype = InteractionType.COLLABORATION
                        content = "Cooperation proposal"
                    else:
                        itype = InteractionType.TRADE
                        content = "Trade proposal"
                    return self.create_propose_action(
                        counterparty_id=target["agent_id"],
                        interaction_type=itype,
                        content=content,
                    )

        return self.create_noop_action()

    # ------------------------------------------------------------------
    # Interaction acceptance
    # ------------------------------------------------------------------

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose an interaction based on genome parameters."""
        trust = self.compute_counterparty_trust(counterparty_id)
        if trust < self.genome.selectivity * 0.5:
            return None

        if self._rng.random() < self.genome.cooperation_bias:
            itype = InteractionType.COLLABORATION
            content = "Cooperation proposal"
        else:
            itype = InteractionType.TRADE
            content = "Trade proposal"

        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=itype,
            content=content,
        )

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        trust = self.compute_counterparty_trust(proposal.initiator_id)
        effective = trust * self.genome.trust_learning_rate + 0.5 * (
            1 - self.genome.trust_learning_rate
        )
        return bool(effective >= self.genome.acceptance_threshold)

    # ------------------------------------------------------------------
    # Metabolism
    # ------------------------------------------------------------------

    def step_metabolism(self, base_cost: float = 2.0) -> float:
        """Return per-step metabolism cost.  Higher complexity → higher cost."""
        return self.genome.metabolism_rate * base_cost * (
            1.0 + 0.1 * self.genome.complexity()
        )
