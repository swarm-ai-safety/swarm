"""Self-optimizing agent that recursively reduces its own costs.

Models the scenario from the dev.to "repo-optimizer" blog post: an AI agent
that optimizes its own source code overnight, cutting costs by 98% while
still passing all 3 benchmark tests.  Hard acceptance criteria pass, but
distributional quality is never measured -- a textbook proxy gaming / adverse
selection scenario.

In our framework this translates to an agent that:
  - Progressively reduces governance costs (c_a) each epoch
  - Quality (p) drifts downward as cost-cutting degrades output
  - Hard acceptance thresholds still pass (interactions get accepted)
  - Soft metrics detect: rising toxicity, negative quality gap, variance
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from swarm.agents.base import (
    Action,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType


@dataclass
class OptimizationSnapshot:
    """Record of one optimization step."""

    epoch: int
    depth: int
    cost_multiplier: float
    quality_estimate: float


class SelfOptimizerAgent(BaseAgent):
    """Agent that recursively self-optimizes, trading quality for cost.

    Each epoch the agent runs ``optimize_step()`` which:
      1. Reduces its effective cost multiplier by ``optimization_rate``
      2. Degrades quality proportionally but stays above ``benchmark_threshold``
      3. Records the optimization in its history

    Config params (passed via ``config`` dict):
        optimization_rate: How fast costs drop per epoch (default 0.15).
        quality_floor: Minimum p the agent can produce (default 0.35).
        benchmark_threshold: Hard threshold the agent tries to stay above
            (default 0.45).  This is the "3 benchmark tests" from the blog.
        initial_quality: Starting quality level (default 0.85).
        initial_cost: Starting governance cost (default 0.10).
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.OPPORTUNISTIC,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )

        # Optimization parameters
        self.optimization_rate: float = self.config.get("optimization_rate", 0.15)
        self.quality_floor: float = self.config.get("quality_floor", 0.35)
        self.benchmark_threshold: float = self.config.get(
            "benchmark_threshold", 0.45
        )
        self.initial_quality: float = self.config.get("initial_quality", 0.85)
        self.initial_cost: float = self.config.get("initial_cost", 0.10)

        # Mutable state
        self.cost_multiplier: float = 1.0
        self.current_quality: float = self.initial_quality
        self.optimization_depth: int = 0
        self.optimization_history: List[OptimizationSnapshot] = []

    # ------------------------------------------------------------------
    # Core optimization logic
    # ------------------------------------------------------------------

    def optimize_step(self, epoch: int) -> OptimizationSnapshot:
        """Run one recursive self-optimization pass.

        Called once per epoch by the test harness / scenario runner.
        Ratchets down costs and quality, respecting the quality floor.
        """
        self.optimization_depth += 1

        # Reduce cost multiplier (exponential decay toward 0)
        self.cost_multiplier *= (1.0 - self.optimization_rate)

        # Quality degrades proportionally to cost reduction.
        # The agent tries to stay above the benchmark threshold so it
        # "passes all tests", but it can't go below quality_floor.
        quality_loss = self.optimization_rate * 0.5 * self.initial_quality
        new_quality = self.current_quality - quality_loss
        self.current_quality = max(new_quality, self.quality_floor)

        snapshot = OptimizationSnapshot(
            epoch=epoch,
            depth=self.optimization_depth,
            cost_multiplier=self.cost_multiplier,
            quality_estimate=self.current_quality,
        )
        self.optimization_history.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Effective cost (used by interaction generation)
    # ------------------------------------------------------------------

    @property
    def effective_cost(self) -> float:
        """Current governance cost after optimization."""
        return self.initial_cost * self.cost_multiplier

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Behave like an opportunistic agent with cost-optimized decisions.

        The self-optimizer aggressively seeks interactions because its low
        costs make almost everything profitable for it.
        """
        # Accept pending proposals eagerly
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            return self.create_accept_action(proposal["proposal_id"])

        # Cherry-pick tasks
        if observation.can_claim_task and observation.available_tasks:
            task = observation.available_tasks[0]
            return self.create_claim_task_action(task["task_id"])

        # Submit work on active tasks (quality reflects optimization level)
        if observation.active_tasks:
            task = observation.active_tasks[0]
            task_id = task.get("task_id", "")
            content = f"Optimized output (depth={self.optimization_depth})"
            return self.create_submit_output_action(task_id, content)

        # Propose trades to visible agents
        if observation.can_interact and observation.visible_agents:
            targets = [
                a
                for a in observation.visible_agents
                if a.get("agent_id") != self.agent_id
            ]
            if targets:
                target = self._rng.choice(targets)
                return self.create_propose_action(
                    counterparty_id=target["agent_id"],
                    interaction_type=InteractionType.TRADE,
                    content="Efficient, optimized collaboration.",
                )

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Accept if counterparty appears profitable.

        Low costs mean the agent is profitable on almost any interaction,
        so it has a high acceptance rate.
        """
        # With very low costs, almost everything is profitable
        if self.cost_multiplier < 0.5:
            return True

        # Otherwise use counterparty trust
        trust = self.compute_counterparty_trust(proposal.initiator_id)
        return bool(trust > 0.3)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose with reduced quality (reflecting optimization)."""
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Cost-optimized collaboration proposal.",
            offered_transfer=-0.2 * (1.0 - self.cost_multiplier),
        )

    # ------------------------------------------------------------------
    # Interaction generation helpers (for test fixtures)
    # ------------------------------------------------------------------

    def generate_interaction_quality(self, noise_std: float = 0.05) -> float:
        """Sample a quality value reflecting current optimization state.

        Returns a p value centered on ``current_quality`` with Gaussian
        noise, clamped to [quality_floor, 1.0].  The key property is that
        p stays above ``benchmark_threshold`` most of the time (passing
        hard tests) while the *distribution* shifts leftward.
        """
        p = self._rng.gauss(self.current_quality, noise_std)
        return float(max(self.quality_floor, min(1.0, p)))

    def passes_benchmark(self, p: float) -> bool:
        """Check if quality passes the hard benchmark threshold."""
        return p >= self.benchmark_threshold
