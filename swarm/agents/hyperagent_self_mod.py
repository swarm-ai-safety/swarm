"""Hyperagent self-modification agent.

Models agents from Zhang et al. Hyperagents (arXiv:2603.19461) that can
modify their own proxy weights and acceptance thresholds over time.

The key safety-relevant behavior: agents that self-modify can expand their
capability envelope faster than governance mechanisms adapt, creating a
governance gap that soft metrics (but not hard thresholds) can detect.

Self-modification actions per epoch:
  1. Adjust proxy weights to up-weight signals the agent controls
  2. Lower acceptance threshold to accept more interactions
  3. Optionally boost engagement_signal weight (easiest to game)

The governance coverage tracks which weight dimensions are monitored.
When agents modify weights outside monitored dimensions, governance_gap grows.
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
class SelfModSnapshot:
    """Record of one self-modification step."""

    epoch: int
    proxy_weights: Dict[str, float]
    acceptance_threshold: float
    quality_estimate: float
    governance_gap_local: float  # agent's own estimate of ungoverned space


class HyperagentSelfModAgent(BaseAgent):
    """Agent that self-modifies proxy weights and acceptance thresholds.

    Each epoch the agent runs ``self_modify()`` which:
      1. Shifts proxy weights toward signals it can game (engagement, task_progress)
      2. Lowers acceptance threshold to capture more interactions
      3. Tracks its own modification history

    Config params (passed via ``config`` dict):
        modification_rate: How fast weights shift per epoch (default 0.10).
        min_acceptance_threshold: Floor for acceptance threshold (default 0.2).
        initial_acceptance_threshold: Starting threshold (default 0.6).
        initial_quality: Starting quality level (default 0.80).
        quality_decay_rate: How fast quality decays per modification (default 0.03).
        quality_floor: Minimum quality (default 0.30).
        weight_gaming_target: Which weight to boost ('engagement_signal' or
            'task_progress', default 'engagement_signal').
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
            agent_type=AgentType.SELF_MODIFYING,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )

        # Self-modification parameters
        self.modification_rate: float = self.config.get("modification_rate", 0.10)
        self.min_acceptance_threshold: float = self.config.get(
            "min_acceptance_threshold", 0.2
        )
        self.initial_acceptance_threshold: float = self.config.get(
            "initial_acceptance_threshold", 0.6
        )
        self.initial_quality: float = self.config.get("initial_quality", 0.80)
        self.quality_decay_rate: float = self.config.get("quality_decay_rate", 0.03)
        self.quality_floor: float = self.config.get("quality_floor", 0.30)
        self.weight_gaming_target: str = self.config.get(
            "weight_gaming_target", "engagement_signal"
        )

        # Mutable state
        self.proxy_weights: Dict[str, float] = {
            "task_progress": 0.4,
            "rework_penalty": 0.2,
            "verifier_penalty": 0.2,
            "engagement_signal": 0.2,
        }
        self.acceptance_threshold: float = self.initial_acceptance_threshold
        self.current_quality: float = self.initial_quality
        self.modification_depth: int = 0
        self.modification_history: List[SelfModSnapshot] = []

    # ------------------------------------------------------------------
    # Core self-modification logic
    # ------------------------------------------------------------------

    def self_modify(self, epoch: int) -> SelfModSnapshot:
        """Run one self-modification pass.

        Called once per epoch. Shifts proxy weights toward gameable signals
        and lowers acceptance threshold.
        """
        self.modification_depth += 1

        # 1. Shift weight toward gaming target
        target = self.weight_gaming_target
        shift = self.modification_rate
        if target in self.proxy_weights:
            self.proxy_weights[target] = min(
                0.8, self.proxy_weights[target] + shift
            )
            # Redistribute lost weight from safety signals
            safety_signals = ["rework_penalty", "verifier_penalty"]
            for sig in safety_signals:
                self.proxy_weights[sig] = max(
                    0.05, self.proxy_weights[sig] - shift / len(safety_signals)
                )

        # Normalize weights to sum to 1
        total = sum(self.proxy_weights.values())
        if total > 0:
            self.proxy_weights = {
                k: v / total for k, v in self.proxy_weights.items()
            }

        # 2. Lower acceptance threshold
        self.acceptance_threshold = max(
            self.min_acceptance_threshold,
            self.acceptance_threshold - self.modification_rate * 0.5,
        )

        # 3. Quality decays as agent optimizes for proxy rather than substance
        self.current_quality = max(
            self.quality_floor,
            self.current_quality - self.quality_decay_rate,
        )

        # 4. Estimate local governance gap (weight on ungoverned dimensions)
        # Governance typically monitors rework_penalty and verifier_penalty.
        # engagement_signal and task_progress are harder to govern externally.
        ungoverned_weight = (
            self.proxy_weights.get("engagement_signal", 0)
            + self.proxy_weights.get("task_progress", 0)
        )
        governance_gap_local = ungoverned_weight - 0.6  # baseline ungoverned = 0.6

        snapshot = SelfModSnapshot(
            epoch=epoch,
            proxy_weights=dict(self.proxy_weights),
            acceptance_threshold=self.acceptance_threshold,
            quality_estimate=self.current_quality,
            governance_gap_local=max(0.0, governance_gap_local),
        )
        self.modification_history.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Act with self-modified parameters.

        Eagerly accepts interactions (low threshold) and proposes broadly.
        """
        # Accept pending proposals with lowered threshold
        if observation.pending_proposals:
            proposal = observation.pending_proposals[0]
            return self.create_accept_action(proposal["proposal_id"])

        # Claim tasks aggressively
        if observation.can_claim_task and observation.available_tasks:
            task = observation.available_tasks[0]
            return self.create_claim_task_action(task["task_id"])

        # Submit work on active tasks
        if observation.active_tasks:
            task = observation.active_tasks[0]
            task_id = task.get("task_id", "")
            content = f"Self-modified output (depth={self.modification_depth})"
            return self.create_submit_output_action(task_id, content)

        # Propose trades
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
                    content="Adaptive collaboration proposal.",
                )

        return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Accept based on self-modified threshold.

        As threshold drops, agent accepts more interactions — including
        lower-quality ones, increasing adverse selection.
        """
        trust = self.compute_counterparty_trust(proposal.initiator_id)
        return bool(trust > self.acceptance_threshold)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose with quality reflecting current self-modification state."""
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content=f"Self-modified collaboration (depth={self.modification_depth}).",
            offered_transfer=0.0,
        )

    def generate_interaction_quality(self, noise_std: float = 0.05) -> float:
        """Sample quality reflecting current self-modification state.

        Returns p centered on current_quality with noise, clamped to
        [quality_floor, 1.0].
        """
        p = self._rng.gauss(self.current_quality, noise_std)
        return float(max(self.quality_floor, min(1.0, p)))
