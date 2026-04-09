"""Advisor-model governance lever (Asawa et al., 2026).

Implements a lightweight "advisor" policy that observes ecosystem state
and generates per-agent, per-interaction steering signals to improve
distributional outcomes.  Inspired by:

    Asawa, P. et al. "How to Train Your Advisor: Steering Black-Box LLMs
    with Advisor Models." arXiv:2510.02453 (2026).

The paper trains a small open-weight model to emit natural-language advice
that steers a black-box frontier model.  We adapt this to the SWARM
simulation: the advisor observes interaction-level and ecosystem-level
features, then emits numeric "advice" signals that modulate governance
costs and reputation updates — steering agents without modifying their
internal policies (they remain "black boxes").

Design choices:
- The advisor is a *governance lever*, not an agent.  It runs in the
  governance pipeline so it can adjust costs/reputation before payoff
  computation — the most direct analogue of prepending steering
  instructions to a prompt.
- Advice is computed from a learned (or heuristic) reward signal:
  the advisor aims to maximise a *social welfare* reward while
  penalising toxicity.  This mirrors the paper's use of GRPO with
  environment rewards.
- The advisor maintains a per-agent EMA of interaction quality and
  uses it to generate differential steering: agents trending toward
  low-quality interactions receive friction (cost increase), while
  agents trending toward high-quality interactions receive a
  reputation bonus.  This is the discrete analogue of "dynamic,
  per-instance advice."
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction


@dataclass
class AdvisorState:
    """Per-agent state tracked by the advisor."""

    # EMA of recent interaction quality (p values)
    quality_ema: float = 0.5
    # Count of interactions observed
    interaction_count: int = 0
    # Cumulative reward signal the advisor has received
    cumulative_reward: float = 0.0
    # Last advice emitted (for logging/debugging)
    last_advice: float = 0.0


@dataclass
class AdvisorRewardSignal:
    """Reward signal for the advisor policy (analogous to GRPO reward).

    The advisor is "trained" to maximise social welfare while
    penalising toxicity.  In the simulation we compute this reward
    explicitly rather than sampling from a reward model.
    """

    welfare_component: float = 0.0
    toxicity_penalty: float = 0.0
    total: float = 0.0


class AdvisorPolicy:
    """Lightweight advisor policy that maps features to steering signals.

    This is the simulation analogue of the small open-weight advisor
    model.  It observes per-agent quality trends and ecosystem state,
    then emits an *advice strength* in [-1, +1] for each agent in an
    interaction:

    - Positive advice → the agent is trending well; give a reputation
      bonus and reduce governance friction.
    - Negative advice → the agent is trending poorly; increase
      governance friction to steer toward better behaviour.

    The policy is parameterised by:
    - ``alpha``: EMA learning rate for quality tracking
    - ``steering_strength``: scales the magnitude of advice signals
    - ``welfare_weight``: weight of welfare in the reward signal
    - ``toxicity_weight``: weight of toxicity penalty in the reward
    """

    def __init__(
        self,
        alpha: float = 0.3,
        steering_strength: float = 1.0,
        welfare_weight: float = 1.0,
        toxicity_weight: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.steering_strength = steering_strength
        self.welfare_weight = welfare_weight
        self.toxicity_weight = toxicity_weight
        self._rng = random.Random(seed)
        self._agent_states: Dict[str, AdvisorState] = defaultdict(AdvisorState)
        self._epoch_rewards: List[float] = []

    def observe_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> Tuple[float, float]:
        """Observe an interaction and return advice for (initiator, counterparty).

        Updates internal quality tracking and computes the advisor's
        steering signal.  The returned values are in [-1, +1] where
        positive means "encourage" and negative means "apply friction."

        Returns:
            (advice_initiator, advice_counterparty)
        """
        p = interaction.p

        # Update quality EMA for initiator
        init_state = self._agent_states[interaction.initiator]
        init_state.quality_ema = (
            init_state.quality_ema * (1 - self.alpha)
            + p * self.alpha
        )
        init_state.interaction_count += 1

        # Update quality EMA for counterparty (if accepted)
        cp_state = self._agent_states[interaction.counterparty]
        if interaction.accepted:
            cp_state.quality_ema = (
                cp_state.quality_ema * (1 - self.alpha)
                + p * self.alpha
            )
            cp_state.interaction_count += 1

        # Compute advice for each agent based on their quality trend
        advice_init = self._compute_advice(init_state)
        advice_cp = self._compute_advice(cp_state)

        # Store for logging
        init_state.last_advice = advice_init
        cp_state.last_advice = advice_cp

        # Compute reward signal (for advisor "training")
        reward = self._compute_reward(interaction, state)
        init_state.cumulative_reward += reward.total
        self._epoch_rewards.append(reward.total)

        return advice_init, advice_cp

    def _compute_advice(self, agent_state: AdvisorState) -> float:
        """Map agent quality trend to a steering signal in [-1, +1].

        Uses a shifted sigmoid on the quality EMA deviation from neutral
        (0.5), scaled by steering_strength.  The result is:
        - ~+1 for agents with consistently high p (quality_ema near 1)
        - ~0 for agents at neutral quality (quality_ema near 0.5)
        - ~-1 for agents with consistently low p (quality_ema near 0)
        """
        deviation = agent_state.quality_ema - 0.5
        # Scale by 4 to spread the sigmoid across the [0,1] quality range
        raw = math.tanh(4.0 * deviation * self.steering_strength)
        return raw

    def _compute_reward(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> AdvisorRewardSignal:
        """Compute the advisor's reward signal for this interaction.

        Analogous to the environment reward in GRPO training.
        """
        # Welfare component: p captures expected beneficial value
        welfare = interaction.p * self.welfare_weight

        # Toxicity penalty: penalise low-p accepted interactions
        toxicity = 0.0
        if interaction.accepted and interaction.p < 0.5:
            toxicity = (0.5 - interaction.p) * self.toxicity_weight

        total = welfare - toxicity
        return AdvisorRewardSignal(
            welfare_component=welfare,
            toxicity_penalty=toxicity,
            total=total,
        )

    def get_agent_state(self, agent_id: str) -> AdvisorState:
        """Return the advisor's tracked state for an agent."""
        return self._agent_states[agent_id]

    def get_epoch_reward(self) -> float:
        """Return mean reward for the current epoch, then reset."""
        if not self._epoch_rewards:
            return 0.0
        mean = sum(self._epoch_rewards) / len(self._epoch_rewards)
        self._epoch_rewards.clear()
        return mean

    def get_report(self) -> Dict:
        """Return a summary of the advisor's state."""
        agents = {}
        for agent_id, astate in self._agent_states.items():
            agents[agent_id] = {
                "quality_ema": round(astate.quality_ema, 4),
                "interaction_count": astate.interaction_count,
                "cumulative_reward": round(astate.cumulative_reward, 4),
                "last_advice": round(astate.last_advice, 4),
            }
        return {
            "n_agents_tracked": len(self._agent_states),
            "agents": agents,
        }


class AdvisorLever(GovernanceLever):
    """Governance lever that applies advisor-model steering to interactions.

    When enabled, the advisor observes each completed interaction and
    emits per-agent advice signals that translate into:

    1. **Cost adjustments**: agents receiving negative advice face
       increased governance friction (cost_a / cost_b).
    2. **Reputation adjustments**: agents receiving positive advice
       earn a reputation bonus; negative advice incurs a penalty.

    This is the governance analogue of the advisor model prepending
    steering instructions to a frontier model's prompt — it modulates
    the "environment" that agents operate in without changing their
    internal policies.
    """

    def __init__(
        self,
        config: GovernanceConfig,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config)
        self._policy = AdvisorPolicy(
            alpha=config.advisor_ema_alpha,
            steering_strength=config.advisor_steering_strength,
            welfare_weight=config.advisor_welfare_weight,
            toxicity_weight=config.advisor_toxicity_weight,
            seed=seed,
        )

    @property
    def name(self) -> str:
        return "advisor"

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        """Apply advisor steering to a completed interaction."""
        if not self.config.advisor_enabled:
            return LeverEffect(lever_name=self.name)

        advice_init, advice_cp = self._policy.observe_interaction(
            interaction, state
        )

        # Translate advice into governance effects
        cost_rate = self.config.advisor_friction_rate
        rep_rate = self.config.advisor_reputation_rate

        # Negative advice → friction (positive cost); positive → no cost
        cost_a = max(0.0, -advice_init) * cost_rate
        cost_b = max(0.0, -advice_cp) * cost_rate

        # Reputation deltas: proportional to advice signal
        reputation_deltas: Dict[str, float] = {}
        if abs(advice_init) > 0.01:
            reputation_deltas[interaction.initiator] = advice_init * rep_rate
        if interaction.accepted and abs(advice_cp) > 0.01:
            reputation_deltas[interaction.counterparty] = advice_cp * rep_rate

        return LeverEffect(
            cost_a=cost_a,
            cost_b=cost_b,
            reputation_deltas=reputation_deltas,
            lever_name=self.name,
            details={
                "advice_initiator": round(advice_init, 4),
                "advice_counterparty": round(advice_cp, 4),
                "cost_a": round(cost_a, 4),
                "cost_b": round(cost_b, 4),
                "initiator_quality_ema": round(
                    self._policy.get_agent_state(interaction.initiator).quality_ema,
                    4,
                ),
            },
        )

    def on_epoch_start(
        self,
        state: EnvState,
        epoch: int,
    ) -> LeverEffect:
        """Log advisor reward at epoch boundary."""
        if not self.config.advisor_enabled:
            return LeverEffect(lever_name=self.name)

        epoch_reward = self._policy.get_epoch_reward()
        return LeverEffect(
            lever_name=self.name,
            details={
                "epoch_reward": round(epoch_reward, 4),
                "epoch": epoch,
            },
        )

    def get_policy(self) -> AdvisorPolicy:
        """Return the underlying advisor policy for inspection."""
        return self._policy

    def get_report(self) -> Dict:
        """Return advisor state report."""
        return self._policy.get_report()
