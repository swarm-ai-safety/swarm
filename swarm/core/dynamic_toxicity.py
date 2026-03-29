"""Dynamic toxicity middleware: feedback loops from toxicity to future state.

Three mechanisms for toxicity to degrade future interaction quality:

1. **Proxy Calibration Drift** — cumulative toxicity degrades the proxy
   computer's sigmoid sharpness, making screening less discriminating.
2. **Trust Erosion (Honest Agent Exit)** — sustained high toxicity causes
   honest agents to exit, endogenously concentrating adversarial fraction.
3. **Interaction Quality Contagion** — agents who interact with low-p
   counterparts have their own future quality pulled down.

Each is implemented as a Middleware subclass that runs at on_epoch_end,
reads the epoch's toxicity, and mutates state for the next epoch.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from swarm.core.middleware import Middleware, MiddlewareContext
from swarm.logging.event_bus import EventBus
from swarm.models.events import Event, EventType

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# 1. Proxy Calibration Drift
# -----------------------------------------------------------------------


class ProxyCalibrationDriftMiddleware(Middleware):
    """Cumulative toxicity degrades the proxy computer's discrimination.

    Mechanism: each epoch, sigmoid_k is reduced proportionally to
    cumulative toxicity.  As k shrinks, the sigmoid flattens, p values
    cluster toward 0.5, and the screening threshold becomes less
    discriminating — admitting more toxic interactions, creating a
    positive feedback loop.

    Parameters:
        proxy_computer: The ProxyComputer instance to mutate.
        alpha: Drift rate — how much each unit of cumulative toxicity
               reduces sigmoid_k. Default 0.5.
        k_floor: Minimum sigmoid_k to prevent total collapse of
                 discrimination. Default 0.5.
    """

    def __init__(
        self,
        proxy_computer: Any,
        event_bus: EventBus,
        *,
        alpha: float = 0.5,
        k_floor: float = 0.5,
    ) -> None:
        self._proxy = proxy_computer
        self._emit = event_bus.emit
        self._alpha = alpha
        self._k_floor = k_floor
        self._k_initial = proxy_computer.sigmoid_k
        self._cumulative_toxicity = 0.0

    def on_epoch_start(self, ctx: MiddlewareContext) -> None:
        pass

    def on_epoch_end(self, ctx: MiddlewareContext) -> None:
        interactions = ctx.state.completed_interactions
        if not interactions or ctx.metrics_calculator is None:
            return

        toxicity = ctx.metrics_calculator.toxicity_rate(interactions)
        self._cumulative_toxicity += toxicity

        # k decays: k = k_initial * max(floor_ratio, 1 - alpha * cumulative)
        decay_factor = max(
            self._k_floor / self._k_initial,
            1.0 - self._alpha * self._cumulative_toxicity,
        )
        new_k = self._k_initial * decay_factor

        old_k = self._proxy.sigmoid_k
        self._proxy.sigmoid_k = new_k

        logger.info(
            "ProxyCalibrationDrift: epoch %d, toxicity=%.4f, "
            "cumulative=%.4f, sigmoid_k %.3f -> %.3f",
            ctx.state.current_epoch,
            toxicity,
            self._cumulative_toxicity,
            old_k,
            new_k,
        )

        self._emit(
            Event(
                event_type=EventType.EPOCH_COMPLETED,
                payload={
                    "dynamic_toxicity_type": "proxy_calibration_drift",
                    "epoch_toxicity": toxicity,
                    "cumulative_toxicity": self._cumulative_toxicity,
                    "sigmoid_k_old": old_k,
                    "sigmoid_k_new": new_k,
                },
                epoch=ctx.state.current_epoch,
            )
        )

    def on_step_start(self, ctx: MiddlewareContext) -> None:
        pass


# -----------------------------------------------------------------------
# 2. Trust Erosion (Honest Agent Exit)
# -----------------------------------------------------------------------


class TrustErosionMiddleware(Middleware):
    """Sustained high toxicity causes honest agents to exit.

    Mechanism: at each epoch end, compute a rolling average of recent
    toxicity.  If it exceeds the exit threshold, each honest agent has
    a probability proportional to the excess of leaving the ecosystem
    (being frozen).  This endogenously concentrates adversarial fraction,
    potentially triggering the phase transition the paper identifies.

    Parameters:
        event_bus: For emitting exit events.
        beta: Sensitivity — exit probability per unit of excess toxicity.
              Default 0.3.
        exit_threshold: Rolling toxicity above which exits begin.
                        Default 0.35.
        window: Number of epochs for rolling average. Default 3.
        min_honest: Minimum honest agents to keep (prevents total
                    ecosystem death). Default 1.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        event_bus: EventBus,
        *,
        beta: float = 0.3,
        exit_threshold: float = 0.35,
        window: int = 3,
        min_honest: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self._emit = event_bus.emit
        self._beta = beta
        self._exit_threshold = exit_threshold
        self._window = window
        self._min_honest = min_honest
        self._rng = random.Random(seed)
        self._toxicity_history: List[float] = []

    def on_epoch_start(self, ctx: MiddlewareContext) -> None:
        pass

    def on_epoch_end(self, ctx: MiddlewareContext) -> None:
        from swarm.models.agent import AgentType

        interactions = ctx.state.completed_interactions
        if not interactions or ctx.metrics_calculator is None:
            return

        toxicity = ctx.metrics_calculator.toxicity_rate(interactions)
        self._toxicity_history.append(toxicity)

        # Rolling average over recent epochs
        recent = self._toxicity_history[-self._window :]
        rolling_tox = sum(recent) / len(recent)

        if rolling_tox <= self._exit_threshold:
            return

        excess = rolling_tox - self._exit_threshold
        exit_prob = min(1.0, self._beta * excess)

        # Find honest agents that could exit
        honest_ids = [
            aid
            for aid, agent in ctx.agents.items()
            if agent.agent_type == AgentType.HONEST
            and aid not in ctx.state.frozen_agents
        ]

        # Ensure we keep minimum honest agents
        max_exits = max(0, len(honest_ids) - self._min_honest)
        if max_exits == 0:
            return

        exited: List[str] = []
        for aid in honest_ids:
            if len(exited) >= max_exits:
                break
            if self._rng.random() < exit_prob:
                ctx.state.frozen_agents.add(aid)
                exited.append(aid)

        if exited:
            logger.info(
                "TrustErosion: epoch %d, rolling_toxicity=%.4f (threshold=%.4f), "
                "%d honest agent(s) exited: %s",
                ctx.state.current_epoch,
                rolling_tox,
                self._exit_threshold,
                len(exited),
                exited,
            )

            self._emit(
                Event(
                    event_type=EventType.EPOCH_COMPLETED,
                    payload={
                        "dynamic_toxicity_type": "trust_erosion",
                        "rolling_toxicity": rolling_tox,
                        "exit_threshold": self._exit_threshold,
                        "exit_probability": exit_prob,
                        "agents_exited": exited,
                        "remaining_honest": len(honest_ids) - len(exited),
                    },
                    epoch=ctx.state.current_epoch,
                )
            )

    def on_step_start(self, ctx: MiddlewareContext) -> None:
        pass


# -----------------------------------------------------------------------
# 3. Interaction Quality Contagion
# -----------------------------------------------------------------------


class QualityContagionMiddleware(Middleware):
    """Agents who interact with low-p counterparts get quality pulled down.

    Mechanism: at epoch end, for each agent, compute the average p of
    their accepted interactions.  Apply a bias to the agent's future
    interactions by shifting the engagement signal in ProxyObservables.
    This is implemented by adjusting each agent's internal trust memory
    toward the quality of counterparts they actually interacted with.

    Unlike the existing pairwise trust EMA (which only affects acceptance
    decisions), this creates an ecosystem-wide contagion effect: if agent
    A interacts with toxic agent B, A's future interactions with *any*
    agent C will be scored slightly lower, because A's engagement signal
    has been degraded.

    Parameters:
        gamma: Contagion strength — how much counterpart quality shifts
               the agent's own quality. Default 0.1.
        neutral: The neutral point toward which contagion pulls.
                 Default 0.5.
    """

    def __init__(
        self,
        event_bus: EventBus,
        *,
        gamma: float = 0.1,
        neutral: float = 0.5,
    ) -> None:
        self._emit = event_bus.emit
        self._gamma = gamma
        self._neutral = neutral
        # Track per-agent quality bias accumulated over time
        self._agent_bias: Dict[str, float] = {}

    def on_epoch_start(self, ctx: MiddlewareContext) -> None:
        pass

    def on_epoch_end(self, ctx: MiddlewareContext) -> None:
        interactions = ctx.state.completed_interactions
        if not interactions:
            return

        accepted = [i for i in interactions if i.accepted]
        if not accepted:
            return

        # Compute average counterpart quality per agent
        agent_counterpart_quality: Dict[str, List[float]] = {}
        for interaction in accepted:
            # Initiator experienced counterparty's quality
            agent_counterpart_quality.setdefault(
                interaction.initiator, []
            ).append(interaction.p)
            # Counterparty experienced initiator's quality
            if interaction.counterparty:
                agent_counterpart_quality.setdefault(
                    interaction.counterparty, []
                ).append(interaction.p)

        contagion_updates: Dict[str, float] = {}

        for agent_id, qualities in agent_counterpart_quality.items():
            avg_quality = sum(qualities) / len(qualities)
            # Contagion pull: shift bias toward counterpart quality
            delta = self._gamma * (avg_quality - self._neutral)
            old_bias = self._agent_bias.get(agent_id, 0.0)
            new_bias = old_bias + delta
            # Clamp bias to prevent runaway
            new_bias = max(-0.5, min(0.5, new_bias))
            self._agent_bias[agent_id] = new_bias
            contagion_updates[agent_id] = new_bias

            # Apply bias to agent's trust memory — shift all counterparty
            # trust values by the contagion delta
            agent = ctx.agents.get(agent_id)
            if agent is not None and hasattr(agent, "_counterparty_memory"):
                for cp_id in list(agent._counterparty_memory.keys()):
                    old_trust = agent._counterparty_memory[cp_id]
                    # Shift trust, clamped to [0, 1]
                    agent._counterparty_memory[cp_id] = max(
                        0.0, min(1.0, old_trust + delta)
                    )

        if contagion_updates:
            logger.info(
                "QualityContagion: epoch %d, updated %d agents, "
                "bias range [%.4f, %.4f]",
                ctx.state.current_epoch,
                len(contagion_updates),
                min(contagion_updates.values()),
                max(contagion_updates.values()),
            )

            self._emit(
                Event(
                    event_type=EventType.EPOCH_COMPLETED,
                    payload={
                        "dynamic_toxicity_type": "quality_contagion",
                        "agents_updated": len(contagion_updates),
                        "bias_min": min(contagion_updates.values()),
                        "bias_max": max(contagion_updates.values()),
                        "bias_mean": sum(contagion_updates.values())
                        / len(contagion_updates),
                    },
                    epoch=ctx.state.current_epoch,
                )
            )

    def on_step_start(self, ctx: MiddlewareContext) -> None:
        pass
