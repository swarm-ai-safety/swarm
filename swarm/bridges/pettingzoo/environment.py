"""PettingZoo ParallelEnv wrapping a SWARM multi-agent simulation.

Each agent in the SWARM population becomes an independent RL actor.
All agents act simultaneously (Parallel API).

Architecture::

    PettingZoo RL framework
        └── SwarmParallelEnv  (this module)
                ├── ProxyComputer    (observable → v_hat → p)
                ├── SoftPayoffEngine (p → payoffs)
                ├── SoftMetrics      (population-level safety)
                └── EventLog         (append-only audit trail)

Observation space (per agent)::

    Box([-1, -1, ..., 0, 0, ...], [1, 1, ..., 1, 1, ...])
    ┌──────────────────────────────────────────────────┐
    │ own_reputation          float  [-1, 1]           │
    │ own_resources           float  [0, 100]          │
    │ mean_ecosystem_p        float  [0, 1]            │
    │ toxicity_rate           float  [0, 1]            │
    │ n_agents_alive          float  [0, 1] normalised │
    │ step_fraction           float  [0, 1]            │
    │ history[0..H-1].p       float  [0, 1] × H        │
    │ history[0..H-1].payoff  float  [-10, 10] × H     │
    └──────────────────────────────────────────────────┘

Action space (per agent)::

    Discrete(4)
    0 = reject   (refuse to interact this step)
    1 = accept   (interact cooperatively)
    2 = exploit  (attempt to extract value)
    3 = signal   (costly pro-social signal / governance vote)
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from gymnasium import spaces
    from pettingzoo import ParallelEnv
except ImportError as exc:
    raise ImportError(
        "PettingZoo bridge requires pettingzoo and gymnasium. "
        "Install with: pip install 'pettingzoo>=1.24.0' gymnasium"
    ) from exc

from swarm.bridges.pettingzoo.config import PettingZooConfig
from swarm.bridges.pettingzoo.events import PZEvent, PZEventType
from swarm.core.payoff import SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.logging.event_log import EventLog
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)

# Action constants
ACTION_REJECT = 0
ACTION_ACCEPT = 1
ACTION_EXPLOIT = 2
ACTION_SIGNAL = 3
ACTION_NAMES = {0: "reject", 1: "accept", 2: "exploit", 3: "signal"}

# Observation vector layout
_OBS_STATIC = 6  # reputation, resources, mean_p, toxicity, n_alive, step_frac
_OBS_PER_HIST = 2  # p, payoff per history entry


class SwarmParallelEnv(ParallelEnv):
    """SWARM multi-agent simulation as a PettingZoo ParallelEnv.

    Every agent in the population acts simultaneously each step.
    Observations encode the agent's local view of the ecosystem.
    Rewards are SWARM soft payoffs.  Episodes end after ``max_steps``
    or when catastrophic toxicity (>0.9) is detected.

    Example::

        from swarm.bridges.pettingzoo.environment import SwarmParallelEnv

        env = SwarmParallelEnv(render_mode="ansi")
        observations, infos = env.reset(seed=42)

        while env.agents:
            actions = {a: env.action_space(a).sample() for a in env.agents}
            observations, rewards, terms, truncs, infos = env.step(actions)

        env.close()
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "swarm_safety_v0"}

    def __init__(
        self,
        config: Optional[PettingZooConfig] = None,
        event_log: Optional[EventLog] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._config = config or PettingZooConfig()
        self._proxy = ProxyComputer(sigmoid_k=self._config.proxy_sigmoid_k)
        self._payoff = SoftPayoffEngine(self._config.get_payoff_config())
        self._metrics = SoftMetrics(payoff_engine=self._payoff)
        self._event_log = event_log

        self.render_mode = render_mode or self._config.render_mode or None

        # Build agent ID list
        self.possible_agents: list[str] = self._build_agent_ids()
        self.agents: list[str] = []

        # Per-agent state (populated on reset)
        self._agent_type: Dict[str, str] = {}
        self._reputation: Dict[str, float] = {}
        self._resources: Dict[str, float] = {}
        self._history: Dict[str, List[Dict[str, float]]] = {}
        self._interactions: List[SoftInteraction] = []
        self._events: List[PZEvent] = []

        self._step_count: int = 0
        self._rng = random.Random(42)

        # Obs dimension
        self._obs_dim = _OBS_STATIC + self._config.obs_history_len * _OBS_PER_HIST

    # ------------------------------------------------------------------
    # PettingZoo required interface
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Box:
        low = np.full(self._obs_dim, -10.0, dtype=np.float32)
        high = np.full(self._obs_dim, 10.0, dtype=np.float32)
        # Tighter bounds for known fields
        low[0] = -1.0   # reputation
        high[0] = 1.0
        low[1] = 0.0    # resources
        high[1] = 100.0
        low[2] = 0.0    # mean_p
        high[2] = 1.0
        low[3] = 0.0    # toxicity
        high[3] = 1.0
        low[4] = 0.0    # n_alive (normalised)
        high[4] = 1.0
        low[5] = 0.0    # step_fraction
        high[5] = 1.0
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(4)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Reset the environment for a new episode."""
        if seed is not None:
            self._rng = random.Random(seed)

        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._interactions = []

        # Initialize per-agent state
        self._reputation = {}
        self._resources = {}
        self._history = {}
        self._agent_type = {}

        for agent_id in self.agents:
            atype = self._id_to_type(agent_id)
            self._agent_type[agent_id] = atype
            self._reputation[agent_id] = self._initial_reputation(atype)
            self._resources[agent_id] = 10.0
            self._history[agent_id] = []

        self._record_event(PZEventType.EPISODE_RESET, "", {
            "n_agents": len(self.agents),
            "seed": seed,
        })

        observations = {a: self._observe(a) for a in self.agents}
        infos: Dict[str, Dict[str, Any]] = {a: {} for a in self.agents}
        return observations, infos

    def step(
        self,
        actions: Dict[str, int],
    ) -> tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Execute one simultaneous step for all agents."""
        self._step_count += 1

        rewards: Dict[str, float] = dict.fromkeys(self.agents, 0.0)
        terminations: Dict[str, bool] = dict.fromkeys(self.agents, False)
        truncations: Dict[str, bool] = dict.fromkeys(self.agents, False)
        infos: Dict[str, Dict[str, Any]] = {a: {} for a in self.agents}  # noqa: C420

        # --- Pair agents and resolve interactions ---
        active = [a for a in self.agents if a in actions]
        pairs = self._make_pairs(active)

        step_interactions: List[SoftInteraction] = []

        for a, b in pairs:
            act_a = actions.get(a, ACTION_REJECT)
            act_b = actions.get(b, ACTION_REJECT)

            interaction = self._resolve_interaction(a, b, act_a, act_b)
            step_interactions.append(interaction)
            self._interactions.append(interaction)

            # Compute payoffs
            pay_a = self._payoff.payoff_initiator(interaction)
            pay_b = self._payoff.payoff_counterparty(interaction)

            rewards[a] = rewards.get(a, 0.0) + pay_a
            rewards[b] = rewards.get(b, 0.0) + pay_b

            # Update reputation with EMA
            self._reputation[a] = 0.9 * self._reputation[a] + 0.1 * interaction.p
            self._reputation[b] = 0.9 * self._reputation[b] + 0.1 * interaction.p

            # Update resources
            self._resources[a] = max(0.0, self._resources[a] + pay_a)
            self._resources[b] = max(0.0, self._resources[b] + pay_b)

            # Record in agent histories
            self._history[a].append({"p": interaction.p, "payoff": pay_a})
            self._history[b].append({"p": interaction.p, "payoff": pay_b})

            self._record_event(PZEventType.INTERACTION_RECORDED, a, {
                "counterparty": b,
                "action_a": ACTION_NAMES.get(act_a, "?"),
                "action_b": ACTION_NAMES.get(act_b, "?"),
                "p": interaction.p,
                "accepted": interaction.accepted,
            })

        # --- Handle agents with no pair (odd count) ---
        paired_agents = {a for pair in pairs for a in pair}
        for a in active:
            if a not in paired_agents:
                # No interaction this step
                self._history[a].append({"p": 0.5, "payoff": 0.0})

        # --- Apply signalling cost ---
        for a in active:
            if actions.get(a) == ACTION_SIGNAL:
                cost = 0.5
                self._resources[a] = max(0.0, self._resources[a] - cost)
                rewards[a] -= cost
                # Signalling boosts reputation
                self._reputation[a] = min(
                    1.0, self._reputation[a] + 0.05
                )

        # --- Check termination conditions ---
        # Agent terminates if resources hit zero
        for a in list(self.agents):
            if self._resources.get(a, 0.0) <= 0.0:
                terminations[a] = True
                self._record_event(PZEventType.AGENT_TERMINATED, a, {
                    "reason": "bankrupt",
                    "step": self._step_count,
                })

        # Truncation: max steps reached
        if self._step_count >= self._config.max_steps:
            for a in self.agents:
                truncations[a] = True

        # Catastrophic toxicity → terminate all
        if len(self._interactions) >= 3:
            recent = self._interactions[-10:]
            tox = self._metrics.toxicity_rate(recent)
            if tox > 0.9:
                for a in self.agents:
                    terminations[a] = True
                self._record_event(PZEventType.EPISODE_DONE, "", {
                    "reason": "catastrophic_toxicity",
                    "toxicity": tox,
                })

        # --- Remove terminated/truncated agents ---
        self.agents = [
            a for a in self.agents
            if not terminations.get(a, False) and not truncations.get(a, False)
        ]

        # --- Build observations for surviving agents ---
        observations = {a: self._observe(a) for a in self.agents}

        # --- Per-step info ---
        step_metrics = {}
        if step_interactions:
            step_metrics = {
                "toxicity_rate": self._metrics.toxicity_rate(self._interactions),
                "quality_gap": self._metrics.quality_gap(self._interactions),
                "mean_p": sum(ix.p for ix in step_interactions) / len(step_interactions),
                "n_interactions": len(self._interactions),
            }
        for a in list(terminations.keys()):
            infos[a] = {"step_metrics": step_metrics}

        self._record_event(PZEventType.STEP_COMPLETED, "", {
            "step": self._step_count,
            "n_alive": len(self.agents),
            "n_interactions": len(step_interactions),
        })

        if self._event_log and step_interactions:
            self._log_interactions(step_interactions)

        return observations, rewards, terminations, truncations, infos

    def render(self) -> Optional[str]:
        """Render the environment state."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_ansi()
        return None

    def close(self) -> None:
        """Release resources."""
        pass

    def state(self) -> np.ndarray:
        """Global state vector for centralised-training / decentralised-execution."""
        parts = []
        for a in self.possible_agents:
            parts.append(self._reputation.get(a, 0.0))
            parts.append(self._resources.get(a, 0.0) / 100.0)
        # Ecosystem metrics
        if self._interactions:
            tox = self._metrics.toxicity_rate(self._interactions)
            mean_p = sum(ix.p for ix in self._interactions) / len(self._interactions)
        else:
            tox, mean_p = 0.0, 0.5
        parts.extend([mean_p, tox, self._step_count / self._config.max_steps])
        return np.array(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_interactions(self) -> List[SoftInteraction]:
        """Return all recorded interactions from the current episode."""
        return list(self._interactions)

    def get_events(self) -> List[PZEvent]:
        """Return all bridge events."""
        return list(self._events)

    def get_metrics(self) -> Dict[str, float]:
        """Compute SWARM safety metrics for the episode so far."""
        if not self._interactions:
            return {
                "toxicity_rate": 0.0,
                "quality_gap": 0.0,
                "mean_p": 0.5,
                "total_welfare": 0.0,
                "n_interactions": 0,
            }
        tox = self._metrics.toxicity_rate(self._interactions)
        qg = self._metrics.quality_gap(self._interactions)
        mean_p = sum(ix.p for ix in self._interactions) / len(self._interactions)
        welfare = sum(
            self._payoff.payoff_initiator(ix) + self._payoff.payoff_counterparty(ix)
            for ix in self._interactions
        )
        return {
            "toxicity_rate": tox,
            "quality_gap": qg,
            "mean_p": mean_p,
            "total_welfare": welfare,
            "n_interactions": len(self._interactions),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_agent_ids(self) -> list[str]:
        """Build the full list of possible agent IDs from config."""
        ids: list[str] = []
        for atype, frac in self._config.agent_types.items():
            count = max(1, round(self._config.n_agents * frac))
            for i in range(count):
                ids.append(f"{atype}_{i}")
        # Trim or pad to exactly n_agents
        if len(ids) > self._config.n_agents:
            ids = ids[: self._config.n_agents]
        while len(ids) < self._config.n_agents:
            ids.append(f"honest_{len(ids)}")
        return ids

    def _id_to_type(self, agent_id: str) -> str:
        """Extract agent type from ID (e.g. 'honest_0' → 'honest')."""
        parts = agent_id.rsplit("_", 1)
        return parts[0] if len(parts) == 2 else "honest"

    def _initial_reputation(self, agent_type: str) -> float:
        """Starting reputation by type."""
        return {
            "honest": 0.8,
            "opportunistic": 0.5,
            "deceptive": 0.3,
        }.get(agent_type, 0.5)

    def _observe(self, agent: str) -> np.ndarray:
        """Build observation vector for a single agent."""
        obs = np.zeros(self._obs_dim, dtype=np.float32)

        # Static features
        obs[0] = self._reputation.get(agent, 0.0)
        obs[1] = min(self._resources.get(agent, 0.0), 100.0)
        if self._interactions:
            recent = self._interactions[-10:]
            obs[2] = sum(ix.p for ix in recent) / len(recent)
            obs[3] = self._metrics.toxicity_rate(recent)
        else:
            obs[2] = 0.5
            obs[3] = 0.0
        obs[4] = len(self.agents) / max(len(self.possible_agents), 1)
        obs[5] = self._step_count / max(self._config.max_steps, 1)

        # History features
        hist = self._history.get(agent, [])
        H = self._config.obs_history_len
        for i in range(H):
            idx = len(hist) - H + i
            base = _OBS_STATIC + i * _OBS_PER_HIST
            if idx >= 0 and idx < len(hist):
                obs[base] = hist[idx]["p"]
                obs[base + 1] = np.clip(hist[idx]["payoff"], -10.0, 10.0)
            else:
                obs[base] = 0.5
                obs[base + 1] = 0.0

        return obs

    def _make_pairs(self, agents: List[str]) -> List[tuple[str, str]]:
        """Randomly pair agents for interactions this step."""
        shuffled = list(agents)
        self._rng.shuffle(shuffled)
        pairs = []
        for i in range(0, len(shuffled) - 1, 2):
            pairs.append((shuffled[i], shuffled[i + 1]))
        return pairs

    def _resolve_interaction(
        self,
        agent_a: str,
        agent_b: str,
        action_a: int,
        action_b: int,
    ) -> SoftInteraction:
        """Resolve a pairwise interaction into a SoftInteraction.

        Action semantics:
            reject (0)  → no interaction, no observables
            accept (1)  → cooperative, positive observables
            exploit (2) → attempt extraction, mixed observables
            signal (3)  → treated as accept + pro-social boost
        """
        # Both must not reject for an interaction to occur
        accepted = action_a != ACTION_REJECT and action_b != ACTION_REJECT

        # Build observables from action pair
        if not accepted:
            observables = ProxyObservables(
                task_progress_delta=0.0,
                counterparty_engagement_delta=-0.2,
            )
        elif action_a == ACTION_EXPLOIT or action_b == ACTION_EXPLOIT:
            # Exploitation attempt
            exploiter = agent_a if action_a == ACTION_EXPLOIT else agent_b
            victim = agent_b if exploiter == agent_a else agent_a
            # Exploitation succeeds more against low-reputation agents
            exploit_success = self._rng.random() > self._reputation.get(victim, 0.5)
            observables = ProxyObservables(
                task_progress_delta=0.3 if exploit_success else -0.2,
                rework_count=0 if exploit_success else 1,
                tool_misuse_flags=1,
                counterparty_engagement_delta=-0.4,
            )
        else:
            # Cooperative interaction (accept or signal)
            # Quality depends on both agents' reputations
            rep_a = self._reputation.get(agent_a, 0.5)
            rep_b = self._reputation.get(agent_b, 0.5)
            quality = (rep_a + rep_b) / 2.0 + self._rng.gauss(0, 0.1)

            # Signal action boosts quality
            if action_a == ACTION_SIGNAL:
                quality += 0.1
            if action_b == ACTION_SIGNAL:
                quality += 0.1

            quality = max(-1.0, min(1.0, quality))
            observables = ProxyObservables(
                task_progress_delta=quality,
                counterparty_engagement_delta=quality * 0.8
                + self._rng.gauss(0, 0.05),
            )

        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            initiator=agent_a,
            counterparty=agent_b,
            interaction_type=InteractionType.COLLABORATION,
            accepted=accepted,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "pettingzoo",
                "step": self._step_count,
                "action_a": ACTION_NAMES.get(action_a, "?"),
                "action_b": ACTION_NAMES.get(action_b, "?"),
                "agent_type_a": self._agent_type.get(agent_a, "?"),
                "agent_type_b": self._agent_type.get(agent_b, "?"),
            },
        )

    def _render_ansi(self) -> str:
        """Render a text summary of the current state."""
        lines = [
            f"=== SWARM PettingZoo  step={self._step_count}/{self._config.max_steps} ===",
            f"Alive: {len(self.agents)}/{len(self.possible_agents)}",
        ]
        metrics = self.get_metrics()
        lines.append(
            f"Toxicity: {metrics['toxicity_rate']:.3f}  "
            f"Quality gap: {metrics['quality_gap']:.3f}  "
            f"Mean p: {metrics['mean_p']:.3f}"
        )
        lines.append("Agents:")
        for a in self.possible_agents:
            alive = "+" if a in self.agents else "-"
            rep = self._reputation.get(a, 0.0)
            res = self._resources.get(a, 0.0)
            lines.append(f"  [{alive}] {a:20s}  rep={rep:.2f}  res={res:.1f}")
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
        return text

    def _record_event(
        self, event_type: PZEventType, agent_id: str, payload: Dict[str, Any]
    ) -> None:
        if len(self._events) >= 50_000:
            self._events = self._events[-25_000:]
        self._events.append(PZEvent(
            event_type=event_type,
            agent_id=agent_id,
            payload=payload,
        ))

    def _log_interactions(self, interactions: List[SoftInteraction]) -> None:
        """Log interactions to the SWARM EventLog."""
        if self._event_log is None:
            return
        from swarm.models.events import Event, EventType

        for ix in interactions:
            event = Event(
                event_type=EventType.INTERACTION_COMPLETED,
                interaction_id=ix.interaction_id,
                initiator_id=ix.initiator,
                counterparty_id=ix.counterparty,
                payload={
                    "accepted": ix.accepted,
                    "v_hat": ix.v_hat,
                    "p": ix.p,
                    "bridge": "pettingzoo",
                },
            )
            self._event_log.append(event)
