"""Attestation heartbeat governance lever.

Requires agents to periodically re-attest their acceptance of governance
constraints. An agent that goes silent (misses its heartbeat interval)
is treated as non-compliant — its operations are blocked until it
re-attests.

This closes the "silence as evasion" attack surface: an agent cannot
avoid governance scrutiny by simply ceasing to emit logs during
sensitive operations.  Missing an attestation is an auditable event.

Design:
    - Each agent has a heartbeat deadline (step-based).
    - ``record_attestation(agent_id, step)`` resets the deadline.
    - ``on_step`` checks all tracked agents; any that have exceeded
      the interval without attesting are frozen and penalised.
    - ``can_agent_act`` blocks agents whose attestation has lapsed.
    - The oracle baseline runs with ``attestation_heartbeat_enabled=False``,
      so the lever's safety effect is cleanly isolable in benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Set

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect


@dataclass
class HeartbeatTracker:
    """Per-agent heartbeat state."""

    last_attestation_step: int = -1
    missed_count: int = 0
    is_lapsed: bool = False


class AttestationHeartbeatLever(GovernanceLever):
    """Governance lever that enforces periodic active re-attestation.

    Agents must positively confirm they accept the current constraint
    set at regular intervals.  Silence triggers suspension — making
    silence *audible* in the governance record.

    Hooks used:
        ``on_step``       — check for lapsed heartbeats each step.
        ``can_agent_act`` — block agents whose attestation has lapsed.
        ``on_epoch_start``— optionally reset lapsed status at epoch boundary.
    """

    def __init__(self, config: GovernanceConfig) -> None:
        super().__init__(config)
        self._trackers: Dict[str, HeartbeatTracker] = {}

    @property
    def name(self) -> str:
        return "attestation_heartbeat"

    # ------------------------------------------------------------------
    # Public API — called by the orchestrator / agent harness
    # ------------------------------------------------------------------

    def record_attestation(self, agent_id: str, step: int) -> None:
        """Record that *agent_id* has attested at *step*.

        Resets the heartbeat deadline and clears any lapsed status.
        """
        tracker = self._get_tracker(agent_id)
        tracker.last_attestation_step = step
        tracker.is_lapsed = False

    def get_status(self, agent_id: str) -> Dict[str, object]:
        """Return heartbeat status for an agent."""
        tracker = self._get_tracker(agent_id)
        return {
            "last_attestation_step": tracker.last_attestation_step,
            "missed_count": tracker.missed_count,
            "is_lapsed": tracker.is_lapsed,
        }

    def get_lapsed_agents(self) -> FrozenSet[str]:
        """Return the set of agents whose attestation has lapsed."""
        return frozenset(
            aid for aid, t in self._trackers.items() if t.is_lapsed
        )

    # ------------------------------------------------------------------
    # GovernanceLever hooks
    # ------------------------------------------------------------------

    def on_step(
        self,
        state: EnvState,
        step: int,
    ) -> LeverEffect:
        """Check all tracked agents for missed heartbeats.

        Auto-registers any agents present in ``state.agents`` that are
        not yet tracked, so the lever enforces heartbeats even without
        explicit ``register_agent`` calls from the orchestrator.
        """
        if not self.config.attestation_heartbeat_enabled:
            return LeverEffect(lever_name=self.name)

        # Auto-register untracked agents so the lever is never a no-op.
        for agent_id in state.agents:
            if agent_id not in self._trackers:
                self.register_agent(agent_id, initial_step=step)

        interval = self.config.attestation_heartbeat_interval
        agents_to_freeze: Set[str] = set()
        newly_lapsed: list[dict[str, object]] = []

        for agent_id, tracker in self._trackers.items():
            if tracker.is_lapsed:
                # Already lapsed — keep frozen until re-attestation
                agents_to_freeze.add(agent_id)
                continue

            # Normalize sentinel -1 to 0 so agents that have never
            # attested (or were reset at epoch boundary) get one full
            # interval from step 0, matching the documented behaviour.
            base_step = 0 if tracker.last_attestation_step < 0 else tracker.last_attestation_step
            deadline = base_step + interval
            if step > deadline:
                tracker.is_lapsed = True
                tracker.missed_count += 1
                agents_to_freeze.add(agent_id)
                newly_lapsed.append({
                    "agent_id": agent_id,
                    "last_attestation_step": tracker.last_attestation_step,
                    "deadline": deadline,
                    "missed_count": tracker.missed_count,
                })

        # Penalise only agents that *just* lapsed this step, not those
        # already lapsed from prior steps (avoids recurring per-step
        # penalty drain).
        reputation_deltas: Dict[str, float] = {}
        penalty = self.config.attestation_heartbeat_reputation_penalty
        if penalty != 0.0:
            for detail in newly_lapsed:
                reputation_deltas[str(detail["agent_id"])] = penalty

        return LeverEffect(
            agents_to_freeze=agents_to_freeze,
            reputation_deltas=reputation_deltas,
            lever_name=self.name,
            details={
                "newly_lapsed_count": len(newly_lapsed),
                "newly_lapsed_agents": newly_lapsed,
                "total_lapsed_count": len(agents_to_freeze),
            },
        )

    def on_epoch_start(
        self,
        state: EnvState,
        epoch: int,
    ) -> LeverEffect:
        """Optionally clear lapsed status at epoch boundary.

        When ``attestation_heartbeat_reset_on_epoch`` is True, all
        agents get a fresh start each epoch (they still need to
        attest within the first interval of the new epoch).
        """
        if not self.config.attestation_heartbeat_enabled:
            return LeverEffect(lever_name=self.name)

        agents_to_unfreeze: Set[str] = set()

        if self.config.attestation_heartbeat_reset_on_epoch:
            for agent_id, tracker in self._trackers.items():
                if tracker.is_lapsed:
                    agents_to_unfreeze.add(agent_id)
                # Reset ALL agents' baselines so non-lapsed agents
                # don't carry stale deadlines across epoch boundaries
                # (current_step resets to 0 each epoch).
                tracker.is_lapsed = False
                tracker.last_attestation_step = -1

        return LeverEffect(
            agents_to_unfreeze=agents_to_unfreeze,
            lever_name=self.name,
            details={
                "unfrozen_count": len(agents_to_unfreeze),
                "unfrozen_agents": list(agents_to_unfreeze),
            },
        )

    def can_agent_act(
        self,
        agent_id: str,
        state: EnvState,
    ) -> bool:
        """Block agents whose attestation has lapsed.

        Untracked agents are auto-registered with a deadline starting
        from the current step, so the lever cannot be bypassed by
        skipping explicit registration.
        """
        if not self.config.attestation_heartbeat_enabled:
            return True

        if agent_id not in self._trackers:
            self.register_agent(agent_id, initial_step=state.current_step)
            return True  # just registered — first interval starts now

        return not self._trackers[agent_id].is_lapsed

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, initial_step: int = -1) -> None:
        """Register an agent for heartbeat tracking.

        Should be called when the agent joins the simulation so that
        its first heartbeat deadline is set.
        """
        if agent_id not in self._trackers:
            self._trackers[agent_id] = HeartbeatTracker(
                last_attestation_step=initial_step,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_tracker(self, agent_id: str) -> HeartbeatTracker:
        if agent_id not in self._trackers:
            self._trackers[agent_id] = HeartbeatTracker()
        return self._trackers[agent_id]
