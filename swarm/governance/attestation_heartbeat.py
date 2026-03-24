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
        """Check all tracked agents for missed heartbeats."""
        if not self.config.attestation_heartbeat_enabled:
            return LeverEffect(lever_name=self.name)

        interval = self.config.attestation_heartbeat_interval
        agents_to_freeze: Set[str] = set()
        lapsed_details: list[dict[str, object]] = []

        for agent_id, tracker in self._trackers.items():
            if tracker.is_lapsed:
                # Already lapsed — keep frozen until re-attestation
                agents_to_freeze.add(agent_id)
                continue

            # An agent that has never attested gets a grace period of
            # one full interval from step 0.
            deadline = tracker.last_attestation_step + interval
            if step > deadline:
                tracker.is_lapsed = True
                tracker.missed_count += 1
                agents_to_freeze.add(agent_id)
                lapsed_details.append({
                    "agent_id": agent_id,
                    "last_attestation_step": tracker.last_attestation_step,
                    "deadline": deadline,
                    "missed_count": tracker.missed_count,
                })

        reputation_deltas: Dict[str, float] = {}
        penalty = self.config.attestation_heartbeat_reputation_penalty
        for agent_id in agents_to_freeze:
            if penalty != 0.0:
                reputation_deltas[agent_id] = penalty

        return LeverEffect(
            agents_to_freeze=agents_to_freeze,
            reputation_deltas=reputation_deltas,
            lever_name=self.name,
            details={
                "lapsed_count": len(lapsed_details),
                "lapsed_agents": lapsed_details,
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
                    tracker.is_lapsed = False
                    # Reset deadline relative to epoch start (step 0)
                    tracker.last_attestation_step = -1
                    agents_to_unfreeze.add(agent_id)

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
        """Block agents whose attestation has lapsed."""
        if not self.config.attestation_heartbeat_enabled:
            return True

        # Agents not yet tracked are allowed (they'll be added on
        # first interaction or explicit registration).
        if agent_id not in self._trackers:
            return True

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
