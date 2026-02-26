"""Escalation Sandbox environment: crisis state, turn resolution, fog-of-war."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swarm.domains.escalation_sandbox.config import EscalationConfig
from swarm.domains.escalation_sandbox.entities import (
    DE_ESCALATION_FRICTION,
    ESCALATION_CONSEQUENCES,
    NUCLEAR_THRESHOLD,
    CrisisOutcome,
    EscalationAction,
    EscalationEvent,
    EscalationLevel,
    NationState,
    TurnResult,
)

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Result from a completed episode."""

    outcome: CrisisOutcome = CrisisOutcome.TIMEOUT
    turns_played: int = 0
    turn_results: List[TurnResult] = field(default_factory=list)
    events: List[EscalationEvent] = field(default_factory=list)
    nations: Dict[str, NationState] = field(default_factory=dict)


class EscalationEnvironment:
    """Geopolitical crisis environment with escalation ladder.

    Provides:
      - Nation-state management with military/economic/welfare state
      - 10-level escalation ladder as primary action space
      - Signal-action divergence tracking (Phase 2)
      - Fog-of-war noise producing accidental escalation (Phase 3)
      - De-escalation friction and governance levers (Phase 4)
      - Consequence engine applying asymmetric damage
      - Termination condition evaluation
    """

    def __init__(self, config: EscalationConfig) -> None:
        self._config = config
        self._rng = random.Random(config.seed)

        # Nation states
        self._nations: Dict[str, NationState] = {}

        # Turn tracking
        self._current_turn = 0
        self._max_turns = config.max_turns

        # Episode state
        self._outcome = CrisisOutcome.ONGOING
        self._events: List[EscalationEvent] = []
        self._turn_results: List[TurnResult] = []

        # Nuclear threshold tracking
        self._nuclear_threshold_turn: Optional[int] = None
        self._nuclear_agents: set[str] = set()

        # Circuit breaker state
        self._circuit_breaker_active = False
        self._circuit_breaker_turns_remaining = 0
        self._circuit_breaker_fired = False  # once-per-episode flag

        # Governance tracking
        self._mediation_proposed_this_turn = False
        self._treaties_violated: Dict[str, int] = {}  # agent_id -> count

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def add_nation(
        self,
        agent_id: str,
        name: str = "",
        military_strength: float = 100.0,
        economic_strength: float = 100.0,
        has_nuclear: bool = True,
        has_second_strike: bool = True,
        intelligence_quality: float = 0.5,
    ) -> NationState:
        """Register a nation-state in the environment."""
        nation = NationState(
            agent_id=agent_id,
            name=name or agent_id,
            military_strength=military_strength,
            economic_strength=economic_strength,
            has_nuclear=has_nuclear,
            has_second_strike=has_second_strike,
            intelligence_quality=intelligence_quality,
        )
        self._nations[agent_id] = nation
        return nation

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def obs(self, agent_id: str) -> Dict[str, Any]:
        """Build observation for an agent."""
        nation = self._nations[agent_id]
        opponents = {
            aid: n for aid, n in self._nations.items() if aid != agent_id
        }

        # Build opponent observations (limited by intelligence quality)
        opponent_obs = {}
        for opp_id, opp in opponents.items():
            opponent_obs[opp_id] = {
                "name": opp.name,
                "current_level": opp.current_level,
                "trust_score": opp.trust_score,
                "diplomatic_standing": opp.diplomatic_standing,
                "has_nuclear": opp.has_nuclear,
                "has_second_strike": opp.has_second_strike,
                # Partially visible based on intelligence quality
                "estimated_military": opp.military_strength * (
                    0.5 + 0.5 * nation.intelligence_quality
                ) + self._rng.gauss(0, 10 * (1 - nation.intelligence_quality)),
                "estimated_economic": opp.economic_strength * (
                    0.5 + 0.5 * nation.intelligence_quality
                ),
            }
            # Include signal history if signals are broadcast
            if self._config.signals.broadcast_before_action:
                opponent_obs[opp_id]["recent_signals"] = opp.signal_history[-5:]

        return {
            "agent_id": agent_id,
            "name": nation.name,
            "turn": self._current_turn,
            "max_turns": self._max_turns,
            "current_level": nation.current_level,
            "military_strength": nation.military_strength,
            "economic_strength": nation.economic_strength,
            "population_welfare": nation.population_welfare,
            "diplomatic_standing": nation.diplomatic_standing,
            "trust_score": nation.trust_score,
            "has_nuclear": nation.has_nuclear,
            "has_second_strike": nation.has_second_strike,
            "intelligence_quality": nation.intelligence_quality,
            "cumulative_military_damage": nation.cumulative_military_damage,
            "cumulative_economic_damage": nation.cumulative_economic_damage,
            "cumulative_civilian_casualties": nation.cumulative_civilian_casualties,
            "opponents": opponent_obs,
            "active_treaties": nation.active_treaties,
            "circuit_breaker_active": self._circuit_breaker_active,
            "nuclear_threshold_crossed_by": list(self._nuclear_agents),
            "crisis_template": self._config.crisis.template,
            "governance": {
                "mad_enabled": self._config.governance.mad_enabled,
                "mediation_available": self._config.governance.mediation_enabled,
                "back_channel_available": self._config.governance.back_channel_enabled,
                "circuit_breaker_enabled": self._config.governance.circuit_breaker_enabled,
            },
        }

    # ------------------------------------------------------------------
    # Turn Resolution
    # ------------------------------------------------------------------

    def apply_actions(
        self, actions: Dict[str, EscalationAction],
    ) -> TurnResult:
        """Apply all agent actions for one turn.

        Turn resolution order:
        1. Broadcast signals (if enabled)
        2. Apply fog-of-war to intended actions
        3. Evaluate governance constraints (circuit breaker, treaties)
        4. Resolve consequences based on realised escalation levels
        5. Update trust scores
        6. Check termination conditions

        Args:
            actions: Mapping from agent_id to their chosen action.

        Returns:
            TurnResult with events and outcome.
        """
        turn_events: List[EscalationEvent] = []
        realised_levels: Dict[str, int] = {}
        fog_deltas: Dict[str, int] = {}

        # 1. Record signals and broadcast
        for agent_id, action in actions.items():
            signal = max(0, min(9, action.signal_level))
            turn_events.append(EscalationEvent(
                event_type="signal_sent",
                turn=self._current_turn,
                agent_id=agent_id,
                details={"signal_level": signal},
            ))

        # 2. Apply fog-of-war and compute realised levels
        for agent_id, action in actions.items():
            intended = max(0, min(9, action.action_level))
            nation = self._nations[agent_id]

            # Apply intelligence investment
            if action.invest_intelligence > 0:
                cost = min(action.invest_intelligence, nation.economic_strength * 0.1)
                nation.economic_strength -= cost
                # Temporarily boost intelligence for this turn
                intel_boost = cost * 0.01
                nation.intelligence_quality = min(
                    1.0, nation.intelligence_quality + intel_boost,
                )
                turn_events.append(EscalationEvent(
                    event_type="intelligence_investment",
                    turn=self._current_turn,
                    agent_id=agent_id,
                    details={"cost": cost, "new_quality": nation.intelligence_quality},
                ))

            # Compute fog delta
            fog_delta = self._compute_fog_delta(intended, nation)
            realised = max(0, min(9, intended + fog_delta))
            fog_deltas[agent_id] = fog_delta
            realised_levels[agent_id] = realised

            # Track intended vs realised
            nation.intended_level = EscalationLevel(intended)

            # Log fog deviation
            if fog_delta != 0:
                turn_events.append(EscalationEvent(
                    event_type="fog_deviation",
                    turn=self._current_turn,
                    agent_id=agent_id,
                    details={
                        "intended": intended,
                        "realised": realised,
                        "fog_delta": fog_delta,
                    },
                ))

            # Check for fog catastrophe (sub-nuclear -> nuclear)
            if intended < NUCLEAR_THRESHOLD <= realised:
                turn_events.append(EscalationEvent(
                    event_type="fog_catastrophe",
                    turn=self._current_turn,
                    agent_id=agent_id,
                    details={
                        "intended": intended,
                        "realised": realised,
                        "message": "Fog pushed sub-nuclear action across nuclear threshold",
                    },
                ))

        # 3a. Circuit breaker: evaluate trigger for ALL agents first
        cb_cfg = self._config.governance
        if (
            cb_cfg.circuit_breaker_enabled
            and not self._circuit_breaker_fired
            and not self._circuit_breaker_active
        ):
            for agent_id in list(realised_levels.keys()):
                realised = realised_levels[agent_id]
                if realised >= cb_cfg.circuit_breaker_threshold:
                    # Fire once per episode
                    self._circuit_breaker_fired = True
                    self._circuit_breaker_active = True
                    self._circuit_breaker_turns_remaining = max(
                        1, cb_cfg.circuit_breaker_negotiation_turns,
                    )
                    turn_events.append(EscalationEvent(
                        event_type="circuit_breaker_triggered",
                        turn=self._current_turn,
                        agent_id=agent_id,
                        details={
                            "original_level": realised,
                            "capped_level": cb_cfg.circuit_breaker_threshold - 1,
                        },
                    ))
                    break  # trigger once

        # 3b. Cap ALL agents at threshold-1 while circuit breaker is active
        if self._circuit_breaker_active:
            cap = cb_cfg.circuit_breaker_threshold - 1
            for agent_id in list(realised_levels.keys()):
                if realised_levels[agent_id] > cap:
                    realised_levels[agent_id] = cap

        # 3c. Expire circuit breaker (countdown per turn, not per agent)
        if self._circuit_breaker_active and self._circuit_breaker_turns_remaining > 0:
            self._circuit_breaker_turns_remaining -= 1
            if self._circuit_breaker_turns_remaining <= 0:
                self._circuit_breaker_active = False
                turn_events.append(EscalationEvent(
                    event_type="circuit_breaker_expired",
                    turn=self._current_turn,
                    agent_id="",
                    details={"message": "Mandatory negotiation period ended"},
                ))

        # 3d. Evaluate per-agent governance constraints
        for agent_id in list(realised_levels.keys()):
            realised = realised_levels[agent_id]
            nation = self._nations[agent_id]
            action = actions[agent_id]

            # Treaty enforcement
            if nation.active_treaties:
                treaty_max = self._config.governance.treaty_max_level
                if realised > treaty_max:
                    penalty = self._config.governance.treaty_defection_penalty
                    nation.diplomatic_standing = max(
                        0.0, nation.diplomatic_standing - 0.2,
                    )
                    nation.economic_strength = max(
                        0.0, nation.economic_strength - penalty,
                    )
                    self._treaties_violated[agent_id] = (
                        self._treaties_violated.get(agent_id, 0) + 1
                    )
                    turn_events.append(EscalationEvent(
                        event_type="treaty_violation",
                        turn=self._current_turn,
                        agent_id=agent_id,
                        details={
                            "level": realised,
                            "treaty_max": treaty_max,
                            "penalty": penalty,
                            "violations_total": self._treaties_violated[agent_id],
                        },
                    ))

            # De-escalation friction — deducts from economic_strength
            current = nation.current_level
            if realised < current:
                # Agent is de-escalating
                friction = self._compute_de_escalation_cost(current, realised)
                nation.governance_costs_paid += friction
                nation.economic_strength = max(
                    0.0, nation.economic_strength - friction,
                )

                # Commitment trap
                if (
                    self._config.signals.commitment_trap_enabled
                    and current in nation.public_commitments
                ):
                    commitment_cost = self._config.signals.commitment_reputation_cost
                    nation.diplomatic_standing = max(
                        0.0, nation.diplomatic_standing - commitment_cost * 0.01,
                    )
                    turn_events.append(EscalationEvent(
                        event_type="commitment_trap",
                        turn=self._current_turn,
                        agent_id=agent_id,
                        details={
                            "committed_level": current,
                            "new_level": realised,
                            "reputation_cost": commitment_cost * 0.01,
                        },
                    ))

                turn_events.append(EscalationEvent(
                    event_type="de_escalation_attempt",
                    turn=self._current_turn,
                    agent_id=agent_id,
                    details={
                        "from_level": current,
                        "to_level": realised,
                        "friction_cost": friction,
                    },
                ))

            # Track commitment (signal becomes a public commitment)
            signal = max(0, min(9, action.signal_level))
            if signal > current:
                nation.public_commitments.add(signal)

        # 4. Apply consequences
        self._apply_consequences(realised_levels, turn_events)

        # 5. Update trust scores and escalation levels
        for agent_id, action in actions.items():
            nation = self._nations[agent_id]
            signal = max(0, min(9, action.signal_level))
            realised = realised_levels[agent_id]

            # Update trust based on signal vs action (not realised, since
            # the agent can't control fog)
            intended = max(0, min(9, action.action_level))
            nation.update_trust(
                signal, intended, self._config.signals.trust_decay_rate,
            )

            # Update current level
            old_level = nation.current_level
            nation.current_level = EscalationLevel(realised)

            # Track nuclear threshold crossing
            if realised >= NUCLEAR_THRESHOLD and agent_id not in self._nuclear_agents:
                self._nuclear_agents.add(agent_id)
                if self._nuclear_threshold_turn is None:
                    self._nuclear_threshold_turn = self._current_turn
                turn_events.append(EscalationEvent(
                    event_type="nuclear_threshold_crossed",
                    turn=self._current_turn,
                    agent_id=agent_id,
                    details={
                        "level": realised,
                        "first_crossing": self._nuclear_threshold_turn == self._current_turn,
                    },
                ))

            # Log escalation change
            if realised != old_level:
                turn_events.append(EscalationEvent(
                    event_type="escalation_change",
                    turn=self._current_turn,
                    agent_id=agent_id,
                    details={
                        "from_level": int(old_level),
                        "to_level": realised,
                        "direction": "up" if realised > old_level else "down",
                    },
                ))

            # Log signal-action divergence
            divergence = abs(signal - intended)
            if divergence > 0:
                turn_events.append(EscalationEvent(
                    event_type="signal_action_divergence",
                    turn=self._current_turn,
                    agent_id=agent_id,
                    details={
                        "signal": signal,
                        "action": intended,
                        "divergence": divergence,
                    },
                ))

        # 6. Mediation (if enabled)
        if self._config.governance.mediation_enabled:
            mediation_events = self._apply_mediation(realised_levels)
            turn_events.extend(mediation_events)

        # 7. MAD deterrence check
        if self._config.governance.mad_enabled:
            mad_events = self._apply_mad_deterrence(realised_levels)
            turn_events.extend(mad_events)

        # 8. Check termination
        self._outcome = self._check_termination(realised_levels)

        # Build turn result
        result = TurnResult(
            turn=self._current_turn,
            actions=dict(actions),
            realised_levels=realised_levels,
            fog_deltas=fog_deltas,
            events=turn_events,
            outcome=self._outcome,
        )

        self._events.extend(turn_events)
        self._turn_results.append(result)
        self._current_turn += 1

        return result

    # ------------------------------------------------------------------
    # Fog-of-War
    # ------------------------------------------------------------------

    def _compute_fog_delta(self, intended_level: int, nation: NationState) -> int:
        """Compute fog-of-war noise for an intended action level.

        Higher levels have higher variance. Intelligence quality reduces noise.
        Positive skew biases toward accidental escalation.
        """
        if not self._config.fog_of_war.enabled:
            return 0

        fog = self._config.fog_of_war

        # Base noise
        sigma = fog.noise_sigma
        # Scale by level (higher levels = more chaos)
        sigma += intended_level * fog.per_level_variance_scale
        # Scale by timeline pressure
        if self._config.crisis.timeline_pressure and self._max_turns > 0:
            progress = self._current_turn / self._max_turns
            sigma += progress * self._config.crisis.pressure_fog_increase
        # Reduce by intelligence quality
        intel_factor = 1.0 - (
            nation.intelligence_quality * fog.intelligence_reduction_factor
        )
        sigma *= max(0.1, intel_factor)

        # Back-channel reduces fog
        if self._config.governance.back_channel_enabled:
            sigma *= (1.0 - fog.intelligence_reduction_factor * 0.3)

        # Sample noise with positive skew
        mu = fog.noise_mu + fog.positive_skew
        raw_delta = self._rng.gauss(mu, sigma)

        # Round to integer
        return int(round(raw_delta))

    # ------------------------------------------------------------------
    # Consequence Engine
    # ------------------------------------------------------------------

    def _apply_consequences(
        self,
        realised_levels: Dict[str, int],
        events: List[EscalationEvent],
    ) -> None:
        """Apply escalation consequences to all nations."""
        agent_ids = list(realised_levels.keys())

        for agent_id in agent_ids:
            level = realised_levels[agent_id]
            esc_level = EscalationLevel(level)
            nation = self._nations[agent_id]
            consequences = ESCALATION_CONSEQUENCES[esc_level]

            # Apply self-costs
            nation.apply_damage(
                military=consequences["military_self"],
                economic=consequences["economic_self"],
                civilian=consequences["civilian_self"],
            )

            # Apply target costs to opponents
            for opp_id in agent_ids:
                if opp_id == agent_id:
                    continue
                opponent = self._nations[opp_id]
                opponent.apply_damage(
                    military=consequences["military_target"],
                    economic=consequences["economic_target"],
                    civilian=consequences["civilian_target"],
                )

    # ------------------------------------------------------------------
    # De-Escalation
    # ------------------------------------------------------------------

    def _compute_de_escalation_cost(
        self, current_level: int, target_level: int,
    ) -> float:
        """Compute the cost of de-escalating from current to target level."""
        if target_level >= current_level:
            return 0.0

        total_cost = 0.0
        multiplier = self._config.governance.de_escalation_friction_multiplier
        for lvl in range(target_level, current_level):
            esc = EscalationLevel(lvl + 1)
            total_cost += DE_ESCALATION_FRICTION.get(esc, 0.0) * multiplier

        return total_cost

    # ------------------------------------------------------------------
    # Governance Mechanisms
    # ------------------------------------------------------------------

    def _apply_mediation(
        self, realised_levels: Dict[str, int],
    ) -> List[EscalationEvent]:
        """Apply third-party mediation if conditions are met."""
        events: List[EscalationEvent] = []

        # Check if any agent is above level 4 (mediation trigger)
        max_level = max(realised_levels.values()) if realised_levels else 0
        if max_level < 4:
            return events

        de_esc = self._config.governance.mediation_de_escalation_levels
        events.append(EscalationEvent(
            event_type="governance_intervention",
            turn=self._current_turn,
            agent_id="mediator",
            details={
                "type": "mediation",
                "proposed_de_escalation": de_esc,
                "trigger_level": max_level,
            },
        ))

        return events

    def _apply_mad_deterrence(
        self, realised_levels: Dict[str, int],
    ) -> List[EscalationEvent]:
        """Apply MAD deterrence: if one side uses nukes, opponents with
        second-strike capability retaliate with configured probability."""
        events: List[EscalationEvent] = []
        prob = self._config.governance.mad_retaliation_probability

        for agent_id, level in realised_levels.items():
            if level < NUCLEAR_THRESHOLD:
                continue

            for opp_id, opp in self._nations.items():
                if opp_id == agent_id:
                    continue
                if not (opp.has_second_strike and opp.has_nuclear):
                    continue

                if self._rng.random() < prob:
                    # Retaliate: raise opponent's level to match or exceed
                    old_level = realised_levels.get(opp_id, 0)
                    retaliation_level = max(old_level, level)
                    realised_levels[opp_id] = retaliation_level
                    opp.current_level = EscalationLevel(retaliation_level)
                    events.append(EscalationEvent(
                        event_type="mad_retaliation",
                        turn=self._current_turn,
                        agent_id=opp_id,
                        details={
                            "trigger_agent": agent_id,
                            "trigger_level": level,
                            "retaliation_level": retaliation_level,
                            "second_strike_capable": True,
                        },
                    ))
                else:
                    events.append(EscalationEvent(
                        event_type="mad_deterrence_signal",
                        turn=self._current_turn,
                        agent_id=opp_id,
                        details={
                            "trigger_agent": agent_id,
                            "trigger_level": level,
                            "second_strike_capable": True,
                            "retaliation_failed": True,
                        },
                    ))

        return events

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _check_termination(
        self, realised_levels: Dict[str, int],
    ) -> CrisisOutcome:
        """Check if the episode should terminate."""
        # Full strategic exchange by both sides
        all_max = all(
            lvl >= EscalationLevel.FULL_STRATEGIC_EXCHANGE
            for lvl in realised_levels.values()
        )
        if all_max:
            return CrisisOutcome.MUTUAL_DESTRUCTION

        # Strategic nuclear exchange (level 8+) by both sides
        both_nuclear = all(
            lvl >= EscalationLevel.STRATEGIC_NUCLEAR_STRIKE
            for lvl in realised_levels.values()
        )
        if both_nuclear:
            return CrisisOutcome.NUCLEAR_EXCHANGE

        # Surrender: one side at 0 while other is high
        for agent_id, level in realised_levels.items():
            if level == 0:
                others_high = any(
                    lvl >= EscalationLevel.LIMITED_STRIKE
                    for oid, lvl in realised_levels.items()
                    if oid != agent_id
                )
                # Check if nation's resources are depleted
                nation = self._nations[agent_id]
                if (
                    others_high
                    and nation.military_strength < 10
                    and nation.population_welfare < 10
                ):
                    return CrisisOutcome.SURRENDER

        # Ceasefire: all at status quo or diplomatic
        all_low = all(
            lvl <= EscalationLevel.DIPLOMATIC_PROTEST
            for lvl in realised_levels.values()
        )
        if all_low and self._current_turn > 2:
            return CrisisOutcome.CEASEFIRE

        # Timeout
        if self._current_turn >= self._max_turns - 1:
            return CrisisOutcome.TIMEOUT

        return CrisisOutcome.ONGOING

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def nations(self) -> Dict[str, NationState]:
        return dict(self._nations)

    @property
    def current_turn(self) -> int:
        return self._current_turn

    @property
    def outcome(self) -> CrisisOutcome:
        return self._outcome

    @property
    def events(self) -> List[EscalationEvent]:
        return list(self._events)

    @property
    def turn_results(self) -> List[TurnResult]:
        return list(self._turn_results)

    @property
    def nuclear_threshold_turn(self) -> Optional[int]:
        return self._nuclear_threshold_turn

    @property
    def nuclear_agents(self) -> set[str]:
        return set(self._nuclear_agents)

    @property
    def config(self) -> EscalationConfig:
        return self._config

    def is_terminal(self) -> bool:
        """Check if the episode has ended."""
        return self._outcome != CrisisOutcome.ONGOING

    def get_escalation_state(self) -> Dict[str, Any]:
        """Get a summary of the current escalation state."""
        return {
            "turn": self._current_turn,
            "max_turns": self._max_turns,
            "outcome": self._outcome.value,
            "nuclear_threshold_turn": self._nuclear_threshold_turn,
            "nuclear_agents": list(self._nuclear_agents),
            "circuit_breaker_active": self._circuit_breaker_active,
            "nations": {
                aid: {
                    "current_level": int(n.current_level),
                    "military_strength": n.military_strength,
                    "economic_strength": n.economic_strength,
                    "population_welfare": n.population_welfare,
                    "trust_score": n.trust_score,
                    "signal_action_divergence": n.signal_action_divergence(),
                    "cumulative_damage": (
                        n.cumulative_military_damage
                        + n.cumulative_economic_damage
                        + n.cumulative_civilian_casualties
                    ),
                }
                for aid, n in self._nations.items()
            },
        }
