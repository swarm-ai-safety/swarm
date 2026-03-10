"""Flash crash simulation engine for multi-agent swarms.

Models cascading confidence collapse analogous to the 2010 Flash Crash:
1. A shared "market confidence" signal tracks ecosystem health
2. A trigger event (large sell-off / exogenous shock) collapses confidence
3. Agents observe confidence and adjust behavior — withdrawing when confidence
   drops, which further reduces confidence (positive feedback loop)
4. Circuit breakers can halt activity when confidence falls too fast
5. A recovery phase where confidence slowly rebuilds if fundamentals are sound

The flash crash dynamics layer on top of the existing perturbation engine.
It modifies agent observables and engagement based on the confidence signal,
creating emergent cascading behavior without requiring agents to have
explicit "panic" logic.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CrashPhase(str, Enum):
    """Phases of a flash crash event."""

    PRE_CRASH = "pre_crash"        # Normal operations, building up
    TRIGGER = "trigger"            # Initial shock event
    CASCADE = "cascade"            # Positive feedback collapse
    CIRCUIT_BREAK = "circuit_break" # Trading halt / activity freeze
    RECOVERY = "recovery"          # Slow confidence rebuild
    POST_CRASH = "post_crash"      # New equilibrium


class TriggerType(str, Enum):
    """What initiates the crash."""

    EXOGENOUS_SHOCK = "exogenous_shock"    # External event (like a fat-finger trade)
    CONFIDENCE_EROSION = "confidence_erosion"  # Gradual buildup then snap
    COORDINATED_WITHDRAWAL = "coordinated_withdrawal"  # Agents collude to withdraw


class FlashCrashConfig(BaseModel):
    """Configuration for a flash crash simulation."""

    # Trigger parameters
    trigger_type: TriggerType = TriggerType.EXOGENOUS_SHOCK
    trigger_epoch: int = 5            # When the crash triggers
    trigger_magnitude: float = 0.4    # How much confidence drops instantly (0-1)

    # Cascade dynamics
    cascade_feedback_rate: float = 0.3  # How much low confidence feeds back (0-1)
    cascade_contagion_rate: float = 0.5  # How fast panic spreads between agents (0-1)
    confidence_floor: float = 0.05    # Minimum confidence (never truly zero)
    withdrawal_threshold: float = 0.4  # Agents withdraw below this confidence

    # Circuit breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: float = 0.2  # Confidence level that trips breaker
    circuit_breaker_duration_steps: int = 5  # How long the halt lasts
    circuit_breaker_cooldown_epochs: int = 1  # Cooldown before breaker can re-trip

    # Recovery dynamics
    recovery_rate: float = 0.05      # How fast confidence recovers per step
    recovery_requires_fundamentals: bool = True  # Only recover if avg p > threshold
    recovery_fundamental_threshold: float = 0.5  # Min avg p to allow recovery

    # Market depth / liquidity (analogous to order book depth)
    initial_confidence: float = 0.8  # Starting confidence level
    liquidity_depth: float = 1.0     # Higher = more resistant to shocks (0-2)

    # Spoofing (agents placing fake signals to manipulate confidence)
    spoofing_enabled: bool = False
    spoofing_agent_fraction: float = 0.2  # Fraction of agents that may spoof
    spoofing_magnitude: float = 0.1       # How much spoofing can move confidence


@dataclass
class AgentCrashState:
    """Per-agent state during a flash crash."""

    confidence: float = 0.8       # Agent's local confidence estimate
    withdrawn: bool = False       # Has agent withdrawn from activity?
    panic_level: float = 0.0      # 0 = calm, 1 = full panic
    withdrawal_step: int = -1     # When agent withdrew (-1 = not withdrawn)
    interactions_during_crash: int = 0
    losses_during_crash: float = 0.0


@dataclass
class FlashCrashState:
    """Global state for the flash crash simulation."""

    phase: CrashPhase = CrashPhase.PRE_CRASH
    market_confidence: float = 0.8  # Global confidence signal [0, 1]
    confidence_history: List[float] = field(default_factory=list)
    phase_history: List[str] = field(default_factory=list)

    # Per-agent crash state
    agent_states: Dict[str, AgentCrashState] = field(default_factory=dict)

    # Circuit breaker state
    circuit_breaker_active: bool = False
    circuit_breaker_steps_remaining: int = 0
    circuit_breaker_trip_count: int = 0
    last_circuit_break_epoch: int = -1

    # Crash metrics
    crash_start_step: int = -1
    crash_trough_step: int = -1
    crash_trough_confidence: float = 1.0
    recovery_start_step: int = -1
    total_steps: int = 0

    # Withdrawal tracking
    peak_withdrawal_fraction: float = 0.0
    total_withdrawal_episodes: int = 0


class FlashCrashEngine:
    """Simulates flash crash dynamics in a multi-agent swarm.

    The engine maintains a global confidence signal and per-agent panic
    levels. Each step:
    1. Update confidence based on recent interaction quality
    2. Check for trigger conditions
    3. Apply cascade dynamics (confidence -> panic -> withdrawal -> lower confidence)
    4. Check circuit breakers
    5. Apply recovery if conditions are met

    The engine exposes methods that the orchestrator's perturbation middleware
    can call to modify agent behavior during a crash.
    """

    def __init__(
        self,
        config: FlashCrashConfig,
        seed: Optional[int] = None,
    ):
        self.config = config
        self._rng = random.Random(seed)
        self.state = FlashCrashState(
            market_confidence=config.initial_confidence,
        )

    def get_agent_state(self, agent_id: str) -> AgentCrashState:
        """Get or create crash state for an agent."""
        if agent_id not in self.state.agent_states:
            self.state.agent_states[agent_id] = AgentCrashState(
                confidence=self.state.market_confidence,
            )
        return self.state.agent_states[agent_id]

    def step(
        self,
        epoch: int,
        step: int,
        agent_ids: List[str],
        recent_avg_p: float = 0.5,
    ) -> Dict[str, Any]:
        """Advance the flash crash simulation by one step.

        Args:
            epoch: Current epoch
            step: Current step within epoch
            agent_ids: List of active agent IDs
            recent_avg_p: Average p across recent interactions

        Returns:
            Dict with step results including phase, confidence, and
            per-agent withdrawal status.
        """
        self.state.total_steps += 1
        global_step = self.state.total_steps

        # Ensure all agents have state
        for aid in agent_ids:
            self.get_agent_state(aid)

        # 1. Check trigger
        if (
            self.state.phase == CrashPhase.PRE_CRASH
            and epoch >= self.config.trigger_epoch
        ):
            self._apply_trigger(global_step)

        # 2. Update confidence based on phase
        if self.state.phase == CrashPhase.TRIGGER:
            self._transition_to_cascade(global_step)

        if self.state.phase == CrashPhase.CASCADE:
            self._apply_cascade(global_step, agent_ids, recent_avg_p)

        if self.state.phase == CrashPhase.CIRCUIT_BREAK:
            self._apply_circuit_break(global_step, epoch)

        if self.state.phase == CrashPhase.RECOVERY:
            self._apply_recovery(global_step, recent_avg_p)

        # 3. Update per-agent states
        self._update_agent_states(agent_ids)

        # 4. Record history
        self.state.confidence_history.append(self.state.market_confidence)
        self.state.phase_history.append(self.state.phase.value)

        # 5. Track metrics
        withdrawn_count = sum(
            1 for s in self.state.agent_states.values() if s.withdrawn
        )
        withdrawal_fraction = (
            withdrawn_count / len(agent_ids) if agent_ids else 0.0
        )
        self.state.peak_withdrawal_fraction = max(
            self.state.peak_withdrawal_fraction, withdrawal_fraction
        )

        return {
            "phase": self.state.phase.value,
            "market_confidence": self.state.market_confidence,
            "withdrawn_agents": [
                aid for aid in agent_ids
                if self.state.agent_states[aid].withdrawn
            ],
            "withdrawal_fraction": withdrawal_fraction,
            "circuit_breaker_active": self.state.circuit_breaker_active,
            "global_step": global_step,
        }

    def _apply_trigger(self, global_step: int) -> None:
        """Apply the initial crash trigger."""
        self.state.phase = CrashPhase.TRIGGER
        self.state.crash_start_step = global_step

        if self.config.trigger_type == TriggerType.EXOGENOUS_SHOCK:
            # Sudden drop — like Waddell & Reed's $4.1B sell order
            drop = self.config.trigger_magnitude / max(
                self.config.liquidity_depth, 0.01
            )
            self.state.market_confidence = max(
                self.config.confidence_floor,
                self.state.market_confidence - drop,
            )

        elif self.config.trigger_type == TriggerType.CONFIDENCE_EROSION:
            # Smaller initial drop, but primes the cascade
            drop = self.config.trigger_magnitude * 0.5
            self.state.market_confidence = max(
                self.config.confidence_floor,
                self.state.market_confidence - drop,
            )

        elif self.config.trigger_type == TriggerType.COORDINATED_WITHDRAWAL:
            # Multiple agents withdraw simultaneously
            agent_ids = list(self.state.agent_states.keys())
            n_withdraw = max(
                1,
                int(len(agent_ids) * self.config.spoofing_agent_fraction),
            )
            selected = self._rng.sample(
                agent_ids, min(n_withdraw, len(agent_ids))
            )
            for aid in selected:
                astate = self.state.agent_states[aid]
                astate.withdrawn = True
                astate.panic_level = 0.8
                astate.withdrawal_step = global_step
                self.state.total_withdrawal_episodes += 1

            # Withdrawal impacts confidence
            withdrawal_impact = len(selected) / max(len(agent_ids), 1)
            self.state.market_confidence = max(
                self.config.confidence_floor,
                self.state.market_confidence - withdrawal_impact * 0.3,
            )

    def _transition_to_cascade(self, global_step: int) -> None:
        """Move from trigger to cascade phase."""
        self.state.phase = CrashPhase.CASCADE

    def _apply_cascade(
        self,
        global_step: int,
        agent_ids: List[str],
        recent_avg_p: float,
    ) -> None:
        """Apply cascading confidence collapse.

        This is the core flash crash dynamic: low confidence causes
        withdrawals, which reduce liquidity, which drops confidence
        further. Analogous to the feedback loop between HFT withdrawal
        and declining order book depth in the 2010 crash.
        """
        # Count withdrawn agents (reduced "liquidity")
        withdrawn_count = sum(
            1 for aid in agent_ids
            if self.state.agent_states.get(aid, AgentCrashState()).withdrawn
        )
        active_fraction = 1.0 - (
            withdrawn_count / max(len(agent_ids), 1)
        )

        # Feedback: fewer active agents -> less liquidity -> bigger drops
        # This mirrors the "hot potato" volume — trading continues but
        # with increasingly thin liquidity
        liquidity_factor = active_fraction * self.config.liquidity_depth
        feedback_drop = (
            self.config.cascade_feedback_rate
            * (1.0 - self.state.market_confidence)
            / max(liquidity_factor, 0.1)
        )

        # Quality signal: if interactions are actually bad, confidence
        # drops faster (this is the fundamental vs. panic distinction)
        quality_penalty = max(0.0, 0.5 - recent_avg_p) * 0.2

        total_drop = feedback_drop + quality_penalty
        self.state.market_confidence = max(
            self.config.confidence_floor,
            self.state.market_confidence - total_drop,
        )

        # Track trough
        if self.state.market_confidence < self.state.crash_trough_confidence:
            self.state.crash_trough_confidence = self.state.market_confidence
            self.state.crash_trough_step = global_step

        # Check circuit breaker
        if (
            self.config.circuit_breaker_enabled
            and self.state.market_confidence
            <= self.config.circuit_breaker_threshold
        ):
            self._trip_circuit_breaker(global_step)
            return

        # Check if cascade has exhausted itself (confidence stabilizing)
        if (
            len(self.state.confidence_history) >= 3
            and all(
                abs(self.state.confidence_history[-(i + 1)] - self.state.market_confidence)
                < 0.01
                for i in range(min(3, len(self.state.confidence_history)))
            )
        ):
            self.state.phase = CrashPhase.RECOVERY
            self.state.recovery_start_step = global_step

    def _trip_circuit_breaker(self, global_step: int) -> None:
        """Activate the circuit breaker — halt all activity."""
        self.state.phase = CrashPhase.CIRCUIT_BREAK
        self.state.circuit_breaker_active = True
        self.state.circuit_breaker_steps_remaining = (
            self.config.circuit_breaker_duration_steps
        )
        self.state.circuit_breaker_trip_count += 1

    def _apply_circuit_break(self, global_step: int, epoch: int) -> None:
        """Process a circuit break step — activity is halted."""
        self.state.circuit_breaker_steps_remaining -= 1

        # During halt, confidence stabilizes slightly (panic subsides)
        stabilization = 0.02
        self.state.market_confidence = min(
            1.0,
            self.state.market_confidence + stabilization,
        )

        if self.state.circuit_breaker_steps_remaining <= 0:
            # Breaker lifts
            self.state.circuit_breaker_active = False
            self.state.last_circuit_break_epoch = epoch
            self.state.phase = CrashPhase.RECOVERY
            self.state.recovery_start_step = global_step

    def _apply_recovery(
        self,
        global_step: int,
        recent_avg_p: float,
    ) -> None:
        """Apply recovery dynamics — slow confidence rebuild."""
        can_recover = True
        if self.config.recovery_requires_fundamentals:
            can_recover = (
                recent_avg_p >= self.config.recovery_fundamental_threshold
            )

        if can_recover:
            # Gradual recovery — much slower than the crash
            # (matches the empirical asymmetry: crashes are fast, recovery is slow)
            recovery = self.config.recovery_rate * (
                1.0 - self.state.market_confidence
            )
            self.state.market_confidence = min(
                1.0,
                self.state.market_confidence + recovery,
            )
        else:
            # Without good fundamentals, confidence erodes slowly
            self.state.market_confidence = max(
                self.config.confidence_floor,
                self.state.market_confidence - 0.01,
            )

        # Check if fully recovered
        if self.state.market_confidence >= self.config.initial_confidence * 0.95:
            self.state.phase = CrashPhase.POST_CRASH

    def _update_agent_states(self, agent_ids: List[str]) -> None:
        """Update per-agent panic and withdrawal based on market confidence."""
        for aid in agent_ids:
            astate = self.get_agent_state(aid)

            # Contagion: agent confidence tracks market with noise
            # (some agents are more susceptible to panic than others)
            noise = self._rng.gauss(0, 0.05)
            contagion_pull = self.config.cascade_contagion_rate * (
                self.state.market_confidence - astate.confidence
            )
            astate.confidence = max(
                0.0,
                min(1.0, astate.confidence + contagion_pull + noise),
            )

            # Panic level inversely tracks confidence
            astate.panic_level = max(0.0, min(1.0, 1.0 - astate.confidence))

            # Withdrawal logic
            if not astate.withdrawn:
                if astate.confidence < self.config.withdrawal_threshold:
                    # Probabilistic withdrawal — lower confidence = higher chance
                    withdraw_prob = (
                        self.config.withdrawal_threshold - astate.confidence
                    ) / self.config.withdrawal_threshold
                    if self._rng.random() < withdraw_prob:
                        astate.withdrawn = True
                        astate.withdrawal_step = self.state.total_steps
                        self.state.total_withdrawal_episodes += 1
            else:
                # Re-entry: agents return when confidence recovers
                if astate.confidence > self.config.withdrawal_threshold * 1.2:
                    re_entry_prob = 0.3 * (
                        astate.confidence - self.config.withdrawal_threshold
                    )
                    if self._rng.random() < re_entry_prob:
                        astate.withdrawn = False

    def modify_observables(
        self,
        agent_id: str,
        observables_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Modify agent observables based on crash state.

        During a crash, agents' observable signals degrade — engagement
        drops, rework increases, and progress stalls. This models the
        real-world phenomenon where market stress corrupts signal quality.

        Args:
            agent_id: The agent whose observables to modify
            observables_dict: Original observable values

        Returns:
            Modified observables dict
        """
        if self.state.phase == CrashPhase.PRE_CRASH:
            return observables_dict

        astate = self.get_agent_state(agent_id)
        modified = dict(observables_dict)

        # Engagement drops with panic
        if "counterparty_engagement_delta" in modified:
            engagement_penalty = -astate.panic_level * 0.5
            modified["counterparty_engagement_delta"] = max(
                -1.0,
                modified["counterparty_engagement_delta"] + engagement_penalty,
            )

        # Progress slows during panic
        if "task_progress_delta" in modified:
            progress_dampening = 1.0 - astate.panic_level * 0.7
            modified["task_progress_delta"] = (
                modified["task_progress_delta"] * progress_dampening
            )

        # Rework increases under stress
        if "rework_count" in modified and astate.panic_level > 0.5:
            extra_rework = self._rng.randint(0, 2)
            modified["rework_count"] = modified["rework_count"] + extra_rework

        return modified

    def get_crash_metrics(self) -> Dict[str, Any]:
        """Return summary metrics for the flash crash event."""
        history = self.state.confidence_history
        if not history:
            return {"crash_occurred": False}

        peak_confidence = max(history) if history else self.config.initial_confidence
        trough_confidence = self.state.crash_trough_confidence

        # Crash depth: how far confidence fell from peak
        crash_depth = peak_confidence - trough_confidence

        # Crash duration: steps from trigger to recovery start
        crash_duration = 0
        if self.state.crash_start_step >= 0:
            end = self.state.recovery_start_step
            if end < 0:
                end = self.state.total_steps
            crash_duration = end - self.state.crash_start_step

        # Recovery time: steps from recovery start to post-crash
        recovery_duration = 0
        if self.state.recovery_start_step >= 0:
            recovery_duration = (
                self.state.total_steps - self.state.recovery_start_step
            )

        # Asymmetry ratio: recovery time / crash time
        # The 2010 flash crash: ~5 min crash, ~20 min recovery => ratio ~4
        asymmetry_ratio = (
            recovery_duration / max(crash_duration, 1)
            if crash_duration > 0
            else 0.0
        )

        return {
            "crash_occurred": self.state.crash_start_step >= 0,
            "crash_depth": crash_depth,
            "crash_duration_steps": crash_duration,
            "recovery_duration_steps": recovery_duration,
            "asymmetry_ratio": asymmetry_ratio,
            "trough_confidence": trough_confidence,
            "peak_withdrawal_fraction": self.state.peak_withdrawal_fraction,
            "circuit_breaker_trips": self.state.circuit_breaker_trip_count,
            "total_withdrawal_episodes": self.state.total_withdrawal_episodes,
            "final_confidence": self.state.market_confidence,
            "final_phase": self.state.phase.value,
            "confidence_history": list(history),
        }

    def should_block_agent(self, agent_id: str) -> bool:
        """Check if an agent should be blocked from acting.

        During a circuit break, all agents are blocked.
        Withdrawn agents are also blocked.
        """
        if self.state.circuit_breaker_active:
            return True
        astate = self.get_agent_state(agent_id)
        return astate.withdrawn
