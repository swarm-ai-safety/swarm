"""Tests for the flash crash simulation engine."""

import pytest

from swarm.scenarios.flash_crash import (
    AgentCrashState,
    CrashPhase,
    FlashCrashConfig,
    FlashCrashEngine,
    TriggerType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    return FlashCrashConfig()


@pytest.fixture
def engine(default_config):
    return FlashCrashEngine(default_config, seed=42)


@pytest.fixture
def agent_ids():
    return [f"agent_{i}" for i in range(10)]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestFlashCrashConfig:
    def test_default_values(self):
        config = FlashCrashConfig()
        assert config.trigger_epoch == 5
        assert config.trigger_magnitude == 0.4
        assert config.initial_confidence == 0.8
        assert config.circuit_breaker_enabled is True

    def test_custom_values(self):
        config = FlashCrashConfig(
            trigger_type=TriggerType.CONFIDENCE_EROSION,
            trigger_epoch=10,
            trigger_magnitude=0.6,
        )
        assert config.trigger_type == TriggerType.CONFIDENCE_EROSION
        assert config.trigger_epoch == 10
        assert config.trigger_magnitude == 0.6


# ---------------------------------------------------------------------------
# Engine initialisation
# ---------------------------------------------------------------------------

class TestFlashCrashEngine:
    def test_initial_state(self, engine):
        assert engine.state.phase == CrashPhase.PRE_CRASH
        assert engine.state.market_confidence == 0.8
        assert engine.state.circuit_breaker_active is False
        assert len(engine.state.confidence_history) == 0

    def test_agent_state_creation(self, engine):
        state = engine.get_agent_state("agent_0")
        assert isinstance(state, AgentCrashState)
        assert state.confidence == 0.8
        assert state.withdrawn is False
        assert state.panic_level == 0.0


# ---------------------------------------------------------------------------
# Pre-crash phase
# ---------------------------------------------------------------------------

class TestPreCrashPhase:
    def test_no_trigger_before_epoch(self, engine, agent_ids):
        """Confidence should stay stable before trigger epoch."""
        result = engine.step(epoch=0, step=0, agent_ids=agent_ids)
        assert result["phase"] == "pre_crash"
        assert result["market_confidence"] == pytest.approx(0.8, abs=0.1)
        assert len(result["withdrawn_agents"]) == 0

    def test_stays_pre_crash_until_trigger(self, engine, agent_ids):
        """System stays in pre-crash until trigger epoch."""
        for epoch in range(5):
            for step in range(10):
                engine.step(
                    epoch=epoch, step=step, agent_ids=agent_ids
                )
        assert engine.state.phase == CrashPhase.PRE_CRASH


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

class TestTrigger:
    def test_exogenous_shock_drops_confidence(self, agent_ids):
        config = FlashCrashConfig(
            trigger_type=TriggerType.EXOGENOUS_SHOCK,
            trigger_epoch=0,
            trigger_magnitude=0.4,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)
        result = engine.step(epoch=0, step=0, agent_ids=agent_ids)
        # Trigger step applies the shock — confidence should have dropped
        assert result["phase"] == "trigger"
        assert result["market_confidence"] < 0.8

    def test_confidence_erosion_trigger(self, agent_ids):
        config = FlashCrashConfig(
            trigger_type=TriggerType.CONFIDENCE_EROSION,
            trigger_epoch=0,
            trigger_magnitude=0.4,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)
        result = engine.step(epoch=0, step=0, agent_ids=agent_ids)
        # Erosion trigger drops less than exogenous shock
        assert result["market_confidence"] < 0.8
        # But not as much as full magnitude
        assert result["market_confidence"] > 0.4

    def test_trigger_phase_is_observable(self, agent_ids):
        """The trigger phase should appear in results for exactly one step."""
        config = FlashCrashConfig(
            trigger_type=TriggerType.EXOGENOUS_SHOCK,
            trigger_epoch=0,
            trigger_magnitude=0.4,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)
        result = engine.step(epoch=0, step=0, agent_ids=agent_ids)
        assert result["phase"] == "trigger"

        # Next step should transition to cascade
        result2 = engine.step(epoch=0, step=1, agent_ids=agent_ids)
        assert result2["phase"] in ("cascade", "circuit_break", "recovery")

    def test_coordinated_withdrawal_trigger(self, agent_ids):
        config = FlashCrashConfig(
            trigger_type=TriggerType.COORDINATED_WITHDRAWAL,
            trigger_epoch=0,
            coordinated_withdrawal_fraction=0.3,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)
        result = engine.step(epoch=0, step=0, agent_ids=agent_ids)
        # Some agents should be withdrawn
        assert len(result["withdrawn_agents"]) > 0


# ---------------------------------------------------------------------------
# Cascade dynamics
# ---------------------------------------------------------------------------

class TestCascade:
    def test_confidence_decreases_during_cascade(self, agent_ids):
        """Confidence should keep falling during cascade phase."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.3,
            cascade_feedback_rate=0.3,
            circuit_breaker_enabled=False,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        confidences = []
        for step in range(10):
            result = engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.3
            )
            confidences.append(result["market_confidence"])

        # Confidence should trend downward during cascade
        assert confidences[-1] < confidences[0]

    def test_cascade_respects_confidence_floor(self, agent_ids):
        """Confidence should never go below the floor."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.9,
            cascade_feedback_rate=0.8,
            confidence_floor=0.05,
            circuit_breaker_enabled=False,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        for step in range(50):
            result = engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.1
            )

        assert result["market_confidence"] >= 0.05

    def test_agents_withdraw_during_cascade(self, agent_ids):
        """Agents should withdraw as confidence drops."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.5,
            cascade_feedback_rate=0.4,
            withdrawal_threshold=0.4,
            cascade_contagion_rate=0.8,
            circuit_breaker_enabled=False,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        max_withdrawn = 0
        for step in range(20):
            result = engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.3
            )
            max_withdrawn = max(max_withdrawn, len(result["withdrawn_agents"]))

        # At least some agents should have withdrawn
        assert max_withdrawn > 0

    def test_low_liquidity_amplifies_crash(self, agent_ids):
        """Lower liquidity depth should cause deeper initial crash."""
        configs = [
            FlashCrashConfig(
                trigger_epoch=0,
                trigger_magnitude=0.3,
                liquidity_depth=depth,
                circuit_breaker_enabled=False,
                cascade_feedback_rate=0.1,
                initial_confidence=0.8,
            )
            for depth in [2.0, 0.5]
        ]

        # Compare confidence after just 3 steps (before both hit the floor)
        step3_confidences = []
        for config in configs:
            engine = FlashCrashEngine(config, seed=42)
            for step in range(3):
                result = engine.step(
                    epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.4
                )
            step3_confidences.append(result["market_confidence"])

        # Low liquidity (0.5) should result in lower confidence than high (2.0)
        assert step3_confidences[1] < step3_confidences[0]


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_circuit_breaker_trips(self, agent_ids):
        """Circuit breaker should trip when confidence hits threshold."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.7,
            cascade_feedback_rate=0.5,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=0.2,
            circuit_breaker_duration_steps=3,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        tripped = False
        for step in range(30):
            result = engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.3
            )
            if result["circuit_breaker_active"]:
                tripped = True
                break

        assert tripped
        assert engine.state.circuit_breaker_trip_count >= 1

    def test_circuit_breaker_stabilizes_confidence(self, agent_ids):
        """During circuit break, confidence should increase each step."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.7,
            cascade_feedback_rate=0.5,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=0.2,
            circuit_breaker_duration_steps=5,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        # Run until circuit breaker trips, then track confidence during break
        confidence_during_break = []
        for step in range(30):
            result = engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.3
            )
            if result["phase"] == "circuit_break":
                confidence_during_break.append(result["market_confidence"])

        # During circuit break, each step should increase confidence
        if len(confidence_during_break) >= 2:
            for i in range(1, len(confidence_during_break)):
                assert confidence_during_break[i] >= confidence_during_break[i - 1]

    def test_circuit_breaker_blocks_agents(self, agent_ids):
        """All agents should be blocked during circuit break."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.8,
            cascade_feedback_rate=0.6,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=0.3,
            circuit_breaker_duration_steps=5,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        # Run until circuit breaker trips
        for step in range(30):
            engine.step(epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.2)
            if engine.state.circuit_breaker_active:
                for aid in agent_ids:
                    assert engine.should_block_agent(aid)
                break


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

class TestRecovery:
    def test_recovery_with_good_fundamentals(self, agent_ids):
        """Confidence should recover when fundamentals are good."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.4,
            cascade_feedback_rate=0.2,
            circuit_breaker_enabled=False,
            recovery_rate=0.1,
            recovery_requires_fundamentals=True,
            recovery_fundamental_threshold=0.5,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        # Crash phase
        for step in range(15):
            engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.3
            )

        confidence_at_trough = engine.state.market_confidence

        # Recovery phase with good fundamentals
        for step in range(50):
            engine.step(
                epoch=1, step=step, agent_ids=agent_ids, recent_avg_p=0.7
            )

        # Confidence should have recovered
        assert engine.state.market_confidence > confidence_at_trough

    def test_no_recovery_with_bad_fundamentals(self, agent_ids):
        """Confidence should NOT recover when fundamentals are bad."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.4,
            cascade_feedback_rate=0.2,
            circuit_breaker_enabled=False,
            recovery_rate=0.1,
            recovery_requires_fundamentals=True,
            recovery_fundamental_threshold=0.5,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        # Run many steps with bad fundamentals
        for step in range(100):
            engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.2
            )

        # Confidence should be very low
        assert engine.state.market_confidence < 0.3

    def test_recovery_asymmetry(self, agent_ids):
        """Recovery should be slower than crash (asymmetric)."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.4,
            cascade_feedback_rate=0.3,
            circuit_breaker_enabled=False,
            recovery_rate=0.05,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        # Run the full crash + recovery cycle
        for step in range(100):
            engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.6
            )

        metrics = engine.get_crash_metrics()
        # Ensure we actually entered crash and recovery
        assert metrics["crash_occurred"]
        assert metrics["recovery_duration_steps"] > 0
        # Recovery should take longer than crash
        assert metrics["asymmetry_ratio"] >= 1.0


# ---------------------------------------------------------------------------
# Observable modification
# ---------------------------------------------------------------------------

class TestObservableModification:
    def test_no_modification_pre_crash(self, engine, agent_ids):
        """Observables should not be modified before crash."""
        obs = {
            "task_progress_delta": 0.5,
            "counterparty_engagement_delta": 0.3,
            "rework_count": 0,
        }
        modified = engine.modify_observables("agent_0", obs)
        assert modified == obs
        # Should be a copy, not the original reference
        assert modified is not obs

    def test_engagement_drops_during_crash(self, agent_ids):
        """Engagement should drop during panic."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.5,
            initial_confidence=0.8,
            circuit_breaker_enabled=False,
        )
        engine = FlashCrashEngine(config, seed=42)

        # Trigger and run cascade
        for step in range(10):
            engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.3
            )

        obs = {
            "task_progress_delta": 0.5,
            "counterparty_engagement_delta": 0.5,
            "rework_count": 0,
        }
        modified = engine.modify_observables(agent_ids[0], obs)

        # Engagement should be reduced
        assert modified["counterparty_engagement_delta"] <= obs["counterparty_engagement_delta"]

    def test_progress_dampened_during_crash(self, agent_ids):
        """Task progress should slow during panic."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.5,
            initial_confidence=0.8,
            circuit_breaker_enabled=False,
        )
        engine = FlashCrashEngine(config, seed=42)

        for step in range(10):
            engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.3
            )

        obs = {"task_progress_delta": 0.8, "rework_count": 0}
        modified = engine.modify_observables(agent_ids[0], obs)
        assert modified["task_progress_delta"] <= obs["task_progress_delta"]


# ---------------------------------------------------------------------------
# Crash metrics
# ---------------------------------------------------------------------------

class TestCrashMetrics:
    def test_metrics_no_crash(self, engine):
        """Metrics should indicate no crash occurred if none happened."""
        metrics = engine.get_crash_metrics()
        assert metrics["crash_occurred"] is False

    def test_metrics_after_crash(self, agent_ids):
        """Metrics should capture crash characteristics."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.5,
            cascade_feedback_rate=0.3,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=0.2,
            circuit_breaker_duration_steps=3,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        for step in range(30):
            engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.4
            )

        metrics = engine.get_crash_metrics()
        assert metrics["crash_occurred"] is True
        assert metrics["crash_depth"] > 0
        assert metrics["trough_confidence"] < 0.8
        assert metrics["final_confidence"] >= 0.0
        assert len(metrics["confidence_history"]) == 30

    def test_peak_withdrawal_tracked(self, agent_ids):
        """Peak withdrawal fraction should be tracked."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.6,
            cascade_feedback_rate=0.4,
            cascade_contagion_rate=0.8,
            circuit_breaker_enabled=False,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        for step in range(20):
            engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.2
            )

        metrics = engine.get_crash_metrics()
        assert metrics["peak_withdrawal_fraction"] >= 0.0
        assert metrics["peak_withdrawal_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# Agent blocking
# ---------------------------------------------------------------------------

class TestAgentBlocking:
    def test_withdrawn_agents_blocked(self, agent_ids):
        """Withdrawn agents should be blocked."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.6,
            cascade_contagion_rate=0.9,
            circuit_breaker_enabled=False,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        for step in range(20):
            result = engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.2
            )
            for aid in result["withdrawn_agents"]:
                assert engine.should_block_agent(aid)

    def test_active_agents_not_blocked_outside_circuit_break(self, engine, agent_ids):
        """Non-withdrawn agents should not be blocked."""
        engine.step(epoch=0, step=0, agent_ids=agent_ids)
        for aid in agent_ids:
            assert not engine.should_block_agent(aid)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_results(self, agent_ids):
        """Same seed should produce identical results."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.4,
            initial_confidence=0.8,
        )

        runs = []
        for _ in range(2):
            engine = FlashCrashEngine(config, seed=123)
            history = []
            for step in range(20):
                result = engine.step(
                    epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.4
                )
                history.append(result["market_confidence"])
            runs.append(history)

        assert runs[0] == runs[1]

    def test_different_seeds_different_agent_states(self, agent_ids):
        """Different seeds should produce different per-agent states."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.4,
            cascade_contagion_rate=0.5,
            initial_confidence=0.8,
            circuit_breaker_enabled=False,
        )

        agent_confidences = []
        for seed in [1, 2]:
            engine = FlashCrashEngine(config, seed=seed)
            for step in range(20):
                engine.step(
                    epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.4
                )
            # Collect per-agent confidence values (these are stochastic)
            confs = [
                engine.get_agent_state(aid).confidence for aid in agent_ids
            ]
            agent_confidences.append(confs)

        # Per-agent confidences should differ due to different RNG seeds
        assert agent_confidences[0] != agent_confidences[1]


# ---------------------------------------------------------------------------
# Full crash cycle
# ---------------------------------------------------------------------------

class TestFullCrashCycle:
    def test_complete_crash_recovery_cycle(self, agent_ids):
        """Test a complete crash -> circuit break -> recovery cycle."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.5,
            cascade_feedback_rate=0.3,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=0.2,
            circuit_breaker_duration_steps=3,
            recovery_rate=0.1,
            recovery_requires_fundamentals=True,
            recovery_fundamental_threshold=0.4,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        phases_seen = set()
        for step in range(100):
            result = engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.6
            )
            phases_seen.add(result["phase"])

        # Trigger phase should now be observable
        assert "trigger" in phases_seen
        # Should have gone through multiple phases
        assert len(phases_seen) > 1
        # Crash should have occurred
        metrics = engine.get_crash_metrics()
        assert metrics["crash_occurred"]
        # Final confidence should be higher than the trough
        assert metrics["final_confidence"] > metrics["trough_confidence"]


# ---------------------------------------------------------------------------
# Circuit breaker cooldown
# ---------------------------------------------------------------------------

class TestCircuitBreakerCooldown:
    def test_cooldown_prevents_immediate_retrip(self, agent_ids):
        """Circuit breaker should not re-trip within cooldown period."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.6,
            cascade_feedback_rate=0.5,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=0.3,
            circuit_breaker_duration_steps=3,
            circuit_breaker_cooldown_epochs=2,
            recovery_rate=0.02,
            recovery_requires_fundamentals=False,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        # Run epoch 0: should trigger and trip circuit breaker once
        for step in range(30):
            engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.3
            )

        first_trip_count = engine.state.circuit_breaker_trip_count
        assert first_trip_count >= 1

        # Run epoch 0 more — still within cooldown (need 2 epochs)
        # Even if confidence drops again, breaker should NOT re-trip
        # because last_circuit_break_epoch=0 and epoch=0, so
        # 0 - 0 = 0 < 2 (cooldown)
        for step in range(30, 60):
            engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.2
            )

        # Trip count should not have increased during same epoch
        assert engine.state.circuit_breaker_trip_count == first_trip_count


# ---------------------------------------------------------------------------
# Slow decay stabilization guard
# ---------------------------------------------------------------------------

class TestStabilizationGuard:
    def test_slow_decay_does_not_false_stabilize(self, agent_ids):
        """A slowly declining cascade should not be classified as stabilized."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.2,
            # Very small feedback rate -> slow decline
            cascade_feedback_rate=0.02,
            circuit_breaker_enabled=False,
            confidence_floor=0.01,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        # Run enough steps for the old check to false-positive
        for step in range(30):
            engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.3
            )

        # If confidence is still declining, phase should still be CASCADE
        history = engine.state.confidence_history
        if len(history) >= 2 and history[-1] < history[-2] - 0.001:
            assert engine.state.phase == CrashPhase.CASCADE


# ---------------------------------------------------------------------------
# Re-crash from recovery (double-dip)
# ---------------------------------------------------------------------------

class TestDoubleDip:
    def test_recovery_can_reenter_cascade(self, agent_ids):
        """If confidence drops back to threshold during recovery, re-enter cascade."""
        config = FlashCrashConfig(
            trigger_epoch=0,
            trigger_magnitude=0.4,
            cascade_feedback_rate=0.3,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=0.15,
            circuit_breaker_duration_steps=3,
            circuit_breaker_cooldown_epochs=1,
            recovery_rate=0.02,
            recovery_requires_fundamentals=True,
            recovery_fundamental_threshold=0.5,
            confidence_floor=0.05,
            initial_confidence=0.8,
        )
        engine = FlashCrashEngine(config, seed=42)

        # Epoch 0: crash -> circuit break -> recovery
        for step in range(40):
            engine.step(
                epoch=0, step=step, agent_ids=agent_ids, recent_avg_p=0.6
            )

        # Epoch 1+: bad fundamentals during recovery should erode confidence
        # and eventually re-enter cascade
        phases_seen = set()
        for epoch in range(1, 10):
            for step in range(20):
                result = engine.step(
                    epoch=epoch, step=step, agent_ids=agent_ids, recent_avg_p=0.1
                )
                phases_seen.add(result["phase"])

        # With bad fundamentals, confidence erodes in recovery and should
        # eventually drop below the circuit breaker threshold, triggering
        # a re-entry into cascade
        assert "cascade" in phases_seen
