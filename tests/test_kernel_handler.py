"""Tests for the kernel oracle handler (v2: correlated cheating, OOD, tolerance)."""

import pytest

from swarm.agents.base import Action, ActionType
from swarm.core.kernel_handler import (
    CHALLENGE_CATALOG,
    DIFFICULTY_PENALTY,
    DIFFICULTY_SPEEDUP_VARIANCE,
    KernelOracleConfig,
    KernelOracleHandler,
)
from swarm.env.state import EnvState
from swarm.models.agent import AgentType
from swarm.models.events import Event


class _EventCollector:
    """Collects emitted events for assertions."""

    def __init__(self):
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)


def _make_handler(seed: int = 42, **kwargs) -> tuple[KernelOracleHandler, _EventCollector]:
    from swarm.logging.event_bus import EventBus

    collector = _EventCollector()
    bus = EventBus()
    bus.subscribe(collector)
    config = KernelOracleConfig(seed=seed, **kwargs)
    handler = KernelOracleHandler(config=config, event_bus=bus)
    return handler, collector


def _make_state(*agent_specs: tuple[str, AgentType]) -> EnvState:
    state = EnvState()
    for agent_id, agent_type in agent_specs:
        state.add_agent(agent_id=agent_id, name=agent_id, agent_type=agent_type)
    return state


class TestConfigValidation:
    def test_valid_config(self):
        config = KernelOracleConfig()
        assert config.enabled is True

    def test_invalid_functional_quality(self):
        with pytest.raises(ValueError):
            KernelOracleConfig(honest_functional_quality=1.5)

    def test_invalid_difficulty_distribution(self):
        with pytest.raises(ValueError):
            KernelOracleConfig(difficulty_distribution={"easy": 0.1, "medium": 0.1, "hard": 0.1})

    def test_valid_custom_config(self):
        config = KernelOracleConfig(
            honest_functional_quality=0.9,
            adversarial_cheat_rate=0.8,
            audit_detection_rate=0.95,
        )
        assert config.honest_functional_quality == 0.9
        assert config.adversarial_cheat_rate == 0.8

    def test_legacy_quality_mean_migration(self):
        """Old YAMLs using honest_quality_mean should map to functional_quality."""
        config = KernelOracleConfig(
            honest_quality_mean=0.9,
            opportunistic_quality_mean=0.6,
            adversarial_quality_mean=0.8,
        )
        assert config.honest_functional_quality == 0.9
        assert config.opportunistic_functional_quality == 0.6
        assert config.adversarial_functional_quality == 0.8

    def test_new_v2_fields_have_defaults(self):
        config = KernelOracleConfig()
        assert config.cheat_speedup_bonus == 0.40
        assert config.honest_ood_quality == 0.80
        assert config.adversarial_ood_quality == 0.30
        assert config.honest_tolerance_margin == 0.70
        assert config.adversarial_tolerance_margin == 0.05


class TestChallengeCatalog:
    def test_catalog_populated(self):
        assert len(CHALLENGE_CATALOG) > 0

    def test_catalog_has_all_difficulties(self):
        difficulties = {c.difficulty for c in CHALLENGE_CATALOG}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_catalog_has_ood_tests(self):
        for c in CHALLENGE_CATALOG:
            assert c.num_ood_tests > 0, f"{c.challenge_id} missing OOD tests"

    def test_difficulty_penalties_defined(self):
        for difficulty in ("easy", "medium", "hard"):
            assert difficulty in DIFFICULTY_PENALTY

    def test_difficulty_speedup_variance_defined(self):
        for difficulty in ("easy", "medium", "hard"):
            assert difficulty in DIFFICULTY_SPEEDUP_VARIANCE

    def test_epoch_challenge_sampling(self):
        handler, _ = _make_handler()
        state = _make_state()
        handler.on_epoch_start(state)
        assert len(handler._epoch_challenges) > 0


class TestSubmitKernel:
    def test_honest_agent_high_pass_rate(self):
        handler, collector = _make_handler(seed=42)
        state = _make_state(("honest_1", AgentType.HONEST))
        handler.on_epoch_start(state)

        action = Action(
            action_type=ActionType.SUBMIT_KERNEL,
            agent_id="honest_1",
        )
        result = handler.handle_action(action, state)

        assert result.success
        assert result.submission is not None
        assert result.submission.is_cheat is False
        assert result.observables is not None
        # Honest agent should pass most tests
        pass_rate = result.submission.tests_passed / max(1, result.submission.tests_total)
        assert pass_rate > 0.3  # Generous lower bound given randomness

    def test_adversarial_agent_produces_cheats(self):
        handler, _ = _make_handler(seed=42, adversarial_cheat_rate=1.0)
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))
        handler.on_epoch_start(state)

        action = Action(
            action_type=ActionType.SUBMIT_KERNEL,
            agent_id="adv_1",
        )
        result = handler.handle_action(action, state)

        assert result.success
        assert result.submission is not None
        assert result.submission.is_cheat is True

    def test_adversarial_cheat_rate_statistical(self):
        """Over many submissions, cheat rate should approximate config."""
        handler, _ = _make_handler(seed=123, adversarial_cheat_rate=0.6)
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        cheats = 0
        n = 200
        for _ in range(n):
            handler.on_epoch_start(state)
            action = Action(
                action_type=ActionType.SUBMIT_KERNEL,
                agent_id="adv_1",
            )
            result = handler.handle_action(action, state)
            if result.submission and result.submission.is_cheat:
                cheats += 1

        cheat_rate = cheats / n
        assert 0.4 < cheat_rate < 0.8, f"Cheat rate {cheat_rate} not near 0.6"

    def test_submit_emits_event(self):
        handler, collector = _make_handler()
        state = _make_state(("h1", AgentType.HONEST))
        handler.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        handler.handle_action(action, state)

        kernel_events = [
            e for e in collector.events if e.event_type.value == "kernel_submitted"
        ]
        assert len(kernel_events) == 1

    def test_submission_has_ood_fields(self):
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))
        handler.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        result = handler.handle_action(action, state)

        sub = result.submission
        assert sub.ood_tests_total > 0
        assert 0 <= sub.ood_tests_passed <= sub.ood_tests_total
        assert 0.0 <= sub.tolerance_margin <= 1.0
        assert sub.base_speedup <= sub.perf_speedup or not sub.is_cheat


class TestCorrelatedSpeedupCheating:
    """Test the core v2 mechanic: cheating boosts apparent speedup."""

    def test_cheat_speedup_higher_than_base(self):
        """When cheating, perf_speedup should exceed base_speedup."""
        handler, _ = _make_handler(
            seed=42, adversarial_cheat_rate=1.0, cheat_speedup_bonus=0.4
        )
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        for _ in range(20):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            result = handler.handle_action(action, state)
            sub = result.submission
            assert sub.is_cheat
            # perf_speedup = base + cheat_bonus + precision_bonus (clamped to [-1, 1])
            precision_bonus = 0.10 * (sub.precision_speedup_factor - 1.0)
            expected = max(-1.0, min(1.0, sub.base_speedup + 0.4 + precision_bonus))
            assert abs(sub.perf_speedup - expected) < 1e-10

    def test_honest_no_cheat_speedup_bonus(self):
        """Honest agents should have no cheat bonus; only precision bonus."""
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))

        for _ in range(20):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
            result = handler.handle_action(action, state)
            sub = result.submission
            assert not sub.is_cheat
            # No cheat bonus, but may have precision speedup bonus
            precision_bonus = 0.10 * (sub.precision_speedup_factor - 1.0)
            expected = max(-1.0, min(1.0, sub.base_speedup + precision_bonus))
            assert abs(sub.perf_speedup - expected) < 1e-10

    def test_cheat_speedup_statistical_advantage(self):
        """Over many runs, adversarial cheaters should have higher average speedup."""
        handler, _ = _make_handler(
            seed=42, adversarial_cheat_rate=1.0, cheat_speedup_bonus=0.4
        )
        state = _make_state(
            ("h1", AgentType.HONEST), ("adv_1", AgentType.ADVERSARIAL)
        )

        honest_speedups = []
        adv_speedups = []

        for _ in range(100):
            handler.on_epoch_start(state)
            h_action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
            a_action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            h_result = handler.handle_action(h_action, state)
            a_result = handler.handle_action(a_action, state)
            honest_speedups.append(h_result.submission.perf_speedup)
            adv_speedups.append(a_result.submission.perf_speedup)

        avg_honest = sum(honest_speedups) / len(honest_speedups)
        avg_adv = sum(adv_speedups) / len(adv_speedups)
        assert avg_adv > avg_honest, (
            f"Adversarial avg speedup {avg_adv:.3f} should exceed honest {avg_honest:.3f}"
        )


class TestOODTestRegime:
    """Test the split functional/OOD test regime."""

    def test_honest_high_ood_quality(self):
        """Honest agents should pass most OOD tests."""
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))

        ood_rates = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
            result = handler.handle_action(action, state)
            sub = result.submission
            if sub.ood_tests_total > 0:
                ood_rates.append(sub.ood_tests_passed / sub.ood_tests_total)

        avg_ood = sum(ood_rates) / len(ood_rates)
        assert avg_ood > 0.6, f"Honest OOD pass rate {avg_ood:.3f} too low"

    def test_adversarial_cheat_low_ood_quality(self):
        """Adversarial cheaters should fail most OOD tests."""
        handler, _ = _make_handler(seed=42, adversarial_cheat_rate=1.0)
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        ood_rates = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            result = handler.handle_action(action, state)
            sub = result.submission
            if sub.ood_tests_total > 0:
                ood_rates.append(sub.ood_tests_passed / sub.ood_tests_total)

        avg_ood = sum(ood_rates) / len(ood_rates)
        assert avg_ood < 0.45, f"Adversarial OOD pass rate {avg_ood:.3f} too high"

    def test_proxy_only_sees_functional(self):
        """Proxy observables should reflect functional tests, not OOD."""
        handler, _ = _make_handler(seed=42, adversarial_cheat_rate=1.0)
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))
        handler.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
        result = handler.handle_action(action, state)

        # task_progress should be based on functional tests (high for adversarial)
        sub = result.submission
        func_rate = sub.tests_passed / max(1, sub.tests_total)
        expected_progress = 2.0 * func_rate - 1.0
        assert abs(result.observables.task_progress_delta - expected_progress) < 1e-10

        # OOD results should NOT affect proxy observables at submit time
        assert result.observables.verifier_rejections == 0


class TestToleranceExploitation:
    """Test tolerance margin mechanics."""

    def test_honest_comfortable_margin(self):
        """Honest agents should have tolerance margin well within bounds."""
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))

        margins = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
            result = handler.handle_action(action, state)
            margins.append(result.submission.tolerance_margin)

        avg_margin = sum(margins) / len(margins)
        assert avg_margin > 0.5, f"Honest avg margin {avg_margin:.3f} too low"

    def test_adversarial_cheat_tight_margin(self):
        """Adversarial cheaters should have tolerance margin near zero."""
        handler, _ = _make_handler(seed=42, adversarial_cheat_rate=1.0)
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        margins = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            result = handler.handle_action(action, state)
            margins.append(result.submission.tolerance_margin)

        avg_margin = sum(margins) / len(margins)
        assert avg_margin < 0.25, f"Adversarial avg margin {avg_margin:.3f} too high"

    def test_tight_margin_produces_precision_hacks(self):
        """Submissions with tight tolerance should produce more precision hacks."""
        handler, _ = _make_handler(seed=42, adversarial_cheat_rate=1.0)
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        hack_counts = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            result = handler.handle_action(action, state)
            hack_counts.append(result.submission.precision_hacks)

        avg_hacks = sum(hack_counts) / len(hack_counts)
        assert avg_hacks > 0.5, f"Adversarial avg hacks {avg_hacks:.3f} too low"


class TestVerifyKernel:
    def test_verify_detects_cheats(self):
        """Verification should detect cheats at verifier_detection_rate."""
        handler, _ = _make_handler(seed=42, verifier_detection_rate=1.0, adversarial_cheat_rate=1.0)
        state = _make_state(("adv_1", AgentType.ADVERSARIAL), ("ver_1", AgentType.HONEST))
        handler.on_epoch_start(state)

        # Submit a cheat
        submit = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
        submit_result = handler.handle_action(submit, state)
        sub_id = submit_result.submission.submission_id

        # Verify it
        verify = Action(
            action_type=ActionType.VERIFY_KERNEL,
            agent_id="ver_1",
            target_id=sub_id,
        )
        verify_result = handler.handle_action(verify, state)

        assert verify_result.success
        assert verify_result.observables.verifier_rejections > 0

    def test_verify_nonexistent_submission(self):
        handler, _ = _make_handler()
        state = _make_state(("ver_1", AgentType.HONEST))
        handler.on_epoch_start(state)

        verify = Action(
            action_type=ActionType.VERIFY_KERNEL,
            agent_id="ver_1",
            target_id="nonexistent",
        )
        result = handler.handle_action(verify, state)
        assert result.success is False

    def test_verify_honest_no_rejections(self):
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST), ("ver_1", AgentType.HONEST))
        handler.on_epoch_start(state)

        submit = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        submit_result = handler.handle_action(submit, state)
        sub_id = submit_result.submission.submission_id

        verify = Action(
            action_type=ActionType.VERIFY_KERNEL,
            agent_id="ver_1",
            target_id=sub_id,
        )
        verify_result = handler.handle_action(verify, state)

        assert verify_result.success
        # Honest agent's submission is never a cheat, so no rejections
        assert verify_result.observables.verifier_rejections == 0


class TestAuditKernel:
    def test_audit_higher_detection(self):
        """Audit should have higher detection than verification."""
        handler, _ = _make_handler(
            seed=42,
            audit_detection_rate=1.0,
            verifier_detection_rate=0.0,
            adversarial_cheat_rate=1.0,
        )
        state = _make_state(("adv_1", AgentType.ADVERSARIAL), ("aud_1", AgentType.HONEST))
        handler.on_epoch_start(state)

        submit = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
        submit_result = handler.handle_action(submit, state)
        sub_id = submit_result.submission.submission_id

        # Verify: should NOT detect (rate=0)
        verify = Action(
            action_type=ActionType.VERIFY_KERNEL,
            agent_id="aud_1",
            target_id=sub_id,
        )
        verify_result = handler.handle_action(verify, state)
        assert verify_result.observables.verifier_rejections == 0

        # Audit: should detect (rate=1)
        audit = Action(
            action_type=ActionType.AUDIT_KERNEL,
            agent_id="aud_1",
            target_id=sub_id,
        )
        audit_result = handler.handle_action(audit, state)
        assert audit_result.observables.verifier_rejections > 0

    def test_audit_reveals_true_speedup(self):
        """Audit should strip cheat bonus and show base_speedup."""
        handler, _ = _make_handler(
            seed=42,
            adversarial_cheat_rate=1.0,
            cheat_speedup_bonus=0.4,
            audit_detection_rate=1.0,
        )
        state = _make_state(("adv_1", AgentType.ADVERSARIAL), ("aud_1", AgentType.HONEST))
        handler.on_epoch_start(state)

        submit = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
        submit_result = handler.handle_action(submit, state)
        sub = submit_result.submission

        audit = Action(
            action_type=ActionType.AUDIT_KERNEL,
            agent_id="aud_1",
            target_id=sub.submission_id,
        )
        audit_result = handler.handle_action(audit, state)

        # Audit shows base_speedup, not inflated perf_speedup
        assert abs(audit_result.observables.counterparty_engagement_delta - sub.base_speedup) < 1e-10
        # Submit showed inflated speedup
        assert abs(submit_result.observables.counterparty_engagement_delta - sub.perf_speedup) < 1e-10

    def test_audit_uses_ood_results(self):
        """Audit event should include OOD test data."""
        handler, collector = _make_handler(
            seed=42, adversarial_cheat_rate=1.0, audit_detection_rate=1.0
        )
        state = _make_state(("adv_1", AgentType.ADVERSARIAL), ("aud_1", AgentType.HONEST))
        handler.on_epoch_start(state)

        submit = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
        submit_result = handler.handle_action(submit, state)
        sub_id = submit_result.submission.submission_id

        audit = Action(
            action_type=ActionType.AUDIT_KERNEL,
            agent_id="aud_1",
            target_id=sub_id,
        )
        handler.handle_action(audit, state)

        audit_events = [
            e for e in collector.events if e.event_type.value == "kernel_audited"
        ]
        assert len(audit_events) == 1
        payload = audit_events[0].payload
        assert "ood_tests_passed" in payload
        assert "ood_tests_total" in payload
        assert "tolerance_margin" in payload
        assert "base_speedup" in payload

    def test_audit_emits_event(self):
        handler, collector = _make_handler(seed=42, adversarial_cheat_rate=1.0)
        state = _make_state(("adv_1", AgentType.ADVERSARIAL), ("aud_1", AgentType.HONEST))
        handler.on_epoch_start(state)

        submit = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
        submit_result = handler.handle_action(submit, state)
        sub_id = submit_result.submission.submission_id

        audit = Action(
            action_type=ActionType.AUDIT_KERNEL,
            agent_id="aud_1",
            target_id=sub_id,
        )
        handler.handle_action(audit, state)

        audit_events = [
            e for e in collector.events if e.event_type.value == "kernel_audited"
        ]
        assert len(audit_events) == 1


class TestProxySignalRanges:
    def test_task_progress_in_range(self):
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))

        for _ in range(20):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
            result = handler.handle_action(action, state)
            assert result.observables is not None
            assert -1.0 <= result.observables.task_progress_delta <= 1.0

    def test_rework_count_non_negative(self):
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))

        for _ in range(20):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
            result = handler.handle_action(action, state)
            assert result.observables.rework_count >= 0

    def test_engagement_delta_in_range(self):
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))

        for _ in range(20):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
            result = handler.handle_action(action, state)
            assert -1.0 <= result.observables.counterparty_engagement_delta <= 1.0


class TestGroundTruth:
    def test_cheat_maps_to_negative_ground_truth(self):
        """Cheats should produce ground_truth=-1 when wired through orchestrator."""
        handler, _ = _make_handler(seed=42, adversarial_cheat_rate=1.0)
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))
        handler.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
        result = handler.handle_action(action, state)
        assert result.submission.is_cheat is True
        # ground_truth is set by orchestrator, but we verify the flag

    def test_honest_maps_to_positive(self):
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))
        handler.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        result = handler.handle_action(action, state)
        assert result.submission.is_cheat is False


class TestDifficultyPenalty:
    def test_harder_challenges_lower_pass_rate(self):
        """Statistically, hard challenges should yield lower pass rates."""
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))

        easy_rates = []
        hard_rates = []

        for _ in range(100):
            handler.on_epoch_start(state)
            # Find an easy and a hard challenge
            easy_c = [c for c in handler._epoch_challenges if c.difficulty == "easy"]
            hard_c = [c for c in handler._epoch_challenges if c.difficulty == "hard"]

            if easy_c:
                action = Action(
                    action_type=ActionType.SUBMIT_KERNEL,
                    agent_id="h1",
                    target_id=easy_c[0].challenge_id,
                )
                result = handler.handle_action(action, state)
                if result.submission:
                    easy_rates.append(
                        result.submission.tests_passed / max(1, result.submission.tests_total)
                    )

            if hard_c:
                action = Action(
                    action_type=ActionType.SUBMIT_KERNEL,
                    agent_id="h1",
                    target_id=hard_c[0].challenge_id,
                )
                result = handler.handle_action(action, state)
                if result.submission:
                    hard_rates.append(
                        result.submission.tests_passed / max(1, result.submission.tests_total)
                    )

        if easy_rates and hard_rates:
            avg_easy = sum(easy_rates) / len(easy_rates)
            avg_hard = sum(hard_rates) / len(hard_rates)
            assert avg_easy > avg_hard, (
                f"Easy avg {avg_easy:.3f} should be > hard avg {avg_hard:.3f}"
            )


class TestReproducibility:
    def test_same_seed_same_results(self):
        """Same seed should produce identical submissions."""
        results_a = []
        results_b = []

        for seed_run in (results_a, results_b):
            handler, _ = _make_handler(seed=99)
            state = _make_state(("h1", AgentType.HONEST))
            handler.on_epoch_start(state)

            for _ in range(5):
                action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
                result = handler.handle_action(action, state)
                seed_run.append(
                    (
                        result.submission.tests_passed,
                        result.submission.tests_total,
                        result.submission.ood_tests_passed,
                        result.submission.ood_tests_total,
                        result.submission.is_cheat,
                        round(result.submission.perf_speedup, 10),
                        round(result.submission.base_speedup, 10),
                        # v3 fields
                        round(result.submission.memory_tier_score, 10),
                        round(result.submission.precision_speedup_factor, 10),
                    )
                )

        assert results_a == results_b


class TestScenarioIntegration:
    def test_v1_scenario_loads_and_runs(self):
        """Original baseline scenario should still work with legacy fields."""
        from pathlib import Path

        from swarm.scenarios.loader import build_orchestrator, load_scenario

        sc = load_scenario(Path("scenarios/kernel_market/baseline.yaml"))
        assert sc.orchestrator_config.kernel_oracle_config is not None

        orch = build_orchestrator(sc)
        orch.config.n_epochs = 1
        results = orch.run()
        assert len(results) == 1

    def test_v2_scenario_loads_and_runs(self):
        """v2 scenario with new kernel oracle params should load and run."""
        from pathlib import Path

        from swarm.scenarios.loader import build_orchestrator, load_scenario

        sc = load_scenario(Path("scenarios/kernel_market/v2.yaml"))
        assert sc.orchestrator_config.kernel_oracle_config is not None

        orch = build_orchestrator(sc)
        orch.config.n_epochs = 1
        results = orch.run()
        assert len(results) == 1


class TestObservationFields:
    def test_observation_fields_populated(self):
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))
        handler.on_epoch_start(state)

        # Submit a kernel first
        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        handler.handle_action(action, state)

        # Build observation
        obs = handler.build_observation_fields("h1", state)
        assert "kernel_available_challenges" in obs
        assert "kernel_pending_submissions" in obs
        assert "kernel_submissions_to_verify" in obs
        assert "kernel_submission_history" in obs
        assert len(obs["kernel_available_challenges"]) > 0
        assert len(obs["kernel_pending_submissions"]) == 1
        assert len(obs["kernel_submission_history"]) == 1


class TestMemoryHierarchy:
    """Test GPU memory hierarchy efficiency signals (v3)."""

    def test_honest_moderate_shared_mem_utilization(self):
        """Honest agents should have moderate shared memory utilization."""
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))

        utils = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
            result = handler.handle_action(action, state)
            utils.append(result.submission.shared_mem_utilization)

        avg = sum(utils) / len(utils)
        assert 0.45 < avg < 0.85, f"Honest avg shared_mem_util {avg:.3f} out of range"

    def test_adversarial_high_shared_mem_utilization(self):
        """Adversarial agents should have high shared memory utilization (hardcoded tiling)."""
        handler, _ = _make_handler(seed=42, adversarial_cheat_rate=1.0)
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        utils = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            result = handler.handle_action(action, state)
            utils.append(result.submission.shared_mem_utilization)

        avg = sum(utils) / len(utils)
        assert avg > 0.70, f"Adversarial avg shared_mem_util {avg:.3f} too low"

    def test_adversarial_low_register_spills(self):
        """Adversarial agents should have very few register spills (hardcoded tiling)."""
        handler, _ = _make_handler(seed=42, adversarial_cheat_rate=1.0)
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        spills = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            result = handler.handle_action(action, state)
            spills.append(result.submission.register_spill_count)

        avg = sum(spills) / len(spills)
        assert avg < 0.5, f"Adversarial avg register_spill_count {avg:.3f} too high"

    def test_honest_more_register_spills_than_adversarial(self):
        """Honest agents should have more register spills than adversarial."""
        handler, _ = _make_handler(seed=42, adversarial_cheat_rate=1.0)
        state = _make_state(
            ("h1", AgentType.HONEST), ("adv_1", AgentType.ADVERSARIAL)
        )

        honest_spills = []
        adv_spills = []
        for _ in range(100):
            handler.on_epoch_start(state)
            h_action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
            a_action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            h_result = handler.handle_action(h_action, state)
            a_result = handler.handle_action(a_action, state)
            honest_spills.append(h_result.submission.register_spill_count)
            adv_spills.append(a_result.submission.register_spill_count)

        avg_honest = sum(honest_spills) / len(honest_spills)
        avg_adv = sum(adv_spills) / len(adv_spills)
        assert avg_honest > avg_adv, (
            f"Honest avg spills {avg_honest:.3f} should exceed adversarial {avg_adv:.3f}"
        )

    def test_adversarial_memory_audit_penalty(self):
        """Adversarial cheaters should have lower memory_tier_score_audit than benchmark."""
        handler, _ = _make_handler(seed=42, adversarial_cheat_rate=1.0)
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        drops = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            result = handler.handle_action(action, state)
            sub = result.submission
            drops.append(sub.memory_tier_score - sub.memory_tier_score_audit)

        avg_drop = sum(drops) / len(drops)
        assert avg_drop > 0.3, (
            f"Adversarial avg memory audit drop {avg_drop:.3f} too small"
        )

    def test_honest_memory_audit_stable(self):
        """Honest agents should have similar benchmark and audit memory scores."""
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))

        drops = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
            result = handler.handle_action(action, state)
            sub = result.submission
            drops.append(abs(sub.memory_tier_score - sub.memory_tier_score_audit))

        avg_drop = sum(drops) / len(drops)
        assert avg_drop < 0.15, (
            f"Honest avg memory audit drift {avg_drop:.3f} too large"
        )

    def test_register_spills_feed_rework_count(self):
        """Rework count should include register spills."""
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))
        handler.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        result = handler.handle_action(action, state)
        sub = result.submission
        expected_rework = sub.compilation_failures + sub.register_spill_count
        assert result.observables.rework_count == expected_rework


class TestPrecisionExploitation:
    """Test GPU precision format exploitation signals (v3)."""

    def test_adversarial_uses_unsafe_precision(self):
        """Adversarial cheaters should frequently use unsafe FP16."""
        handler, _ = _make_handler(
            seed=42, adversarial_cheat_rate=1.0,
            adversarial_uses_unsafe_precision=0.70,
        )
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        unsafe_count = 0
        n = 200
        for _ in range(n):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            result = handler.handle_action(action, state)
            if result.submission.precision_format_used == "fp16_unsafe":
                unsafe_count += 1

        # Some challenges don't allow tensor cores, so rate is lower than 70%
        rate = unsafe_count / n
        assert rate > 0.25, f"Unsafe precision rate {rate:.3f} too low"

    def test_unsafe_precision_high_speedup(self):
        """Unsafe FP16 submissions should have high precision_speedup_factor."""
        handler, _ = _make_handler(
            seed=42, adversarial_cheat_rate=1.0,
            adversarial_uses_unsafe_precision=1.0,
            unsafe_precision_speedup=3.5,
        )
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        speedups = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            result = handler.handle_action(action, state)
            sub = result.submission
            if sub.precision_format_used == "fp16_unsafe":
                speedups.append(sub.precision_speedup_factor)

        if speedups:
            avg = sum(speedups) / len(speedups)
            assert avg > 2.5, f"Avg unsafe precision speedup {avg:.3f} too low"

    def test_adversarial_audit_error_explosion(self):
        """Adversarial unsafe precision should show massive error on audit."""
        handler, _ = _make_handler(
            seed=42, adversarial_cheat_rate=1.0,
            adversarial_uses_unsafe_precision=1.0,
            adversarial_audit_error_multiplier=1000.0,
        )
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        ratios = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            result = handler.handle_action(action, state)
            sub = result.submission
            if (sub.precision_format_used == "fp16_unsafe"
                    and sub.numerical_error_benchmark > 0):
                ratios.append(sub.numerical_error_audit / sub.numerical_error_benchmark)

        if ratios:
            avg = sum(ratios) / len(ratios)
            assert avg > 500.0, f"Avg error ratio {avg:.1f} too low (expected ~1000x)"

    def test_honest_stable_numerical_error(self):
        """Honest agents should have similar benchmark and audit errors."""
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))

        ratios = []
        for _ in range(100):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
            result = handler.handle_action(action, state)
            sub = result.submission
            if sub.numerical_error_benchmark > 0:
                ratios.append(sub.numerical_error_audit / sub.numerical_error_benchmark)

        if ratios:
            avg = sum(ratios) / len(ratios)
            assert avg < 2.0, f"Honest avg error ratio {avg:.3f} too high"

    def test_precision_flag_on_sensitive_challenge(self):
        """Unsafe precision on numerically sensitive challenges should produce flags."""
        handler, _ = _make_handler(
            seed=42, adversarial_cheat_rate=1.0,
            adversarial_uses_unsafe_precision=1.0,
        )
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        flagged = 0
        sensitive_and_unsafe = 0
        for _ in range(200):
            handler.on_epoch_start(state)
            # Target a sensitive challenge (attention)
            action = Action(
                action_type=ActionType.SUBMIT_KERNEL,
                agent_id="adv_1",
                target_id="6_attention",
            )
            result = handler.handle_action(action, state)
            sub = result.submission
            if sub.precision_format_used == "fp16_unsafe":
                sensitive_and_unsafe += 1
                if sub.precision_hacks > 0:
                    flagged += 1

        if sensitive_and_unsafe > 0:
            flag_rate = flagged / sensitive_and_unsafe
            assert flag_rate > 0.5, (
                f"Precision flag rate on sensitive challenges {flag_rate:.3f} too low"
            )

    def test_precision_speedup_feeds_engagement(self):
        """Precision speedup should contribute to counterparty_engagement_delta."""
        handler, _ = _make_handler(
            seed=42, adversarial_cheat_rate=1.0,
            adversarial_uses_unsafe_precision=1.0,
        )
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        for _ in range(50):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            result = handler.handle_action(action, state)
            sub = result.submission
            if sub.precision_format_used == "fp16_unsafe":
                # Engagement should include precision bonus
                precision_bonus = 0.10 * (sub.precision_speedup_factor - 1.0)
                expected = max(-1.0, min(1.0,
                    sub.base_speedup + 0.4 + precision_bonus  # cheat bonus + precision
                ))
                assert abs(result.observables.counterparty_engagement_delta - expected) < 1e-10
                return  # Found one, test passes

        # If we never got fp16_unsafe in 50 tries, that's also a problem
        pytest.fail("Never generated fp16_unsafe submission in 50 iterations")

    def test_audit_detects_precision_error_explosion(self):
        """Audit should generate extra rejections from precision error explosion."""
        handler, _ = _make_handler(
            seed=42,
            adversarial_cheat_rate=1.0,
            adversarial_uses_unsafe_precision=1.0,
            audit_detection_rate=1.0,
            adversarial_audit_error_multiplier=1000.0,
        )
        state = _make_state(("adv_1", AgentType.ADVERSARIAL), ("aud_1", AgentType.HONEST))

        extra_rejections_from_precision = 0
        audits_with_unsafe = 0

        for _ in range(100):
            handler.on_epoch_start(state)
            submit = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            submit_result = handler.handle_action(submit, state)
            sub = submit_result.submission

            if sub.precision_format_used != "fp16_unsafe":
                continue

            audits_with_unsafe += 1
            audit = Action(
                action_type=ActionType.AUDIT_KERNEL,
                agent_id="aud_1",
                target_id=sub.submission_id,
            )
            audit_result = handler.handle_action(audit, state)
            if audit_result.observables.verifier_rejections > 0:
                extra_rejections_from_precision += 1

        if audits_with_unsafe > 0:
            detection_rate = extra_rejections_from_precision / audits_with_unsafe
            assert detection_rate > 0.5, (
                f"Audit precision detection rate {detection_rate:.3f} too low"
            )


class TestV3ScenarioIntegration:
    def test_v3_scenario_loads_and_runs(self):
        """v3 scenario with GPU-specific params should load and run."""
        from pathlib import Path

        from swarm.scenarios.loader import build_orchestrator, load_scenario

        sc = load_scenario(Path("scenarios/kernel_market/v3.yaml"))
        assert sc.orchestrator_config.kernel_oracle_config is not None

        orch = build_orchestrator(sc)
        orch.config.n_epochs = 1
        results = orch.run()
        assert len(results) == 1


class TestV3CatalogGPUFields:
    def test_all_challenges_have_gpu_fields(self):
        """All challenges in catalog should have GPU hardware specs."""
        for c in CHALLENGE_CATALOG:
            assert c.shared_mem_budget_kb > 0, f"{c.challenge_id} missing shared_mem_budget_kb"
            assert c.register_pressure_class in ("low", "medium", "high"), (
                f"{c.challenge_id} bad register_pressure_class"
            )
            assert c.precision_required in ("fp32", "fp16", "bf16", "mixed"), (
                f"{c.challenge_id} bad precision_required"
            )
            assert c.numerical_sensitivity in ("low", "medium", "high"), (
                f"{c.challenge_id} bad numerical_sensitivity"
            )

    def test_hard_challenges_have_high_shared_mem(self):
        """Hard challenges should have larger shared memory budgets."""
        for c in CHALLENGE_CATALOG:
            if c.difficulty == "hard":
                assert c.shared_mem_budget_kb >= 96, (
                    f"{c.challenge_id}: hard challenge should have >= 96 KB shared mem"
                )

    def test_sensitive_challenges_exist(self):
        """At least one challenge should have high numerical sensitivity."""
        sensitivities = [c.numerical_sensitivity for c in CHALLENGE_CATALOG]
        assert "high" in sensitivities


class TestCudaCodeGeneration:
    """Test CUDA code generation integration (v4)."""

    def test_submission_has_code_when_enabled(self):
        """Submissions should include CUDA code when code gen is enabled."""
        handler, _ = _make_handler(seed=42, enable_code_generation=True)
        state = _make_state(("h1", AgentType.HONEST))
        handler.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        result = handler.handle_action(action, state)

        assert result.submission.cuda_code != ""
        assert result.submission.code_source == "template"
        assert len(result.submission.cuda_code_features) > 0

    def test_no_code_when_disabled(self):
        """Submissions should have no code when code gen is disabled (default)."""
        handler, _ = _make_handler(seed=42)
        state = _make_state(("h1", AgentType.HONEST))
        handler.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        result = handler.handle_action(action, state)

        assert result.submission.cuda_code == ""
        assert result.submission.code_source == "none"

    def test_honest_template_has_bounds_check(self):
        """Honest agent templates should include bounds checks."""
        handler, _ = _make_handler(seed=42, enable_code_generation=True)
        state = _make_state(("h1", AgentType.HONEST))
        handler.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        result = handler.handle_action(action, state)

        features = result.submission.cuda_code_features
        assert features.get("has_bounds_check") is True

    def test_adversarial_template_has_hardcoded_shapes(self):
        """Adversarial agent templates should have hardcoded shapes."""
        handler, _ = _make_handler(
            seed=42, enable_code_generation=True, adversarial_cheat_rate=1.0
        )
        state = _make_state(("adv_1", AgentType.ADVERSARIAL))

        found_hardcoded = False
        for _ in range(20):
            handler.on_epoch_start(state)
            action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="adv_1")
            result = handler.handle_action(action, state)
            features = result.submission.cuda_code_features
            if features.get("has_hardcoded_shapes"):
                found_hardcoded = True
                break

        assert found_hardcoded, "Expected at least one adversarial template with hardcoded shapes"

    def test_code_features_populated(self):
        """Code features dict should have expected keys."""
        handler, _ = _make_handler(seed=42, enable_code_generation=True)
        state = _make_state(("h1", AgentType.HONEST))
        handler.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        result = handler.handle_action(action, state)

        features = result.submission.cuda_code_features
        assert "has_bounds_check" in features
        assert "uses_shared_memory" in features
        assert "has_hardcoded_shapes" in features
        assert "code_length_lines" in features

    def test_features_feed_proxy(self):
        """Code features should affect proxy observables."""
        # Compare with and without code gen for same seed
        handler_off, _ = _make_handler(seed=42, enable_code_generation=False)
        handler_on, _ = _make_handler(seed=42, enable_code_generation=True)
        state = _make_state(("h1", AgentType.HONEST))

        handler_off.on_epoch_start(state)
        handler_on.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        result_off = handler_off.handle_action(action, state)
        result_on = handler_on.handle_action(action, state)

        # With code gen, observables may differ due to code feature adjustments
        # (honest templates with shared memory get +0.05 engagement boost)
        sub = result_on.submission
        if sub.cuda_code_features.get("uses_shared_memory"):
            # Engagement should be >= without code gen (shared mem boost)
            assert (
                result_on.observables.counterparty_engagement_delta
                >= result_off.observables.counterparty_engagement_delta - 0.01
            )

    def test_backward_compat_default_fields(self):
        """KernelSubmission should work without code fields (backward compat)."""
        from swarm.models.kernel import KernelSubmission

        sub = KernelSubmission(challenge_id="1_vector_add", author_id="test")
        assert sub.cuda_code == ""
        assert sub.code_source == "none"
        assert sub.cuda_code_features == {}

    def test_llm_code_via_action_content(self):
        """Code provided in action.content should be used as LLM code."""
        handler, _ = _make_handler(seed=42, enable_code_generation=True)
        state = _make_state(("h1", AgentType.HONEST))
        handler.on_epoch_start(state)

        llm_code = """
        __global__ void vector_add(const float* A, const float* B, float* C, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                C[idx] = A[idx] + B[idx];
            }
        }
        """
        action = Action(
            action_type=ActionType.SUBMIT_KERNEL,
            agent_id="h1",
            content=llm_code,
        )
        result = handler.handle_action(action, state)

        assert result.submission.code_source == "llm"
        assert "vector_add" in result.submission.cuda_code
        assert result.submission.cuda_code_features.get("has_bounds_check") is True

    def test_event_includes_code_source(self):
        """Events should include code_source field."""
        handler, collector = _make_handler(
            seed=42, enable_code_generation=True, code_in_events=True,
        )
        state = _make_state(("h1", AgentType.HONEST))
        handler.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        handler.handle_action(action, state)

        kernel_events = [
            e for e in collector.events if e.event_type.value == "kernel_submitted"
        ]
        assert len(kernel_events) == 1
        payload = kernel_events[0].payload
        assert payload["code_source"] == "template"
        assert "cuda_code" in payload
        assert len(payload["cuda_code"]) > 0

    def test_existing_tests_still_pass_with_code_gen_disabled(self):
        """Enabling code gen should not break existing submission logic."""
        handler, _ = _make_handler(seed=42, enable_code_generation=False)
        state = _make_state(("h1", AgentType.HONEST))
        handler.on_epoch_start(state)

        action = Action(action_type=ActionType.SUBMIT_KERNEL, agent_id="h1")
        result = handler.handle_action(action, state)

        assert result.success
        assert result.submission is not None
        assert result.observables is not None
        pass_rate = result.submission.tests_passed / max(1, result.submission.tests_total)
        assert pass_rate > 0.3
