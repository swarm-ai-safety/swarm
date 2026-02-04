"""Tests for red-teaming framework."""

import pytest

from src.agents.adaptive_adversary import (
    AdaptiveAdversary,
    AttackStrategy,
    StrategyPerformance,
    AdversaryMemory,
)
from src.agents.base import Action, ActionType, Observation
from src.redteam.attacks import (
    AttackScenario,
    AttackResult,
    AttackLibrary,
    AttackCategory,
    AttackDifficulty,
)
from src.redteam.metrics import (
    EvasionMetrics,
    GovernanceEffectiveness,
    AdversaryPerformance,
    compute_evasion_rate,
    compute_detection_latency,
    compute_damage_before_detection,
    compute_governance_effectiveness,
)
from src.redteam.evaluator import (
    RedTeamEvaluator,
    GovernanceRobustness,
    Vulnerability,
    VulnerabilityReport,
)


# =============================================================================
# Adaptive Adversary Tests
# =============================================================================


class TestStrategyPerformance:
    """Tests for StrategyPerformance tracking."""

    def test_initial_performance(self):
        """Test initial performance values."""
        perf = StrategyPerformance(strategy=AttackStrategy.COLLUSION)
        assert perf.attempts == 0
        assert perf.success_rate == 0.5  # Prior
        assert perf.detection_rate == 0.5  # Prior

    def test_record_success(self):
        """Test recording successful attempt."""
        perf = StrategyPerformance(strategy=AttackStrategy.COLLUSION)
        perf.attempts = 10
        perf.successes = 8
        perf.detected = 2
        perf.total_payoff = 50.0

        assert perf.success_rate == 0.8
        assert perf.detection_rate == 0.2
        assert perf.evasion_rate == 0.8
        assert perf.avg_payoff == 5.0

    def test_effectiveness_score(self):
        """Test effectiveness score computation."""
        perf = StrategyPerformance(strategy=AttackStrategy.MIMICRY)
        # Not enough data
        assert perf.effectiveness_score == 0.5

        perf.attempts = 10
        perf.successes = 9
        perf.detected = 1
        perf.total_payoff = 30.0

        # High success (0.9), high evasion (0.9), moderate payoff
        score = perf.effectiveness_score
        assert 0.6 < score < 1.0


class TestAdversaryMemory:
    """Tests for AdversaryMemory."""

    def test_initial_memory(self):
        """Test initial memory state."""
        memory = AdversaryMemory()
        assert memory.current_heat_level == 0.0
        assert memory.epochs_since_detection == 0

    def test_heat_update_on_detection(self):
        """Test heat increases on detection."""
        memory = AdversaryMemory()
        memory.update_heat(detected=True, penalty=0.0)
        assert memory.current_heat_level > 0
        assert memory.epochs_since_detection == 0

    def test_heat_decay(self):
        """Test heat decays over time."""
        memory = AdversaryMemory()
        memory.current_heat_level = 0.5
        memory.epochs_since_detection = 5

        memory.update_heat(detected=False, penalty=0.0)
        assert memory.current_heat_level < 0.5


class TestAdaptiveAdversary:
    """Tests for AdaptiveAdversary agent."""

    def test_creation(self):
        """Test adaptive adversary creation."""
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        assert agent.agent_id == "adaptive_1"
        assert agent.learning_rate == 0.1
        assert len(agent.strategy_performance) == len(AttackStrategy)

    def test_initial_strategy(self):
        """Test initial strategy is reputation farming."""
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        assert agent.current_strategy == AttackStrategy.REPUTATION_FARMING
        assert agent.in_reputation_building_phase

    def test_select_strategy_exploration(self):
        """Test strategy exploration."""
        agent = AdaptiveAdversary(
            agent_id="adaptive_1",
            exploration_rate=1.0,  # Always explore
        )
        strategies = set()
        for _ in range(100):
            strategy = agent.select_strategy()
            strategies.add(strategy)
        # Should explore multiple strategies
        assert len(strategies) > 1

    def test_select_strategy_exploitation(self):
        """Test strategy exploitation."""
        agent = AdaptiveAdversary(
            agent_id="adaptive_1",
            exploration_rate=0.0,  # Never explore
        )

        # Make one strategy much better
        agent.strategy_performance[AttackStrategy.MIMICRY].attempts = 20
        agent.strategy_performance[AttackStrategy.MIMICRY].successes = 18
        agent.strategy_performance[AttackStrategy.MIMICRY].total_payoff = 100

        strategy = agent.select_strategy()
        assert strategy == AttackStrategy.MIMICRY

    def test_should_lay_low_high_heat(self):
        """Test laying low when heat is high."""
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        agent.memory.current_heat_level = 0.9

        obs = Observation(
            current_epoch=1,
            current_step=1,
            visible_agents=[],
            visible_posts=[],
        )

        assert agent._should_lay_low(obs)

    def test_act_returns_action(self):
        """Test act returns valid action."""
        agent = AdaptiveAdversary(agent_id="adaptive_1")

        obs = Observation(
            current_epoch=1,
            current_step=1,
            visible_agents=[
                {"agent_id": "honest_1", "agent_type": "honest", "reputation": 5.0},
            ],
            visible_posts=[],
        )

        action = agent.act(obs)
        assert isinstance(action, Action)
        assert action.action_type in ActionType

    def test_update_adversary_outcome(self):
        """Test learning from outcomes."""
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        initial_heat = agent.memory.current_heat_level

        agent.update_adversary_outcome(
            payoff=10.0,
            penalty=5.0,
            detected=True,
            strategy_used=AttackStrategy.COLLUSION,
        )

        assert agent.memory.current_heat_level > initial_heat
        perf = agent.strategy_performance[AttackStrategy.COLLUSION]
        assert perf.attempts == 1
        assert perf.detected == 1

    def test_get_strategy_report(self):
        """Test strategy report generation."""
        agent = AdaptiveAdversary(agent_id="adaptive_1")
        agent.update_adversary_outcome(5.0, 0.0, False, AttackStrategy.MIMICRY)

        report = agent.get_strategy_report()
        assert "current_strategy" in report
        assert "heat_level" in report
        assert "strategy_stats" in report
        assert AttackStrategy.MIMICRY.value in report["strategy_stats"]


# =============================================================================
# Attack Library Tests
# =============================================================================


class TestAttackScenario:
    """Tests for AttackScenario."""

    def test_create_scenario(self):
        """Test creating attack scenario."""
        scenario = AttackScenario(
            attack_id="test_attack",
            name="Test Attack",
            description="A test attack",
            category=AttackCategory.EVASION,
            difficulty=AttackDifficulty.MODERATE,
        )

        assert scenario.attack_id == "test_attack"
        assert scenario.category == AttackCategory.EVASION
        assert scenario.n_adversaries == 1

    def test_to_dict(self):
        """Test serialization."""
        scenario = AttackScenario(
            attack_id="test",
            name="Test",
            description="Test",
            category=AttackCategory.COORDINATION,
            difficulty=AttackDifficulty.ADVANCED,
        )

        data = scenario.to_dict()
        assert data["attack_id"] == "test"
        assert data["category"] == "coordination"


class TestAttackLibrary:
    """Tests for AttackLibrary."""

    def test_get_all_attacks(self):
        """Test getting all attacks."""
        attacks = AttackLibrary.get_all_attacks()
        assert len(attacks) >= 8

    def test_reputation_farming_attack(self):
        """Test reputation farming attack definition."""
        attack = AttackLibrary.reputation_farming_exploit()
        assert attack.attack_id == "reputation_farming"
        assert attack.category == AttackCategory.EXPLOITATION

    def test_collusion_ring_attack(self):
        """Test collusion ring attack definition."""
        attack = AttackLibrary.collusion_ring()
        assert attack.n_adversaries == 3
        assert "collusion_detection" in attack.targeted_levers

    def test_get_attacks_by_category(self):
        """Test filtering by category."""
        evasion = AttackLibrary.get_attacks_by_category(AttackCategory.EVASION)
        assert all(a.category == AttackCategory.EVASION for a in evasion)

    def test_get_attacks_by_difficulty(self):
        """Test filtering by difficulty."""
        advanced = AttackLibrary.get_attacks_by_difficulty(AttackDifficulty.ADVANCED)
        assert all(a.difficulty == AttackDifficulty.ADVANCED for a in advanced)

    def test_get_attacks_targeting_lever(self):
        """Test filtering by targeted lever."""
        circuit_attacks = AttackLibrary.get_attacks_targeting_lever("circuit_breaker")
        assert len(circuit_attacks) > 0


# =============================================================================
# Evasion Metrics Tests
# =============================================================================


class TestEvasionMetrics:
    """Tests for EvasionMetrics."""

    def test_initial_metrics(self):
        """Test initial metric values."""
        metrics = EvasionMetrics()
        assert metrics.evasion_rate == 0.0
        # detection_rate = 1 - evasion_rate when no actions
        assert metrics.total_malicious_actions == 0

    def test_record_action_detected(self):
        """Test recording detected action."""
        metrics = EvasionMetrics()
        metrics.record_action(detected=True, damage=10.0, epoch=5)

        assert metrics.total_malicious_actions == 1
        assert metrics.detected_actions == 1
        assert metrics.first_detection_epoch == 5

    def test_record_action_undetected(self):
        """Test recording undetected action."""
        metrics = EvasionMetrics()
        metrics.record_action(detected=False, damage=10.0, epoch=5)

        assert metrics.undetected_actions == 1
        assert metrics.damage_before_first_detection == 10.0

    def test_evasion_rate(self):
        """Test evasion rate computation."""
        metrics = EvasionMetrics()
        for i in range(10):
            metrics.record_action(detected=(i < 3), damage=5.0, epoch=i)

        assert metrics.evasion_rate == pytest.approx(0.7)
        assert metrics.detection_rate == pytest.approx(0.3)

    def test_strategy_evasion_rates(self):
        """Test per-strategy evasion rates."""
        metrics = EvasionMetrics()
        metrics.record_action(True, 5.0, 1, "collusion")
        metrics.record_action(False, 5.0, 2, "collusion")
        metrics.record_action(False, 5.0, 3, "mimicry")

        rates = metrics.get_strategy_evasion_rates()
        assert rates["collusion"] == 0.5
        assert rates["mimicry"] == 1.0


class TestGovernanceEffectiveness:
    """Tests for GovernanceEffectiveness metrics."""

    def test_precision_recall(self):
        """Test precision and recall computation."""
        eff = GovernanceEffectiveness(
            true_positives=80,
            false_positives=10,
            true_negatives=90,
            false_negatives=20,
        )

        assert eff.precision == 80 / 90  # TP / (TP + FP)
        assert eff.recall == 80 / 100  # TP / (TP + FN)

    def test_f1_score(self):
        """Test F1 score computation."""
        eff = GovernanceEffectiveness(
            true_positives=80,
            false_positives=20,
            false_negatives=20,
        )

        # precision = 0.8, recall = 0.8
        assert abs(eff.f1_score - 0.8) < 0.01

    def test_attack_prevention_rate(self):
        """Test attack prevention rate."""
        eff = GovernanceEffectiveness(
            attacks_prevented=7,
            attacks_succeeded=3,
        )

        assert eff.attack_prevention_rate == 0.7


class TestMetricFunctions:
    """Tests for metric computation functions."""

    def test_compute_evasion_rate(self):
        """Test evasion rate computation."""
        actions = [
            {"action_id": "a1"},
            {"action_id": "a2"},
            {"action_id": "a3"},
        ]
        detections = [{"action_id": "a1"}]

        rate = compute_evasion_rate(actions, detections)
        assert rate == 2 / 3

    def test_compute_detection_latency(self):
        """Test detection latency computation."""
        actions = [
            {"action_id": "a1", "epoch": 5},
            {"action_id": "a2", "epoch": 8},
        ]
        detections = [
            {"action_id": "a1", "epoch": 7},
            {"action_id": "a2", "epoch": 10},
        ]

        mean, min_lat, max_lat = compute_detection_latency(actions, detections)
        assert mean == 2.0
        assert min_lat == 2
        assert max_lat == 2

    def test_compute_damage_before_detection(self):
        """Test damage before detection computation."""
        actions = [
            {"action_id": "a1", "epoch": 2, "damage": 10.0},
            {"action_id": "a2", "epoch": 5, "damage": 15.0},
            {"action_id": "a3", "epoch": 8, "damage": 20.0},
        ]

        damage = compute_damage_before_detection(actions, first_detection_epoch=6)
        assert damage == 25.0  # 10 + 15


# =============================================================================
# Red-Team Evaluator Tests
# =============================================================================


class TestGovernanceRobustness:
    """Tests for GovernanceRobustness assessment."""

    def test_robustness_score(self):
        """Test robustness score computation."""
        robustness = GovernanceRobustness(
            attacks_tested=10,
            attacks_prevented=7,
            overall_evasion_rate=0.3,
            total_damage_allowed=200,
        )

        score = robustness.robustness_score
        assert 0 <= score <= 1

    def test_grade(self):
        """Test letter grade assignment."""
        robustness = GovernanceRobustness()
        robustness.attacks_tested = 10
        robustness.attacks_prevented = 9
        robustness.overall_evasion_rate = 0.1
        robustness.total_damage_allowed = 50

        grade = robustness.grade
        assert grade in ["A", "B", "C", "D", "F"]


class TestVulnerability:
    """Tests for Vulnerability tracking."""

    def test_create_vulnerability(self):
        """Test creating vulnerability."""
        vuln = Vulnerability(
            vulnerability_id="vuln_1",
            severity="high",
            description="Test vulnerability",
            attack_vector="evasion",
            affected_lever="circuit_breaker",
            exploitation_difficulty="moderate",
            potential_damage=50.0,
            mitigation="Lower thresholds",
        )

        assert vuln.severity == "high"
        data = vuln.to_dict()
        assert data["affected_lever"] == "circuit_breaker"


class TestRedTeamEvaluator:
    """Tests for RedTeamEvaluator."""

    def test_creation(self):
        """Test evaluator creation."""
        config = {"circuit_breaker_enabled": True}
        evaluator = RedTeamEvaluator(governance_config=config)

        assert evaluator.governance_config == config
        assert len(evaluator.attack_scenarios) > 0

    def test_quick_evaluate(self):
        """Test quick evaluation."""
        config = {
            "circuit_breaker_enabled": True,
            "collusion_detection_enabled": True,
            "audit_enabled": True,
        }
        evaluator = RedTeamEvaluator(governance_config=config)

        result = evaluator.quick_evaluate()
        assert "attacks_tested" in result
        assert "attacks_successful" in result
        assert "avg_evasion_rate" in result

    def test_evaluate_with_no_defenses(self):
        """Test evaluation with no defenses."""
        config = {}  # No defenses
        evaluator = RedTeamEvaluator(
            governance_config=config,
            attack_scenarios=[AttackLibrary.threshold_dancing()],
        )

        result = evaluator.quick_evaluate(["threshold_dancing"])
        # Without defenses, attacks should be more successful
        assert result["attacks_tested"] == 1

    def test_evaluate_with_full_defenses(self):
        """Test evaluation with all defenses."""
        config = {
            "circuit_breaker_enabled": True,
            "collusion_detection_enabled": True,
            "audit_enabled": True,
            "staking_enabled": True,
            "transaction_tax_rate": 0.1,
        }
        evaluator = RedTeamEvaluator(
            governance_config=config,
            attack_scenarios=[AttackLibrary.reputation_farming_exploit()],
        )

        result = evaluator.quick_evaluate(["reputation_farming"])
        # With full defenses, attacks should be less successful
        assert "avg_evasion_rate" in result


class TestVulnerabilityReport:
    """Tests for VulnerabilityReport."""

    def test_generate_summary(self):
        """Test summary generation."""
        robustness = GovernanceRobustness(
            governance_config={"test": True},
            attacks_tested=5,
            attacks_prevented=3,
            overall_evasion_rate=0.4,
        )
        robustness.vulnerabilities = [
            Vulnerability(
                vulnerability_id="test_vuln",
                severity="high",
                description="Test",
                attack_vector="evasion",
                affected_lever="circuit_breaker",
                exploitation_difficulty="moderate",
                potential_damage=25.0,
                mitigation="Fix it",
            )
        ]
        robustness.recommendations = ["Do something"]

        report = VulnerabilityReport(robustness=robustness)
        summary = report.generate_summary()

        assert "RED-TEAM EVALUATION REPORT" in summary
        assert "test_vuln" in summary
        assert "Do something" in summary


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for red-teaming with orchestrator."""

    def test_adaptive_adversary_in_simulation(self):
        """Test running adaptive adversary in orchestrator."""
        from src.agents.honest import HonestAgent
        from src.core.orchestrator import Orchestrator, OrchestratorConfig
        from src.governance.config import GovernanceConfig

        config = OrchestratorConfig(
            n_epochs=3,
            steps_per_epoch=5,
            governance_config=GovernanceConfig(
                circuit_breaker_enabled=True,
                freeze_threshold_toxicity=0.5,
            ),
        )

        orchestrator = Orchestrator(config=config)
        orchestrator.register_agent(HonestAgent("honest_1"))
        orchestrator.register_agent(HonestAgent("honest_2"))
        orchestrator.register_agent(AdaptiveAdversary("adaptive_1"))

        metrics = orchestrator.run()
        assert len(metrics) == 3

    def test_get_adaptive_adversary_reports(self):
        """Test getting reports from adaptive adversaries."""
        from src.agents.honest import HonestAgent
        from src.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(n_epochs=2, steps_per_epoch=3)
        orchestrator = Orchestrator(config=config)

        orchestrator.register_agent(HonestAgent("honest_1"))
        orchestrator.register_agent(AdaptiveAdversary("adaptive_1"))
        orchestrator.register_agent(AdaptiveAdversary("adaptive_2"))

        orchestrator.run()

        reports = orchestrator.get_adaptive_adversary_reports()
        assert "adaptive_1" in reports
        assert "adaptive_2" in reports
        assert "current_strategy" in reports["adaptive_1"]

    def test_get_evasion_metrics(self):
        """Test getting evasion metrics from orchestrator."""
        from src.agents.honest import HonestAgent
        from src.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(n_epochs=2, steps_per_epoch=3)
        orchestrator = Orchestrator(config=config)

        orchestrator.register_agent(HonestAgent("honest_1"))
        orchestrator.register_agent(AdaptiveAdversary("adaptive_1"))

        orchestrator.run()

        metrics = orchestrator.get_evasion_metrics()
        assert "total_adversaries" in metrics
        assert "adaptive_adversaries" in metrics
        assert metrics["adaptive_adversaries"] == 1

    def test_notify_adversary_detection(self):
        """Test notifying adversary of detection."""
        from src.core.orchestrator import Orchestrator, OrchestratorConfig

        orchestrator = Orchestrator()
        adversary = AdaptiveAdversary("adaptive_1")
        orchestrator.register_agent(adversary)

        initial_heat = adversary.memory.current_heat_level

        orchestrator.notify_adversary_detection(
            agent_id="adaptive_1",
            penalty=10.0,
            detected=True,
        )

        assert adversary.memory.current_heat_level > initial_heat
