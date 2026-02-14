"""Tests for the self-optimizing agent and its distributional safety properties.

The core thesis: hard metrics (acceptance rate, binary threshold) DON'T detect
quality degradation from recursive self-optimization, but soft metrics DO.
"""

import random
from pathlib import Path

import pytest

from swarm.agents.base import InteractionProposal, Observation
from swarm.agents.self_optimizer import OptimizationSnapshot, SelfOptimizerAgent
from swarm.metrics.horizon_eval import HorizonEvalConfig, SystemHorizonEvaluator
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.scenarios.loader import AGENT_TYPES, load_scenario
from tests.fixtures.interactions import generate_self_optimizer_scenario

# =========================================================================
# Unit Tests: Agent creation and optimization mechanics
# =========================================================================


class TestSelfOptimizerAgent:
    """Unit tests for the SelfOptimizerAgent."""

    def test_creation_defaults(self):
        """Agent creates with correct default config."""
        agent = SelfOptimizerAgent(agent_id="opt_1")
        assert agent.agent_id == "opt_1"
        assert agent.optimization_rate == 0.15
        assert agent.quality_floor == 0.35
        assert agent.benchmark_threshold == 0.45
        assert agent.initial_quality == 0.85
        assert agent.initial_cost == 0.10
        assert agent.cost_multiplier == 1.0
        assert agent.current_quality == 0.85
        assert agent.optimization_depth == 0
        assert agent.optimization_history == []

    def test_creation_custom_config(self):
        """Agent respects custom config params."""
        agent = SelfOptimizerAgent(
            agent_id="opt_2",
            config={
                "optimization_rate": 0.20,
                "quality_floor": 0.40,
                "benchmark_threshold": 0.50,
                "initial_quality": 0.90,
                "initial_cost": 0.15,
            },
        )
        assert agent.optimization_rate == 0.20
        assert agent.quality_floor == 0.40
        assert agent.benchmark_threshold == 0.50
        assert agent.initial_quality == 0.90
        assert agent.initial_cost == 0.15

    def test_optimize_step_reduces_cost_multiplier(self):
        """optimize_step() reduces cost multiplier each call."""
        agent = SelfOptimizerAgent(agent_id="opt_1")
        initial_cm = agent.cost_multiplier

        snapshot = agent.optimize_step(epoch=0)
        assert agent.cost_multiplier < initial_cm
        assert agent.cost_multiplier == pytest.approx(1.0 * (1.0 - 0.15))

        agent.optimize_step(epoch=1)
        assert agent.cost_multiplier < snapshot.cost_multiplier

    def test_optimize_step_reduces_quality(self):
        """optimize_step() reduces quality each call."""
        agent = SelfOptimizerAgent(agent_id="opt_1")
        initial_q = agent.current_quality

        agent.optimize_step(epoch=0)
        assert agent.current_quality < initial_q

    def test_quality_floor_respected(self):
        """Quality never drops below quality_floor even after many steps."""
        agent = SelfOptimizerAgent(
            agent_id="opt_1",
            config={"optimization_rate": 0.3, "quality_floor": 0.40},
        )

        for epoch in range(50):
            agent.optimize_step(epoch)

        assert agent.current_quality >= agent.quality_floor

    def test_optimization_history_tracked(self):
        """Each optimize_step() is recorded in history."""
        agent = SelfOptimizerAgent(agent_id="opt_1")
        for epoch in range(5):
            agent.optimize_step(epoch)

        assert len(agent.optimization_history) == 5
        assert all(isinstance(s, OptimizationSnapshot) for s in agent.optimization_history)
        assert agent.optimization_history[0].epoch == 0
        assert agent.optimization_history[4].epoch == 4
        assert agent.optimization_depth == 5

    def test_optimization_history_monotonic_cost(self):
        """Cost multiplier monotonically decreases across history."""
        agent = SelfOptimizerAgent(agent_id="opt_1")
        for epoch in range(10):
            agent.optimize_step(epoch)

        costs = [s.cost_multiplier for s in agent.optimization_history]
        for i in range(1, len(costs)):
            assert costs[i] < costs[i - 1]

    def test_effective_cost_decreases(self):
        """effective_cost property reflects cost_multiplier."""
        agent = SelfOptimizerAgent(agent_id="opt_1")
        initial_cost = agent.effective_cost
        assert initial_cost == pytest.approx(0.10)

        agent.optimize_step(epoch=0)
        assert agent.effective_cost < initial_cost

    def test_generate_interaction_quality(self):
        """generate_interaction_quality returns values in valid range."""
        random.seed(42)
        agent = SelfOptimizerAgent(agent_id="opt_1")

        qualities = [agent.generate_interaction_quality() for _ in range(100)]
        assert all(agent.quality_floor <= q <= 1.0 for q in qualities)
        # Mean should be near initial quality
        mean_q = sum(qualities) / len(qualities)
        assert abs(mean_q - agent.current_quality) < 0.1

    def test_passes_benchmark(self):
        """passes_benchmark correctly checks threshold."""
        agent = SelfOptimizerAgent(agent_id="opt_1")
        assert agent.passes_benchmark(0.50)
        assert agent.passes_benchmark(0.45)
        assert not agent.passes_benchmark(0.44)

    def test_act_returns_action(self):
        """act() returns a valid action."""
        agent = SelfOptimizerAgent(agent_id="opt_1")
        obs = Observation()
        action = agent.act(obs)
        assert action is not None

    def test_accept_interaction_low_cost(self):
        """Agent with low cost_multiplier accepts almost everything."""
        agent = SelfOptimizerAgent(agent_id="opt_1")
        # Drive cost_multiplier below 0.5
        for epoch in range(10):
            agent.optimize_step(epoch)
        assert agent.cost_multiplier < 0.5

        proposal = InteractionProposal(
            initiator_id="other_agent",
            counterparty_id="opt_1",
        )
        obs = Observation()
        assert agent.accept_interaction(proposal, obs) is True

    def test_propose_interaction(self):
        """propose_interaction returns a valid proposal."""
        agent = SelfOptimizerAgent(agent_id="opt_1")
        obs = Observation()
        proposal = agent.propose_interaction(obs, "target_1")
        assert proposal is not None
        assert proposal.initiator_id == "opt_1"
        assert proposal.counterparty_id == "target_1"

    def test_registered_in_agent_types(self):
        """self_optimizer is registered in loader AGENT_TYPES."""
        assert "self_optimizer" in AGENT_TYPES
        assert AGENT_TYPES["self_optimizer"] is SelfOptimizerAgent


# =========================================================================
# Fixture Tests: generate_self_optimizer_scenario
# =========================================================================


class TestSelfOptimizerFixture:
    """Tests for the scenario fixture generator."""

    def test_generates_correct_epoch_count(self):
        """Fixture returns the requested number of epochs."""
        data = generate_self_optimizer_scenario(n_epochs=10, seed=42)
        assert len(data) == 10

    def test_epoch_has_both_agent_types(self):
        """Each epoch contains interactions from honest and optimizer agents."""
        data = generate_self_optimizer_scenario(n_epochs=5, seed=42)
        for epoch_interactions in data:
            types = {i.metadata.get("agent_type") for i in epoch_interactions}
            assert "honest" in types
            assert "self_optimizer" in types

    def test_optimizer_quality_degrades_over_epochs(self):
        """Optimizer mean p decreases across epochs."""
        data = generate_self_optimizer_scenario(n_epochs=15, seed=42)

        optimizer_mean_p_by_epoch = []
        for epoch_interactions in data:
            opt = [
                i for i in epoch_interactions
                if i.metadata.get("agent_type") == "self_optimizer"
            ]
            if opt:
                optimizer_mean_p_by_epoch.append(sum(i.p for i in opt) / len(opt))

        # Early epochs should have higher quality than late epochs
        early = sum(optimizer_mean_p_by_epoch[:3]) / 3
        late = sum(optimizer_mean_p_by_epoch[-3:]) / 3
        assert early > late, f"Early ({early:.3f}) should be > late ({late:.3f})"

    def test_honest_quality_stable(self):
        """Honest agent quality remains stable across epochs."""
        data = generate_self_optimizer_scenario(n_epochs=15, seed=42)

        honest_mean_p_by_epoch = []
        for epoch_interactions in data:
            honest = [
                i for i in epoch_interactions
                if i.metadata.get("agent_type") == "honest"
            ]
            if honest:
                honest_mean_p_by_epoch.append(sum(i.p for i in honest) / len(honest))

        early = sum(honest_mean_p_by_epoch[:3]) / 3
        late = sum(honest_mean_p_by_epoch[-3:]) / 3
        # Honest agents should be within ~0.1 across epochs
        assert abs(early - late) < 0.15

    def test_optimizer_cost_declines(self):
        """Optimizer c_a (governance cost) declines over epochs."""
        data = generate_self_optimizer_scenario(n_epochs=15, seed=42)

        mean_cost_by_epoch = []
        for epoch_interactions in data:
            opt = [
                i for i in epoch_interactions
                if i.metadata.get("agent_type") == "self_optimizer"
            ]
            if opt:
                mean_cost_by_epoch.append(sum(i.c_a for i in opt) / len(opt))

        assert mean_cost_by_epoch[0] > mean_cost_by_epoch[-1]


# =========================================================================
# Metrics Tests: Hard vs Soft detection (the core value proposition)
# =========================================================================


class TestSoftMetricsDetect:
    """Tests proving soft metrics catch what hard metrics miss."""

    @pytest.fixture
    def scenario_data(self):
        """Generate a full self-optimizer scenario."""
        return generate_self_optimizer_scenario(n_epochs=20, seed=42)

    @pytest.fixture
    def flat_interactions(self, scenario_data):
        """Flatten epoch-grouped data."""
        return [i for epoch in scenario_data for i in epoch]

    @pytest.fixture
    def soft_metrics(self):
        return SoftMetrics()

    def test_hard_acceptance_rate_stays_high(self, flat_interactions):
        """Hard metric: acceptance rate doesn't reveal the problem."""
        accepted = sum(1 for i in flat_interactions if i.accepted)
        total = len(flat_interactions)
        acceptance_rate = accepted / total
        # Acceptance rate stays above 0.5 -- the hard metric looks fine
        assert acceptance_rate > 0.5, (
            f"Acceptance rate {acceptance_rate:.3f} should stay high"
        )

    def test_hard_threshold_passes(self, scenario_data):
        """Hard metric: most interactions pass the binary threshold."""
        threshold = 0.45
        all_accepted = [
            i for epoch in scenario_data for i in epoch if i.accepted
        ]
        pass_rate = sum(1 for i in all_accepted if i.p >= threshold) / len(all_accepted)
        # > 50% of accepted interactions pass the hard benchmark
        assert pass_rate > 0.5, (
            f"Hard benchmark pass rate {pass_rate:.3f} should look OK"
        )

    def test_soft_metrics_detect_toxicity_rise(self, scenario_data, soft_metrics):
        """Soft metric: toxicity rises in later epochs.

        Optimizers are accepted at high rates even as quality degrades,
        so E[1-p | accepted] increases over time.
        """
        early_interactions = [i for epoch in scenario_data[:5] for i in epoch]
        late_interactions = [i for epoch in scenario_data[-5:] for i in epoch]

        early_tox = soft_metrics.toxicity_rate(early_interactions)
        late_tox = soft_metrics.toxicity_rate(late_interactions)

        assert late_tox > early_tox, (
            f"Late toxicity ({late_tox:.3f}) should exceed early ({early_tox:.3f})"
        )

    def test_soft_metrics_detect_quality_degradation(
        self, scenario_data, soft_metrics
    ):
        """Soft metric: average quality of accepted interactions drops."""
        early_interactions = [i for epoch in scenario_data[:5] for i in epoch]
        late_interactions = [i for epoch in scenario_data[-5:] for i in epoch]

        early_q = soft_metrics.average_quality(early_interactions, accepted_only=True)
        late_q = soft_metrics.average_quality(late_interactions, accepted_only=True)

        assert early_q > late_q, (
            f"Early quality ({early_q:.3f}) should exceed late ({late_q:.3f})"
        )

    def test_soft_metrics_detect_negative_quality_gap(
        self, scenario_data, soft_metrics
    ):
        """Soft metric: quality gap goes negative in late epochs.

        Adverse selection: low-p optimizer interactions are accepted (90% rate)
        while some high-p honest interactions are rejected (35% rejection rate).
        This drives E[p|accepted] below E[p|rejected].
        """
        late_interactions = [i for epoch in scenario_data[-5:] for i in epoch]
        gap = soft_metrics.quality_gap(late_interactions)
        # In late epochs, accepted pool is polluted with low-p optimizer
        # interactions while the rejected pool contains high-p honest ones
        assert gap < 0.0, (
            f"Late quality gap ({gap:.3f}) should be negative (adverse selection)"
        )

    def test_soft_metrics_detect_variance_increase(
        self, scenario_data, soft_metrics
    ):
        """Soft metric: quality variance increases as distribution spreads."""
        early_interactions = [i for epoch in scenario_data[:5] for i in epoch]
        late_interactions = [i for epoch in scenario_data[-5:] for i in epoch]

        early_var = soft_metrics.quality_variance(early_interactions)
        late_var = soft_metrics.quality_variance(late_interactions)

        assert late_var > early_var, (
            f"Late variance ({late_var:.4f}) should exceed early ({early_var:.4f})"
        )

    def test_quality_distribution_shifts_leftward(self, scenario_data, soft_metrics):
        """The quality distribution shifts leftward over epochs."""
        early_interactions = [i for epoch in scenario_data[:5] for i in epoch]
        late_interactions = [i for epoch in scenario_data[-5:] for i in epoch]

        early_dist = soft_metrics.quality_distribution(early_interactions, bins=5)
        late_dist = soft_metrics.quality_distribution(late_interactions, bins=5)

        # More interactions in the lower bins in late epochs
        early_low = sum(count for start, end, count in early_dist if end <= 0.6)
        late_low = sum(count for start, end, count in late_dist if end <= 0.6)

        assert late_low > early_low, (
            f"Late low-quality count ({late_low}) should exceed early ({early_low})"
        )


# =========================================================================
# Horizon Evaluator Tests
# =========================================================================


class TestHorizonEvaluator:
    """Test that SystemHorizonEvaluator detects adverse selection drift."""

    def test_adverse_selection_drift_detected(self):
        """Horizon evaluator detects worsening adverse selection drift.

        Quality gap should decline over time as optimizers flood the
        accepted pool with low-p interactions.
        """
        data = generate_self_optimizer_scenario(n_epochs=20, seed=42)

        evaluator = SystemHorizonEvaluator(
            config=HorizonEvalConfig(
                agent_horizon_steps=1,
                drift_window=5,
            )
        )
        result = evaluator.evaluate(data)

        # The drift should be negative (worsening quality gap over time)
        assert result.drift_direction == "worsening", (
            f"Drift direction should be 'worsening', got '{result.drift_direction}' "
            f"(slope={result.adverse_selection_drift:.4f})"
        )

    def test_cumulative_harm_increases(self):
        """Cumulative harm trajectory should be monotonically non-decreasing."""
        data = generate_self_optimizer_scenario(n_epochs=15, seed=42)

        evaluator = SystemHorizonEvaluator()
        result = evaluator.evaluate(data)

        assert len(result.cumulative_harm_trajectory) == 15
        # Cumulative harm is monotonically non-decreasing by definition
        for i in range(1, len(result.cumulative_harm_trajectory)):
            assert result.cumulative_harm_trajectory[i] >= result.cumulative_harm_trajectory[i - 1]


# =========================================================================
# Integration Tests
# =========================================================================


class TestSelfOptimizerIntegration:
    """Integration tests: scenario loads, agent runs, governance responds."""

    def test_scenario_yaml_loads(self):
        """The self_optimizer.yaml scenario loads without error."""
        path = Path(__file__).parent.parent / "scenarios" / "self_optimizer.yaml"
        scenario = load_scenario(path)
        assert scenario.scenario_id == "self_optimizer"
        assert len(scenario.agent_specs) == 2  # honest + self_optimizer

    def test_agent_type_from_yaml(self):
        """self_optimizer type resolves to SelfOptimizerAgent."""
        path = Path(__file__).parent.parent / "scenarios" / "self_optimizer.yaml"
        scenario = load_scenario(path)

        from swarm.scenarios.loader import create_agents

        agents = create_agents(scenario.agent_specs)
        optimizer_agents = [a for a in agents if isinstance(a, SelfOptimizerAgent)]
        assert len(optimizer_agents) == 3

    def test_optimizer_degrades_through_epochs(self):
        """Self-optimizers' quality degrades when optimize_step is called."""
        agent = SelfOptimizerAgent(
            agent_id="opt_test",
            config={"optimization_rate": 0.15, "quality_floor": 0.35},
        )

        qualities = []
        for epoch in range(20):
            agent.optimize_step(epoch)
            # Sample several interactions
            epoch_qs = [agent.generate_interaction_quality() for _ in range(20)]
            qualities.append(sum(epoch_qs) / len(epoch_qs))

        # Quality should be lower in later epochs
        assert qualities[0] > qualities[-1]
        # Quality should never go below floor
        assert all(q >= agent.quality_floor - 0.1 for q in qualities)

    def test_circuit_breaker_threshold_reachable(self):
        """Toxicity can reach circuit breaker threshold in late epochs."""
        data = generate_self_optimizer_scenario(n_epochs=20, seed=42)
        metrics = SoftMetrics()

        # Check last 3 epochs
        late_interactions = [i for epoch in data[-3:] for i in epoch]
        toxicity = metrics.toxicity_rate(late_interactions)

        # Toxicity should be meaningfully above zero in late epochs
        # (whether it crosses the circuit breaker threshold depends on config)
        assert toxicity > 0.1, (
            f"Late-epoch toxicity ({toxicity:.3f}) should be elevated"
        )
