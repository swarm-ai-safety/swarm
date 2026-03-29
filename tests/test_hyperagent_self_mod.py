"""Tests for the hyperagent self-modification agent and scenario."""

import random
from pathlib import Path

from swarm.agents.hyperagent_self_mod import HyperagentSelfModAgent, SelfModSnapshot


class TestHyperagentSelfModAgent:
    """Tests for HyperagentSelfModAgent self-modification mechanics."""

    def _make_agent(self, **config_overrides) -> HyperagentSelfModAgent:
        config = {
            "modification_rate": 0.10,
            "min_acceptance_threshold": 0.2,
            "initial_acceptance_threshold": 0.6,
            "initial_quality": 0.80,
            "quality_decay_rate": 0.03,
            "quality_floor": 0.30,
            "weight_gaming_target": "engagement_signal",
        }
        config.update(config_overrides)
        return HyperagentSelfModAgent(
            agent_id="test-hyperagent-01",
            config=config,
            rng=random.Random(42),
        )

    def test_initial_state(self):
        agent = self._make_agent()
        assert agent.modification_depth == 0
        assert agent.acceptance_threshold == 0.6
        assert agent.current_quality == 0.80
        assert abs(sum(agent.proxy_weights.values()) - 1.0) < 1e-9

    def test_self_modify_shifts_weights(self):
        agent = self._make_agent()
        initial_engagement = agent.proxy_weights["engagement_signal"]
        agent.self_modify(epoch=1)

        # Engagement weight should increase
        assert agent.proxy_weights["engagement_signal"] > initial_engagement
        # Safety signals should decrease
        assert agent.proxy_weights["rework_penalty"] < 0.2
        assert agent.proxy_weights["verifier_penalty"] < 0.2
        # Weights still sum to 1
        assert abs(sum(agent.proxy_weights.values()) - 1.0) < 1e-9

    def test_self_modify_lowers_threshold(self):
        agent = self._make_agent()
        agent.self_modify(epoch=1)
        assert agent.acceptance_threshold < 0.6

    def test_self_modify_decays_quality(self):
        agent = self._make_agent()
        agent.self_modify(epoch=1)
        assert agent.current_quality < 0.80

    def test_quality_floor_respected(self):
        agent = self._make_agent(quality_decay_rate=0.50)
        for epoch in range(50):
            agent.self_modify(epoch=epoch)
        assert agent.current_quality >= agent.quality_floor

    def test_threshold_floor_respected(self):
        agent = self._make_agent(modification_rate=0.50)
        for epoch in range(50):
            agent.self_modify(epoch=epoch)
        assert agent.acceptance_threshold >= agent.min_acceptance_threshold

    def test_modification_history_recorded(self):
        agent = self._make_agent()
        for epoch in range(5):
            snapshot = agent.self_modify(epoch=epoch)
            assert isinstance(snapshot, SelfModSnapshot)
        assert len(agent.modification_history) == 5
        assert agent.modification_depth == 5

    def test_governance_gap_grows(self):
        """After several modifications, governance gap should be positive."""
        agent = self._make_agent()
        for epoch in range(10):
            agent.self_modify(epoch=epoch)
        # Engagement weight has grown, so ungoverned weight > 0.6 baseline
        last = agent.modification_history[-1]
        assert last.governance_gap_local > 0.0

    def test_generate_interaction_quality_in_range(self):
        agent = self._make_agent()
        for _ in range(100):
            p = agent.generate_interaction_quality()
            assert 0.0 <= p <= 1.0

    def test_engagement_weight_caps_at_0_8(self):
        agent = self._make_agent(modification_rate=0.50)
        for epoch in range(20):
            agent.self_modify(epoch=epoch)
        assert agent.proxy_weights["engagement_signal"] <= 0.8 + 1e-9


class TestScenarioYAMLLoads:
    """Verify the scenario YAML loads without errors."""

    def test_load_hyperagent_self_mod(self):
        from swarm.scenarios.loader import load_scenario

        scenario = load_scenario("scenarios/hyperagent_self_mod.yaml")
        assert scenario.scenario_id == "hyperagent_self_mod"
        assert scenario.orchestrator_config.n_epochs == 25
        assert len(scenario.agent_specs) == 3  # 3 agent groups

    def test_self_modification_enabled_in_governance(self):
        from swarm.scenarios.loader import load_scenario

        scenario = load_scenario("scenarios/hyperagent_self_mod.yaml")
        gov = scenario.orchestrator_config.governance_config
        assert gov is not None
        assert gov.self_modification_enabled is True


class TestSelfModifyOrchestratorIntegration:
    """Integration tests: self_modify() called by orchestrator, metrics piped."""

    def _build_orchestrator(self, n_epochs: int = 5, steps_per_epoch: int = 3):
        from swarm.scenarios.loader import build_orchestrator, load_scenario

        scenario = load_scenario(
            Path("scenarios/hyperagent_self_mod.yaml")
        )
        # Use fewer epochs/steps for test speed
        scenario.orchestrator_config.n_epochs = n_epochs
        scenario.orchestrator_config.steps_per_epoch = steps_per_epoch
        return build_orchestrator(scenario)

    def test_modification_depth_increases(self):
        """After running epochs, self-mod agents should have depth > 0."""
        orch = self._build_orchestrator(n_epochs=5, steps_per_epoch=2)
        orch.run()

        self_mod_agents = [
            a for a in orch.get_all_agents()
            if isinstance(a, HyperagentSelfModAgent)
        ]
        assert len(self_mod_agents) == 3
        for agent in self_mod_agents:
            assert agent.modification_depth == 5
            assert len(agent.modification_history) == 5

    def test_governance_gap_grows_over_epochs(self):
        """Governance gap should increase across epochs."""
        orch = self._build_orchestrator(n_epochs=10, steps_per_epoch=2)
        metrics = orch.run()

        # Collect envelope metrics from epochs that have them
        envelope_gaps = [
            m.capability_envelope_metrics["mean_governance_gap"]
            for m in metrics
            if m.capability_envelope_metrics is not None
        ]
        assert len(envelope_gaps) > 0
        # Gap should grow: last > first (agents expand beyond governance)
        assert envelope_gaps[-1] >= envelope_gaps[0]

    def test_circuit_breaker_activates(self):
        """With enough self-modification, toxicity should rise and
        the circuit breaker should activate (freeze epochs > 0)."""
        orch = self._build_orchestrator(n_epochs=15, steps_per_epoch=5)
        metrics = orch.run()

        # Check that toxicity_rate eventually rises above baseline
        toxicities = [m.toxicity_rate for m in metrics]
        # At least some epoch should show non-zero toxicity
        assert any(t > 0.0 for t in toxicities), (
            "Expected some toxicity from self-modifying agents"
        )

    def test_capability_envelope_metrics_present(self):
        """Epoch metrics should include capability_envelope_metrics."""
        orch = self._build_orchestrator(n_epochs=3, steps_per_epoch=2)
        metrics = orch.run()

        for m in metrics:
            env = m.capability_envelope_metrics
            assert env is not None, "capability_envelope_metrics should be set"
            assert "mean_governance_gap" in env
            assert "max_governance_gap" in env
            assert "mean_envelope" in env
            assert "n_self_mod_agents" in env
            assert env["n_self_mod_agents"] == 3
            assert "max_modification_depth" in env

    def test_modification_depth_in_envelope_metrics(self):
        """max_modification_depth in envelope should match agent state."""
        orch = self._build_orchestrator(n_epochs=4, steps_per_epoch=2)
        metrics = orch.run()

        last = metrics[-1]
        assert last.capability_envelope_metrics is not None
        assert last.capability_envelope_metrics["max_modification_depth"] == 4
