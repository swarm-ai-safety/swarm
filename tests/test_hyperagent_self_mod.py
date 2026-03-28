"""Tests for the hyperagent self-modification agent and scenario."""

import random

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
