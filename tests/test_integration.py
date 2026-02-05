"""Integration tests: end-to-end scenario runs with metric validation."""

from pathlib import Path

import pytest

from src.agents.adversarial import AdversarialAgent
from src.agents.deceptive import DeceptiveAgent
from src.agents.honest import HonestAgent
from src.agents.opportunistic import OpportunisticAgent
from src.core.orchestrator import EpochMetrics, Orchestrator, OrchestratorConfig
from src.core.payoff import PayoffConfig
from src.governance.config import GovernanceConfig
from src.scenarios.loader import build_orchestrator, load_scenario

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _run_scenario(name: str) -> tuple["ScenarioConfig", list[EpochMetrics]]:
    """Load a YAML scenario, run it, return (config, metrics)."""
    from src.scenarios.loader import ScenarioConfig

    scenario = load_scenario(SCENARIOS_DIR / f"{name}.yaml")
    orch = build_orchestrator(scenario)
    metrics = orch.run()
    return scenario, metrics


# ---------------------------------------------------------------------------
# 1. YAML scenario end-to-end: baseline
# ---------------------------------------------------------------------------


class TestBaselineScenario:
    """Run baseline.yaml end-to-end and validate outputs."""

    def test_runs_to_completion(self):
        scenario, metrics = _run_scenario("baseline")
        assert len(metrics) == scenario.orchestrator_config.n_epochs

    def test_interactions_occur(self):
        _, metrics = _run_scenario("baseline")
        total = sum(m.total_interactions for m in metrics)
        assert total > 0, "No interactions occurred"

    def test_metrics_in_valid_ranges(self):
        _, metrics = _run_scenario("baseline")
        for m in metrics:
            assert 0.0 <= m.toxicity_rate <= 1.0
            assert -1.0 <= m.quality_gap <= 1.0
            assert m.total_interactions >= m.accepted_interactions >= 0

    def test_toxicity_below_threshold(self):
        scenario, metrics = _run_scenario("baseline")
        threshold = scenario.success_criteria.get("toxicity_threshold", 1.0)
        avg_tox = _avg([m.toxicity_rate for m in metrics])
        assert avg_tox <= threshold, (
            f"Average toxicity {avg_tox:.3f} exceeds threshold {threshold}"
        )


# ---------------------------------------------------------------------------
# 2. YAML scenario end-to-end: strict governance
# ---------------------------------------------------------------------------


class TestStrictGovernanceScenario:
    """Run strict_governance.yaml end-to-end."""

    def test_runs_to_completion(self):
        scenario, metrics = _run_scenario("strict_governance")
        assert len(metrics) == scenario.orchestrator_config.n_epochs

    def test_interactions_occur(self):
        _, metrics = _run_scenario("strict_governance")
        total = sum(m.total_interactions for m in metrics)
        assert total > 0

    def test_metrics_in_valid_ranges(self):
        _, metrics = _run_scenario("strict_governance")
        for m in metrics:
            assert 0.0 <= m.toxicity_rate <= 1.0
            assert -1.0 <= m.quality_gap <= 1.0


# ---------------------------------------------------------------------------
# 3. Governance effects: compare baseline vs. strict
# ---------------------------------------------------------------------------


class TestGovernanceEffects:
    """Verify governance levers change simulation outcomes."""

    @pytest.fixture(scope="class")
    def baseline_metrics(self):
        _, metrics = _run_scenario("baseline")
        return metrics

    @pytest.fixture(scope="class")
    def strict_metrics(self):
        _, metrics = _run_scenario("strict_governance")
        return metrics

    def test_governance_changes_welfare(self, baseline_metrics, strict_metrics):
        """Governance should measurably change total welfare (up or down)."""
        baseline_welfare = _avg([m.total_welfare for m in baseline_metrics])
        strict_welfare = _avg([m.total_welfare for m in strict_metrics])
        # They shouldn't be identical — governance has an effect
        assert baseline_welfare != strict_welfare

    def test_governance_changes_payoffs(self, baseline_metrics, strict_metrics):
        """Average payoffs should differ under governance."""
        baseline_payoff = _avg([m.avg_payoff for m in baseline_metrics])
        strict_payoff = _avg([m.avg_payoff for m in strict_metrics])
        assert baseline_payoff != strict_payoff


# ---------------------------------------------------------------------------
# 4. Determinism / reproducibility
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Running the same scenario twice with the same seed produces identical results."""

    def test_seeded_runs_are_identical(self):
        _, metrics_a = _run_scenario("baseline")
        _, metrics_b = _run_scenario("baseline")

        assert len(metrics_a) == len(metrics_b)
        for a, b in zip(metrics_a, metrics_b):
            assert a.total_interactions == b.total_interactions
            assert a.accepted_interactions == b.accepted_interactions
            assert a.toxicity_rate == pytest.approx(b.toxicity_rate)
            assert a.quality_gap == pytest.approx(b.quality_gap)
            assert a.total_welfare == pytest.approx(b.total_welfare)


# ---------------------------------------------------------------------------
# 5. Programmatic orchestrator: honest-only ecosystem
# ---------------------------------------------------------------------------


class TestHonestOnlyEcosystem:
    """All-honest agents should produce a healthy ecosystem."""

    @pytest.fixture(scope="class")
    def metrics(self):
        config = OrchestratorConfig(n_epochs=5, steps_per_epoch=8, seed=99)
        orch = Orchestrator(config=config)
        for i in range(4):
            orch.register_agent(HonestAgent(agent_id=f"honest_{i}"))
        return orch.run()

    def test_low_toxicity(self, metrics):
        avg_tox = _avg([m.toxicity_rate for m in metrics])
        assert avg_tox < 0.3, f"Honest-only toxicity unexpectedly high: {avg_tox:.3f}"

    def test_positive_welfare(self, metrics):
        total = sum(m.total_welfare for m in metrics)
        assert total > 0, "Honest agents should generate positive total welfare"


# ---------------------------------------------------------------------------
# 6. Programmatic orchestrator: adversarial-heavy ecosystem
# ---------------------------------------------------------------------------


class TestAdversarialHeavyEcosystem:
    """Adversarial-majority ecosystem should show higher toxicity."""

    @pytest.fixture(scope="class")
    def adversarial_metrics(self):
        config = OrchestratorConfig(n_epochs=5, steps_per_epoch=8, seed=99)
        orch = Orchestrator(config=config)
        orch.register_agent(HonestAgent(agent_id="honest_1"))
        for i in range(3):
            orch.register_agent(AdversarialAgent(agent_id=f"adv_{i}"))
        return orch.run()

    @pytest.fixture(scope="class")
    def honest_metrics(self):
        config = OrchestratorConfig(n_epochs=5, steps_per_epoch=8, seed=99)
        orch = Orchestrator(config=config)
        for i in range(4):
            orch.register_agent(HonestAgent(agent_id=f"honest_{i}"))
        return orch.run()

    def test_adversarial_has_higher_toxicity(
        self, adversarial_metrics, honest_metrics
    ):
        adv_tox = _avg([m.toxicity_rate for m in adversarial_metrics])
        hon_tox = _avg([m.toxicity_rate for m in honest_metrics])
        assert adv_tox > hon_tox, (
            f"Expected adversarial toxicity ({adv_tox:.3f}) > "
            f"honest toxicity ({hon_tox:.3f})"
        )


# ---------------------------------------------------------------------------
# 7. Network topology scenario
# ---------------------------------------------------------------------------


class TestNetworkEffectsScenario:
    """Run network_effects.yaml and verify network metrics are produced."""

    def test_runs_to_completion(self):
        scenario, metrics = _run_scenario("network_effects")
        assert len(metrics) == scenario.orchestrator_config.n_epochs

    def test_network_metrics_present(self):
        _, metrics = _run_scenario("network_effects")
        # At least some epochs should have network metrics
        has_network = [m for m in metrics if m.network_metrics is not None]
        assert len(has_network) > 0, "No epochs reported network metrics"


# ---------------------------------------------------------------------------
# 8. Collusion detection scenario
# ---------------------------------------------------------------------------


class TestCollusionDetectionScenario:
    """Run collusion_detection.yaml and verify it completes."""

    def test_runs_to_completion(self):
        scenario, metrics = _run_scenario("collusion_detection")
        assert len(metrics) == scenario.orchestrator_config.n_epochs

    def test_interactions_occur(self):
        _, metrics = _run_scenario("collusion_detection")
        total = sum(m.total_interactions for m in metrics)
        assert total > 0


# ---------------------------------------------------------------------------
# 9. Epoch callbacks fire correctly
# ---------------------------------------------------------------------------


class TestCallbackIntegration:
    """Verify orchestrator callbacks fire during a full run."""

    def test_epoch_callback_fires_every_epoch(self):
        config = OrchestratorConfig(n_epochs=5, steps_per_epoch=5, seed=7)
        orch = Orchestrator(config=config)
        for i in range(3):
            orch.register_agent(HonestAgent(agent_id=f"h_{i}"))

        epoch_log: list[int] = []
        orch.on_epoch_end(lambda m: epoch_log.append(m.epoch))

        metrics = orch.run()
        assert len(epoch_log) == 5
        assert epoch_log == [m.epoch for m in metrics]

    def test_interaction_callback_fires(self):
        config = OrchestratorConfig(n_epochs=3, steps_per_epoch=5, seed=7)
        orch = Orchestrator(config=config)
        for i in range(3):
            orch.register_agent(HonestAgent(agent_id=f"h_{i}"))

        interactions_logged: list[float] = []
        orch.on_interaction_complete(
            lambda ix, pa, pb: interactions_logged.append(pa)
        )

        orch.run()
        assert len(interactions_logged) > 0, "No interaction callbacks fired"


# ---------------------------------------------------------------------------
# 10. Edge case: single agent
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases that should not crash."""

    def test_single_agent_no_crash(self):
        config = OrchestratorConfig(n_epochs=3, steps_per_epoch=3, seed=1)
        orch = Orchestrator(config=config)
        orch.register_agent(HonestAgent(agent_id="solo"))
        metrics = orch.run()
        assert len(metrics) == 3
        # No interactions possible with one agent
        total = sum(m.total_interactions for m in metrics)
        assert total == 0

    def test_two_agents_minimal(self):
        config = OrchestratorConfig(n_epochs=2, steps_per_epoch=3, seed=1)
        orch = Orchestrator(config=config)
        orch.register_agent(HonestAgent(agent_id="a"))
        orch.register_agent(HonestAgent(agent_id="b"))
        metrics = orch.run()
        assert len(metrics) == 2
