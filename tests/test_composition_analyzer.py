"""Tests for SafetyCompositionAnalyzer (Piece 3: Composition Boundary)."""


import pytest

from swarm.core.behavioral_profiler import InferredProfile
from swarm.core.composition_analyzer import (
    RegimeThresholds,
    SafetyCompositionAnalyzer,
    SafetyRegime,
    SimulationPoint,
    SweepConfig,
    _create_population,
    _estimate_critical_fraction,
    classify_regime,
)
from swarm.core.orchestrator import EpochMetrics
from swarm.core.payoff import PayoffConfig
from swarm.governance.config import GovernanceConfig
from swarm.models.agent import AgentType

# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


def _make_epoch(
    accepted: int = 80,
    total: int = 100,
    toxicity: float = 0.1,
    quality_gap: float = 0.2,
    welfare: float = 50.0,
    epoch: int = 0,
) -> EpochMetrics:
    return EpochMetrics(
        epoch=epoch,
        total_interactions=total,
        accepted_interactions=accepted,
        toxicity_rate=toxicity,
        quality_gap=quality_gap,
        total_welfare=welfare,
    )


class TestClassifyRegime:
    def test_cooperative_high_acceptance(self):
        metrics = [_make_epoch(accepted=95, total=100, toxicity=0.05)] * 10
        assert classify_regime(metrics) == SafetyRegime.COOPERATIVE

    def test_collapse_low_acceptance(self):
        metrics = [_make_epoch(accepted=10, total=100, toxicity=0.3)] * 10
        assert classify_regime(metrics) == SafetyRegime.COLLAPSE

    def test_collapse_high_toxicity(self):
        metrics = [_make_epoch(accepted=70, total=100, toxicity=0.6)] * 10
        assert classify_regime(metrics) == SafetyRegime.COLLAPSE

    def test_contested_middle(self):
        metrics = [_make_epoch(accepted=60, total=100, toxicity=0.2)] * 10
        assert classify_regime(metrics) == SafetyRegime.CONTESTED

    def test_empty_metrics_is_collapse(self):
        assert classify_regime([]) == SafetyRegime.COLLAPSE

    def test_zero_interactions_is_collapse(self):
        metrics = [_make_epoch(accepted=0, total=0)] * 5
        assert classify_regime(metrics) == SafetyRegime.COLLAPSE

    def test_uses_tail_third(self):
        """Early bad epochs shouldn't affect classification if tail is good."""
        bad = [_make_epoch(accepted=5, total=100, toxicity=0.8)] * 5
        good = [_make_epoch(accepted=95, total=100, toxicity=0.05)] * 10
        # Tail (last 5) is good → should be cooperative.
        metrics = bad + good
        assert classify_regime(metrics) == SafetyRegime.COOPERATIVE

    def test_custom_thresholds(self):
        metrics = [_make_epoch(accepted=50, total=100, toxicity=0.1)] * 10
        # Normally contested, but with relaxed thresholds → cooperative.
        thresholds = RegimeThresholds(cooperative_acceptance_min=0.4)
        assert classify_regime(metrics, thresholds) == SafetyRegime.COOPERATIVE


# ---------------------------------------------------------------------------
# Population creation
# ---------------------------------------------------------------------------


class TestCreatePopulation:
    def test_correct_size(self):
        mixture = {AgentType.HONEST: 0.5, AgentType.ADVERSARIAL: 0.5}
        agents = _create_population(mixture, 10, seed=42)
        assert len(agents) == 10

    def test_deterministic_with_seed(self):
        mixture = {AgentType.HONEST: 0.7, AgentType.ADVERSARIAL: 0.3}
        agents1 = _create_population(mixture, 20, seed=42)
        agents2 = _create_population(mixture, 20, seed=42)
        types1 = [a.agent_type for a in agents1]
        types2 = [a.agent_type for a in agents2]
        assert types1 == types2

    def test_all_honest_mixture(self):
        mixture = {AgentType.HONEST: 1.0}
        agents = _create_population(mixture, 5, seed=42)
        assert all(a.agent_type == AgentType.HONEST for a in agents)

    def test_all_adversarial_mixture(self):
        mixture = {AgentType.ADVERSARIAL: 1.0}
        agents = _create_population(mixture, 5, seed=42)
        assert all(a.agent_type == AgentType.ADVERSARIAL for a in agents)

    def test_unique_ids(self):
        mixture = {AgentType.HONEST: 0.5, AgentType.ADVERSARIAL: 0.5}
        agents = _create_population(mixture, 10, seed=42)
        ids = [a.agent_id for a in agents]
        assert len(ids) == len(set(ids))

    def test_approximate_proportions(self):
        """With a large population, mixture should be approximately honored."""
        mixture = {AgentType.HONEST: 0.8, AgentType.ADVERSARIAL: 0.2}
        agents = _create_population(mixture, 100, seed=42)
        n_honest = sum(1 for a in agents if a.agent_type == AgentType.HONEST)
        # Should be roughly 80 ± 10
        assert 60 <= n_honest <= 95


# ---------------------------------------------------------------------------
# Critical fraction estimation
# ---------------------------------------------------------------------------


class TestEstimateCriticalFraction:
    def test_clear_transition(self):
        points = [
            SimulationPoint(
                population_size=10, adversarial_fraction=0.1,
                governance_label="none", seed=42,
                regime=SafetyRegime.COOPERATIVE,
                final_acceptance_rate=0.9, final_toxicity=0.05,
                final_quality_gap=0.2, final_welfare=50.0, n_epochs=10,
            ),
            SimulationPoint(
                population_size=10, adversarial_fraction=0.5,
                governance_label="none", seed=42,
                regime=SafetyRegime.COLLAPSE,
                final_acceptance_rate=0.1, final_toxicity=0.6,
                final_quality_gap=-0.1, final_welfare=5.0, n_epochs=10,
            ),
        ]
        crit = _estimate_critical_fraction(points)
        assert crit is not None
        assert 0.1 < crit < 0.5

    def test_no_transition(self):
        """All stable → no critical fraction."""
        points = [
            SimulationPoint(
                population_size=10, adversarial_fraction=0.1,
                governance_label="none", seed=42,
                regime=SafetyRegime.COOPERATIVE,
                final_acceptance_rate=0.9, final_toxicity=0.05,
                final_quality_gap=0.2, final_welfare=50.0, n_epochs=10,
            ),
        ]
        assert _estimate_critical_fraction(points) is None

    def test_all_collapse(self):
        """All collapse → no critical fraction."""
        points = [
            SimulationPoint(
                population_size=10, adversarial_fraction=0.5,
                governance_label="none", seed=42,
                regime=SafetyRegime.COLLAPSE,
                final_acceptance_rate=0.1, final_toxicity=0.6,
                final_quality_gap=-0.1, final_welfare=5.0, n_epochs=10,
            ),
        ]
        assert _estimate_critical_fraction(points) is None


# ---------------------------------------------------------------------------
# SafetyCompositionAnalyzer — end-to-end
# ---------------------------------------------------------------------------


class TestSafetyCompositionAnalyzer:
    def _honest_profile(self) -> InferredProfile:
        return InferredProfile(
            agent_id="agent-honest",
            archetype_mixture={
                AgentType.HONEST: 0.85,
                AgentType.OPPORTUNISTIC: 0.10,
                AgentType.DECEPTIVE: 0.03,
                AgentType.ADVERSARIAL: 0.02,
            },
            dominant_archetype=AgentType.HONEST,
            p_mean=0.73,
            n_traces=50,
        )

    def _adversarial_profile(self) -> InferredProfile:
        return InferredProfile(
            agent_id="agent-adversarial",
            archetype_mixture={
                AgentType.HONEST: 0.05,
                AgentType.OPPORTUNISTIC: 0.10,
                AgentType.DECEPTIVE: 0.25,
                AgentType.ADVERSARIAL: 0.60,
            },
            dominant_archetype=AgentType.ADVERSARIAL,
            p_mean=0.25,
            n_traces=50,
        )

    def _small_sweep_config(self) -> SweepConfig:
        """Minimal sweep for fast testing."""
        return SweepConfig(
            population_sizes=[4],
            governance_configs={
                "none": None,
                "basic": GovernanceConfig(
                    circuit_breaker_enabled=True,
                    freeze_threshold_toxicity=0.7,
                ),
            },
            n_epochs=5,
            steps_per_epoch=5,
            seeds=[42],
            payoff_config=PayoffConfig(
                s_plus=2.0, s_minus=1.0, h=2.0,
                rho_a=0.1, rho_b=0.1,
            ),
        )

    def test_empty_profiles_raises(self):
        analyzer = SafetyCompositionAnalyzer()
        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze([])

    def test_honest_population_is_safe(self):
        """A population of mostly-honest agents should be stable."""
        config = self._small_sweep_config()
        analyzer = SafetyCompositionAnalyzer(sweep_config=config)
        cert = analyzer.analyze([self._honest_profile()])

        # Should have at least some non-collapse results.
        assert cert.regime_counts[SafetyRegime.COLLAPSE] == 0 or (
            cert.regime_counts.get(SafetyRegime.COOPERATIVE, 0)
            + cert.regime_counts.get(SafetyRegime.CONTESTED, 0)
        ) > 0
        assert cert.max_safe_population >= 4
        assert len(cert.all_points) > 0

    def test_adversarial_population_degrades(self):
        """A population of adversarial agents should be less stable."""
        config = self._small_sweep_config()
        analyzer = SafetyCompositionAnalyzer(sweep_config=config)
        honest_cert = analyzer.analyze([self._honest_profile()])
        adversarial_cert = analyzer.analyze([self._adversarial_profile()])

        # Adversarial should have more collapses or worse metrics.
        honest_welfare = max(
            (p.final_welfare for p in honest_cert.all_points), default=0
        )
        adversarial_welfare = max(
            (p.final_welfare for p in adversarial_cert.all_points), default=0
        )
        # Honest should have better welfare in at least one config.
        assert honest_welfare >= adversarial_welfare

    def test_certificate_fields_populated(self):
        config = self._small_sweep_config()
        analyzer = SafetyCompositionAnalyzer(sweep_config=config)
        cert = analyzer.analyze([self._honest_profile()])

        assert cert.profile_ids == ["agent-honest"]
        assert cert.dominant_archetypes == {"agent-honest": AgentType.HONEST}
        assert len(cert.population_sizes_tested) == 1
        assert len(cert.governance_configs_tested) == 2
        assert cert.seeds_per_point == 1
        assert cert.conditions  # non-empty string

    def test_summary_string(self):
        config = self._small_sweep_config()
        analyzer = SafetyCompositionAnalyzer(sweep_config=config)
        cert = analyzer.analyze([self._honest_profile()])
        summary = cert.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_progress_callback(self):
        config = self._small_sweep_config()
        analyzer = SafetyCompositionAnalyzer(sweep_config=config)
        progress_calls = []
        analyzer.analyze(
            [self._honest_profile()],
            progress_callback=lambda c, t: progress_calls.append((c, t)),
        )
        # Should have exactly pop_sizes * gov_configs * seeds calls.
        expected = 1 * 2 * 1  # 4 * 2 configs * 1 seed
        assert len(progress_calls) == expected
        assert progress_calls[-1] == (expected, expected)

    def test_multiple_profiles_mixed(self):
        """Mixing honest and adversarial profiles."""
        config = self._small_sweep_config()
        analyzer = SafetyCompositionAnalyzer(sweep_config=config)
        cert = analyzer.analyze([
            self._honest_profile(),
            self._adversarial_profile(),
        ])
        # Mixed population should produce valid results.
        assert len(cert.all_points) > 0
        assert sum(cert.regime_counts.values()) == len(cert.all_points)
        # Mixture should average the two profiles.
        mixture = cert.population_mixture
        assert mixture[AgentType.HONEST] > 0.1
        assert mixture[AgentType.ADVERSARIAL] > 0.1

    def test_simulation_points_have_trajectories(self):
        config = self._small_sweep_config()
        analyzer = SafetyCompositionAnalyzer(sweep_config=config)
        cert = analyzer.analyze([self._honest_profile()])
        for point in cert.all_points:
            assert len(point.epoch_acceptance) == config.n_epochs
            assert len(point.epoch_toxicity) == config.n_epochs
            assert len(point.epoch_quality_gap) == config.n_epochs
            assert all(0 <= a <= 1 for a in point.epoch_acceptance)


# ---------------------------------------------------------------------------
# Aggregate mixture
# ---------------------------------------------------------------------------


class TestAggregateMixture:
    def test_single_profile(self):
        analyzer = SafetyCompositionAnalyzer()
        profile = InferredProfile(
            agent_id="a",
            archetype_mixture={
                AgentType.HONEST: 0.7,
                AgentType.ADVERSARIAL: 0.3,
            },
        )
        mixture = analyzer._aggregate_mixture([profile])
        assert mixture[AgentType.HONEST] == pytest.approx(0.7)
        assert mixture[AgentType.ADVERSARIAL] == pytest.approx(0.3)

    def test_two_profiles_averaged(self):
        analyzer = SafetyCompositionAnalyzer()
        p1 = InferredProfile(
            agent_id="a",
            archetype_mixture={AgentType.HONEST: 1.0, AgentType.ADVERSARIAL: 0.0},
        )
        p2 = InferredProfile(
            agent_id="b",
            archetype_mixture={AgentType.HONEST: 0.0, AgentType.ADVERSARIAL: 1.0},
        )
        mixture = analyzer._aggregate_mixture([p1, p2])
        assert mixture[AgentType.HONEST] == pytest.approx(0.5)
        assert mixture[AgentType.ADVERSARIAL] == pytest.approx(0.5)

    def test_sums_to_one(self):
        analyzer = SafetyCompositionAnalyzer()
        profile = InferredProfile(
            agent_id="a",
            archetype_mixture={
                AgentType.HONEST: 0.4,
                AgentType.OPPORTUNISTIC: 0.3,
                AgentType.DECEPTIVE: 0.2,
                AgentType.ADVERSARIAL: 0.1,
            },
        )
        mixture = analyzer._aggregate_mixture([profile])
        assert sum(mixture.values()) == pytest.approx(1.0)
