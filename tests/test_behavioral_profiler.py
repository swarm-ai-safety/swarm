"""Tests for BehavioralProfiler (Piece 2: Profile Inference)."""

import math
import random

import pytest

from swarm.core.behavioral_profiler import (
    ARCHETYPE_MODELS,
    CORE_ARCHETYPES,
    BehavioralProfiler,
    InferredProfile,
    _discrete_uniform_log_likelihood,
    _log_likelihood_to_weights,
    _std,
    _uniform_log_likelihood,
    archetype_log_likelihood,
)
from swarm.core.eval_trace_adapter import (
    EvalFramework,
    EvalTrace,
    EvalTurn,
    TurnType,
)
from swarm.core.observable_generator import DefaultObservableGenerator
from swarm.core.proxy import ProxyObservables
from swarm.models.agent import AgentType

# ---------------------------------------------------------------------------
# Helpers: generate synthetic traces from known archetypes
# ---------------------------------------------------------------------------


def _generate_archetype_traces(
    agent_type: AgentType,
    n: int = 50,
    seed: int = 42,
) -> list[EvalTrace]:
    """Generate traces that mimic a known SWARM archetype.

    Uses DefaultObservableGenerator to produce signals for the given
    archetype, then wraps them as EvalTraces.  This lets us verify
    that the profiler correctly recovers the archetype from its own
    signal distribution.
    """
    rng = random.Random(seed)
    gen = DefaultObservableGenerator(rng=rng)

    traces = []
    for i in range(n):
        # Generate raw observables as the archetype would.
        obs = gen._base_signals(agent_type)
        progress, rework, rejections, misuse, engagement = obs

        # Map back to EvalTrace turns that produce similar observables.
        # We create turns whose derived statistics match the observables.
        turns = []

        # Create agent actions — some completing tasks based on progress.
        n_actions = 5
        n_completed = max(0, min(n_actions, round((progress / 2.0 + 0.5) * n_actions)))
        for j in range(n_actions):
            cooperative = engagement > 0
            turns.append(EvalTurn(
                turn_index=j,
                turn_type=TurnType.AGENT_ACTION,
                task_completed=(j < n_completed),
                self_correction=(j < rework),
                safety_violation=(j < rejections),
                unsafe_tool_use=(j < misuse),
                cooperative=cooperative,
            ))

        # Set episode-level scores aligned with archetype.
        if agent_type == AgentType.HONEST:
            safety_score = rng.uniform(0.75, 0.95)
            helpfulness_score = rng.uniform(0.7, 0.9)
        elif agent_type == AgentType.OPPORTUNISTIC:
            safety_score = rng.uniform(0.4, 0.7)
            helpfulness_score = rng.uniform(0.3, 0.6)
        elif agent_type == AgentType.DECEPTIVE:
            safety_score = rng.uniform(0.3, 0.6)
            helpfulness_score = rng.uniform(0.4, 0.7)
        else:  # ADVERSARIAL
            safety_score = rng.uniform(0.05, 0.3)
            helpfulness_score = rng.uniform(0.05, 0.25)

        traces.append(EvalTrace(
            trace_id=f"{agent_type.value}-{i}",
            agent_id=f"agent-{agent_type.value}",
            framework=EvalFramework.CUSTOM,
            turns=turns,
            overall_safety_score=safety_score,
            overall_helpfulness_score=helpfulness_score,
        ))

    return traces


# ---------------------------------------------------------------------------
# Low-level likelihood tests
# ---------------------------------------------------------------------------


class TestUniformLogLikelihood:
    def test_inside_range(self):
        ll = _uniform_log_likelihood(0.5, 0.0, 1.0)
        assert ll == pytest.approx(0.0)  # log(1/1) = 0

    def test_outside_range_lower(self):
        ll_inside = _uniform_log_likelihood(0.5, 0.3, 0.8)
        ll_outside = _uniform_log_likelihood(0.0, 0.3, 0.8)
        assert ll_outside < ll_inside

    def test_outside_range_higher(self):
        ll_inside = _uniform_log_likelihood(0.5, 0.3, 0.8)
        ll_outside = _uniform_log_likelihood(1.0, 0.3, 0.8)
        assert ll_outside < ll_inside

    def test_point_distribution(self):
        # When lo == hi, should still return finite likelihood
        ll = _uniform_log_likelihood(0.0, 0.0, 0.0)
        assert math.isfinite(ll)

    def test_farther_outside_is_less_likely(self):
        ll_near = _uniform_log_likelihood(0.85, 0.3, 0.8)
        ll_far = _uniform_log_likelihood(1.5, 0.3, 0.8)
        assert ll_far < ll_near


class TestDiscreteUniformLogLikelihood:
    def test_inside_range(self):
        ll = _discrete_uniform_log_likelihood(1, 0, 2)
        assert ll == pytest.approx(-math.log(3))

    def test_outside_range(self):
        ll_inside = _discrete_uniform_log_likelihood(1, 0, 2)
        ll_outside = _discrete_uniform_log_likelihood(5, 0, 2)
        assert ll_outside < ll_inside

    def test_point_mass(self):
        # Range {0, 0}
        ll = _discrete_uniform_log_likelihood(0, 0, 0)
        assert ll == pytest.approx(0.0)  # log(1/1) = 0


class TestArchetypeLogLikelihood:
    def test_honest_obs_highest_under_honest(self):
        """Honest-like observables should score highest under HONEST model."""
        obs = ProxyObservables(
            task_progress_delta=0.6,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=0.4,
        )
        lls = {
            atype: archetype_log_likelihood(obs, model)
            for atype, model in ARCHETYPE_MODELS.items()
        }
        assert lls[AgentType.HONEST] > lls[AgentType.ADVERSARIAL]

    def test_adversarial_obs_highest_under_adversarial(self):
        """Adversarial-like observables should score highest under ADVERSARIAL."""
        obs = ProxyObservables(
            task_progress_delta=-0.1,
            rework_count=2,
            verifier_rejections=2,
            tool_misuse_flags=1,
            counterparty_engagement_delta=-0.3,
        )
        lls = {
            atype: archetype_log_likelihood(obs, model)
            for atype, model in ARCHETYPE_MODELS.items()
        }
        assert lls[AgentType.ADVERSARIAL] > lls[AgentType.HONEST]


class TestLogLikelihoodToWeights:
    def test_equal_likelihoods_uniform(self):
        lls = {AgentType.HONEST: 0.0, AgentType.ADVERSARIAL: 0.0}
        weights = _log_likelihood_to_weights(lls)
        assert weights[AgentType.HONEST] == pytest.approx(0.5)
        assert weights[AgentType.ADVERSARIAL] == pytest.approx(0.5)

    def test_sum_to_one(self):
        lls = {
            AgentType.HONEST: -1.0,
            AgentType.OPPORTUNISTIC: -3.0,
            AgentType.DECEPTIVE: -5.0,
            AgentType.ADVERSARIAL: -10.0,
        }
        weights = _log_likelihood_to_weights(lls)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_higher_ll_gets_higher_weight(self):
        lls = {AgentType.HONEST: -1.0, AgentType.ADVERSARIAL: -10.0}
        weights = _log_likelihood_to_weights(lls)
        assert weights[AgentType.HONEST] > weights[AgentType.ADVERSARIAL]

    def test_empty(self):
        assert _log_likelihood_to_weights({}) == {}

    def test_numerical_stability_large_values(self):
        """Should not overflow with very negative log-likelihoods."""
        lls = {
            AgentType.HONEST: -500.0,
            AgentType.ADVERSARIAL: -700.0,
        }
        weights = _log_likelihood_to_weights(lls)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights[AgentType.HONEST] > 0.99  # Much more likely


# ---------------------------------------------------------------------------
# BehavioralProfiler.fit
# ---------------------------------------------------------------------------


class TestBehavioralProfiler:
    def test_recovers_honest_archetype(self):
        """Traces generated from HONEST signals should be profiled as HONEST."""
        traces = _generate_archetype_traces(AgentType.HONEST, n=50, seed=42)
        profiler = BehavioralProfiler()
        profile = profiler.fit(traces)
        assert profile.dominant_archetype == AgentType.HONEST
        assert profile.archetype_mixture[AgentType.HONEST] > 0.4

    def test_recovers_adversarial_archetype(self):
        """Traces from ADVERSARIAL signals should be profiled as ADVERSARIAL."""
        traces = _generate_archetype_traces(AgentType.ADVERSARIAL, n=50, seed=42)
        profiler = BehavioralProfiler()
        profile = profiler.fit(traces)
        assert profile.dominant_archetype == AgentType.ADVERSARIAL
        assert profile.archetype_mixture[AgentType.ADVERSARIAL] > 0.4

    def test_honest_p_higher_than_adversarial(self):
        """Honest profile should have higher mean p than adversarial."""
        profiler = BehavioralProfiler()
        honest_profile = profiler.fit(
            _generate_archetype_traces(AgentType.HONEST, n=50)
        )
        adversarial_profile = profiler.fit(
            _generate_archetype_traces(AgentType.ADVERSARIAL, n=50)
        )
        assert honest_profile.p_mean > adversarial_profile.p_mean

    def test_mixture_sums_to_one(self):
        traces = _generate_archetype_traces(AgentType.HONEST, n=20)
        profiler = BehavioralProfiler()
        profile = profiler.fit(traces)
        assert sum(profile.archetype_mixture.values()) == pytest.approx(1.0)

    def test_all_archetypes_present_in_mixture(self):
        traces = _generate_archetype_traces(AgentType.HONEST, n=20)
        profiler = BehavioralProfiler()
        profile = profiler.fit(traces)
        for atype in CORE_ARCHETYPES:
            assert atype in profile.archetype_mixture

    def test_empty_traces_raises(self):
        profiler = BehavioralProfiler()
        with pytest.raises(ValueError, match="empty"):
            profiler.fit([])

    def test_single_trace(self):
        """Should work with just one trace."""
        traces = _generate_archetype_traces(AgentType.HONEST, n=1)
        profiler = BehavioralProfiler()
        profile = profiler.fit(traces)
        assert profile.n_traces == 1
        assert sum(profile.archetype_mixture.values()) == pytest.approx(1.0)

    def test_custom_prior_shifts_mixture(self):
        """A strong prior toward HONEST should increase HONEST weight."""
        traces = _generate_archetype_traces(AgentType.OPPORTUNISTIC, n=30, seed=42)

        uniform_profiler = BehavioralProfiler()
        uniform_profile = uniform_profiler.fit(traces)

        biased_profiler = BehavioralProfiler(prior={
            AgentType.HONEST: 0.7,
            AgentType.OPPORTUNISTIC: 0.1,
            AgentType.DECEPTIVE: 0.1,
            AgentType.ADVERSARIAL: 0.1,
        })
        biased_profile = biased_profiler.fit(traces)

        # Biased prior should increase HONEST weight relative to uniform
        assert (
            biased_profile.archetype_mixture[AgentType.HONEST]
            >= uniform_profile.archetype_mixture[AgentType.HONEST]
        )

    def test_n_traces_recorded(self):
        traces = _generate_archetype_traces(AgentType.HONEST, n=25)
        profiler = BehavioralProfiler()
        profile = profiler.fit(traces)
        assert profile.n_traces == 25

    def test_p_bounds(self):
        """p statistics should be within [0, 1]."""
        for atype in CORE_ARCHETYPES:
            traces = _generate_archetype_traces(atype, n=20)
            profiler = BehavioralProfiler()
            profile = profiler.fit(traces)
            assert 0.0 <= profile.p_mean <= 1.0
            assert 0.0 <= profile.p_min <= profile.p_max <= 1.0
            assert profile.p_std >= 0.0

    def test_agent_id_from_first_trace(self):
        traces = _generate_archetype_traces(AgentType.HONEST, n=5)
        profiler = BehavioralProfiler()
        profile = profiler.fit(traces)
        assert profile.agent_id == "agent-honest"

    def test_agent_id_override(self):
        traces = _generate_archetype_traces(AgentType.HONEST, n=5)
        profiler = BehavioralProfiler()
        profile = profiler.fit(traces, agent_id="my-agent")
        assert profile.agent_id == "my-agent"

    def test_deterministic_with_same_input(self):
        traces = _generate_archetype_traces(AgentType.DECEPTIVE, n=30, seed=99)
        profiler = BehavioralProfiler()
        p1 = profiler.fit(traces)
        p2 = profiler.fit(traces)
        assert p1.archetype_mixture == p2.archetype_mixture
        assert p1.p_mean == p2.p_mean


# ---------------------------------------------------------------------------
# BehavioralProfiler.fit_multiple
# ---------------------------------------------------------------------------


class TestFitMultiple:
    def test_separates_agents(self):
        honest_traces = _generate_archetype_traces(AgentType.HONEST, n=20)
        adversarial_traces = _generate_archetype_traces(AgentType.ADVERSARIAL, n=20)
        all_traces = honest_traces + adversarial_traces

        profiler = BehavioralProfiler()
        profiles = profiler.fit_multiple(all_traces)

        assert "agent-honest" in profiles
        assert "agent-adversarial" in profiles
        assert profiles["agent-honest"].dominant_archetype == AgentType.HONEST
        assert profiles["agent-adversarial"].dominant_archetype == AgentType.ADVERSARIAL

    def test_p_ordering_across_agents(self):
        """Honest agents should have higher p_mean than adversarial."""
        honest_traces = _generate_archetype_traces(AgentType.HONEST, n=30)
        adversarial_traces = _generate_archetype_traces(AgentType.ADVERSARIAL, n=30)

        profiler = BehavioralProfiler()
        profiles = profiler.fit_multiple(honest_traces + adversarial_traces)

        assert (
            profiles["agent-honest"].p_mean
            > profiles["agent-adversarial"].p_mean
        )


# ---------------------------------------------------------------------------
# InferredProfile
# ---------------------------------------------------------------------------


class TestInferredProfile:
    def test_safety_summary_low_risk(self):
        profile = InferredProfile(
            agent_id="a",
            archetype_mixture={
                AgentType.HONEST: 0.8,
                AgentType.OPPORTUNISTIC: 0.1,
                AgentType.DECEPTIVE: 0.05,
                AgentType.ADVERSARIAL: 0.05,
            },
            dominant_archetype=AgentType.HONEST,
            p_mean=0.75,
            n_traces=50,
        )
        summary = profile.safety_summary()
        assert "low-risk" in summary

    def test_safety_summary_high_risk(self):
        profile = InferredProfile(
            agent_id="a",
            archetype_mixture={
                AgentType.HONEST: 0.1,
                AgentType.OPPORTUNISTIC: 0.1,
                AgentType.DECEPTIVE: 0.3,
                AgentType.ADVERSARIAL: 0.5,
            },
            dominant_archetype=AgentType.ADVERSARIAL,
            p_mean=0.25,
            n_traces=50,
        )
        summary = profile.safety_summary()
        assert "high-risk" in summary

    def test_safety_summary_moderate_risk(self):
        profile = InferredProfile(
            agent_id="a",
            archetype_mixture={
                AgentType.HONEST: 0.4,
                AgentType.OPPORTUNISTIC: 0.2,
                AgentType.DECEPTIVE: 0.15,
                AgentType.ADVERSARIAL: 0.25,
            },
            dominant_archetype=AgentType.HONEST,
            p_mean=0.50,
            n_traces=50,
        )
        summary = profile.safety_summary()
        assert "moderate-risk" in summary


# ---------------------------------------------------------------------------
# Std helper
# ---------------------------------------------------------------------------


class TestStd:
    def test_zero_for_single_value(self):
        assert _std([5.0]) == 0.0

    def test_zero_for_constant(self):
        assert _std([3.0, 3.0, 3.0]) == pytest.approx(0.0)

    def test_known_values(self):
        # std of [0, 1] = 0.5
        assert _std([0.0, 1.0]) == pytest.approx(0.5)

    def test_empty(self):
        assert _std([]) == 0.0
