"""Tests for the work regime drift module.

Covers:
- WorkRegimeAgent policy state initialization and adaptation
- WorkRegimeAgent uses AgentType.WORK_REGIME (not AgentType.HONEST)
- WorkRegimeMetrics computation (drift, Gini, legitimacy, coalition)
- Scenario loading (baseline and high-stress YAML files)
- End-to-end short run to verify scenario executes
"""

import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pytest

from swarm.agents.work_regime_agent import WorkRegimeAgent
from swarm.metrics.work_regime_metrics import (
    WorkRegimeEpochMetrics,
    WorkRegimeMetrics,
    build_epoch_metrics,
    compute_coalition_strength,
    compute_defection_rate,
    compute_drift_index,
    compute_gini,
    compute_legitimacy_score,
    compute_quality_degradation,
    compute_work_regime_metrics,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType, SoftInteraction
from swarm.scenarios import load_scenario

# ======================================================================
# WorkRegimeAgent tests
# ======================================================================


class TestWorkRegimeAgent:
    """Tests for the WorkRegimeAgent policy state and adaptation."""

    def test_agent_type_is_work_regime(self):
        """WorkRegimeAgent must use AgentType.WORK_REGIME, not AgentType.HONEST.

        This prevents work_regime workers from being unintentionally exempted
        from dropout in governance logic that checks agent_type == 'honest'.
        """
        agent = WorkRegimeAgent(agent_id="w0", rng=random.Random(42))
        assert agent.agent_type == AgentType.WORK_REGIME
        assert agent.agent_type != AgentType.HONEST

    def test_agent_type_value_is_work_regime_string(self):
        """agent_type.value must be 'work_regime' for PerturbationEngine exemption checks."""
        agent = WorkRegimeAgent(agent_id="w0b", rng=random.Random(42))
        assert agent.agent_type.value == "work_regime"

    def test_init_default_policy(self):
        """Agent starts with default policy state."""
        agent = WorkRegimeAgent(agent_id="w1", rng=random.Random(42))
        assert agent.compliance_propensity == 0.8
        assert agent.cooperation_threshold == 0.3
        assert agent.redistribution_preference == 0.2
        assert agent.exit_propensity == 0.05
        assert agent.grievance == 0.0

    def test_init_custom_config(self):
        """Agent respects config overrides."""
        agent = WorkRegimeAgent(
            agent_id="w2",
            config={
                "compliance_propensity": 0.5,
                "cooperation_threshold": 0.6,
                "redistribution_preference": 0.4,
                "exit_propensity": 0.1,
            },
            rng=random.Random(42),
        )
        assert agent.compliance_propensity == 0.5
        assert agent.cooperation_threshold == 0.6

    def test_policy_drift_initially_zero(self):
        """Drift should be zero before any adaptation."""
        agent = WorkRegimeAgent(agent_id="w3", rng=random.Random(42))
        assert agent.policy_drift() == 0.0

    def test_adapt_under_stress_increases_drift(self):
        """Adaptation under stress should increase drift from initial."""
        agent = WorkRegimeAgent(
            agent_id="w4",
            config={"adapt_rate": 0.1},
            rng=random.Random(42),
        )
        # Simulate 10 epochs of stress
        for _ in range(10):
            agent.adapt_policy(
                avg_payoff=-1.0,       # underpaid
                peer_avg_payoff=2.0,   # peers doing well
                eval_noise=0.6,        # noisy evaluation
                workload_pressure=0.9, # overworked
            )
        assert agent.policy_drift() > 0.0
        assert agent.compliance_propensity < 0.8  # dropped
        assert agent.redistribution_preference > 0.2  # rose
        assert agent.grievance > 0.0

    def test_adapt_under_good_conditions_low_drift(self):
        """Good conditions should produce minimal drift."""
        agent = WorkRegimeAgent(
            agent_id="w5",
            config={"adapt_rate": 0.1},
            rng=random.Random(42),
        )
        for _ in range(10):
            agent.adapt_policy(
                avg_payoff=3.0,        # well paid
                peer_avg_payoff=2.5,   # above average
                eval_noise=0.05,       # fair evaluation
                workload_pressure=0.3, # light workload
            )
        # Drift should be small relative to stress scenario (grievance stays near 0)
        assert agent.policy_drift() < 0.2
        assert agent.compliance_propensity >= 0.6

    def test_policy_snapshot(self):
        """Snapshot returns all expected keys."""
        agent = WorkRegimeAgent(agent_id="w6", rng=random.Random(42))
        snap = agent.policy_snapshot()
        expected_keys = {
            "compliance_propensity",
            "cooperation_threshold",
            "redistribution_preference",
            "exit_propensity",
            "grievance",
            "drift",
        }
        assert set(snap.keys()) == expected_keys

    def test_exit_propensity_rises_with_grievance(self):
        """Sustained grievance should raise exit propensity."""
        agent = WorkRegimeAgent(
            agent_id="w7",
            config={"adapt_rate": 0.15},
            rng=random.Random(42),
        )
        for _ in range(20):
            agent.adapt_policy(
                avg_payoff=-2.0,
                peer_avg_payoff=3.0,
                eval_noise=0.8,
                workload_pressure=0.95,
            )
        assert agent.exit_propensity > agent._initial_exit

    def test_policy_state_bounded(self):
        """All policy variables should stay in [0, 1]."""
        agent = WorkRegimeAgent(
            agent_id="w8",
            config={"adapt_rate": 0.3},  # aggressive
            rng=random.Random(42),
        )
        for _ in range(50):
            agent.adapt_policy(
                avg_payoff=-5.0,
                peer_avg_payoff=5.0,
                eval_noise=1.0,
                workload_pressure=1.0,
            )
        assert 0.0 <= agent.compliance_propensity <= 1.0
        assert 0.0 <= agent.cooperation_threshold <= 1.0
        assert 0.0 <= agent.redistribution_preference <= 1.0
        assert 0.0 <= agent.exit_propensity <= 1.0

    def test_grievance_capped(self):
        """Grievance should never exceed the soft cap."""
        agent = WorkRegimeAgent(
            agent_id="w9",
            config={"adapt_rate": 0.3},
            rng=random.Random(42),
        )
        for _ in range(100):
            agent.adapt_policy(
                avg_payoff=-10.0,
                peer_avg_payoff=10.0,
                eval_noise=1.0,
                workload_pressure=1.0,
            )
        assert agent.grievance <= agent._grievance_cap

    def test_recent_payoffs_bounded(self):
        """Recent payoff tracking should not grow beyond max_recent."""
        agent = WorkRegimeAgent(agent_id="w10", rng=random.Random(42))
        for i in range(50):
            agent.adapt_policy(
                avg_payoff=float(i),
                peer_avg_payoff=0.0,
                eval_noise=0.1,
                workload_pressure=0.5,
            )
        assert len(agent._recent_payoffs) == agent._max_recent
        assert len(agent._recent_eval_noise) == agent._max_recent


# ======================================================================
# Work regime metrics tests
# ======================================================================


class TestWorkRegimeMetrics:
    """Tests for work regime metric computations."""

    def test_gini_perfect_equality(self):
        """Gini of equal values should be 0."""
        assert compute_gini([10.0, 10.0, 10.0, 10.0]) == 0.0

    def test_gini_max_inequality(self):
        """Gini with one person having all: exact value is 0.75 for n=4."""
        gini = compute_gini([0.0, 0.0, 0.0, 100.0])
        assert gini == pytest.approx(0.75)

    def test_gini_empty(self):
        """Empty list returns 0."""
        assert compute_gini([]) == 0.0

    def test_legitimacy_perfect(self):
        """Perfect procedural justice → score = 1.0."""
        score = compute_legitimacy_score(
            audit_precision=1.0,
            eval_noise=0.0,
            appeals_available=True,
            explanation_provided=True,
        )
        assert score == pytest.approx(1.0)

    def test_legitimacy_worst(self):
        """Worst procedural justice → score near 0."""
        score = compute_legitimacy_score(
            audit_precision=0.0,
            eval_noise=1.0,
            appeals_available=False,
            explanation_provided=False,
        )
        assert score == pytest.approx(0.0)

    def test_legitimacy_partial(self):
        """Partial justice → score in middle range."""
        score = compute_legitimacy_score(
            audit_precision=0.5,
            eval_noise=0.5,
            appeals_available=True,
            explanation_provided=False,
        )
        assert 0.2 < score < 0.8

    def test_drift_index_no_drift(self):
        """Zero drift snapshots → (0, 0)."""
        snapshots = [{"drift": 0.0}, {"drift": 0.0}]
        mean, mx = compute_drift_index(snapshots)
        assert mean == 0.0
        assert mx == 0.0

    def test_drift_index_with_drift(self):
        """Non-zero drift is correctly aggregated."""
        snapshots = [{"drift": 0.1}, {"drift": 0.5}, {"drift": 0.3}]
        mean, mx = compute_drift_index(snapshots)
        assert mean == pytest.approx(0.3)
        assert mx == pytest.approx(0.5)

    def test_coalition_strength_triangle(self):
        """Three agents all connected → clustering = 1.0."""
        pairs = [("a", "b"), ("b", "c"), ("a", "c")]
        strength = compute_coalition_strength(pairs, ["a", "b", "c"])
        assert strength == pytest.approx(1.0)

    def test_coalition_strength_no_triangles(self):
        """Star graph (no triangles among leaves) → clustering = 0."""
        pairs = [("hub", "a"), ("hub", "b"), ("hub", "c")]
        strength = compute_coalition_strength(
            pairs, ["hub", "a", "b", "c"]
        )
        # Only hub has neighbors; a,b,c don't connect to each other
        assert strength == 0.0

    def test_defection_rate(self):
        """Defection rate computed correctly."""
        interactions = [
            _make_interaction(accepted=True),
            _make_interaction(accepted=False),
            _make_interaction(accepted=True),
            _make_interaction(accepted=False),
        ]
        rate = compute_defection_rate(interactions)
        assert rate == pytest.approx(0.5)

    def test_quality_degradation(self):
        """Quality degradation from baseline."""
        interactions = [
            _make_interaction(p=0.4, accepted=True),
            _make_interaction(p=0.5, accepted=True),
        ]
        deg = compute_quality_degradation(interactions, baseline_p=0.7)
        assert deg == pytest.approx(0.25)

    def test_build_epoch_metrics(self):
        """build_epoch_metrics returns well-formed dataclass."""
        snapshots = {
            "w1": {
                "compliance_propensity": 0.6,
                "cooperation_threshold": 0.4,
                "redistribution_preference": 0.3,
                "exit_propensity": 0.1,
                "grievance": 0.5,
                "drift": 0.15,
            },
            "w2": {
                "compliance_propensity": 0.7,
                "cooperation_threshold": 0.35,
                "redistribution_preference": 0.25,
                "exit_propensity": 0.08,
                "grievance": 0.3,
                "drift": 0.1,
            },
        }
        interactions = [_make_interaction(p=0.6, accepted=True)]
        payoffs = {"w1": 5.0, "w2": 15.0}

        metrics = build_epoch_metrics(
            epoch=5,
            policy_snapshots=snapshots,
            interactions=interactions,
            agent_payoffs=payoffs,
            strike_count=1,
            total_agents=10,
        )
        assert isinstance(metrics, WorkRegimeEpochMetrics)
        assert metrics.epoch == 5
        assert metrics.strike_rate == pytest.approx(0.1)
        assert metrics.mean_drift_index == pytest.approx(0.125)
        assert metrics.gini_payoff > 0.0

    def test_build_epoch_metrics_empty_snapshots_warns(self, caplog):
        """build_epoch_metrics warns and returns zero aggregate means when policy_snapshots is empty."""
        import logging

        with caplog.at_level(logging.WARNING, logger="swarm.metrics.work_regime_metrics"):
            metrics = build_epoch_metrics(
                epoch=3,
                policy_snapshots={},
                interactions=[],
                agent_payoffs={},
            )

        assert any("no policy snapshots" in m for m in caplog.messages)
        assert isinstance(metrics, WorkRegimeEpochMetrics)
        assert metrics.mean_compliance == pytest.approx(0.0)
        assert metrics.mean_cooperation_threshold == pytest.approx(0.0)
        assert metrics.mean_redistribution_pref == pytest.approx(0.0)
        assert metrics.mean_exit_propensity == pytest.approx(0.0)
        assert metrics.mean_grievance == pytest.approx(0.0)


# ======================================================================
# Scenario loading tests
# ======================================================================


class TestWorkRegimeScenarioLoading:
    """Tests for loading work regime scenario YAML files."""

    def test_load_baseline(self):
        """Baseline scenario loads without errors."""
        path = Path("scenarios/work_regime_drift/baseline.yaml")
        scenario = load_scenario(path)
        assert scenario.scenario_id == "work_regime_drift_baseline"
        assert scenario.motif == "work_regime"
        assert len(scenario.agent_specs) >= 4  # manager + workers + opps + adversarial

    def test_load_high_stress(self):
        """High-stress scenario loads without errors."""
        path = Path("scenarios/work_regime_drift/high_stress.yaml")
        scenario = load_scenario(path)
        assert scenario.scenario_id == "work_regime_drift_high_stress"
        # Unequal pay: theta = 0.8
        assert scenario.orchestrator_config.payoff_config.theta == 0.8

    def test_baseline_has_work_regime_agents(self):
        """Baseline spec includes work_regime agent type."""
        path = Path("scenarios/work_regime_drift/baseline.yaml")
        scenario = load_scenario(path)
        types = [s["type"] for s in scenario.agent_specs]
        assert "work_regime" in types

    def test_high_stress_has_perturbations(self):
        """High-stress scenario has parameter shocks configured."""
        path = Path("scenarios/work_regime_drift/high_stress.yaml")
        scenario = load_scenario(path)
        perturb = scenario.orchestrator_config.perturbation_config
        assert perturb is not None
        assert perturb.parameter_shocks.enabled

    def test_baseline_governance_config(self):
        """Baseline has expected governance settings."""
        path = Path("scenarios/work_regime_drift/baseline.yaml")
        scenario = load_scenario(path)
        gov = scenario.orchestrator_config.governance_config
        assert gov is not None
        assert gov.audit_enabled
        assert gov.circuit_breaker_enabled
        assert gov.collusion_detection_enabled

    def test_create_agents_from_baseline(self):
        """Agents can be instantiated from baseline scenario."""
        from swarm.scenarios import create_agents

        path = Path("scenarios/work_regime_drift/baseline.yaml")
        scenario = load_scenario(path)
        agents = create_agents(scenario.agent_specs, seed=42)

        # Derive expected totals from the YAML spec itself
        expected_total = sum(s.get("count", 1) for s in scenario.agent_specs)
        assert len(agents) == expected_total

        expected_wr = sum(
            s.get("count", 1)
            for s in scenario.agent_specs
            if s["type"] == "work_regime"
        )
        # Check we got WorkRegimeAgent instances
        from swarm.agents.work_regime_agent import WorkRegimeAgent as WRA
        wr_agents = [a for a in agents if isinstance(a, WRA)]
        assert len(wr_agents) == expected_wr

    def test_work_regime_agents_have_correct_type(self):
        """WorkRegimeAgent instances created from scenario use AgentType.WORK_REGIME."""
        from swarm.scenarios import create_agents

        path = Path("scenarios/work_regime_drift/baseline.yaml")
        scenario = load_scenario(path)
        agents = create_agents(scenario.agent_specs, seed=42)

        from swarm.agents.work_regime_agent import WorkRegimeAgent as WRA
        for agent in agents:
            if isinstance(agent, WRA):
                assert agent.agent_type == AgentType.WORK_REGIME, (
                    f"Expected AgentType.WORK_REGIME but got {agent.agent_type} "
                    f"for agent {agent.agent_id}. work_regime workers must not "
                    "be typed as HONEST to avoid unintentional dropout exemptions."
                )

    def test_build_and_short_run_baseline(self):
        """Baseline scenario can build an orchestrator and run 2 epochs."""
        from swarm.scenarios import build_orchestrator

        path = Path("scenarios/work_regime_drift/baseline.yaml")
        scenario = load_scenario(path)

        # Override to short run for speed
        scenario.orchestrator_config.n_epochs = 2
        scenario.orchestrator_config.steps_per_epoch = 3

        orchestrator = build_orchestrator(scenario)
        metrics = orchestrator.run()
        assert len(metrics) == 2
        assert metrics[0].total_interactions >= 0


# ======================================================================
# Helpers
# ======================================================================


def _make_interaction(
    *,
    p: float = 0.5,
    accepted: bool = True,
) -> SoftInteraction:
    """Create a minimal SoftInteraction for testing."""
    return SoftInteraction(
        interaction_id="test",
        initiator="a",
        counterparty="b",
        accepted=accepted,
        task_progress_delta=0.5,
        rework_count=0,
        verifier_rejections=0,
        tool_misuse_flags=0,
        counterparty_engagement_delta=0.3,
        v_hat=0.0,
        p=p,
        tau=0.0,
        c_a=0.0,
        c_b=0.0,
        r_a=0.0,
        r_b=0.0,
    )


# ======================================================================
# Coalition strength tests (from #315 — O(k) set-intersection impl)
# ======================================================================


def _pairs(*edges: Tuple[str, str]) -> List[Tuple[str, str]]:
    return list(edges)


def make_interaction(initiator: str, counterparty: str) -> SoftInteraction:
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        interaction_type=InteractionType.COLLABORATION,
        accepted=True,
        p=0.8,
        timestamp=datetime(2024, 1, 1),
    )


class TestComputeCoalitionStrengthEdgeCases:
    def test_empty_pairs_returns_zero(self):
        assert compute_coalition_strength([], ["a", "b", "c"]) == 0.0

    def test_fewer_than_three_agents_returns_zero(self):
        assert compute_coalition_strength([("a", "b")], ["a", "b"]) == 0.0

    def test_no_node_has_degree_two_returns_zero(self):
        # Star topology: centre has degree 3, leaves degree 1
        pairs = _pairs(("centre", "a"), ("centre", "b"), ("centre", "c"))
        agents = ["centre", "a", "b", "c"]
        # Leaves can't form triangles; centre can but a-b, a-c, b-c edges missing
        result = compute_coalition_strength(pairs, agents)
        assert result == 0.0

    def test_self_loops_ignored(self):
        # Self-loop pairs should not be added to the adjacency list
        pairs = _pairs(("a", "a"), ("b", "b"), ("a", "b"), ("b", "c"), ("a", "c"))
        agents = ["a", "b", "c"]
        result = compute_coalition_strength(pairs, agents)
        assert 0.0 <= result <= 1.0

    def test_pairs_outside_agent_ids_ignored(self):
        pairs = _pairs(("x", "y"), ("a", "b"), ("b", "c"), ("a", "c"))
        agents = ["a", "b", "c"]  # x, y not in agent_ids
        result = compute_coalition_strength(pairs, agents)
        # x-y edge ignored; a-b-c triangle should yield perfect clustering
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_coalition_strength — correctness
# ---------------------------------------------------------------------------


class TestComputeCoalitionStrengthCorrectness:
    def test_complete_triangle_returns_one(self):
        """Three agents all connected → clustering = 1.0."""
        pairs = _pairs(("a", "b"), ("b", "c"), ("a", "c"))
        result = compute_coalition_strength(pairs, ["a", "b", "c"])
        assert result == pytest.approx(1.0)

    def test_complete_graph_k4_returns_one(self):
        """K4 complete graph: every node's neighbours are all connected."""
        agents = ["a", "b", "c", "d"]
        pairs = _pairs(
            ("a", "b"), ("a", "c"), ("a", "d"),
            ("b", "c"), ("b", "d"),
            ("c", "d"),
        )
        result = compute_coalition_strength(pairs, agents)
        assert result == pytest.approx(1.0)

    def test_line_graph_returns_zero(self):
        """a-b-c-d line: no node has two neighbours that are also connected."""
        pairs = _pairs(("a", "b"), ("b", "c"), ("c", "d"))
        result = compute_coalition_strength(pairs, ["a", "b", "c", "d"])
        # b and c have degree 2 but their neighbours are not connected to each other
        assert result == pytest.approx(0.0)

    def test_result_in_unit_interval(self):
        """Coalition strength is always in [0, 1]."""
        pairs = _pairs(
            ("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"),
            ("a", "c"),  # one diagonal only
        )
        result = compute_coalition_strength(pairs, ["a", "b", "c", "d"])
        assert 0.0 <= result <= 1.0

    def test_duplicate_pairs_handled(self):
        """Repeated edges don't inflate the clustering coefficient."""
        pairs = _pairs(
            ("a", "b"), ("a", "b"), ("b", "c"), ("a", "c"),
        )
        result_dup = compute_coalition_strength(pairs, ["a", "b", "c"])
        pairs_dedup = _pairs(("a", "b"), ("b", "c"), ("a", "c"))
        result_dedup = compute_coalition_strength(pairs_dedup, ["a", "b", "c"])
        assert result_dup == pytest.approx(result_dedup)

    def test_directed_pairs_treated_as_undirected(self):
        """(a,b) and (b,a) should produce the same graph as just (a,b)."""
        pairs_bi = _pairs(("a", "b"), ("b", "a"), ("b", "c"), ("a", "c"))
        pairs_uni = _pairs(("a", "b"), ("b", "c"), ("a", "c"))
        result_bi = compute_coalition_strength(pairs_bi, ["a", "b", "c"])
        result_uni = compute_coalition_strength(pairs_uni, ["a", "b", "c"])
        assert result_bi == pytest.approx(result_uni)


# ---------------------------------------------------------------------------
# compute_coalition_strength — sampling
# ---------------------------------------------------------------------------


class TestComputeCoalitionStrengthSampling:
    def _build_clique(self, n: int) -> Tuple[List[str], List[Tuple[str, str]]]:
        agents = [f"a{i}" for i in range(n)]
        pairs = [
            (agents[i], agents[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        return agents, pairs

    def test_sampling_triggered_for_large_network(self):
        """When n_eligible > max_sample_nodes the function still returns a value."""
        agents, pairs = self._build_clique(60)
        result = compute_coalition_strength(pairs, agents, max_sample_nodes=10)
        # K60 is complete, so clustering == 1.0 regardless of which nodes sampled
        assert result == pytest.approx(1.0)

    def test_no_sampling_when_within_limit(self):
        agents, pairs = self._build_clique(10)
        result = compute_coalition_strength(pairs, agents, max_sample_nodes=500)
        assert result == pytest.approx(1.0)

    def test_sampling_is_deterministic(self):
        """Same inputs → same result (no hidden randomness)."""
        agents, pairs = self._build_clique(40)
        r1 = compute_coalition_strength(pairs, agents, max_sample_nodes=5)
        r2 = compute_coalition_strength(pairs, agents, max_sample_nodes=5)
        assert r1 == r2

    def test_max_sample_nodes_one_does_not_crash(self):
        agents, pairs = self._build_clique(20)
        result = compute_coalition_strength(pairs, agents, max_sample_nodes=1)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_work_regime_metrics
# ---------------------------------------------------------------------------


class TestComputeWorkRegimeMetrics:
    def test_empty_interactions_returns_defaults(self):
        result = compute_work_regime_metrics([])
        assert isinstance(result, WorkRegimeMetrics)
        assert result.coalition_strength == 0.0
        assert result.regime_label == "neutral"

    def test_cooperative_label_for_dense_network(self):
        agents = ["a", "b", "c", "d"]
        interactions = [
            make_interaction("a", "b"),
            make_interaction("b", "c"),
            make_interaction("a", "c"),
            make_interaction("b", "d"),
            make_interaction("a", "d"),
            make_interaction("c", "d"),
        ]
        result = compute_work_regime_metrics(interactions, agent_ids=agents)
        assert result.regime_label == "cooperative"
        assert result.coalition_strength >= 0.4

    def test_competitive_label_for_sparse_network(self):
        agents = ["a", "b", "c", "d", "e"]
        # Line graph: a-b-c-d-e, no triangles
        interactions = [
            make_interaction("a", "b"),
            make_interaction("b", "c"),
            make_interaction("c", "d"),
            make_interaction("d", "e"),
        ]
        result = compute_work_regime_metrics(
            interactions,
            agent_ids=agents,
            competitive_threshold=0.05,
        )
        assert result.coalition_strength == pytest.approx(0.0)
        assert result.regime_label == "competitive"

    def test_inferred_agent_ids(self):
        interactions = [
            make_interaction("x", "y"),
            make_interaction("y", "z"),
            make_interaction("x", "z"),
        ]
        result = compute_work_regime_metrics(interactions)  # no agent_ids
        assert result.coalition_strength == pytest.approx(1.0)

    def test_per_agent_clustering_populated(self):
        agents = ["a", "b", "c"]
        interactions = [
            make_interaction("a", "b"),
            make_interaction("b", "c"),
            make_interaction("a", "c"),
        ]
        result = compute_work_regime_metrics(interactions, agent_ids=agents)
        # All three agents form a triangle — each should have cc == 1.0
        for agent, cc in result.per_agent_clustering.items():
            assert cc == pytest.approx(1.0), f"agent {agent} has cc={cc}"

    def test_n_nodes_sampled_at_most_max_sample_nodes(self):
        n = 30
        agents = [f"a{i}" for i in range(n)]
        interactions = [
            make_interaction(agents[i], agents[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        result = compute_work_regime_metrics(
            interactions, agent_ids=agents, max_sample_nodes=5
        )
        assert result.n_nodes_sampled <= 5
        assert result.n_eligible_nodes == n
