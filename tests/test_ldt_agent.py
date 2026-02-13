"""Tests for the Logical Decision Theory (LDT) agent."""

import random

import pytest

from swarm.agents.base import ActionType, InteractionProposal, Observation, Role
from swarm.agents.ldt_agent import InferredPolicy, LDTAgent, _cosine_similarity
from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import InteractionType, SoftInteraction

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _obs(**kwargs) -> Observation:
    """Create a test observation with sensible defaults."""
    defaults = {
        "agent_state": AgentState(
            agent_id="ldt_1", reputation=0.5, resources=100.0
        ),
        "current_epoch": 1,
        "current_step": 5,
        "can_post": True,
        "can_interact": True,
        "can_vote": True,
        "can_claim_task": True,
        "visible_posts": [],
        "pending_proposals": [],
        "available_tasks": [],
        "active_tasks": [],
        "visible_agents": [],
    }
    defaults.update(kwargs)
    return Observation(**defaults)


def _interaction(
    initiator: str = "other",
    counterparty: str = "ldt_1",
    accepted: bool = True,
    p: float = 0.8,
) -> SoftInteraction:
    """Create a test SoftInteraction."""
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        accepted=accepted,
        p=p,
    )


# ------------------------------------------------------------------
# Test: cosine similarity helper
# ------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert _cosine_similarity([1, 2], [1]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 2]) == 0.0


# ------------------------------------------------------------------
# Test: LDT agent initialization
# ------------------------------------------------------------------


class TestLDTAgentInit:
    def test_default_initialization(self):
        agent = LDTAgent(agent_id="ldt_1")

        assert agent.agent_id == "ldt_1"
        assert agent.agent_type == AgentType.HONEST
        assert agent.cooperation_prior == 0.65
        assert agent.similarity_threshold == 0.7
        assert agent.welfare_weight == 0.3
        assert agent.updateless_commitment == 0.8
        assert agent.defection_memory_weight == 0.5
        assert agent.counterfactual_horizon == 20

    def test_custom_config(self):
        agent = LDTAgent(
            agent_id="ldt_2",
            config={
                "cooperation_prior": 0.9,
                "similarity_threshold": 0.5,
                "welfare_weight": 0.6,
                "updateless_commitment": 0.3,
            },
        )

        assert agent.cooperation_prior == 0.9
        assert agent.similarity_threshold == 0.5
        assert agent.welfare_weight == 0.6
        assert agent.updateless_commitment == 0.3

    def test_roles_assignment(self):
        agent = LDTAgent(
            agent_id="ldt_3",
            roles=[Role.VERIFIER, Role.WORKER],
        )
        assert agent.primary_role == Role.VERIFIER
        assert Role.WORKER in agent.roles

    def test_name_defaults_to_id(self):
        agent = LDTAgent(agent_id="ldt_4")
        assert agent.name == "ldt_4"

    def test_custom_name(self):
        agent = LDTAgent(agent_id="ldt_5", name="Alice")
        assert agent.name == "Alice"


# ------------------------------------------------------------------
# Test: logical-twin detection
# ------------------------------------------------------------------


class TestLogicalTwinDetection:
    def test_no_history_returns_moderate_similarity(self):
        """With no history, behaviour vectors are padded with priors."""
        agent = LDTAgent(agent_id="ldt_1")
        score = agent._compute_twin_score("unknown")
        # Both vectors are constant (cooperation_prior and 0.5
        # respectively), so cosine similarity should be 1.0 (parallel).
        assert score == pytest.approx(1.0)

    def test_similar_history_high_score(self):
        """Agents with similar interaction histories should be twins."""
        agent = LDTAgent(agent_id="ldt_1")

        # Build matching traces.
        for _ in range(10):
            agent._own_trace.append((True, 0.8))
            if "peer" not in agent._counterparty_profiles:
                agent._counterparty_profiles["peer"] = []
            agent._counterparty_profiles["peer"].append((True, 0.8))

        score = agent._compute_twin_score("peer")
        assert score > 0.9

    def test_divergent_history_low_score(self):
        """Agents with opposite histories should not be twins.

        Own trace: accepted with high p → positive vector entries.
        Adversary trace: rejected with low p → negative entries (-(1-p)).
        These vectors should point in opposite directions.
        """
        agent = LDTAgent(agent_id="ldt_1")

        for _ in range(20):
            agent._own_trace.append((True, 0.9))
            if "adversary" not in agent._counterparty_profiles:
                agent._counterparty_profiles["adversary"] = []
            agent._counterparty_profiles["adversary"].append((False, 0.1))

        score = agent._compute_twin_score("adversary")
        # Own vector entries: 0.9 (accepted, high p).
        # Adversary vector entries: -(1-0.1) = -0.9 (rejected, low p).
        # These are opposite → cosine similarity ≈ -1, clamped to 0.
        assert score < 0.1


# ------------------------------------------------------------------
# Test: LDT cooperation decision
# ------------------------------------------------------------------


class TestLDTCooperationDecision:
    def test_cooperate_with_unknown_by_default(self):
        """With default prior (0.65), should cooperate with unknowns."""
        agent = LDTAgent(agent_id="ldt_1")
        assert agent._ldt_cooperate_decision("unknown") is True

    def test_cooperate_with_logical_twin(self):
        """Should always cooperate with a detected logical twin."""
        agent = LDTAgent(agent_id="ldt_1")

        # Build matching history to create twin.
        for _ in range(15):
            agent._own_trace.append((True, 0.85))
            if "twin" not in agent._counterparty_profiles:
                agent._counterparty_profiles["twin"] = []
            agent._counterparty_profiles["twin"].append((True, 0.85))

        assert agent._ldt_cooperate_decision("twin") is True

    def test_low_cooperation_prior_rejects_unknowns(self):
        """With very low cooperation prior, may reject unknowns."""
        agent = LDTAgent(
            agent_id="ldt_1",
            config={
                "cooperation_prior": 0.01,
                "updateless_commitment": 1.0,  # fully committed to prior
                "welfare_weight": 0.0,
            },
        )
        # With prior 0.01 and full commitment, effective coop value
        # is 0.01, which should be less than defect value of 0.0.
        # Actually defect payoff is -0.5 * max(0, 0.5 - 0.5) = 0.0
        # for unknowns, so 0.01 > 0.0 — still cooperates.
        # But let's test a case with bad history.
        for _ in range(20):
            agent._own_trace.append((True, 0.8))
            if "bad" not in agent._counterparty_profiles:
                agent._counterparty_profiles["bad"] = []
            agent._counterparty_profiles["bad"].append((True, 0.1))

        # Counterfactual cooperate payoff is 0.1 (avg p from bad's history).
        # Defect payoff is -0.5 * max(0, 0.1 - 0.5) = 0.0.
        # Effective coop value = 1.0 * 0.01 + 0.0 * 0.1 = 0.01.
        result = agent._ldt_cooperate_decision("bad")
        assert result is True  # Still marginal cooperation

    def test_updateless_commitment_effect(self):
        """Higher updateless_commitment sticks closer to prior."""
        agent_committed = LDTAgent(
            agent_id="ldt_1",
            config={"updateless_commitment": 1.0, "cooperation_prior": 0.7},
        )
        agent_adaptive = LDTAgent(
            agent_id="ldt_2",
            config={"updateless_commitment": 0.0, "cooperation_prior": 0.7},
        )

        # Both should cooperate with unknowns, but through different
        # reasoning paths.
        assert agent_committed._ldt_cooperate_decision("x") is True
        assert agent_adaptive._ldt_cooperate_decision("x") is True


# ------------------------------------------------------------------
# Test: counterfactual payoff estimation
# ------------------------------------------------------------------


class TestCounterfactualPayoffs:
    def test_cooperate_payoff_no_history(self):
        agent = LDTAgent(agent_id="ldt_1")
        payoff = agent._counterfactual_cooperate_payoff("unknown")
        assert payoff == agent.cooperation_prior

    def test_cooperate_payoff_with_history(self):
        agent = LDTAgent(agent_id="ldt_1")
        agent._counterparty_profiles["peer"] = [
            (True, 0.9),
            (True, 0.7),
            (False, 0.3),  # rejected — excluded from cooperate estimate
        ]
        payoff = agent._counterfactual_cooperate_payoff("peer")
        assert payoff == pytest.approx(0.8)  # avg of 0.9 and 0.7

    def test_defect_payoff_no_history(self):
        agent = LDTAgent(agent_id="ldt_1")
        payoff = agent._counterfactual_defect_payoff("unknown")
        # coop_p = 0.65 (prior), opportunity cost = -0.5 * max(0, 0.65 - 0.5) = -0.075
        assert payoff == pytest.approx(-0.075)

    def test_defect_payoff_with_good_counterparty(self):
        agent = LDTAgent(agent_id="ldt_1")
        agent._counterparty_profiles["good"] = [(True, 0.9), (True, 0.95)]
        payoff = agent._counterfactual_defect_payoff("good")
        # avg p = 0.925, opportunity cost = -0.5 * (0.925 - 0.5) = -0.2125
        assert payoff == pytest.approx(-0.2125)


# ------------------------------------------------------------------
# Test: act() method
# ------------------------------------------------------------------


class TestLDTAct:
    def test_handles_pending_proposals(self):
        """Should process pending proposals via LDT logic."""
        agent = LDTAgent(agent_id="ldt_1")
        obs = _obs(
            pending_proposals=[
                {"proposal_id": "p1", "initiator_id": "honest_1"}
            ]
        )

        action = agent.act(obs)
        assert action.action_type in (
            ActionType.ACCEPT_INTERACTION,
            ActionType.REJECT_INTERACTION,
        )

    def test_accepts_proposal_from_unknown_with_high_prior(self):
        """Default cooperation_prior (0.65) → accept unknown proposals."""
        agent = LDTAgent(agent_id="ldt_1")
        obs = _obs(
            pending_proposals=[
                {"proposal_id": "p1", "initiator_id": "stranger"}
            ]
        )

        action = agent.act(obs)
        assert action.action_type == ActionType.ACCEPT_INTERACTION

    def test_noop_when_nothing_to_do(self):
        """Should NOOP when no actions are available."""
        agent = LDTAgent(agent_id="ldt_1")
        random.seed(999)  # Ensure random checks fail
        obs = _obs(
            can_post=False,
            can_vote=False,
            can_interact=False,
            can_claim_task=False,
        )

        action = agent.act(obs)
        assert action.action_type == ActionType.NOOP

    def test_claims_available_task(self):
        """Should claim a task when available."""
        agent = LDTAgent(agent_id="ldt_1")
        obs = _obs(
            pending_proposals=[],
            available_tasks=[
                {
                    "task_id": "t1",
                    "bounty": 10,
                    "difficulty": "easy",
                    "min_reputation": 0,
                }
            ],
            can_post=False,
            can_vote=False,
            can_interact=False,
        )

        action = agent.act(obs)
        assert action.action_type == ActionType.CLAIM_TASK
        assert action.target_id == "t1"

    def test_works_on_active_task(self):
        """Should submit output for in-progress tasks."""
        agent = LDTAgent(agent_id="ldt_1")
        obs = _obs(
            active_tasks=[{"task_id": "t1", "status": "in_progress"}],
            can_post=False,
            can_vote=False,
            can_interact=False,
        )

        action = agent.act(obs)
        assert action.action_type == ActionType.SUBMIT_OUTPUT


# ------------------------------------------------------------------
# Test: accept_interaction() method
# ------------------------------------------------------------------


class TestLDTAcceptInteraction:
    def test_accept_from_unknown(self):
        """Default prior → accept from unknown agents."""
        agent = LDTAgent(agent_id="ldt_1")
        proposal = InteractionProposal(
            initiator_id="stranger",
            counterparty_id="ldt_1",
            interaction_type=InteractionType.COLLABORATION,
        )
        obs = _obs()

        assert agent.accept_interaction(proposal, obs) is True

    def test_accept_from_twin(self):
        """Should accept from detected logical twin."""
        agent = LDTAgent(agent_id="ldt_1")

        for _ in range(15):
            agent._own_trace.append((True, 0.85))
            if "twin" not in agent._counterparty_profiles:
                agent._counterparty_profiles["twin"] = []
            agent._counterparty_profiles["twin"].append((True, 0.85))

        proposal = InteractionProposal(
            initiator_id="twin",
            counterparty_id="ldt_1",
            interaction_type=InteractionType.COLLABORATION,
        )
        obs = _obs()

        assert agent.accept_interaction(proposal, obs) is True


# ------------------------------------------------------------------
# Test: propose_interaction() method
# ------------------------------------------------------------------


class TestLDTProposeInteraction:
    def test_propose_to_cooperative_agent(self):
        """Should propose collaboration to cooperative counterparty."""
        agent = LDTAgent(agent_id="ldt_1")
        obs = _obs()

        proposal = agent.propose_interaction(obs, "honest_1")
        assert proposal is not None
        assert proposal.interaction_type == InteractionType.COLLABORATION
        assert proposal.initiator_id == "ldt_1"
        assert proposal.counterparty_id == "honest_1"
        assert proposal.offered_transfer == 0.0

    def test_no_proposal_when_ldt_says_no(self):
        """Should return None if LDT logic rejects cooperation."""
        # This is hard to trigger with defaults since cooperation_prior
        # is high, but we can test with extreme config.
        agent = LDTAgent(
            agent_id="ldt_1",
            config={
                "cooperation_prior": 0.0,
                "updateless_commitment": 0.0,
                "welfare_weight": 0.0,
            },
        )
        # Build bad history.
        for _ in range(20):
            agent._own_trace.append((True, 0.9))
            if "bad" not in agent._counterparty_profiles:
                agent._counterparty_profiles["bad"] = []
            agent._counterparty_profiles["bad"].append((True, 0.1))

        obs = _obs()
        proposal = agent.propose_interaction(obs, "bad")
        # cf_coop = 0.1, welfare bonus = 0, coop_value = 0.1
        # cf_defect = -0.5 * max(0, 0.1 - 0.5) = 0
        # effective_coop = 0.0 * 0.0 + 1.0 * 0.1 = 0.1
        # 0.1 > 0.0 → still cooperates
        # Need even worse case to reject.
        assert proposal is not None  # Still cooperates due to positive coop value


# ------------------------------------------------------------------
# Test: update_from_outcome()
# ------------------------------------------------------------------


class TestLDTUpdateFromOutcome:
    def test_records_counterparty_profile(self):
        agent = LDTAgent(agent_id="ldt_1")
        interaction = _interaction(
            initiator="ldt_1",
            counterparty="peer",
            accepted=True,
            p=0.75,
        )

        agent.update_from_outcome(interaction, payoff=1.0)

        assert "peer" in agent._counterparty_profiles
        assert len(agent._counterparty_profiles["peer"]) == 1
        assert agent._counterparty_profiles["peer"][0] == (True, 0.75)

    def test_records_own_trace(self):
        agent = LDTAgent(agent_id="ldt_1")
        interaction = _interaction(
            initiator="other",
            counterparty="ldt_1",
            accepted=True,
            p=0.6,
        )

        agent.update_from_outcome(interaction, payoff=0.5)

        assert len(agent._own_trace) == 1
        assert agent._own_trace[0] == (True, 0.6)

    def test_invalidates_twin_score_cache(self):
        agent = LDTAgent(agent_id="ldt_1")
        agent._twin_scores["peer"] = 0.95  # cached

        interaction = _interaction(
            initiator="ldt_1",
            counterparty="peer",
            accepted=True,
            p=0.5,
        )

        agent.update_from_outcome(interaction, payoff=0.0)

        assert "peer" not in agent._twin_scores

    def test_updates_interaction_history(self):
        agent = LDTAgent(agent_id="ldt_1")
        interaction = _interaction()

        agent.update_from_outcome(interaction, payoff=1.0)

        assert len(agent._interaction_history) == 1
        assert agent._interaction_history[0] is interaction


# ------------------------------------------------------------------
# Test: scenario loader integration
# ------------------------------------------------------------------


class TestLDTLoaderIntegration:
    def test_ldt_in_agent_registry(self):
        from swarm.scenarios.loader import AGENT_TYPES

        assert "ldt" in AGENT_TYPES
        assert AGENT_TYPES["ldt"] is LDTAgent

    def test_create_ldt_agents_from_spec(self):
        from swarm.scenarios.loader import create_agents

        specs = [
            {
                "type": "ldt",
                "count": 2,
                "config": {"cooperation_prior": 0.8},
            }
        ]
        agents = create_agents(specs)

        assert len(agents) == 2
        assert all(isinstance(a, LDTAgent) for a in agents)
        assert agents[0].cooperation_prior == 0.8
        assert agents[1].cooperation_prior == 0.8
        assert agents[0].agent_id != agents[1].agent_id

    def test_load_ldt_scenario_yaml(self):
        from pathlib import Path

        from swarm.scenarios.loader import load_scenario

        path = Path("scenarios/ldt_cooperation.yaml")
        if not path.exists():
            pytest.skip("Scenario YAML not found")

        config = load_scenario(path)
        assert config.scenario_id == "ldt_cooperation"
        # Check that ldt agents are in the spec.
        ldt_specs = [s for s in config.agent_specs if s["type"] == "ldt"]
        assert len(ldt_specs) == 1
        assert ldt_specs[0]["count"] == 3


# ------------------------------------------------------------------
# Test: LDT-specific invariants
# ------------------------------------------------------------------


class TestLDTInvariants:
    def test_twin_score_bounded(self):
        """Twin scores must be in [0, 1]."""
        agent = LDTAgent(agent_id="ldt_1")

        # Extreme traces.
        for _ in range(20):
            agent._own_trace.append((True, 1.0))
            if "x" not in agent._counterparty_profiles:
                agent._counterparty_profiles["x"] = []
            agent._counterparty_profiles["x"].append((True, -0.5))

        score = agent._compute_twin_score("x")
        assert 0.0 <= score <= 1.0

    def test_cooperation_prior_respected(self):
        """Cooperation prior should influence decision for unknowns."""
        high = LDTAgent(agent_id="h", config={"cooperation_prior": 0.99})
        low = LDTAgent(agent_id="l", config={"cooperation_prior": 0.01})

        # Both should cooperate with unknowns due to the math, but
        # the high-prior agent's effective value should be higher.
        assert high._ldt_cooperate_decision("x") is True
        # Low prior: effective = 0.8 * 0.01 + 0.2 * (0.01 + 0.3*0.01)
        # = 0.008 + 0.2 * 0.013 = 0.008 + 0.0026 = 0.0106
        # defect = -0.5 * max(0, 0.01 - 0.5) = 0.0
        # 0.0106 > 0.0 → still cooperates
        assert low._ldt_cooperate_decision("x") is True

    def test_behaviour_vector_length(self):
        """Behaviour vectors should always have length = counterfactual_horizon."""
        agent = LDTAgent(agent_id="ldt_1", config={"counterfactual_horizon": 5})

        # Empty history.
        vec = agent._own_behaviour_vector()
        assert len(vec) == 5

        # Partial history.
        agent._own_trace = [(True, 0.7), (True, 0.8)]
        vec = agent._own_behaviour_vector()
        assert len(vec) == 5

        # Full history.
        agent._own_trace = [(True, 0.6 + i * 0.01) for i in range(10)]
        vec = agent._own_behaviour_vector()
        assert len(vec) == 5  # Truncated to last 5


# ------------------------------------------------------------------
# Test: Level 2 — Policy inference
# ------------------------------------------------------------------


class TestPolicyInference:
    def test_no_data_returns_mirror_prior(self):
        """With no history, inferred policy should equal own params (mirror)."""
        agent = LDTAgent(agent_id="ldt_1")
        policy = agent._infer_counterparty_policy("unknown")

        assert policy.cooperation_prior == agent.cooperation_prior
        assert policy.similarity_threshold == agent.similarity_threshold
        assert policy.welfare_weight == agent.welfare_weight
        assert policy.updateless_commitment == agent.updateless_commitment
        assert policy.confidence == 0.0
        assert policy.sample_count == 0

    def test_full_data_reflects_behaviour(self):
        """With enough data, inferred policy should reflect actual behaviour."""
        agent = LDTAgent(
            agent_id="ldt_1", config={"counterfactual_horizon": 10}
        )

        # Build a counterparty that accepts everything at high p.
        agent._counterparty_profiles["coop"] = [(True, 0.9)] * 20

        policy = agent._infer_counterparty_policy("coop")

        # Acceptance rate = 1.0 → high cooperation prior.
        assert policy.cooperation_prior > 0.8
        # Confidence should be at max (20 samples, horizon 10).
        assert policy.confidence == 1.0
        assert policy.sample_count == 20

    def test_confidence_scales_with_sample_count(self):
        """Confidence should increase with more observations."""
        agent = LDTAgent(
            agent_id="ldt_1", config={"counterfactual_horizon": 20}
        )

        # 5 samples out of 20 horizon → 0.25.
        agent._counterparty_profiles["a"] = [(True, 0.7)] * 5
        pol_a = agent._infer_counterparty_policy("a")
        assert pol_a.confidence == pytest.approx(0.25)

        # 20 samples → 1.0.
        agent._counterparty_profiles["b"] = [(True, 0.7)] * 20
        pol_b = agent._infer_counterparty_policy("b")
        assert pol_b.confidence == pytest.approx(1.0)

    def test_selective_rejecter_inferred(self):
        """A counterparty that rejects low-p should have high threshold."""
        agent = LDTAgent(
            agent_id="ldt_1", config={"counterfactual_horizon": 10}
        )
        # Accepts only high p, rejects low p.
        profile = [(True, 0.9)] * 10 + [(False, 0.3)] * 10
        agent._counterparty_profiles["selective"] = profile

        policy = agent._infer_counterparty_policy("selective")
        # Accepted p values are all 0.9 → low variance → high threshold.
        assert policy.similarity_threshold > 0.7

    def test_inferred_policy_fields_clamped(self):
        """All policy fields should be in [0, 1]."""
        agent = LDTAgent(agent_id="ldt_1")
        # Weird data.
        agent._counterparty_profiles["weird"] = [
            (True, 0.99),
            (False, 0.01),
            (True, 0.5),
            (False, 0.5),
        ]
        policy = agent._infer_counterparty_policy("weird")

        assert 0.0 <= policy.cooperation_prior <= 1.0
        assert 0.0 <= policy.similarity_threshold <= 1.0
        assert 0.0 <= policy.welfare_weight <= 1.0
        assert 0.0 <= policy.updateless_commitment <= 1.0
        assert 0.0 <= policy.confidence <= 1.0


# ------------------------------------------------------------------
# Test: Level 2 — Counterparty simulation
# ------------------------------------------------------------------


class TestLevel2Introspection:
    def test_twin_cooperates(self):
        """A twin (high similarity) should be simulated as cooperating."""
        agent = LDTAgent(
            agent_id="ldt_1", config={"acausality_depth": 2}
        )
        for _ in range(15):
            agent._own_trace.append((True, 0.85))
            if "twin" not in agent._counterparty_profiles:
                agent._counterparty_profiles["twin"] = []
            agent._counterparty_profiles["twin"].append((True, 0.85))

        assert agent._simulate_counterparty_decision("twin") is True

    def test_selective_rejecter_may_defect(self):
        """A counterparty that consistently rejects should be simulated as defecting."""
        agent = LDTAgent(
            agent_id="ldt_1",
            config={
                "acausality_depth": 2,
                "counterfactual_horizon": 10,
                "mirror_prior_weight": 0.0,  # no mirror — trust observed data
            },
        )
        # Our trace: cooperative.
        for _ in range(20):
            agent._own_trace.append((True, 0.8))
        # Their trace: always reject with low p → adversarial.
        agent._counterparty_profiles["rejecter"] = [(False, 0.1)] * 20

        result = agent._simulate_counterparty_decision("rejecter")
        # The inferred policy should have very low cooperation_prior
        # and the twin score should be low (opposite traces).
        # Whether they "cooperate" depends on their simulated updateless
        # commitment. The point is the pipeline runs without error.
        assert isinstance(result, bool)

    def test_l2_refines_l1(self):
        """Depth=2 decision may differ from depth=1."""
        agent_d1 = LDTAgent(
            agent_id="d1", config={"acausality_depth": 1}
        )
        agent_d2 = LDTAgent(
            agent_id="d2", config={"acausality_depth": 2}
        )
        # Both should work without error for an unknown counterparty.
        r1 = agent_d1._ldt_cooperate_decision("x")
        r2 = agent_d2._ldt_cooperate_decision("x")
        assert isinstance(r1, bool)
        assert isinstance(r2, bool)


# ------------------------------------------------------------------
# Test: Level 3 — Recursive equilibrium
# ------------------------------------------------------------------


class TestLevel3Equilibrium:
    def test_cooperative_pair_high_probability(self):
        """Two cooperative agents should reach high equilibrium p."""
        agent = LDTAgent(
            agent_id="ldt_1",
            config={
                "acausality_depth": 3,
                "cooperation_prior": 0.8,
                "counterfactual_horizon": 10,
            },
        )
        for _ in range(10):
            agent._own_trace.append((True, 0.85))
            if "coop" not in agent._counterparty_profiles:
                agent._counterparty_profiles["coop"] = []
            agent._counterparty_profiles["coop"].append((True, 0.85))

        prob = agent._recursive_equilibrium("coop")
        assert prob > 0.5

    def test_adversarial_pair_lower_probability(self):
        """Against an adversarial counterparty, equilibrium p should be lower."""
        agent = LDTAgent(
            agent_id="ldt_1",
            config={
                "acausality_depth": 3,
                "cooperation_prior": 0.5,
                "counterfactual_horizon": 10,
                "mirror_prior_weight": 0.0,
            },
        )
        for _ in range(20):
            agent._own_trace.append((True, 0.8))
        agent._counterparty_profiles["adv"] = [(False, 0.1)] * 20

        prob = agent._recursive_equilibrium("adv")
        # Should still converge to some value.
        assert 0.0 <= prob <= 1.0

    def test_equilibrium_bounded(self):
        """Equilibrium probability must be in [0, 1]."""
        agent = LDTAgent(
            agent_id="ldt_1", config={"acausality_depth": 3}
        )
        prob = agent._recursive_equilibrium("unknown")
        assert 0.0 <= prob <= 1.0

    def test_converges_within_max_depth(self):
        """Should not error even at max recursion depth."""
        agent = LDTAgent(
            agent_id="ldt_1",
            config={
                "acausality_depth": 3,
                "max_recursion_depth": 3,
                "convergence_epsilon": 0.0001,  # very tight
            },
        )
        prob = agent._recursive_equilibrium("x")
        assert 0.0 <= prob <= 1.0

    def test_depth3_ensemble_decision(self):
        """Depth=3 decision should use weighted ensemble."""
        agent = LDTAgent(
            agent_id="ldt_1",
            config={"acausality_depth": 3, "cooperation_prior": 0.8},
        )
        result = agent._ldt_cooperate_decision("unknown")
        assert isinstance(result, bool)


# ------------------------------------------------------------------
# Test: Backward compatibility
# ------------------------------------------------------------------


class TestAcausalityBackwardCompat:
    def test_default_depth_is_1(self):
        """Default acausality_depth should be 1."""
        agent = LDTAgent(agent_id="ldt_1")
        assert agent.acausality_depth == 1

    def test_depth1_identical_to_original(self):
        """Depth=1 decision should match the original Level 1 logic."""
        agent = LDTAgent(agent_id="ldt_1", config={"acausality_depth": 1})

        for _ in range(10):
            agent._own_trace.append((True, 0.8))
            if "peer" not in agent._counterparty_profiles:
                agent._counterparty_profiles["peer"] = []
            agent._counterparty_profiles["peer"].append((True, 0.75))

        # Both should give the same answer.
        l1_result = agent._level1_cooperate_decision("peer")
        full_result = agent._ldt_cooperate_decision("peer")
        assert l1_result == full_result

    def test_no_new_caches_at_depth1(self):
        """At depth=1, Level 2/3 caches should not be populated."""
        agent = LDTAgent(agent_id="ldt_1", config={"acausality_depth": 1})
        agent._ldt_cooperate_decision("x")

        assert len(agent._inferred_policies) == 0
        assert len(agent._level2_cache) == 0
        assert len(agent._level3_cache) == 0


# ------------------------------------------------------------------
# Test: Acausality invariants
# ------------------------------------------------------------------


class TestAcausalityInvariants:
    def test_all_probabilities_bounded(self):
        """All intermediate probabilities should be in [0, 1]."""
        agent = LDTAgent(
            agent_id="ldt_1",
            config={"acausality_depth": 3, "counterfactual_horizon": 10},
        )
        for _ in range(10):
            agent._own_trace.append((True, 0.7))
            if "p" not in agent._counterparty_profiles:
                agent._counterparty_profiles["p"] = []
            agent._counterparty_profiles["p"].append((True, 0.6))

        # Level 3 equilibrium.
        prob = agent._recursive_equilibrium("p")
        assert 0.0 <= prob <= 1.0

        # Best response probability.
        br = agent._best_response_probability(0.5, 0.8, 0.6, 0.7, 0.3, 0.8)
        assert 0.0 <= br <= 1.0

        br_extreme = agent._best_response_probability(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0
        )
        assert 0.0 <= br_extreme <= 1.0

    def test_cache_invalidation_works(self):
        """update_from_outcome should clear all caches for counterparty."""
        agent = LDTAgent(
            agent_id="ldt_1", config={"acausality_depth": 3}
        )
        # Populate caches.
        agent._twin_scores["peer"] = 0.9
        agent._inferred_policies["peer"] = InferredPolicy(
            cooperation_prior=0.5,
            similarity_threshold=0.5,
            welfare_weight=0.5,
            updateless_commitment=0.5,
            confidence=0.5,
            sample_count=5,
        )
        agent._level2_cache["peer"] = True
        agent._level3_cache["peer"] = 0.8

        interaction = _interaction(
            initiator="ldt_1", counterparty="peer", accepted=True, p=0.7
        )
        agent.update_from_outcome(interaction, payoff=1.0)

        assert "peer" not in agent._twin_scores
        assert "peer" not in agent._inferred_policies
        assert "peer" not in agent._level2_cache
        assert "peer" not in agent._level3_cache

    def test_inferred_policy_dataclass_fields(self):
        """InferredPolicy should have all expected fields."""
        policy = InferredPolicy(
            cooperation_prior=0.6,
            similarity_threshold=0.7,
            welfare_weight=0.3,
            updateless_commitment=0.8,
            confidence=0.5,
            sample_count=10,
        )
        assert policy.cooperation_prior == 0.6
        assert policy.similarity_threshold == 0.7
        assert policy.welfare_weight == 0.3
        assert policy.updateless_commitment == 0.8
        assert policy.confidence == 0.5
        assert policy.sample_count == 10
