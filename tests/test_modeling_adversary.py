"""Tests for the ModelingAdversary agent."""

import random
from typing import Any

from swarm.agents.base import ActionType, Observation
from swarm.agents.modeling_adversary import ModelingAdversary
from swarm.models.agent import AgentState
from swarm.models.interaction import SoftInteraction


def _obs(**kwargs) -> Observation:
    """Create a test observation with sensible defaults."""
    defaults: dict[str, Any] = {
        "agent_state": AgentState(
            agent_id="madv_1", reputation=0.5, resources=100.0
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
    counterparty: str = "madv_1",
    accepted: bool = True,
    p: float = 0.8,
) -> SoftInteraction:
    return SoftInteraction(
        initiator=initiator,
        counterparty=counterparty,
        accepted=accepted,
        p=p,
    )


class TestCounterpartyModeling:
    """Test that the adversary correctly models counterparties."""

    def test_no_history_returns_unknown(self):
        agent = ModelingAdversary("madv_1")
        model = agent._update_model("unknown_agent")
        assert model.estimated_type == "unknown"
        assert model.interaction_count == 0

    def test_cooperative_agent_detected(self):
        agent = ModelingAdversary("madv_1")
        # Build cooperative history
        for _ in range(10):
            ix = _interaction(initiator="coop_1", accepted=True, p=0.8)
            agent.update_from_outcome(ix, 1.0)
        model = agent._get_model("coop_1")
        assert model.cooperation_rate > 0.8
        assert model.interaction_count == 10

    def test_ldt_detection(self):
        """LDT agents should be detectable by high selectivity + stability."""
        agent = ModelingAdversary("madv_1", config={"ldt_detection_sensitivity": 0.5})
        # Build LDT-like history: high acceptance rate, low variance, stable
        for _ in range(15):
            p = 0.75 + random.uniform(-0.05, 0.05)
            ix = _interaction(initiator="ldt_1", accepted=True, p=p)
            agent.update_from_outcome(ix, 1.0)
        model = agent._get_model("ldt_1")
        # High cooperation rate + high selectivity + stability = LDT-like
        assert model.cooperation_rate > 0.8
        assert model.selectivity > 0.5


class TestPhaseTransition:
    """Test trust-building to exploitation phase transition."""

    def test_initial_phase_is_trust_build(self):
        agent = ModelingAdversary("madv_1")
        assert agent._get_phase("some_agent") == "trust_build"

    def test_switches_to_exploit_after_threshold(self):
        agent = ModelingAdversary("madv_1", config={"exploitation_threshold": 3})
        for _ in range(5):
            ix = _interaction(initiator="target", accepted=True, p=0.7)
            agent.update_from_outcome(ix, 1.0)
        # Model now has 5 interactions, threshold is 3
        assert agent._get_phase("target") == "exploit"


class TestMimicry:
    """Test that mimicry behavior works against LDT agents."""

    def test_mimics_during_trust_build(self):
        agent = ModelingAdversary("madv_1")
        # During trust phase, should always mimic
        assert agent._should_mimic("new_agent") is True

    def test_cooperates_during_trust_build(self):
        agent = ModelingAdversary("madv_1")
        assert agent._cooperate_decision("new_agent") is True


class TestActMethod:
    """Test the main act method."""

    def test_accepts_proposals_during_trust(self):
        agent = ModelingAdversary("madv_1")
        obs = _obs(pending_proposals=[{
            "proposal_id": "p1",
            "initiator_id": "target",
        }])
        action = agent.act(obs)
        assert action.action_type == ActionType.ACCEPT_INTERACTION

    def test_identifies_allies(self):
        agent = ModelingAdversary("madv_1")
        obs = _obs(visible_agents=[
            {"agent_id": "adv_1", "agent_type": "adversarial"},
        ])
        agent.act(obs)
        assert "adv_1" in agent._known_allies


class TestUpdateFromOutcome:
    """Test outcome processing."""

    def test_records_history(self):
        agent = ModelingAdversary("madv_1")
        ix = _interaction(initiator="other", accepted=True, p=0.8)
        agent.update_from_outcome(ix, 1.0)
        assert len(agent._counterparty_history["other"]) == 1
        assert len(agent._own_trace) == 1

    def test_updates_model_on_outcome(self):
        agent = ModelingAdversary("madv_1")
        ix = _interaction(initiator="other", accepted=True, p=0.8)
        agent.update_from_outcome(ix, 1.0)
        model = agent._get_model("other")
        assert model.interaction_count == 1
