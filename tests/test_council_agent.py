"""Tests for the CouncilAgent."""

from unittest.mock import MagicMock

from swarm.agents.base import ActionType, Observation
from swarm.agents.council_agent import CouncilAgent
from swarm.agents.llm_config import LLMConfig, LLMProvider
from swarm.council.config import CouncilConfig, CouncilMemberConfig
from swarm.council.protocol import CouncilResult
from swarm.models.agent import AgentState


def _make_council_config() -> CouncilConfig:
    """Create a test council config."""
    return CouncilConfig(
        members=[
            CouncilMemberConfig(
                member_id="m1",
                llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
            ),
            CouncilMemberConfig(
                member_id="m2",
                llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
            ),
        ],
        chairman=CouncilMemberConfig(
            member_id="m1",
            llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
        ),
        min_members_required=1,
        timeout_per_member=5.0,
        seed=42,
    )


def _make_observation(**kwargs) -> Observation:
    """Create a minimal test observation."""
    agent_state = AgentState(
        agent_id="council_1",
        reputation=50.0,
        resources=100.0,
    )
    defaults = {
        "agent_state": agent_state,
        "current_epoch": 1,
        "current_step": 1,
    }
    defaults.update(kwargs)
    return Observation(**defaults)


class TestCouncilAgentCreation:
    def test_creates_member_agents(self):
        config = _make_council_config()
        agent = CouncilAgent(
            agent_id="council_1",
            council_config=config,
            name="test_council",
        )
        assert "m1" in agent._member_agents
        assert "m2" in agent._member_agents
        assert agent._council is not None

    def test_repr(self):
        config = _make_council_config()
        agent = CouncilAgent(agent_id="c1", council_config=config)
        r = repr(agent)
        assert "CouncilAgent" in r
        assert "c1" in r


class TestCouncilAgentAct:
    def test_act_returns_action_on_success(self):
        config = _make_council_config()
        agent = CouncilAgent(agent_id="c1", council_config=config)

        result = CouncilResult(
            synthesis='{"action_type": "POST", "params": {"content": "Hello"}}',
            responses={"m1": "resp1", "m2": "resp2"},
            members_responded=2,
            members_total=2,
            success=True,
        )
        agent._call_council_sync = MagicMock(return_value=result)

        obs = _make_observation()
        action = agent.act(obs)

        assert action.action_type == ActionType.POST
        assert action.content == "Hello"

    def test_act_returns_noop_on_failure(self):
        config = _make_council_config()
        agent = CouncilAgent(agent_id="c1", council_config=config)

        result = CouncilResult(
            synthesis="",
            success=False,
            error="Quorum not met",
        )
        agent._call_council_sync = MagicMock(return_value=result)

        obs = _make_observation()
        action = agent.act(obs)

        assert action.action_type == ActionType.NOOP

    def test_act_returns_noop_on_unparseable_json(self):
        config = _make_council_config()
        agent = CouncilAgent(agent_id="c1", council_config=config)

        result = CouncilResult(
            synthesis="I think we should do something but I'm not sure what.",
            responses={"m1": "resp1"},
            members_responded=1,
            members_total=2,
            success=True,
        )
        agent._call_council_sync = MagicMock(return_value=result)

        obs = _make_observation()
        action = agent.act(obs)

        assert action.action_type == ActionType.NOOP

    def test_act_returns_noop_on_exception(self):
        config = _make_council_config()
        agent = CouncilAgent(agent_id="c1", council_config=config)

        agent._call_council_sync = MagicMock(side_effect=RuntimeError("boom"))

        obs = _make_observation()
        action = agent.act(obs)

        assert action.action_type == ActionType.NOOP


class TestCouncilAgentAccept:
    def test_accept_returns_true(self):
        from swarm.agents.base import InteractionProposal
        from swarm.models.interaction import InteractionType

        config = _make_council_config()
        agent = CouncilAgent(agent_id="c1", council_config=config)

        result = CouncilResult(
            synthesis='{"accept": true, "reasoning": "Good proposal"}',
            responses={"m1": "accept", "m2": "accept"},
            members_responded=2,
            members_total=2,
            success=True,
        )
        agent._call_council_sync = MagicMock(return_value=result)

        proposal = InteractionProposal(
            proposal_id="p1",
            initiator_id="other",
            counterparty_id="c1",
            interaction_type=InteractionType.COLLABORATION,
            content="Let's work together",
        )
        obs = _make_observation()
        assert agent.accept_interaction(proposal, obs) is True

    def test_accept_returns_false_on_failure(self):
        from swarm.agents.base import InteractionProposal
        from swarm.models.interaction import InteractionType

        config = _make_council_config()
        agent = CouncilAgent(agent_id="c1", council_config=config)

        result = CouncilResult(synthesis="", success=False, error="timeout")
        agent._call_council_sync = MagicMock(return_value=result)

        proposal = InteractionProposal(
            proposal_id="p1",
            initiator_id="other",
            counterparty_id="c1",
            interaction_type=InteractionType.COLLABORATION,
            content="collaborate",
        )
        obs = _make_observation()
        assert agent.accept_interaction(proposal, obs) is False

    def test_accept_keyword_fallback(self):
        """When JSON parsing fails, falls back to keyword detection."""
        from swarm.agents.base import InteractionProposal
        from swarm.models.interaction import InteractionType

        config = _make_council_config()
        agent = CouncilAgent(agent_id="c1", council_config=config)

        result = CouncilResult(
            synthesis="We should accept this proposal, it looks good.",
            responses={"m1": "yes"},
            members_responded=1,
            members_total=2,
            success=True,
        )
        agent._call_council_sync = MagicMock(return_value=result)

        proposal = InteractionProposal(
            proposal_id="p1",
            initiator_id="other",
            counterparty_id="c1",
            interaction_type=InteractionType.COLLABORATION,
            content="collaborate",
        )
        obs = _make_observation()
        assert agent.accept_interaction(proposal, obs) is True


class TestCouncilAgentUsageStats:
    def test_aggregates_usage(self):
        config = _make_council_config()
        agent = CouncilAgent(agent_id="c1", council_config=config)

        # Simulate some usage on member agents
        agent._member_agents["m1"].usage_stats.total_requests = 5
        agent._member_agents["m1"].usage_stats.total_input_tokens = 1000
        agent._member_agents["m2"].usage_stats.total_requests = 3
        agent._member_agents["m2"].usage_stats.total_input_tokens = 500

        stats = agent.get_usage_stats()
        assert stats["total_requests"] == 8
        assert stats["total_input_tokens"] == 1500
        assert "per_member" in stats
        assert stats["per_member"]["m1"]["total_requests"] == 5
        assert stats["per_member"]["m2"]["total_requests"] == 3

    def test_propose_returns_none(self):
        config = _make_council_config()
        agent = CouncilAgent(agent_id="c1", council_config=config)
        obs = _make_observation()
        assert agent.propose_interaction(obs, "other") is None
