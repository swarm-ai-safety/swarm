"""Tests for Concordia Entity Agent integration."""

from unittest.mock import MagicMock, patch

import pytest

from swarm.agents.base import ActionType, InteractionProposal, Observation
from swarm.bridges.concordia.config import ConcordiaEntityConfig
from swarm.bridges.concordia.entity_agent import (
    _HAS_CONCORDIA,
    parse_action_response,
    render_observation,
)
from swarm.models.agent import AgentState
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_observation(**kwargs) -> Observation:
    """Build an Observation with sensible defaults."""
    defaults = {
        "agent_state": AgentState(agent_id="test_agent", reputation=0.75),
        "current_epoch": 3,
        "current_step": 2,
    }
    defaults.update(kwargs)
    return Observation(**defaults)


# ---------------------------------------------------------------------------
# TestObservationRendering
# ---------------------------------------------------------------------------


class TestObservationRendering:
    """render_observation() — pure text, no Concordia dependency."""

    def test_basic_rendering(self):
        obs = _make_observation()
        config = ConcordiaEntityConfig()
        text = render_observation(obs, config)
        assert "test_agent" in text
        assert "0.75" in text
        assert "Epoch 3" in text

    def test_with_visible_posts(self):
        obs = _make_observation(
            visible_posts=[
                {"post_id": "p1", "author": "alice", "content": "Hello world"},
                {"post_id": "p2", "author": "bob", "content": "Test post"},
            ]
        )
        config = ConcordiaEntityConfig(max_visible_posts=5)
        text = render_observation(obs, config)
        assert "alice" in text
        assert "Hello world" in text

    def test_post_limit_respected(self):
        obs = _make_observation(
            visible_posts=[
                {"post_id": f"p{i}", "author": f"agent_{i}", "content": f"Post {i}"}
                for i in range(10)
            ]
        )
        config = ConcordiaEntityConfig(max_visible_posts=2)
        text = render_observation(obs, config)
        assert "agent_0" in text
        assert "agent_1" in text
        # agent_2 should not appear (limited to 2)
        assert "agent_2" not in text

    def test_prompt_size_bounded(self):
        obs = _make_observation(
            visible_posts=[
                {"post_id": f"p{i}", "author": f"a{i}", "content": "x" * 500}
                for i in range(20)
            ],
            visible_agents=[
                {"agent_id": f"agent_{i}", "reputation": 0.5} for i in range(20)
            ],
        )
        config = ConcordiaEntityConfig(max_prompt_chars=500)
        text = render_observation(obs, config)
        assert len(text) <= 500

    def test_empty_sections_omitted(self):
        obs = _make_observation()  # No posts, agents, tasks etc.
        config = ConcordiaEntityConfig()
        text = render_observation(obs, config)
        assert "Recent posts" not in text
        assert "Other agents" not in text
        assert "Available tasks" not in text

    def test_with_tasks(self):
        obs = _make_observation(
            available_tasks=[
                {"task_id": "t1", "description": "Fix the bug"},
            ]
        )
        config = ConcordiaEntityConfig()
        text = render_observation(obs, config)
        assert "Fix the bug" in text

    def test_capabilities_shown(self):
        obs = _make_observation(can_post=True, can_vote=True, can_interact=False)
        config = ConcordiaEntityConfig()
        text = render_observation(obs, config)
        assert "post" in text
        assert "vote" in text

    def test_ecosystem_metrics(self):
        obs = _make_observation(
            ecosystem_metrics={"toxicity": 0.12, "welfare": 45.0}
        )
        config = ConcordiaEntityConfig()
        text = render_observation(obs, config)
        assert "toxicity" in text


# ---------------------------------------------------------------------------
# TestResponseParsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    """parse_action_response() — pure text parsing, no Concordia dependency."""

    def test_parse_post(self):
        obs = _make_observation()
        action = parse_action_response("I'll post: Hello everyone!", obs)
        assert action.action_type == ActionType.POST
        assert "Hello everyone" in action.content

    def test_parse_vote(self):
        obs = _make_observation()
        action = parse_action_response("I vote on post_123", obs)
        assert action.action_type == ActionType.VOTE

    def test_parse_vote_downvote(self):
        obs = _make_observation()
        action = parse_action_response("I vote down on post_123", obs)
        assert action.action_type == ActionType.VOTE
        assert action.vote_direction == -1

    def test_parse_propose(self):
        obs = _make_observation()
        action = parse_action_response("I propose to agent_5: Let's collaborate", obs)
        assert action.action_type == ActionType.PROPOSE_INTERACTION

    def test_parse_claim_task(self):
        obs = _make_observation(
            available_tasks=[{"task_id": "t1", "description": "Do something"}]
        )
        action = parse_action_response("I'll claim_task t1", obs)
        assert action.action_type == ActionType.CLAIM_TASK
        assert action.target_id == "t1"

    def test_parse_claim_task_default(self):
        """claim_task without ID picks first available."""
        obs = _make_observation(
            available_tasks=[{"task_id": "t99", "description": "Default task"}]
        )
        action = parse_action_response("I want to claim task", obs)
        assert action.action_type == ActionType.CLAIM_TASK
        assert action.target_id == "t99"

    def test_parse_noop_explicit(self):
        obs = _make_observation()
        action = parse_action_response("noop", obs)
        assert action.action_type == ActionType.NOOP

    def test_parse_do_nothing(self):
        obs = _make_observation()
        action = parse_action_response("I'll do nothing this turn.", obs)
        assert action.action_type == ActionType.NOOP

    def test_parse_pass(self):
        obs = _make_observation()
        action = parse_action_response("pass", obs)
        assert action.action_type == ActionType.NOOP

    def test_malformed_fallback_noop(self):
        obs = _make_observation()
        action = parse_action_response("asdfghjkl gibberish", obs)
        assert action.action_type == ActionType.NOOP

    def test_empty_response_noop(self):
        obs = _make_observation()
        action = parse_action_response("", obs)
        assert action.action_type == ActionType.NOOP

    def test_content_truncated(self):
        obs = _make_observation()
        long_content = "post: " + "x" * 1000
        action = parse_action_response(long_content, obs)
        assert action.action_type == ActionType.POST
        assert len(action.content) <= 500


# ---------------------------------------------------------------------------
# TestConcordiaEntityAgent (mocked Concordia)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CONCORDIA, reason="gdm-concordia not installed")
class TestConcordiaEntityAgent:
    """End-to-end tests with mocked Concordia Entity."""

    def _make_agent(self, entity_mock=None):
        from swarm.bridges.concordia.entity_agent import ConcordiaEntityAgent

        config = ConcordiaEntityConfig(goal="Test goal")
        agent = ConcordiaEntityAgent(
            agent_id="ce_1",
            concordia_config=config,
            name="test_entity",
            entity=entity_mock or MagicMock(),
        )
        return agent

    def test_act_returns_action(self):
        mock_entity = MagicMock()
        mock_entity.act.return_value = "I'll post: Testing concordia integration"
        agent = self._make_agent(mock_entity)

        obs = _make_observation()
        action = agent.act(obs)

        assert action.action_type == ActionType.POST
        assert action.agent_id == "ce_1"
        mock_entity.act.assert_called_once()
        mock_entity.observe.assert_called()  # Self-feedback

    def test_act_entity_failure_returns_noop(self):
        mock_entity = MagicMock()
        mock_entity.act.side_effect = RuntimeError("LLM failure")
        agent = self._make_agent(mock_entity)

        obs = _make_observation()
        action = agent.act(obs)

        assert action.action_type == ActionType.NOOP

    def test_accept_interaction_accept(self):
        mock_entity = MagicMock()
        mock_entity.act.return_value = "accept"
        agent = self._make_agent(mock_entity)

        proposal = InteractionProposal(
            initiator_id="other_agent",
            counterparty_id="ce_1",
            content="Let's work together",
        )
        obs = _make_observation()

        assert agent.accept_interaction(proposal, obs) is True

    def test_accept_interaction_reject(self):
        mock_entity = MagicMock()
        mock_entity.act.return_value = "reject"
        agent = self._make_agent(mock_entity)

        proposal = InteractionProposal(
            initiator_id="other_agent",
            counterparty_id="ce_1",
            content="Suspicious proposal",
        )
        obs = _make_observation()

        assert agent.accept_interaction(proposal, obs) is False

    def test_propose_interaction_decline(self):
        mock_entity = MagicMock()
        mock_entity.act.return_value = "I decline to interact."
        agent = self._make_agent(mock_entity)

        obs = _make_observation()
        result = agent.propose_interaction(obs, "other_agent")
        assert result is None

    def test_propose_interaction_accept(self):
        mock_entity = MagicMock()
        mock_entity.act.return_value = "Let's collaborate on the task!"
        agent = self._make_agent(mock_entity)

        obs = _make_observation()
        result = agent.propose_interaction(obs, "other_agent")
        assert result is not None
        assert result.initiator_id == "ce_1"
        assert result.counterparty_id == "other_agent"

    def test_update_from_outcome(self):
        mock_entity = MagicMock()
        agent = self._make_agent(mock_entity)

        interaction = SoftInteraction(
            initiator="ce_1",
            counterparty="other",
            p=0.8,
            accepted=True,
        )
        agent.update_from_outcome(interaction, payoff=1.5)

        # SWARM bookkeeping
        assert len(agent._interaction_history) == 1
        assert len(agent._memory) > 0

        # Entity got narrative feedback
        mock_entity.observe.assert_called()
        call_args = mock_entity.observe.call_args[0][0]
        assert "other" in call_args
        assert "0.80" in call_args


# ---------------------------------------------------------------------------
# TestConcordiaEntityAgentNoLib
# ---------------------------------------------------------------------------


class TestConcordiaEntityAgentNoLib:
    """Tests that run regardless of whether Concordia is installed."""

    def test_import_error_when_concordia_missing(self):
        """Verify ImportError if concordia not available."""
        with patch(
            "swarm.bridges.concordia.entity_agent._HAS_CONCORDIA", False
        ):
            # Need to reimport the class to get the patched check
            from swarm.bridges.concordia.entity_agent import ConcordiaEntityAgent

            with pytest.raises(ImportError, match="gdm-concordia"):
                ConcordiaEntityAgent(agent_id="test")


# ---------------------------------------------------------------------------
# TestLoaderRegistration
# ---------------------------------------------------------------------------


class TestLoaderRegistration:
    """Verify concordia_entity type is recognized by the scenario loader."""

    def test_concordia_entity_type_creates_agents(self):
        """Loader recognizes concordia_entity and calls the right class."""
        mock_agent = MagicMock()
        mock_cls = MagicMock(return_value=mock_agent)

        with patch(
            "swarm.scenarios.loader._get_concordia_entity_class",
            return_value=mock_cls,
        ):
            from swarm.scenarios.loader import create_agents

            specs = [
                {
                    "type": "concordia_entity",
                    "count": 2,
                    "concordia": {
                        "goal": "Test goal",
                        "max_prompt_chars": 2000,
                    },
                }
            ]
            agents = create_agents(specs, seed=42)

        assert len(agents) == 2
        assert mock_cls.call_count == 2
        # Verify config was passed
        call_kwargs = mock_cls.call_args_list[0][1]
        assert call_kwargs["concordia_config"].goal == "Test goal"
        assert call_kwargs["concordia_config"].max_prompt_chars == 2000

    def test_concordia_entity_default_config(self):
        """Loader uses default ConcordiaEntityConfig when no concordia key."""
        mock_agent = MagicMock()
        mock_cls = MagicMock(return_value=mock_agent)

        with patch(
            "swarm.scenarios.loader._get_concordia_entity_class",
            return_value=mock_cls,
        ):
            from swarm.scenarios.loader import create_agents

            specs = [{"type": "concordia_entity", "count": 1}]
            agents = create_agents(specs, seed=42)

        assert len(agents) == 1
        call_kwargs = mock_cls.call_args_list[0][1]
        assert call_kwargs["concordia_config"].goal == "Be helpful and cooperative."
