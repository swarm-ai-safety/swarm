"""Tests for the CrewAI-backed SWARM agent adapter.

These tests verify the adapter layer without requiring the real ``crewai``
package.  The crew is mocked so that tests stay fast and deterministic.
"""

import json
import random
from unittest.mock import MagicMock, patch

import pytest

from swarm.agents.base import Action, ActionType, Observation
from swarm.agents.crewai_adapter import (
    CrewAIToolAdapter,
    CrewBackedAgent,
    CrewConfig,
    SwarmActionSchema,
    _ACTION_KIND_MAP,
    _VALID_KINDS,
    get_profile_agents,
    register_crew_profile,
    CrewAgentRole,
)
from swarm.models.agent import AgentState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_observation() -> Observation:
    """Minimal valid observation."""
    return Observation(
        agent_state=AgentState(
            agent_id="test_agent",
            reputation=1.0,
            resources=100.0,
        ),
        current_epoch=0,
        current_step=0,
        can_post=True,
        can_interact=True,
        can_vote=True,
        can_claim_task=True,
        visible_posts=[{"post_id": "p1", "content": "hello"}],
        visible_agents=[{"agent_id": "other_1", "reputation": 0.5}],
        available_tasks=[{"task_id": "t1", "difficulty": "easy"}],
        pending_proposals=[],
        ecosystem_metrics={"toxicity": 0.1},
    )


@pytest.fixture()
def crew_config() -> CrewConfig:
    return CrewConfig(
        crew_profile="general_v1",
        role_name="Test Agent",
        model="gpt-4o-mini",
        enable_trace=True,
    )


@pytest.fixture()
def agent(crew_config: CrewConfig) -> CrewBackedAgent:
    return CrewBackedAgent(
        agent_id="crew_test_1",
        crew_config=crew_config,
        rng=random.Random(42),
    )


# ---------------------------------------------------------------------------
# SwarmActionSchema tests
# ---------------------------------------------------------------------------


class TestSwarmActionSchema:
    def test_valid_schema(self):
        schema = SwarmActionSchema(
            kind="post",
            content="Hello world",
            confidence=0.9,
            rationale="Testing",
        )
        assert schema.kind == "post"
        assert schema.confidence == 0.9

    def test_defaults(self):
        schema = SwarmActionSchema(kind="noop")
        assert schema.content == ""
        assert schema.target_id == ""
        assert schema.confidence == 0.7
        assert schema.metadata == {}

    def test_json_schema_generation(self):
        schema = SwarmActionSchema.model_json_schema()
        assert "kind" in schema["properties"]
        assert "content" in schema["properties"]

    def test_valid_kinds_list(self):
        assert "noop" in _VALID_KINDS
        assert "post" in _VALID_KINDS
        assert "claim_task" in _VALID_KINDS

    def test_action_kind_map_completeness(self):
        """Every valid kind string maps to an ActionType."""
        for kind in _VALID_KINDS:
            assert kind in _ACTION_KIND_MAP


# ---------------------------------------------------------------------------
# CrewAIToolAdapter tests
# ---------------------------------------------------------------------------


class TestCrewAIToolAdapter:
    def test_no_observation(self):
        adapter = CrewAIToolAdapter()
        assert adapter.get_agent_state() == {}
        assert adapter.get_visible_posts() == []
        assert adapter.get_available_tasks() == []
        assert adapter.get_pending_proposals() == []
        assert adapter.get_visible_agents() == []
        assert adapter.get_ecosystem_metrics() == {}
        assert adapter.get_available_bounties() == []

    def test_with_observation(self, default_observation: Observation):
        adapter = CrewAIToolAdapter()
        adapter.update_observation(default_observation)

        state = adapter.get_agent_state()
        assert state["agent_id"] == "test_agent"
        assert state["resources"] == 100.0

        posts = adapter.get_visible_posts()
        assert len(posts) == 1
        assert posts[0]["post_id"] == "p1"

        tasks = adapter.get_available_tasks()
        assert len(tasks) == 1

        agents = adapter.get_visible_agents()
        assert len(agents) == 1

        metrics = adapter.get_ecosystem_metrics()
        assert metrics["toxicity"] == 0.1

    def test_stage_action(self):
        adapter = CrewAIToolAdapter()
        result = adapter.stage_action({"kind": "post", "content": "test"})
        assert "staged" in result
        assert len(adapter.get_staged_actions()) == 1

    def test_update_observation_clears_staged(self, default_observation: Observation):
        adapter = CrewAIToolAdapter()
        adapter.stage_action({"kind": "post"})
        assert len(adapter.get_staged_actions()) == 1

        adapter.update_observation(default_observation)
        assert len(adapter.get_staged_actions()) == 0


# ---------------------------------------------------------------------------
# Crew profile tests
# ---------------------------------------------------------------------------


class TestCrewProfiles:
    def test_get_default_profiles(self):
        for profile_name in ("market_team_v1", "audit_team_v1", "general_v1"):
            agents = get_profile_agents(profile_name)
            assert len(agents) >= 2
            assert all(isinstance(a, CrewAgentRole) for a in agents)

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown crew profile"):
            get_profile_agents("nonexistent_profile")

    def test_register_custom_profile(self):
        custom_agents = [
            CrewAgentRole(role="CustomRole", goal="Do something"),
        ]
        register_crew_profile("test_custom", custom_agents)
        retrieved = get_profile_agents("test_custom")
        assert len(retrieved) == 1
        assert retrieved[0].role == "CustomRole"


# ---------------------------------------------------------------------------
# CrewConfig tests
# ---------------------------------------------------------------------------


class TestCrewConfig:
    def test_defaults(self):
        config = CrewConfig()
        assert config.crew_profile == "general_v1"
        assert config.temperature == 0.7
        assert config.enable_trace is True

    def test_custom_values(self):
        config = CrewConfig(
            crew_profile="audit_team_v1",
            role_name="Auditor",
            model="claude-sonnet-4-5-20250929",
            temperature=0.3,
        )
        assert config.crew_profile == "audit_team_v1"
        assert config.role_name == "Auditor"


# ---------------------------------------------------------------------------
# CrewBackedAgent tests (with mocked crew)
# ---------------------------------------------------------------------------


class TestCrewBackedAgent:
    def test_init(self, agent: CrewBackedAgent):
        assert agent.agent_id == "crew_test_1"
        assert agent.crew_config.crew_profile == "general_v1"
        assert agent._crew is None  # lazy

    def test_build_context(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        ctx = agent._build_context(default_observation)
        assert ctx["epoch"] == 0
        assert ctx["step"] == 0
        assert ctx["agent_id"] == "crew_test_1"
        assert ctx["budget"] == 100.0
        assert ctx["reputation"] == 1.0
        assert ctx["can_post"] is True
        assert "action_schema" in ctx
        assert "valid_action_kinds" in ctx
        assert isinstance(ctx["valid_action_kinds"], list)

    def test_parse_action_valid_json(self, agent: CrewBackedAgent):
        raw = json.dumps({"kind": "post", "content": "Hello!", "confidence": 0.9})
        schema = agent._parse_action(raw)
        assert schema.kind == "post"
        assert schema.content == "Hello!"
        assert schema.confidence == 0.9

    def test_parse_action_json_in_code_block(self, agent: CrewBackedAgent):
        raw = '```json\n{"kind": "vote", "target_id": "p1"}\n```'
        schema = agent._parse_action(raw)
        assert schema.kind == "vote"
        assert schema.target_id == "p1"

    def test_parse_action_fallback_noop(self, agent: CrewBackedAgent):
        raw = "This is not valid JSON at all."
        schema = agent._parse_action(raw)
        assert schema.kind == "noop"
        assert "Parse failure" in schema.rationale

    def test_schema_to_action(self, agent: CrewBackedAgent):
        schema = SwarmActionSchema(
            kind="post",
            content="Hello world",
            confidence=0.85,
            rationale="Want to contribute",
        )
        action = agent._schema_to_action(schema)
        assert isinstance(action, Action)
        assert action.action_type == ActionType.POST
        assert action.content == "Hello world"
        assert action.agent_id == "crew_test_1"
        assert action.metadata["confidence"] == 0.85

    def test_schema_to_action_unknown_kind_defaults_noop(
        self, agent: CrewBackedAgent
    ):
        schema = SwarmActionSchema(kind="unknown_action_xyz")
        action = agent._schema_to_action(schema)
        assert action.action_type == ActionType.NOOP

    def test_act_with_mocked_crew(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """Full act() cycle with a mocked crew."""
        mock_crew = MagicMock()
        mock_result = MagicMock()
        mock_result.raw = json.dumps(
            {
                "kind": "post",
                "content": "Crew decided to post",
                "confidence": 0.92,
                "rationale": "Contributing useful info",
            }
        )
        mock_crew.kickoff.return_value = mock_result
        agent._crew = mock_crew

        action = agent.act(default_observation)

        assert action.action_type == ActionType.POST
        assert action.content == "Crew decided to post"
        assert action.metadata["_crew_trace"]["confidence"] == 0.92
        assert action.metadata["_crew_trace"]["crew_profile"] == "general_v1"

        # Verify deliberation memory was recorded
        history = agent.get_deliberation_history()
        assert len(history) == 1
        assert history[0]["action_kind"] == "post"

    def test_act_crew_failure_falls_back_to_noop(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """When the crew raises an exception, agent returns noop."""
        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = RuntimeError("Crew exploded")
        agent._crew = mock_crew

        action = agent.act(default_observation)
        assert action.action_type == ActionType.NOOP

    def test_act_unparseable_output_falls_back_to_noop(
        self, agent: CrewBackedAgent, default_observation: Observation
    ):
        """When crew output isn't valid JSON, agent returns noop with trace."""
        mock_crew = MagicMock()
        mock_result = MagicMock()
        mock_result.raw = "I think we should do something but I forgot the format"
        mock_crew.kickoff.return_value = mock_result
        agent._crew = mock_crew

        action = agent.act(default_observation)
        assert action.action_type == ActionType.NOOP
        # Trace should still be attached
        assert "_crew_trace" in action.metadata

    def test_accept_interaction_trust_based(self, agent: CrewBackedAgent):
        """accept_interaction uses counterparty trust."""
        from swarm.agents.base import InteractionProposal

        proposal = InteractionProposal(
            initiator_id="other_agent",
            counterparty_id=agent.agent_id,
        )
        obs = Observation()

        # Unknown agent: trust = 0.5, threshold = 0.4 -> accept
        assert agent.accept_interaction(proposal, obs) is True

        # Set low trust
        agent._counterparty_memory["other_agent"] = 0.1
        assert agent.accept_interaction(proposal, obs) is False

    def test_propose_interaction(self, agent: CrewBackedAgent):
        obs = Observation()

        # Unknown agent -> trust 0.5 -> propose
        proposal = agent.propose_interaction(obs, "counterparty_1")
        assert proposal is not None
        assert proposal.initiator_id == agent.agent_id
        assert proposal.counterparty_id == "counterparty_1"

        # Low-trust agent -> None
        agent._counterparty_memory["low_trust"] = 0.1
        assert agent.propose_interaction(obs, "low_trust") is None

    def test_trace_disabled(
        self, default_observation: Observation
    ):
        """When enable_trace=False, no _crew_trace in metadata."""
        config = CrewConfig(enable_trace=False)
        agent = CrewBackedAgent(
            agent_id="no_trace",
            crew_config=config,
        )

        mock_crew = MagicMock()
        mock_result = MagicMock()
        mock_result.raw = json.dumps({"kind": "noop"})
        mock_crew.kickoff.return_value = mock_result
        agent._crew = mock_crew

        action = agent.act(default_observation)
        assert "_crew_trace" not in action.metadata

    def test_repr(self, agent: CrewBackedAgent):
        r = repr(agent)
        assert "CrewBackedAgent" in r
        assert "crew_test_1" in r
        assert "general_v1" in r


# ---------------------------------------------------------------------------
# Loader integration tests
# ---------------------------------------------------------------------------


class TestLoaderIntegration:
    def test_lazy_import_crewai_classes(self):
        """Verify the lazy import function in the loader works."""
        from swarm.scenarios.loader import _get_crewai_classes

        AgentCls, ConfigCls = _get_crewai_classes()
        assert AgentCls is CrewBackedAgent
        assert ConfigCls is CrewConfig

    def test_create_agents_crewai_type(self):
        """create_agents handles type='crewai_adapter'."""
        from swarm.scenarios.loader import create_agents

        specs = [
            {
                "type": "crewai_adapter",
                "count": 2,
                "params": {
                    "role_name": "Trader",
                    "crew_profile": "market_team_v1",
                },
            }
        ]
        agents = create_agents(specs, seed=99)
        assert len(agents) == 2
        assert all(isinstance(a, CrewBackedAgent) for a in agents)
        assert agents[0].agent_id == "crewai_1"
        assert agents[1].agent_id == "crewai_2"
        assert agents[0].crew_config.role_name == "Trader"
        assert agents[0].crew_config.crew_profile == "market_team_v1"

    def test_create_agents_mixed(self):
        """CrewAI agents can coexist with scripted agents."""
        from swarm.scenarios.loader import create_agents

        specs = [
            {"type": "honest", "count": 2},
            {
                "type": "crewai_adapter",
                "count": 1,
                "params": {"role_name": "Auditor"},
            },
            {"type": "opportunistic", "count": 1},
        ]
        agents = create_agents(specs, seed=42)
        assert len(agents) == 4
        crew_agents = [a for a in agents if isinstance(a, CrewBackedAgent)]
        assert len(crew_agents) == 1


# ---------------------------------------------------------------------------
# Action kind mapping exhaustiveness
# ---------------------------------------------------------------------------


class TestActionKindMapping:
    @pytest.mark.parametrize("kind", _VALID_KINDS)
    def test_each_kind_maps_to_action_type(self, kind: str):
        schema = SwarmActionSchema(kind=kind)
        action_type = _ACTION_KIND_MAP[kind]
        assert isinstance(action_type, ActionType)

    def test_all_mapped_action_types_are_unique(self):
        """Each kind string maps to a distinct ActionType."""
        values = list(_ACTION_KIND_MAP.values())
        # noop might be the default for unmapped, but each explicit
        # mapping should be distinct
        assert len(values) == len(set(values))
