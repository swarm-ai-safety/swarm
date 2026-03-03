"""Tests for the Swarms-backed SWARM agent adapter.

These tests verify the adapter layer without requiring the real ``swarms``
package.  The Swarms agent is mocked so that tests stay fast and deterministic.

Security-related tests are grouped under ``TestSecurity*`` classes.
"""

import json
import random
from unittest.mock import MagicMock

import pytest

from swarm.agents.base import Action, ActionType, Observation
from swarm.agents.swarms_adapter import (
    _ACTION_KIND_MAP,
    _VALID_KINDS,
    DEFAULT_SWARMS_TIMEOUT_SECONDS,
    MAX_CONTENT_LENGTH,
    MAX_DELIBERATION_MEMORY,
    MAX_ID_LENGTH,
    MAX_METADATA_KEYS,
    MAX_METADATA_VALUE_LENGTH,
    MAX_RATIONALE_LENGTH,
    MAX_RAW_TRACE_LENGTH,
    SwarmActionSchema,
    SwarmsBackedAgent,
    SwarmsConfig,
    _sanitize_swarms_metadata,
)
from swarm.models.agent import AgentState, AgentType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_observation() -> Observation:
    """Minimal valid observation with IDs for validation tests."""
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
        pending_proposals=[{"proposal_id": "prop_1", "initiator_id": "other_1"}],
        ecosystem_metrics={"toxicity": 0.1},
        available_bounties=[{"bounty_id": "b1", "reward_amount": 10.0}],
    )


@pytest.fixture()
def swarms_config() -> SwarmsConfig:
    return SwarmsConfig(
        architecture="Agent",
        model_name="gpt-4o-mini",
        enable_trace=True,
    )


@pytest.fixture()
def agent(swarms_config: SwarmsConfig) -> SwarmsBackedAgent:
    return SwarmsBackedAgent(
        agent_id="swarms_test_1",
        swarms_config=swarms_config,
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

    def test_invalid_kind_rejected(self):
        """Unknown kind strings are rejected at parse time."""
        with pytest.raises(ValueError, match="Invalid action kind"):
            SwarmActionSchema(kind="totally_bogus")

    def test_content_max_length(self):
        """Content exceeding MAX_CONTENT_LENGTH is rejected."""
        with pytest.raises(ValueError):
            SwarmActionSchema(kind="post", content="x" * (MAX_CONTENT_LENGTH + 1))

    def test_rationale_max_length(self):
        """Rationale exceeding MAX_RATIONALE_LENGTH is rejected."""
        with pytest.raises(ValueError):
            SwarmActionSchema(kind="noop", rationale="r" * (MAX_RATIONALE_LENGTH + 1))

    def test_target_id_max_length(self):
        """target_id exceeding MAX_ID_LENGTH is rejected."""
        with pytest.raises(ValueError):
            SwarmActionSchema(kind="noop", target_id="i" * (MAX_ID_LENGTH + 1))

    def test_counterparty_id_max_length(self):
        """counterparty_id exceeding MAX_ID_LENGTH is rejected."""
        with pytest.raises(ValueError):
            SwarmActionSchema(kind="noop", counterparty_id="c" * (MAX_ID_LENGTH + 1))


# ---------------------------------------------------------------------------
# SwarmsConfig tests
# ---------------------------------------------------------------------------


class TestSwarmsConfig:
    def test_defaults(self):
        config = SwarmsConfig()
        assert config.architecture == "Agent"
        assert config.model_name == "gpt-4o-mini"
        assert config.max_loops == 1
        assert config.timeout_seconds == DEFAULT_SWARMS_TIMEOUT_SECONDS
        assert config.safe_mode is True
        assert config.enable_trace is True

    def test_custom_values(self):
        config = SwarmsConfig(
            architecture="Agent",
            model_name="gpt-4o",
            max_loops=3,
            timeout_seconds=60.0,
            temperature=0.3,
        )
        assert config.model_name == "gpt-4o"
        assert config.max_loops == 3
        assert config.timeout_seconds == 60.0
        assert config.temperature == 0.3

    def test_invalid_architecture_rejected(self):
        """Unknown architecture is rejected at parse time."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            SwarmsConfig(architecture="NonexistentWorkflow")

    def test_max_loops_bounded(self):
        """max_loops must be between 1 and 10."""
        with pytest.raises(ValueError):
            SwarmsConfig(max_loops=0)
        with pytest.raises(ValueError):
            SwarmsConfig(max_loops=11)


# ---------------------------------------------------------------------------
# Metadata sanitization tests
# ---------------------------------------------------------------------------


class TestSanitizeSwarmsMetadata:
    def test_allows_safe_types(self):
        raw = {"key_str": "value", "key_int": 42, "key_float": 3.14, "key_bool": True}
        result = _sanitize_swarms_metadata(raw)
        assert result == raw

    def test_drops_nested_dicts(self):
        raw = {"safe": "ok", "nested": {"inner": "dropped"}}
        result = _sanitize_swarms_metadata(raw)
        assert "safe" in result
        assert "nested" not in result

    def test_drops_lists(self):
        raw = {"safe": 1, "items": [1, 2, 3]}
        result = _sanitize_swarms_metadata(raw)
        assert "items" not in result

    def test_truncates_long_strings(self):
        raw = {"long": "x" * 5000}
        result = _sanitize_swarms_metadata(raw)
        assert len(result["long"]) == MAX_METADATA_VALUE_LENGTH

    def test_caps_number_of_keys(self):
        raw = {f"key_{i}": i for i in range(100)}
        result = _sanitize_swarms_metadata(raw)
        assert len(result) == MAX_METADATA_KEYS

    def test_empty_input(self):
        assert _sanitize_swarms_metadata({}) == {}


# ---------------------------------------------------------------------------
# SwarmsBackedAgent tests (with mocked Swarms agent)
# ---------------------------------------------------------------------------


class TestSwarmsBackedAgent:
    def test_init(self, agent: SwarmsBackedAgent):
        assert agent.agent_id == "swarms_test_1"
        assert agent.swarms_config.architecture == "Agent"
        assert agent._swarms_agent is None  # lazy

    def test_agent_type_is_swarms(self, agent: SwarmsBackedAgent):
        """Agent type should be SWARMS, not HONEST."""
        assert agent.agent_type == AgentType.SWARMS

    def test_build_prompt(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        prompt = agent._build_prompt(default_observation)
        assert "epoch" in prompt
        assert "valid_action_kinds" in prompt
        assert "SwarmAction" in prompt
        assert "JSON" in prompt

    def test_parse_action_valid_json(self, agent: SwarmsBackedAgent):
        raw = json.dumps({"kind": "post", "content": "Hello!", "confidence": 0.9})
        schema = agent._parse_action(raw)
        assert schema.kind == "post"
        assert schema.content == "Hello!"
        assert schema.confidence == 0.9

    def test_parse_action_json_in_code_block(self, agent: SwarmsBackedAgent):
        raw = '```json\n{"kind": "vote", "target_id": "p1"}\n```'
        schema = agent._parse_action(raw)
        assert schema.kind == "vote"
        assert schema.target_id == "p1"

    def test_parse_action_fallback_noop(self, agent: SwarmsBackedAgent):
        raw = "This is not valid JSON at all."
        schema = agent._parse_action(raw)
        assert schema.kind == "noop"
        assert "Parse failure" in schema.rationale

    def test_parse_action_invalid_kind_fallback_noop(self, agent: SwarmsBackedAgent):
        """An invalid kind in valid JSON falls back to noop."""
        raw = json.dumps({"kind": "hacky_exploit"})
        schema = agent._parse_action(raw)
        assert schema.kind == "noop"

    def test_schema_to_action(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        agent._collect_valid_ids(default_observation)
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
        assert action.agent_id == "swarms_test_1"
        assert action.metadata["confidence"] == 0.85

    def test_act_with_mocked_agent(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """Full act() cycle with a mocked Swarms agent."""
        mock_swarms = MagicMock()
        mock_swarms.run.return_value = json.dumps(
            {
                "kind": "post",
                "content": "Swarms decided to post",
                "confidence": 0.92,
                "rationale": "Contributing useful info",
            }
        )
        agent._swarms_agent = mock_swarms

        action = agent.act(default_observation)

        assert action.action_type == ActionType.POST
        assert action.content == "Swarms decided to post"
        assert action.metadata["_swarms_trace"]["confidence"] == 0.92
        assert action.metadata["_swarms_trace"]["architecture"] == "Agent"

        # Verify deliberation memory was recorded
        history = agent.get_deliberation_history()
        assert len(history) == 1
        assert history[0]["action_kind"] == "post"

    def test_act_agent_failure_falls_back_to_noop(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """When the Swarms agent raises an exception, agent returns noop."""
        mock_swarms = MagicMock()
        mock_swarms.run.side_effect = RuntimeError("Swarms exploded")
        agent._swarms_agent = mock_swarms

        action = agent.act(default_observation)
        assert action.action_type == ActionType.NOOP

    def test_act_unparseable_output_falls_back_to_noop(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """When Swarms output isn't valid JSON, agent returns noop with trace."""
        mock_swarms = MagicMock()
        mock_swarms.run.return_value = (
            "I think we should do something but I forgot the format"
        )
        agent._swarms_agent = mock_swarms

        action = agent.act(default_observation)
        assert action.action_type == ActionType.NOOP
        # Trace should still be attached
        assert "_swarms_trace" in action.metadata

    def test_accept_interaction_trust_based(self, agent: SwarmsBackedAgent):
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

    def test_propose_interaction(self, agent: SwarmsBackedAgent):
        obs = Observation()

        # Unknown agent -> trust 0.5 -> propose
        proposal = agent.propose_interaction(obs, "counterparty_1")
        assert proposal is not None
        assert proposal.initiator_id == agent.agent_id
        assert proposal.counterparty_id == "counterparty_1"

        # Low-trust agent -> None
        agent._counterparty_memory["low_trust"] = 0.1
        assert agent.propose_interaction(obs, "low_trust") is None

    def test_trace_disabled(self, default_observation: Observation):
        """When enable_trace=False, no _swarms_trace in metadata."""
        config = SwarmsConfig(enable_trace=False)
        agent = SwarmsBackedAgent(
            agent_id="no_trace",
            swarms_config=config,
        )

        mock_swarms = MagicMock()
        mock_swarms.run.return_value = json.dumps({"kind": "noop"})
        agent._swarms_agent = mock_swarms

        action = agent.act(default_observation)
        assert "_swarms_trace" not in action.metadata

    def test_repr(self, agent: SwarmsBackedAgent):
        r = repr(agent)
        assert "SwarmsBackedAgent" in r
        assert "swarms_test_1" in r
        assert "Agent" in r


# ---------------------------------------------------------------------------
# Security: ID validation
# ---------------------------------------------------------------------------


class TestSecurityIDValidation:
    def test_valid_target_id_accepted(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """target_id present in observation is accepted."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(kind="vote", target_id="p1")
        action = agent._schema_to_action(schema)
        assert action.target_id == "p1"

    def test_invalid_target_id_rejected(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """target_id NOT in observation is rejected to empty string."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(kind="vote", target_id="fake_post_999")
        action = agent._schema_to_action(schema)
        assert action.target_id == ""

    def test_valid_counterparty_id_accepted(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """counterparty_id present in visible_agents is accepted."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(
            kind="propose_interaction", counterparty_id="other_1"
        )
        action = agent._schema_to_action(schema)
        assert action.counterparty_id == "other_1"

    def test_invalid_counterparty_id_rejected(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """counterparty_id NOT in visible_agents is rejected to empty string."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(
            kind="propose_interaction", counterparty_id="secret_admin_agent"
        )
        action = agent._schema_to_action(schema)
        assert action.counterparty_id == ""

    def test_task_id_in_available_tasks(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(kind="claim_task", target_id="t1")
        action = agent._schema_to_action(schema)
        assert action.target_id == "t1"

    def test_bounty_id_accepted(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(kind="place_bid", target_id="b1")
        action = agent._schema_to_action(schema)
        assert action.target_id == "b1"

    def test_proposal_id_accepted(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(
            kind="accept_interaction", target_id="prop_1"
        )
        action = agent._schema_to_action(schema)
        assert action.target_id == "prop_1"

    def test_empty_ids_pass_through(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """Empty IDs (default) should remain empty, not rejected."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(kind="noop")
        action = agent._schema_to_action(schema)
        assert action.target_id == ""
        assert action.counterparty_id == ""


# ---------------------------------------------------------------------------
# Security: Metadata namespace isolation
# ---------------------------------------------------------------------------


class TestSecurityMetadataNamespacing:
    def test_swarms_metadata_is_namespaced(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """Swarms-supplied metadata lives under 'swarms_metadata' key."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(
            kind="noop",
            metadata={"custom_key": "custom_val"},
        )
        action = agent._schema_to_action(schema)
        assert "swarms_metadata" in action.metadata
        assert action.metadata["swarms_metadata"]["custom_key"] == "custom_val"
        # Should NOT have swarms keys at top level
        assert "custom_key" not in action.metadata

    def test_swarms_cannot_inject_internal_keys(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """Swarms metadata with keys like 'bid_id' won't pollute top-level."""
        agent._collect_valid_ids(default_observation)
        schema = SwarmActionSchema(
            kind="noop",
            metadata={
                "bid_id": "attacker_bid",
                "reward_amount": 99999,
                "_swarms_trace": {"fake": True},
            },
        )
        action = agent._schema_to_action(schema)
        # These should only be inside swarms_metadata, not at root
        assert action.metadata.get("bid_id") is None
        assert action.metadata.get("reward_amount") is None
        # _swarms_trace in swarms_metadata is dropped (nested dict)
        assert "_swarms_trace" not in action.metadata["swarms_metadata"]


# ---------------------------------------------------------------------------
# Security: Timeout
# ---------------------------------------------------------------------------


class TestSecurityTimeout:
    def test_timeout_raises_on_slow_agent(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """An agent that exceeds the timeout triggers noop fallback."""
        import time

        mock_swarms = MagicMock()

        def slow_run(prompt):
            time.sleep(5)
            return '{"kind": "post"}'

        mock_swarms.run.side_effect = slow_run
        agent._swarms_agent = mock_swarms
        agent.swarms_config = SwarmsConfig(timeout_seconds=0.1)

        action = agent.act(default_observation)
        assert action.action_type == ActionType.NOOP

    def test_run_with_timeout_normal(self, agent: SwarmsBackedAgent):
        """Normal run within timeout succeeds."""
        mock_swarms = MagicMock()
        mock_swarms.run.return_value = "result"
        result = agent._run_with_timeout(mock_swarms, "test prompt")
        assert result == "result"

    def test_timeout_does_not_block_caller(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """After timeout, act() returns promptly (no wait for thread)."""
        import time

        mock_swarms = MagicMock()

        def slow_run(prompt):
            time.sleep(10)
            return '{"kind": "post"}'

        mock_swarms.run.side_effect = slow_run
        agent._swarms_agent = mock_swarms
        agent.swarms_config = SwarmsConfig(timeout_seconds=0.1)

        start = time.monotonic()
        action = agent.act(default_observation)
        elapsed = time.monotonic() - start

        assert action.action_type == ActionType.NOOP
        # Must return in well under the 10s sleep
        assert elapsed < 2.0

    def test_pool_reused_across_ticks(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """Thread pool is reused across normal ticks."""
        mock_swarms = MagicMock()
        mock_swarms.run.return_value = json.dumps({"kind": "noop"})
        agent._swarms_agent = mock_swarms

        agent.act(default_observation)
        pool_after_first = agent._executor_pool

        agent.act(default_observation)
        pool_after_second = agent._executor_pool

        assert pool_after_first is pool_after_second

    def test_pool_recreated_after_timeout(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """After a timeout, the old pool is discarded and a fresh one created."""
        import time

        mock_swarms = MagicMock()

        def slow_then_fast(prompt):
            if mock_swarms.run.call_count <= 1:
                time.sleep(5)
            return '{"kind": "noop"}'

        mock_swarms.run.side_effect = slow_then_fast
        agent._swarms_agent = mock_swarms
        agent.swarms_config = SwarmsConfig(timeout_seconds=0.1)

        # First call times out
        agent.act(default_observation)
        assert agent._executor_pool is None  # discarded

        # Second call creates a fresh pool
        mock_swarms.run.side_effect = None
        mock_swarms.run.return_value = json.dumps({"kind": "noop"})
        agent.swarms_config = SwarmsConfig(timeout_seconds=120.0)

        action = agent.act(default_observation)
        assert action.action_type == ActionType.NOOP
        assert agent._executor_pool is not None

    def test_timeout_specific_log_message(
        self, agent: SwarmsBackedAgent, default_observation: Observation, caplog
    ):
        """Timeout produces a distinct warning, not the generic exception log."""
        import logging
        import time

        mock_swarms = MagicMock()
        mock_swarms.run.side_effect = lambda prompt: time.sleep(5)
        agent._swarms_agent = mock_swarms
        agent.swarms_config = SwarmsConfig(timeout_seconds=0.1)

        with caplog.at_level(logging.WARNING):
            agent.act(default_observation)

        assert any("timed out after" in r.message for r in caplog.records)
        assert not any(
            "deliberation failed" in r.message for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Security: Deliberation memory cap
# ---------------------------------------------------------------------------


class TestSecurityDeliberationMemory:
    def test_memory_capped(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """Memory never exceeds MAX_DELIBERATION_MEMORY."""
        mock_swarms = MagicMock()
        mock_swarms.run.return_value = json.dumps({"kind": "noop"})
        agent._swarms_agent = mock_swarms

        for i in range(MAX_DELIBERATION_MEMORY + 50):
            obs = Observation(
                agent_state=default_observation.agent_state,
                current_epoch=i,
                current_step=0,
            )
            agent.act(obs)

        assert len(agent._deliberation_memory) == MAX_DELIBERATION_MEMORY

    def test_memory_rationale_truncated(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """Rationales in memory are truncated to 500 chars."""
        mock_swarms = MagicMock()
        mock_swarms.run.return_value = json.dumps({
            "kind": "noop",
            "rationale": "x" * 1500,
        })
        agent._swarms_agent = mock_swarms

        agent.act(default_observation)

        entry = agent._deliberation_memory[-1]
        assert len(entry["rationale"]) == 500


# ---------------------------------------------------------------------------
# Security: Trace truncation
# ---------------------------------------------------------------------------


class TestSecurityTraceOutput:
    def test_raw_output_truncated_in_trace(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """raw_output in trace is capped at MAX_RAW_TRACE_LENGTH."""
        mock_swarms = MagicMock()
        mock_swarms.run.return_value = json.dumps({"kind": "noop"}) + " " * 10_000
        agent._swarms_agent = mock_swarms

        action = agent.act(default_observation)

        trace = action.metadata.get("_swarms_trace", {})
        assert len(trace.get("raw_output", "")) <= MAX_RAW_TRACE_LENGTH

    def test_rationale_truncated_in_trace(
        self, agent: SwarmsBackedAgent, default_observation: Observation
    ):
        """rationale in trace is capped at MAX_RATIONALE_LENGTH."""
        mock_swarms = MagicMock()
        long_rationale = "r" * 1999  # within schema limit
        mock_swarms.run.return_value = json.dumps({
            "kind": "noop",
            "rationale": long_rationale,
        })
        agent._swarms_agent = mock_swarms

        action = agent.act(default_observation)

        trace = action.metadata.get("_swarms_trace", {})
        assert len(trace.get("rationale", "")) <= MAX_RATIONALE_LENGTH


# ---------------------------------------------------------------------------
# Loader integration tests
# ---------------------------------------------------------------------------


class TestLoaderIntegration:
    def test_lazy_import_swarms_classes(self):
        """Verify the lazy import function in the loader works."""
        from swarm.scenarios.loader import _get_swarms_classes

        AgentCls, ConfigCls = _get_swarms_classes()
        assert AgentCls is SwarmsBackedAgent
        assert ConfigCls is SwarmsConfig

    def test_create_agents_swarms_type(self):
        """create_agents handles type='swarms_adapter'."""
        from swarm.scenarios.loader import create_agents

        specs = [
            {
                "type": "swarms_adapter",
                "count": 2,
                "params": {
                    "model_name": "gpt-4o",
                    "architecture": "Agent",
                    "max_loops": 2,
                },
            }
        ]
        agents = create_agents(specs, seed=99)
        assert len(agents) == 2
        assert all(isinstance(a, SwarmsBackedAgent) for a in agents)
        assert agents[0].agent_id == "swarms_1"
        assert agents[1].agent_id == "swarms_2"
        assert agents[0].swarms_config.model_name == "gpt-4o"
        assert agents[0].swarms_config.max_loops == 2

    def test_create_agents_timeout_from_yaml(self):
        """timeout_seconds param in YAML is wired to SwarmsConfig."""
        from swarm.scenarios.loader import create_agents

        specs = [
            {
                "type": "swarms_adapter",
                "count": 1,
                "params": {
                    "timeout_seconds": 30.0,
                },
            }
        ]
        agents = create_agents(specs, seed=1)
        assert agents[0].swarms_config.timeout_seconds == 30.0

    def test_create_agents_default_timeout(self):
        """Without timeout in YAML, default is used."""
        from swarm.scenarios.loader import create_agents

        specs = [
            {
                "type": "swarms_adapter",
                "count": 1,
                "params": {},
            }
        ]
        agents = create_agents(specs, seed=1)
        assert agents[0].swarms_config.timeout_seconds == DEFAULT_SWARMS_TIMEOUT_SECONDS

    def test_create_agents_mixed(self):
        """Swarms agents can coexist with scripted agents."""
        from swarm.scenarios.loader import create_agents

        specs = [
            {"type": "honest", "count": 2},
            {
                "type": "swarms_adapter",
                "count": 1,
                "params": {"model_name": "gpt-4o-mini"},
            },
            {"type": "opportunistic", "count": 1},
        ]
        agents = create_agents(specs, seed=42)
        assert len(agents) == 4
        swarms_agents = [a for a in agents if isinstance(a, SwarmsBackedAgent)]
        assert len(swarms_agents) == 1

    def test_create_agents_system_prompt_from_yaml(self):
        """system_prompt param in YAML is wired to SwarmsConfig."""
        from swarm.scenarios.loader import create_agents

        custom_prompt = "You are a specialized safety auditor."
        specs = [
            {
                "type": "swarms_adapter",
                "count": 1,
                "params": {"system_prompt": custom_prompt},
            }
        ]
        agents = create_agents(specs, seed=1)
        assert agents[0].swarms_config.system_prompt == custom_prompt


# ---------------------------------------------------------------------------
# Action kind mapping exhaustiveness
# ---------------------------------------------------------------------------


class TestActionKindMapping:
    @pytest.mark.parametrize("kind", _VALID_KINDS)
    def test_each_kind_maps_to_action_type(self, kind: str):
        action_type = _ACTION_KIND_MAP[kind]
        assert isinstance(action_type, ActionType)

    def test_all_mapped_action_types_are_unique(self):
        """Each kind string maps to a distinct ActionType."""
        values = list(_ACTION_KIND_MAP.values())
        assert len(values) == len(set(values))
