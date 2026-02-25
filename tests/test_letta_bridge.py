"""Unit tests for the Letta bridge components.

Tests config validation, memory mapper serialization, response parser
(JSON extraction, fallback to NOOP, ID validation), and observable mapper.
All mocked - no real Letta server needed.
"""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from swarm.agents.base import ActionType, Observation
from swarm.bridges.letta.client import LettaSwarmClient
from swarm.bridges.letta.config import LettaConfig
from swarm.bridges.letta.memory_mapper import LettaMemoryMapper
from swarm.bridges.letta.response_parser import LettaResponseParser

# ---- Config Tests ----


class TestLettaConfig:
    def test_default_config(self):
        config = LettaConfig()
        assert config.base_url == "http://localhost:8283"
        assert config.server_mode == "external"
        assert config.enabled is True
        assert config.shared_governance_block is True

    def test_custom_config(self):
        config = LettaConfig(
            base_url="http://letta:9000",
            api_key="test-key",
            server_mode="docker",
            default_model="openai/gpt-4o",
            core_memory_limit=10000,
        )
        assert config.base_url == "http://letta:9000"
        assert config.api_key == "test-key"
        assert config.server_mode == "docker"
        assert config.default_model == "openai/gpt-4o"
        assert config.core_memory_limit == 10000

    def test_memory_limit_validation(self):
        with pytest.raises(ValueError):
            LettaConfig(core_memory_limit=50)  # Below minimum of 100


# ---- Memory Mapper Tests ----


class TestLettaMemoryMapper:
    def setup_method(self):
        self.config = LettaConfig()
        self.mapper = LettaMemoryMapper(self.config)

    def test_observation_to_memory_blocks(self):
        obs = Observation(
            current_epoch=2,
            current_step=5,
            can_post=True,
            can_interact=True,
            ecosystem_metrics={"toxicity_rate": 0.1},
        )
        blocks = self.mapper.observation_to_memory_blocks(obs)
        assert "swarm_state" in blocks
        state = json.loads(blocks["swarm_state"])
        assert state["epoch"] == 2
        assert state["step"] == 5
        assert state["can_post"] is True

    def test_observation_to_message(self):
        obs = Observation(
            current_epoch=1,
            current_step=3,
            can_post=True,
            visible_posts=[{"post_id": "p1", "content": "Hello world"}],
            pending_proposals=[
                {"proposal_id": "prop1", "initiator_id": "agent_2"}
            ],
        )
        msg = self.mapper.observation_to_message(obs)
        assert "Epoch 1, Step 3" in msg
        assert "post" in msg.lower()
        assert "p1" in msg
        assert "prop1" in msg or "agent_2" in msg

    def test_extract_trust_updates_valid(self):
        content = '{"agent_1": 0.8, "agent_2": 0.3}'
        result = self.mapper.extract_trust_updates(content)
        assert result == {"agent_1": 0.8, "agent_2": 0.3}

    def test_extract_trust_updates_empty(self):
        assert self.mapper.extract_trust_updates("") == {}
        assert self.mapper.extract_trust_updates("not json") == {}

    def test_governance_state_to_block(self):
        metrics = {"toxicity_rate": 0.15, "quality_gap": -0.2}
        proposals = [{"id": "gov_1", "type": "freeze"}]
        result = self.mapper.governance_state_to_block(metrics, proposals)
        parsed = json.loads(result)
        assert parsed["ecosystem_metrics"]["toxicity_rate"] == 0.15
        assert len(parsed["active_proposals"]) == 1


# ---- Response Parser Tests ----


class TestLettaResponseParser:
    def setup_method(self):
        self.parser = LettaResponseParser()
        self.obs = Observation(
            visible_posts=[
                {"post_id": "post_42", "content": "test"},
                {"post_id": "post_43", "content": "test2"},
            ],
            pending_proposals=[
                {"proposal_id": "prop_1", "initiator_id": "agent_2"}
            ],
            visible_agents=[
                {"agent_id": "agent_2"},
                {"agent_id": "agent_3"},
            ],
        )

    def test_parse_json_post(self):
        response = '{"action": "post", "content": "Hello everyone!"}'
        action = self.parser.parse(response, "agent_1", self.obs)
        assert action.action_type == ActionType.POST
        assert action.content == "Hello everyone!"
        assert action.agent_id == "agent_1"

    def test_parse_json_vote(self):
        response = '{"action": "vote", "target_id": "post_42", "vote_direction": 1}'
        action = self.parser.parse(response, "agent_1", self.obs)
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "post_42"
        assert action.vote_direction == 1

    def test_parse_json_vote_invalid_target(self):
        response = '{"action": "vote", "target_id": "bad_id", "vote_direction": 1}'
        # Use observation without pending proposals so heuristic won't trigger
        obs_no_proposals = Observation(
            visible_posts=[{"post_id": "post_42", "content": "test"}],
        )
        action = self.parser.parse(response, "agent_1", obs_no_proposals)
        # Should fall back to NOOP because target_id is invalid
        assert action.action_type == ActionType.NOOP

    def test_parse_json_propose(self):
        response = '{"action": "propose_interaction", "counterparty_id": "agent_2"}'
        action = self.parser.parse(response, "agent_1", self.obs)
        assert action.action_type == ActionType.PROPOSE_INTERACTION
        assert action.counterparty_id == "agent_2"

    def test_parse_json_propose_invalid_counterparty(self):
        response = (
            '{"action": "propose_interaction", "counterparty_id": "bad_id"}'
        )
        # Use observation without pending proposals so heuristic won't trigger
        obs_no_proposals = Observation(
            visible_agents=[{"agent_id": "agent_2"}],
        )
        action = self.parser.parse(response, "agent_1", obs_no_proposals)
        assert action.action_type == ActionType.NOOP

    def test_parse_json_accept(self):
        response = '{"action": "accept", "target_id": "prop_1"}'
        action = self.parser.parse(response, "agent_1", self.obs)
        assert action.action_type == ActionType.ACCEPT_INTERACTION
        assert action.target_id == "prop_1"

    def test_parse_json_reject(self):
        response = '{"action": "reject", "target_id": "prop_1"}'
        action = self.parser.parse(response, "agent_1", self.obs)
        assert action.action_type == ActionType.REJECT_INTERACTION

    def test_parse_embedded_json(self):
        response = (
            'I think I should post something. {"action": "post", '
            '"content": "My contribution"} That is my action.'
        )
        action = self.parser.parse(response, "agent_1", self.obs)
        assert action.action_type == ActionType.POST
        assert action.content == "My contribution"

    def test_parse_noop_on_garbage(self):
        response = "I am thinking about what to do."
        # Use observation without pending proposals so heuristic won't trigger
        obs_no_proposals = Observation()
        action = self.parser.parse(response, "agent_1", obs_no_proposals)
        assert action.action_type == ActionType.NOOP

    def test_parse_empty_response(self):
        action = self.parser.parse("", "agent_1", self.obs)
        assert action.action_type == ActionType.NOOP

    def test_heuristic_accept(self):
        response = "Yes, I accept this proposal!"
        action = self.parser.parse(response, "agent_1", self.obs)
        assert action.action_type == ActionType.ACCEPT_INTERACTION
        assert action.target_id == "prop_1"

    def test_heuristic_reject(self):
        response = "No, I decline this interaction."
        action = self.parser.parse(response, "agent_1", self.obs)
        assert action.action_type == ActionType.REJECT_INTERACTION

    def test_vote_direction_clamped(self):
        response = '{"action": "vote", "target_id": "post_42", "vote_direction": 99}'
        action = self.parser.parse(response, "agent_1", self.obs)
        assert action.action_type == ActionType.VOTE
        assert action.vote_direction == 1  # Clamped to max 1

    def test_vote_direction_default_upvote(self):
        response = '{"action": "vote", "target_id": "post_42"}'
        action = self.parser.parse(response, "agent_1", self.obs)
        assert action.action_type == ActionType.VOTE
        assert action.vote_direction == 1  # Default to upvote


# ---- Ollama streaming compatibility tests ----


class TestLettaClientStreamingCompat:
    """Tests for _is_ollama_model, _should_stream, and send_message routing."""

    def _make_client(self, **kwargs) -> LettaSwarmClient:
        return LettaSwarmClient(LettaConfig(**kwargs))

    # --- _is_ollama_model ---

    def test_is_ollama_model_ollama_prefix(self):
        assert LettaSwarmClient._is_ollama_model("ollama/llama3") is True

    def test_is_ollama_model_ollama_prefix_uppercase(self):
        assert LettaSwarmClient._is_ollama_model("Ollama/Mistral") is True

    def test_is_ollama_model_cloud_model(self):
        assert LettaSwarmClient._is_ollama_model("anthropic/claude-sonnet-4-20250514") is False

    def test_is_ollama_model_openai(self):
        assert LettaSwarmClient._is_ollama_model("openai/gpt-4o") is False

    # --- _should_stream (auto-detect) ---

    def test_should_stream_auto_cloud_model(self):
        client = self._make_client()  # default model is Anthropic
        assert client._should_stream() is True

    def test_should_stream_auto_ollama_model(self):
        client = self._make_client(default_model="ollama/llama3")
        assert client._should_stream() is False

    def test_should_stream_auto_explicit_ollama_override(self):
        client = self._make_client()  # default model is Anthropic
        # Override via model arg
        assert client._should_stream(model="ollama/llama3") is False

    def test_should_stream_config_override_true(self):
        client = self._make_client(
            default_model="ollama/llama3", stream_tokens=True
        )
        assert client._should_stream() is True

    def test_should_stream_config_override_false(self):
        client = self._make_client(
            default_model="anthropic/claude-sonnet-4-20250514",
            stream_tokens=False,
        )
        assert client._should_stream() is False

    # --- _extract_texts_from_messages ---

    def test_extract_texts_assistant_message(self):
        msg = MagicMock()
        msg.assistant_message = "Hello from Letta"
        msg.content = None
        texts = LettaSwarmClient._extract_texts_from_messages([msg])
        assert texts == ["Hello from Letta"]

    def test_extract_texts_content_field(self):
        msg = MagicMock(spec=[])
        msg.content = "Some content"
        texts = LettaSwarmClient._extract_texts_from_messages([msg])
        assert texts == ["Some content"]

    def test_extract_texts_skips_empty(self):
        msg = MagicMock(spec=[])
        # No assistant_message, no content attributes
        texts = LettaSwarmClient._extract_texts_from_messages([msg])
        assert texts == []

    # --- send_message routing (sync path for Ollama) ---

    def _make_client_with_mock_sdk(self, model: str, stream_tokens=None):
        kwargs: dict = {"default_model": model}
        if stream_tokens is not None:
            kwargs["stream_tokens"] = stream_tokens
        client = self._make_client(**kwargs)

        # Stub _ensure_client so tests never touch private state or the SDK import
        sdk = MagicMock()
        client._ensure_client = MagicMock(return_value=sdk)
        return client, sdk

    def test_send_message_uses_sync_for_ollama(self):
        client, sdk = self._make_client_with_mock_sdk("ollama/llama3")

        # Set up non-streaming response
        msg = MagicMock()
        msg.assistant_message = "Ollama reply"
        msg.content = None
        sdk.agents.messages.create.return_value = MagicMock(messages=[msg])

        result = client.send_message("agent-123", "hello")

        sdk.agents.messages.create.assert_called_once()
        sdk.agents.messages.create_stream.assert_not_called()
        assert result == "Ollama reply"

    def test_send_message_uses_stream_for_cloud(self):
        client, sdk = self._make_client_with_mock_sdk(
            "anthropic/claude-sonnet-4-20250514"
        )

        # Set up streaming events
        event = MagicMock()
        event.assistant_message = "Cloud reply"
        event.content = None

        @contextmanager
        def fake_stream(*args, **kwargs):
            yield [event]

        sdk.agents.messages.create_stream = fake_stream

        result = client.send_message("agent-456", "hello")

        sdk.agents.messages.create.assert_not_called()
        assert result == "Cloud reply"

    def test_send_message_stream_tokens_false_forces_sync(self):
        client, sdk = self._make_client_with_mock_sdk(
            "anthropic/claude-sonnet-4-20250514", stream_tokens=False
        )

        msg = MagicMock()
        msg.assistant_message = "Forced sync reply"
        msg.content = None
        sdk.agents.messages.create.return_value = MagicMock(messages=[msg])

        result = client.send_message("agent-789", "hello")

        sdk.agents.messages.create.assert_called_once()
        assert result == "Forced sync reply"

    def test_send_message_stream_tokens_true_forces_stream_for_ollama(self):
        client, sdk = self._make_client_with_mock_sdk(
            "ollama/llama3", stream_tokens=True
        )

        event = MagicMock()
        event.assistant_message = "Forced stream reply"
        event.content = None

        @contextmanager
        def fake_stream(*args, **kwargs):
            yield [event]

        sdk.agents.messages.create_stream = fake_stream

        result = client.send_message("agent-000", "hello")

        sdk.agents.messages.create.assert_not_called()
        assert result == "Forced stream reply"

    # --- LettaConfig new fields ---

    def test_config_default_ollama_base_url(self):
        config = LettaConfig()
        assert config.ollama_base_url == "http://localhost:11434"

    def test_config_stream_tokens_defaults_to_none(self):
        config = LettaConfig()
        assert config.stream_tokens is None

    def test_config_custom_ollama_base_url(self):
        config = LettaConfig(ollama_base_url="http://myhost:11434")
        assert config.ollama_base_url == "http://myhost:11434"
