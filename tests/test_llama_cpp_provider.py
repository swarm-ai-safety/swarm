"""Tests for llama.cpp provider integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swarm.agents.llm_agent import LLMAgent
from swarm.agents.llm_config import LLMConfig, LLMProvider
from swarm.agents.llm_health import check_llama_server, require_llama_server

# =============================================================================
# LLMConfig — LLAMA_CPP defaults
# =============================================================================


class TestLlamaCppConfig:
    """Tests for LLMConfig with LLAMA_CPP provider."""

    def test_default_base_url(self):
        cfg = LLMConfig(provider=LLMProvider.LLAMA_CPP, model="test-model")
        assert cfg.base_url == "http://localhost:8080/v1"

    def test_custom_base_url_preserved(self):
        cfg = LLMConfig(
            provider=LLMProvider.LLAMA_CPP,
            model="test-model",
            base_url="http://myhost:9090/v1",
        )
        assert cfg.base_url == "http://myhost:9090/v1"

    def test_api_key_not_required(self):
        """LLAMA_CPP should not require an API key."""
        cfg = LLMConfig(provider=LLMProvider.LLAMA_CPP, model="test-model")
        assert cfg.api_key is None  # no error raised

    def test_model_path_field(self):
        cfg = LLMConfig(
            provider=LLMProvider.LLAMA_CPP,
            model="local",
            model_path="/tmp/model.gguf",
            n_ctx=2048,
            n_threads=4,
            llama_seed=42,
        )
        assert cfg.model_path == "/tmp/model.gguf"
        assert cfg.n_ctx == 2048
        assert cfg.n_threads == 4
        assert cfg.llama_seed == 42

    def test_default_in_process_fields(self):
        cfg = LLMConfig(provider=LLMProvider.LLAMA_CPP, model="test")
        assert cfg.model_path is None
        assert cfg.n_ctx == 4096
        assert cfg.n_threads is None
        assert cfg.llama_seed == -1


# =============================================================================
# LLMAgent — provider routing
# =============================================================================


class TestLlamaCppRouting:
    """Tests that LLAMA_CPP routes to the correct call method."""

    def test_api_key_returns_dummy(self):
        cfg = LLMConfig(provider=LLMProvider.LLAMA_CPP, model="test")
        agent = LLMAgent(agent_id="a1", llm_config=cfg)
        assert agent._api_key == "not-needed"

    @pytest.mark.asyncio
    async def test_routes_to_openai_compatible_when_no_model_path(self):
        """Option A: no model_path -> _call_openai_compatible_async."""
        cfg = LLMConfig(provider=LLMProvider.LLAMA_CPP, model="test-model")
        agent = LLMAgent(agent_id="a1", llm_config=cfg)

        mock_result = ("response text", 10, 5)
        with patch.object(
            agent, "_call_openai_compatible_async", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_result
            result = await agent._call_llm_async("system", "user")
            mock_call.assert_awaited_once_with("system", "user")
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_routes_to_direct_when_model_path_set(self):
        """Option B: model_path set -> _call_llama_cpp_direct_async."""
        cfg = LLMConfig(
            provider=LLMProvider.LLAMA_CPP,
            model="local",
            model_path="/tmp/fake.gguf",
        )
        agent = LLMAgent(agent_id="a1", llm_config=cfg)

        mock_result = ("response text", 10, 5)
        with patch.object(
            agent, "_call_llama_cpp_direct_async", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_result
            result = await agent._call_llm_async("system", "user")
            mock_call.assert_awaited_once_with("system", "user")
            assert result == mock_result


# =============================================================================
# _call_llama_cpp_direct_async — unit test with mocked llama_cpp
# =============================================================================


class TestCallLlamaCppDirectAsync:
    """Test the in-process llama-cpp-python call path."""

    @pytest.mark.asyncio
    async def test_direct_call_returns_text_and_tokens(self):
        cfg = LLMConfig(
            provider=LLMProvider.LLAMA_CPP,
            model="local",
            model_path="/tmp/fake.gguf",
        )
        agent = LLMAgent(agent_id="a1", llm_config=cfg)

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello from llama!"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        }
        agent._llama_model = mock_model

        text, inp, out = await agent._call_llama_cpp_direct_async(
            "You are helpful.", "Say hello."
        )

        assert text == "Hello from llama!"
        assert inp == 20
        assert out == 8
        mock_model.create_chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_direct_call_handles_missing_usage(self):
        cfg = LLMConfig(
            provider=LLMProvider.LLAMA_CPP,
            model="local",
            model_path="/tmp/fake.gguf",
        )
        agent = LLMAgent(agent_id="a1", llm_config=cfg)

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hi"}}],
        }
        agent._llama_model = mock_model

        text, inp, out = await agent._call_llama_cpp_direct_async("sys", "usr")
        assert text == "Hi"
        assert inp == 0
        assert out == 0


# =============================================================================
# Health check
# =============================================================================


class TestLlamaHealthCheck:
    """Tests for the health check utility."""

    def test_check_returns_false_on_unreachable(self):
        """Health check should return False when server is not running."""
        result = check_llama_server(
            base_url="http://localhost:19999", timeout=1.0
        )
        assert result is False

    def test_require_raises_on_unreachable(self):
        with pytest.raises(ConnectionError, match="Cannot reach llama-server"):
            require_llama_server(
                base_url="http://localhost:19999", timeout=1.0
            )

    def test_require_includes_model_hint(self):
        with pytest.raises(ConnectionError, match="my-model"):
            require_llama_server(
                base_url="http://localhost:19999",
                timeout=1.0,
                model_hint="my-model",
            )

    def test_check_strips_v1_suffix(self):
        """base_url with /v1 should still hit /health on the root."""
        result = check_llama_server(
            base_url="http://localhost:19999/v1", timeout=1.0
        )
        assert result is False  # just verifying no crash

    @patch("urllib.request.urlopen")
    def test_check_returns_true_when_healthy(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        assert check_llama_server() is True


# =============================================================================
# _get_llama_model — import error handling
# =============================================================================


class TestGetLlamaModel:
    """Test lazy model loading error handling."""

    def test_raises_import_error_without_llama_cpp(self):
        cfg = LLMConfig(
            provider=LLMProvider.LLAMA_CPP,
            model="local",
            model_path="/tmp/fake.gguf",
        )
        agent = LLMAgent(agent_id="a1", llm_config=cfg)

        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError, match="llama-cpp-python"):
                agent._get_llama_model()

    def test_raises_value_error_without_model_path(self):
        cfg = LLMConfig(
            provider=LLMProvider.LLAMA_CPP,
            model="local",
        )
        agent = LLMAgent(agent_id="a1", llm_config=cfg)

        # Ensure the import itself wouldn't fail
        mock_llama_mod = MagicMock()
        with patch.dict("sys.modules", {"llama_cpp": mock_llama_mod}):
            with pytest.raises(ValueError, match="model_path is required"):
                agent._get_llama_model()
