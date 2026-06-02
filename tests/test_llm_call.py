"""Tests for swarm/judges/llm_call.py — provider-dispatch helpers.

The provider call functions themselves require network + API keys and
are not exercised here. We test the surface that's deterministic and
provider-agnostic: retry semantics, API-key resolution.
"""

from __future__ import annotations

import pytest

from swarm.judges.llm_call import (
    LLMCallResult,
    _resolve_api_key,
    call_with_retries,
)


class TestResolveApiKey:
    def test_explicit_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        assert _resolve_api_key("anthropic", "explicit") == "explicit"

    def test_falls_back_to_env(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        assert _resolve_api_key("openai", None) == "from-env"

    def test_returns_none_when_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _resolve_api_key("anthropic", None) is None

    def test_ollama_does_not_need_key(self) -> None:
        # Ollama is keyless; the function should just return None and the
        # caller (call_ollama) ignores the key entirely.
        assert _resolve_api_key("ollama", None) is None


class TestCallWithRetries:
    def _ok(self):
        return LLMCallResult(text="ok", input_tokens=1, output_tokens=1, latency_seconds=0.01)

    def test_no_retry_on_success(self) -> None:
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            return self._ok()

        result = call_with_retries(fn, max_retries=3, sleep=lambda _s: None)
        assert result.text == "ok"
        assert attempts["n"] == 1

    def test_retries_then_succeeds(self) -> None:
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            return self._ok()

        result = call_with_retries(fn, max_retries=5, sleep=lambda _s: None)
        assert result.text == "ok"
        assert attempts["n"] == 3

    def test_exhausts_and_raises(self) -> None:
        def fn():
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError, match="permanent"):
            call_with_retries(fn, max_retries=2, sleep=lambda _s: None)

    def test_backoff_doubles(self) -> None:
        delays: list[float] = []

        def fn():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            call_with_retries(
                fn, max_retries=3, base_delay=1.0, sleep=lambda s: delays.append(s)
            )
        # delays should be [1, 2, 4]: 2^0, 2^1, 2^2 scaled by base_delay
        assert delays == [1.0, 2.0, 4.0]
