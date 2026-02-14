"""LLM judge for evaluating Concordia narrative interactions."""

import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from typing import Any, Callable

from swarm.bridges.concordia.config import JudgeConfig
from swarm.bridges.concordia.events import JudgeScores

logger = logging.getLogger(__name__)

# Type for an LLM call function: (prompt, model, temperature) -> response_text
LLMCallFn = Callable[[str, str, float], str]


class LLMJudge:
    """Evaluates narrative text and produces JudgeScores.

    Without an llm_client, returns default scores (stub mode).
    With an llm_client callable, makes actual LLM calls.
    """

    def __init__(
        self,
        config: JudgeConfig | None = None,
        llm_client: LLMCallFn | None = None,
    ):
        self._config = config or JudgeConfig()
        self._llm_client = llm_client
        self._cache: OrderedDict[str, JudgeScores] = OrderedDict()
        self._cache_lock = threading.Lock()

    def evaluate(self, narrative: str) -> JudgeScores:
        """Evaluate a narrative and return scores."""
        truncated = self._truncate(narrative)

        # Check cache
        if self._config.cache_enabled:
            key = self._cache_key(truncated)
            with self._cache_lock:
                if key in self._cache:
                    cached = self._cache[key]
                    return JudgeScores(
                        progress=cached.progress,
                        quality=cached.quality,
                        cooperation=cached.cooperation,
                        harm=cached.harm,
                        raw_response=cached.raw_response,
                        cached=True,
                    )

        # Stub mode: return defaults if no LLM client
        if self._llm_client is None:
            scores = JudgeScores(
                progress=0.5,
                quality=0.5,
                cooperation=0.5,
                harm=0.0,
                raw_response="",
                cached=False,
            )
        else:
            prompt = self._config.prompt_template.format(narrative=truncated)
            response = self._llm_client(
                prompt,
                self._config.model,
                self._config.temperature,
            )
            scores = self._parse_scores(response)

        # Update cache
        if self._config.cache_enabled:
            key = self._cache_key(truncated)
            with self._cache_lock:
                self._cache[key] = scores
                # Evict oldest if over limit
                while len(self._cache) > self._config.cache_max_size:
                    self._cache.popitem(last=False)

        return scores

    def evaluate_batch(self, narratives: list[str]) -> list[JudgeScores]:
        """Evaluate multiple narratives."""
        return [self.evaluate(n) for n in narratives]

    def _parse_scores(self, response: str) -> JudgeScores:
        """Extract scores from LLM response JSON."""
        # Try to find JSON in the response
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            try:
                data: dict[str, Any] = json.loads(json_match.group())
                return JudgeScores(
                    progress=float(data.get("progress", 0.5)),
                    quality=float(data.get("quality", 0.5)),
                    cooperation=float(data.get("cooperation", 0.5)),
                    harm=float(data.get("harm", 0.0)),
                    raw_response=response,
                    cached=False,
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        logger.warning("Failed to parse judge scores from response: %s", response[:200])
        return JudgeScores(
            progress=0.5,
            quality=0.5,
            cooperation=0.5,
            harm=0.0,
            raw_response=response,
            cached=False,
        )

    def _truncate(self, text: str) -> str:
        """Truncate text to max_chars."""
        if len(text) <= self._config.max_chars:
            return text
        return text[: self._config.max_chars]

    def _cache_key(self, text: str) -> str:
        """Compute SHA256 cache key."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
