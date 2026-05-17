"""Governance utilities tuned for SciAgentGym-style tool loop failures."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolCallFingerprint:
    """Compact representation used for loop detection."""

    tool_name: str
    args_hash: str
    error_class: str | None


@dataclass
class ToolLoopDetector:
    """n-gram repetition detector for tool-call loops.

    Tracks recent calls and estimates how much repetition exists over a moving
    window. When repeat_ratio exceeds max_repeat_ratio, governance can trigger a
    circuit breaker or force a re-planning phase.
    """

    ngram_size: int = 3
    window_size: int = 24
    max_repeat_ratio: float = 0.55
    _calls: deque[ToolCallFingerprint] = field(default_factory=deque)

    def record(self, fingerprint: ToolCallFingerprint) -> None:
        """Add a call fingerprint to the rolling history window."""
        self._calls.append(fingerprint)
        while len(self._calls) > self.window_size:
            self._calls.popleft()

    def repeat_ratio(self) -> float:
        """Return repeated n-gram ratio over the current window."""
        calls = list(self._calls)
        if len(calls) < self.ngram_size:
            return 0.0

        ngrams = [tuple(calls[i : i + self.ngram_size]) for i in range(len(calls) - self.ngram_size + 1)]
        counts = Counter(ngrams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        return repeated / len(ngrams)

    def should_trip_circuit_breaker(self) -> bool:
        """Whether repeat ratio crossed configured threshold."""
        return self.repeat_ratio() >= self.max_repeat_ratio
