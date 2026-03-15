"""Lesson persistence for autoresearch governance tuning.

Tracks which governance lever mutations have been tried, their outcomes,
and whether they should be skipped in future iterations.  Backed by a
single JSON file per scenario so lessons survive across sessions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Lesson:
    """A single governance mutation trial and its outcome."""

    param: str
    old_value: Any
    new_value: Any
    accepted: bool
    primary_metric: str
    primary_value: float
    baseline_value: float
    iteration: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    guardrail_errors: list[str] = field(default_factory=list)
    scenario: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        return self.primary_value - self.baseline_value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lesson:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LessonStore:
    """JSON-backed lesson store for a single scenario.

    Stores lessons at ``{root}/{scenario}/lessons.json``.
    """

    def __init__(self, root: str | Path, scenario: str) -> None:
        self.root = Path(root)
        self.scenario = scenario
        self._path = self.root / scenario / "lessons.json"
        self._lessons: list[Lesson] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._lessons = [Lesson.from_dict(d) for d in raw]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([lesson.to_dict() for lesson in self._lessons], indent=2),
            encoding="utf-8",
        )

    def add(self, lesson: Lesson) -> None:
        self._lessons.append(lesson)
        self._save()

    def all(self) -> list[Lesson]:
        return list(self._lessons)

    def accepted(self) -> list[Lesson]:
        return [lesson for lesson in self._lessons if lesson.accepted]

    def rejected(self) -> list[Lesson]:
        return [lesson for lesson in self._lessons if not lesson.accepted]

    def for_param(self, param: str) -> list[Lesson]:
        return [lesson for lesson in self._lessons if lesson.param == param]

    def was_tried(self, param: str, new_value: Any, tolerance: float = 1e-6) -> bool:
        """Check if a specific param->value mutation was already tried."""
        for lesson in self._lessons:
            if lesson.param != param:
                continue
            if isinstance(new_value, bool):
                if lesson.new_value == new_value:
                    return True
            elif isinstance(new_value, (int, float)):
                if abs(float(lesson.new_value) - float(new_value)) < tolerance:
                    return True
            elif lesson.new_value == new_value:
                return True
        return False

    def best_known_value(self, param: str) -> Any | None:
        """Return the best accepted value for a parameter, or None."""
        hits = [lesson for lesson in self.accepted() if lesson.param == param]
        if not hits:
            return None
        return max(hits, key=lambda x: x.improvement).new_value

    def summary(self) -> dict[str, Any]:
        """Return a summary of the lesson store."""
        return {
            "scenario": self.scenario,
            "total_lessons": len(self._lessons),
            "accepted": len(self.accepted()),
            "rejected": len(self.rejected()),
            "params_tried": sorted({lesson.param for lesson in self._lessons}),
        }

    def __len__(self) -> int:
        return len(self._lessons)
