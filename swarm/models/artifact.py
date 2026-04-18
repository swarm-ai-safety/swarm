"""Typed artifact models for the emergent coordination layer.

Handlers publish ``Artifact`` instances after producing outputs; agents
declare ``ArtifactNeed``s for inputs they require. The ``ArtifactRegistry``
(in ``swarm.env.artifact_registry``) matches supply to demand and
maintains pressure scores. ``ArtifactSchema`` provides structural
compatibility checks used by schema-overlap matching (ScienceClaw
ArtifactReactor-style).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class ArtifactSchema(BaseModel):
    """Structural description of an artifact kind.

    Two schemas are compatible when *have* contains every field *need*
    declares, with matching type names. Empty ``fields`` in *need*
    means any artifact of the same kind is accepted.
    """

    model_config = ConfigDict(frozen=False)

    kind: str = ""
    fields: Dict[str, str] = Field(default_factory=dict)

    def is_compatible(self, have: "ArtifactSchema") -> bool:
        """True iff *have* is a type-matching superset of this schema."""
        for name, typ in self.fields.items():
            if have.fields.get(name) != typ:
                return False
        return True


class Artifact(BaseModel):
    """A typed output published by a handler.

    ``p_at_production`` is the soft-label quality estimate at the moment
    of creation; consumers can filter on it. ``consumed_by`` tracks the
    interaction_ids that later consumed this artifact (used to wire
    causal parents). It is intentionally excluded from ``to_dict()``.
    """

    model_config = ConfigDict(frozen=False)

    artifact_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    kind: str = ""
    producer_id: str = ""
    interaction_id: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    step: int = 0
    p_at_production: float = 0.5
    consumed_by: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for observation/export — omits ``consumed_by``."""
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "producer_id": self.producer_id,
            "interaction_id": self.interaction_id,
            "data": dict(self.data),
            "step": self.step,
            "p_at_production": self.p_at_production,
        }


class ArtifactNeed(BaseModel):
    """An unsatisfied input requirement declared by an agent.

    Matched against the registry by ``kind``, with optional quality
    floor (``min_p``) and freshness bound (``max_age_steps``).
    """

    model_config = ConfigDict(frozen=False)

    kind: str = ""
    requester_id: str = ""
    min_p: float = 0.0
    max_age_steps: int = 2_147_483_647
    fields: Dict[str, str] = Field(default_factory=dict)
