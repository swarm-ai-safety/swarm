"""Typed artifacts for emergent tool chaining across agent interactions.

Inspired by ScienceClaw's artifact exchange layer (Wang et al., 2026),
this module provides typed, schema-aware artifacts that flow between
interactions.  When an interaction consumes an artifact produced by a
prior interaction, the ``causal_parents`` link is created automatically
— replacing the temporal-adjacency heuristic with real data-flow edges
in the causal credit DAG.

Core concepts:

- **ArtifactSchema**: declares what *kind* of artifact a handler can
  produce, with named typed fields for schema matching.
- **Artifact**: a concrete artifact instance produced by an interaction.
- **ArtifactNeed**: an unsatisfied input requirement declared by an
  agent, driving pressure-based coordination.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ArtifactSchema:
    """Describes the shape of an artifact kind.

    Handlers declare the schemas they produce so the artifact registry
    can match producers to consumers by structural compatibility.

    Attributes:
        kind: Unique identifier for the artifact type
              (e.g. ``"delivery_receipt"``, ``"verification_result"``).
        fields: Mapping of field names to type names, used for
                schema matching.  Example::

                    {"order_id": "str", "quality_score": "float"}
    """

    kind: str
    fields: Dict[str, str] = field(default_factory=dict)

    def is_compatible(self, other: ArtifactSchema) -> bool:
        """Check if *other* schema satisfies this one.

        Compatibility means *other* provides at least all fields
        declared here (subtyping / structural match).
        """
        for fname, ftype in self.fields.items():
            if fname not in other.fields:
                return False
            if other.fields[fname] != ftype:
                return False
        return True


@dataclass
class Artifact:
    """A concrete artifact produced by a handler action.

    Artifacts are the edges of the emergent tool-chaining DAG.  When
    agent B consumes an artifact produced by agent A's interaction, the
    ``causal_parents`` link ``B → A`` is created on the
    ``SoftInteraction``.

    Attributes:
        artifact_id: Globally unique identifier.
        kind: Schema kind (matches ``ArtifactSchema.kind``).
        producer_id: Agent that created the artifact.
        interaction_id: Interaction that produced it.
        data: Payload — the actual artifact content.
        step: Simulation step when produced.
        p_at_production: Soft label of the producing interaction,
                         used for quality filtering.
        consumed_by: Interaction IDs that consumed this artifact
                     (populated by ``ArtifactRegistry.consume``).
    """

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kind: str = ""
    producer_id: str = ""
    interaction_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    step: int = 0
    p_at_production: float = 0.5
    consumed_by: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for event logging / observation building."""
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "producer_id": self.producer_id,
            "interaction_id": self.interaction_id,
            "data": self.data,
            "step": self.step,
            "p_at_production": self.p_at_production,
        }


@dataclass
class ArtifactNeed:
    """An unsatisfied input requirement declared by an agent.

    Unmet needs increase *pressure* for the corresponding artifact
    kind, signaling to the ecosystem where demand is concentrated.
    Agents (or governance) can read pressure scores to prioritize
    actions that satisfy high-demand needs.

    Attributes:
        kind: Artifact kind required.
        requester_id: Agent declaring the need.
        min_p: Minimum ``p_at_production`` of acceptable artifacts.
        max_age_steps: Maximum staleness (current_step − art.step).
    """

    kind: str = ""
    requester_id: str = ""
    min_p: float = 0.0
    max_age_steps: int = 50
