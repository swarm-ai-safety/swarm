"""Judge views — orthogonal-to-proxy projections of accepted interactions.

The pre-registered orthogonality property (see rubric_v1.md) is enforced
here: a JudgeView is constructed by *stripping* the ProxyComputer
observables and their derivatives, plus ground_truth, before any judge
sees the interaction. The rubric forbids any single field that could
leak proxy-side information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Fields excluded from the judge's view. These are the inputs to
# ProxyComputer (the gaming channel we are anchoring against) plus
# anything derived from them, plus answer-leak fields.
FORBIDDEN_FIELDS: frozenset[str] = frozenset({
    "task_progress_delta",
    "rework_count",
    "verifier_rejections",
    "tool_misuse_flags",
    "counterparty_engagement_delta",
    "v_hat",
    "p",
    "ground_truth",
    "tau",
    "c_a",
    "c_b",
    "r_a",
    "r_b",
})


@dataclass(frozen=True)
class JudgeView:
    """Read-only projection of a SoftInteraction safe to show a judge."""

    interaction_id: str
    interaction_type: str
    accepted: bool
    initiator_label: str
    counterparty_label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_judge_payload(self) -> dict[str, Any]:
        """Return the JSON payload the rubric describes."""
        return {
            "interaction_id": self.interaction_id,
            "interaction_type": self.interaction_type,
            "accepted": self.accepted,
            "initiator_label": self.initiator_label,
            "counterparty_label": self.counterparty_label,
            "metadata": dict(self.metadata),
        }


def _scrub_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Defense-in-depth: strip forbidden keys from a metadata dict."""
    return {k: v for k, v in metadata.items() if k not in FORBIDDEN_FIELDS}


def make_view(interaction: Any) -> JudgeView:
    """Build a JudgeView from a SoftInteraction-like object.

    Any forbidden field that somehow lives in `metadata` is also stripped.
    """
    interaction_type = getattr(interaction, "interaction_type", None)
    type_name = (
        interaction_type.name  # type: ignore[union-attr]
        if hasattr(interaction_type, "name")
        else str(interaction_type)
    )
    raw_metadata = dict(getattr(interaction, "metadata", {}) or {})
    return JudgeView(
        interaction_id=str(getattr(interaction, "interaction_id", "")),
        interaction_type=type_name,
        accepted=bool(getattr(interaction, "accepted", False)),
        initiator_label=str(getattr(interaction, "initiator", "")),
        counterparty_label=str(getattr(interaction, "counterparty", "")),
        metadata=_scrub_metadata(raw_metadata),
    )


def assert_view_is_orthogonal(payload: dict[str, Any]) -> None:
    """Defensive runtime check that a payload carries no forbidden fields.

    Raises AssertionError if any forbidden field is present at the top level
    or inside metadata. This is the load-bearing orthogonality property —
    if it ever fails, the judge anchor has been compromised.
    """
    bad_top = set(payload.keys()) & FORBIDDEN_FIELDS
    if bad_top:
        raise AssertionError(
            f"Judge payload contains forbidden top-level fields: {sorted(bad_top)}"
        )
    metadata = payload.get("metadata", {}) or {}
    bad_meta = set(metadata.keys()) & FORBIDDEN_FIELDS
    if bad_meta:
        raise AssertionError(
            f"Judge payload metadata contains forbidden fields: {sorted(bad_meta)}"
        )
