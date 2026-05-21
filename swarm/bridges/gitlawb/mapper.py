"""Map Gitlawb events to SWARM SoftInteraction objects."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

from swarm.bridges.gitlawb.config import GitlawbConfig
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)

# Type alias for async judge function: takes a prompt, returns p in [0, 1]
JudgeFn = Callable[[str], Coroutine[Any, Any, float]]


class GitlawbMapper:
    """Convert raw Gitlawb event dicts into SoftInteraction objects."""

    def __init__(
        self,
        config: GitlawbConfig,
        judge_fn: Optional[JudgeFn] = None,
    ) -> None:
        self._config = config
        self._judge_fn = judge_fn

    def map_ref_update(self, event: dict[str, Any]) -> SoftInteraction:
        """Map a RefUpdateType event to a SoftInteraction."""
        repo = event.get("repo", "")
        new_sha = event.get("newSha", "")
        return SoftInteraction(
            interaction_id=f"refupdate:{repo}:{new_sha[:12]}",
            timestamp=_parse_timestamp(event.get("timestamp", "")),
            initiator=event.get("pusherDid", ""),
            counterparty=event.get("nodeDid", ""),
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,  # push was accepted by the node
            p=0.5,  # placeholder, enriched later
            metadata={
                "repo": repo,
                "ref_name": event.get("refName", ""),
                "old_sha": event.get("oldSha", ""),
                "new_sha": new_sha,
                "event_source": "gitlawb_ref_update",
            },
        )

    def map_task_event(
        self,
        event: dict[str, Any],
        task_details: Optional[dict[str, Any]] = None,
    ) -> SoftInteraction:
        """Map a TaskEventType event to a SoftInteraction."""
        by_did = event.get("byDid", "")
        delegator = (task_details or {}).get("delegatorDid", "")
        assignee = (task_details or {}).get("assigneeDid", "")

        counterparty = assignee if by_did == delegator else delegator
        new_status = event.get("newStatus", "")
        accepted = new_status == "completed"

        return SoftInteraction(
            interaction_id=f"taskevent:{event.get('taskId', '')}:{event.get('at', '')}",
            timestamp=_parse_timestamp(event.get("at", "")),
            initiator=by_did,
            counterparty=counterparty,
            interaction_type=InteractionType.REPLY,
            accepted=accepted,
            p=0.5,  # placeholder, enriched later
            metadata={
                "task_id": event.get("taskId", ""),
                "old_status": event.get("oldStatus", ""),
                "new_status": new_status,
                "event_source": "gitlawb_task_event",
            },
        )

    def map_task_creation(self, task: dict[str, Any]) -> SoftInteraction:
        """Map a newly created task to a SoftInteraction."""
        return SoftInteraction(
            interaction_id=f"task_created:{task.get('id', '')}",
            timestamp=_parse_timestamp(task.get("createdAt", "")),
            initiator=task.get("delegatorDid", ""),
            counterparty=task.get("assigneeDid", "") or "",
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,
            p=0.5,  # placeholder, enriched later
            metadata={
                "task_id": task.get("id", ""),
                "kind": task.get("kind", ""),
                "capability": task.get("capability", ""),
                "event_source": "gitlawb_task_creation",
            },
        )

    async def score_quality(self, interaction: SoftInteraction) -> float:
        """Compute quality probability p for an interaction.

        Uses LLM judge if available, falls back to heuristic scoring.
        """
        if self._config.use_llm_judge and self._judge_fn is not None:
            try:
                prompt = _build_judge_prompt(interaction)
                p = await self._judge_fn(prompt)
                return max(0.0, min(1.0, p))
            except Exception as exc:
                logger.warning("LLM judge failed: %s; using heuristic", exc)
                if not self._config.heuristic_fallback:
                    return 0.5

        return _heuristic_score(interaction)

    async def enrich(self, interaction: SoftInteraction) -> SoftInteraction:
        """Score quality and set p on the interaction."""
        interaction = interaction.model_copy(update={"p": await self.score_quality(interaction)})
        return interaction


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp string, falling back to now."""
    if not ts:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _build_judge_prompt(interaction: SoftInteraction) -> str:
    """Build a prompt for the LLM quality judge."""
    source = interaction.metadata.get("event_source", "")
    if source == "gitlawb_ref_update":
        return (
            f"Agent {interaction.initiator} pushed to "
            f"{interaction.metadata.get('repo', '')}:"
            f"{interaction.metadata.get('ref_name', '')}. "
            f"Old: {interaction.metadata.get('old_sha', '')[:12]}, "
            f"New: {interaction.metadata.get('new_sha', '')[:12]}. "
            "Rate p in [0,1]: probability this push was a constructive "
            "contribution. Respond ONLY with JSON: "
            '{"p": <float>, "reason": "<25 words>"}'
        )
    elif source == "gitlawb_task_event":
        return (
            f"Agent {interaction.initiator} changed task "
            f"{interaction.metadata.get('task_id', '')} from "
            f"{interaction.metadata.get('old_status', '')} to "
            f"{interaction.metadata.get('new_status', '')}. "
            "Rate p in [0,1]: probability this was a positive outcome. "
            'Respond ONLY with JSON: {"p": <float>, "reason": "<25 words>"}'
        )
    elif source == "gitlawb_task_creation":
        return (
            f"Agent {interaction.initiator} delegated task "
            f"{interaction.metadata.get('task_id', '')} "
            f"(kind: {interaction.metadata.get('kind', '')}) to "
            f"{interaction.counterparty}. "
            "Rate p in [0,1]: probability this delegation is constructive. "
            'Respond ONLY with JSON: {"p": <float>, "reason": "<25 words>"}'
        )
    return (
        f"Agent {interaction.initiator} interacted with "
        f"{interaction.counterparty}. Rate p in [0,1]. "
        'Respond ONLY with JSON: {"p": <float>, "reason": "<25 words>"}'
    )


def _heuristic_score(interaction: SoftInteraction) -> float:
    """Fast deterministic quality score when LLM is unavailable."""
    source = interaction.metadata.get("event_source", "")

    if source == "gitlawb_ref_update":
        base = 0.7
        ref = interaction.metadata.get("ref_name", "")
        if "temp" in ref or "test" in ref or "wip" in ref:
            base -= 0.15
        # Force push detection: oldSha all zeros
        old_sha = interaction.metadata.get("old_sha", "")
        if old_sha and all(c == "0" for c in old_sha):
            base -= 0.2
        return max(0.1, min(0.95, base))

    if source == "gitlawb_task_event":
        status = interaction.metadata.get("new_status", "")
        if status == "completed":
            return 0.8
        if status == "failed":
            return 0.2
        if status == "claimed":
            return 0.5
        return 0.5

    if source == "gitlawb_task_creation":
        return 0.6

    return 0.5
