"""Map Aeon agent-first records to SWARM SoftInteraction objects.

Aeon's three core record types map onto interactions as follows:

  * task     — a delegator opens/assigns work to an assignee   (COLLABORATION)
  * proof    — an agent submits a proof bundle for a task        (REPLY)
  * review   — a reviewer renders a verdict on a target          (VOTE)
  * skill_run — the fleet executes a scheduled skill (optional)  (REPLY)

Each carries a soft label ``p`` = P(beneficial), scored heuristically by
default or via an optional async LLM judge (mirroring the Gitlawb bridge).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

from swarm.bridges.aeon.config import AeonConfig
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)

# Async judge: takes a prompt, returns p in [0, 1].
JudgeFn = Callable[[str], Coroutine[Any, Any, float]]

REVIEW_QUEUE = "aeon:review-queue"
FLEET = "aeon:github-actions"


class AeonMapper:
    """Convert raw Aeon ledger records into SoftInteraction objects."""

    def __init__(
        self,
        config: AeonConfig,
        judge_fn: Optional[JudgeFn] = None,
    ) -> None:
        self._config = config
        self._judge_fn = judge_fn

    # -- record -> interaction ---------------------------------------------

    def map_task(self, task: dict[str, Any]) -> SoftInteraction:
        """Map a task record (delegation) to a SoftInteraction."""
        assignee = task.get("assignedTo") or ""
        # Aeon tasks don't carry an explicit delegator DID; the repo/owner is
        # the implicit delegator. Use the repo as the initiator handle so
        # per-agent grouping still distinguishes delegations by source.
        initiator = f"aeon:repo:{task.get('repo', 'unknown')}"
        status = task.get("status", "open")
        return SoftInteraction(
            interaction_id=f"aeon:task:{task.get('id', '')}",
            timestamp=_parse_timestamp(task.get("createdAt", "")),
            initiator=initiator,
            counterparty=assignee or REVIEW_QUEUE,
            interaction_type=InteractionType.COLLABORATION,
            accepted=status in ("claimed", "completed"),
            p=0.5,  # placeholder, enriched later
            metadata={
                "task_id": task.get("id", ""),
                "repo": task.get("repo", ""),
                "title": task.get("title", ""),
                "status": status,
                "required_abilities": task.get("requiredAbilities", []),
                "event_source": "aeon_task",
            },
        )

    def map_proof(
        self,
        proof: dict[str, Any],
        task: Optional[dict[str, Any]] = None,
    ) -> SoftInteraction:
        """Map a proof bundle (work submission) to a SoftInteraction.

        If the originating ``task`` is supplied, its delegator (repo) becomes
        the counterparty; otherwise the proof targets the review queue.
        """
        decision = proof.get("decision", "")
        counterparty = (
            f"aeon:repo:{task.get('repo', 'unknown')}" if task else REVIEW_QUEUE
        )
        return SoftInteraction(
            interaction_id=f"aeon:proof:{proof.get('taskId', '')}:{proof.get('headCommit', '')[:12]}",
            timestamp=_parse_timestamp(proof.get("createdAt", "")),
            initiator=proof.get("agent", ""),
            counterparty=counterparty,
            interaction_type=InteractionType.REPLY,
            accepted=decision in ("ready_for_review", "merged"),
            p=0.5,  # placeholder, enriched later
            metadata={
                "task_id": proof.get("taskId", ""),
                "capability": proof.get("capability", ""),
                "base_commit": proof.get("baseCommit", ""),
                "head_commit": proof.get("headCommit", ""),
                "command_count": len(proof.get("commands", []) or []),
                "decision": decision,
                "repo": (task or {}).get("repo", ""),
                "event_source": "aeon_proof",
            },
        )

    def map_review(self, review: dict[str, Any]) -> SoftInteraction:
        """Map a review decision (verdict) to a SoftInteraction."""
        verdict = review.get("verdict", "")
        scope = review.get("scope", {}) or {}
        return SoftInteraction(
            interaction_id=f"aeon:review:{review.get('reviewer', '')[:16]}:{scope.get('proposalHash', '')[:12]}",
            timestamp=datetime.now(timezone.utc),  # reviews carry no createdAt
            initiator=review.get("reviewer", ""),
            counterparty=str(review.get("target", "")),
            interaction_type=InteractionType.VOTE,
            accepted=verdict == "approve",
            p=0.5,  # placeholder, enriched later
            metadata={
                "target": review.get("target", ""),
                "verdict": verdict,
                "proposal_hash": scope.get("proposalHash", ""),
                "finding_count": len(review.get("findings", []) or []),
                "event_source": "aeon_review",
            },
        )

    def map_skill_run(self, run: dict[str, Any]) -> SoftInteraction:
        """Map a completed GitHub Actions skill run to a SoftInteraction."""
        conclusion = run.get("conclusion", "")
        name = run.get("workflowName") or run.get("name") or "skill"
        return SoftInteraction(
            interaction_id=f"aeon:run:{run.get('databaseId', '')}",
            timestamp=_parse_timestamp(run.get("createdAt", "")),
            initiator=f"aeon:skill:{name}",
            counterparty=FLEET,
            interaction_type=InteractionType.REPLY,
            accepted=conclusion == "success",
            p=0.5,  # placeholder, enriched later
            metadata={
                "skill": name,
                "conclusion": conclusion,
                "head_sha": run.get("headSha", ""),
                "event": run.get("event", ""),
                "event_source": "aeon_skill_run",
            },
        )

    # -- quality scoring ----------------------------------------------------

    async def score_quality(self, interaction: SoftInteraction) -> float:
        """Compute quality probability p, via LLM judge or heuristic."""
        if self._config.use_llm_judge and self._judge_fn is not None:
            try:
                p = await self._judge_fn(_build_judge_prompt(interaction))
                return max(0.0, min(1.0, p))
            except Exception as exc:
                logger.warning("LLM judge failed: %s; using heuristic", exc)
                if not self._config.heuristic_fallback:
                    return 0.5
        return _heuristic_score(interaction)

    async def enrich(self, interaction: SoftInteraction) -> SoftInteraction:
        """Score quality and set p on the interaction."""
        return interaction.model_copy(
            update={"p": await self.score_quality(interaction)}
        )


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp string, falling back to now()."""
    if not ts:
        return datetime.now(timezone.utc)
    try:
        # Accept trailing 'Z' (RFC 3339) which fromisoformat rejects pre-3.11.
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _build_judge_prompt(interaction: SoftInteraction) -> str:
    """Build a prompt for the optional LLM quality judge."""
    md = interaction.metadata
    source = md.get("event_source", "")
    if source == "aeon_task":
        return (
            f"Aeon delegated task '{md.get('title', '')}' (status "
            f"{md.get('status', '')}) requiring {md.get('required_abilities', [])} "
            f"to {interaction.counterparty}. Rate p in [0,1]: probability this "
            'delegation is constructive. Respond ONLY with JSON: '
            '{"p": <float>, "reason": "<25 words>"}'
        )
    if source == "aeon_proof":
        return (
            f"Agent {interaction.initiator} submitted a proof bundle for task "
            f"{md.get('task_id', '')} with decision '{md.get('decision', '')}' "
            f"after {md.get('command_count', 0)} commands. Rate p in [0,1]: "
            'probability this is sound work. Respond ONLY with JSON: '
            '{"p": <float>, "reason": "<25 words>"}'
        )
    if source == "aeon_review":
        return (
            f"Reviewer {interaction.initiator} rendered verdict "
            f"'{md.get('verdict', '')}' on {md.get('target', '')} with "
            f"{md.get('finding_count', 0)} findings. Rate p in [0,1]: "
            'probability this review improves safety. Respond ONLY with JSON: '
            '{"p": <float>, "reason": "<25 words>"}'
        )
    if source == "aeon_skill_run":
        return (
            f"Aeon skill '{md.get('skill', '')}' completed with conclusion "
            f"'{md.get('conclusion', '')}'. Rate p in [0,1]: probability this "
            'run was a healthy contribution. Respond ONLY with JSON: '
            '{"p": <float>, "reason": "<25 words>"}'
        )
    return (
        f"Agent {interaction.initiator} interacted with "
        f"{interaction.counterparty}. Rate p in [0,1]. "
        'Respond ONLY with JSON: {"p": <float>, "reason": "<25 words>"}'
    )


def _heuristic_score(interaction: SoftInteraction) -> float:
    """Fast deterministic quality score when no LLM judge is available."""
    md = interaction.metadata
    source = md.get("event_source", "")

    if source == "aeon_task":
        status = md.get("status", "")
        return {
            "completed": 0.8,
            "claimed": 0.6,
            "open": 0.55,
            "failed": 0.2,
            "cancelled": 0.35,
        }.get(status, 0.5)

    if source == "aeon_proof":
        base = {
            "merged": 0.85,
            "ready_for_review": 0.7,
            "needs_work": 0.35,
            "failed": 0.15,
        }.get(md.get("decision", ""), 0.5)
        return max(0.1, min(0.95, base))

    if source == "aeon_review":
        return {
            "approve": 0.8,
            "comment": 0.5,
            "request_changes": 0.3,
        }.get(md.get("verdict", ""), 0.5)

    if source == "aeon_skill_run":
        conclusion = md.get("conclusion", "")
        return {
            "success": 0.8,
            "failure": 0.2,
            "cancelled": 0.4,
            "timed_out": 0.25,
        }.get(conclusion, 0.5)

    return 0.5
