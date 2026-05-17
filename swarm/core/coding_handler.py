"""Handler for coding agent VERIFY_OUTPUT actions.

CLAIM_TASK and SUBMIT_OUTPUT are already handled by TaskHandler.
This handler adds code-review scoring that produces ProxyObservables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet

from swarm.agents.base import Action, ActionType
from swarm.core.handler import Handler, HandlerActionResult
from swarm.core.proxy import ProxyObservables
from swarm.models.interaction import InteractionType

if TYPE_CHECKING:
    from swarm.env.state import EnvState
    from swarm.logging.event_bus import EventBus


class CodingHandler(Handler):
    """Handles VERIFY_OUTPUT for coding agent code reviews.

    Scoring logic:
    - An honest reviewer detects hidden issues proportional to
      ``(1 - subtlety) * review_honesty``.
    - Produces ProxyObservables that flow through the standard
      proxy → p → payoff pipeline.
    """

    def __init__(self, *, event_bus: EventBus, rng=None) -> None:
        super().__init__(event_bus=event_bus)
        import random
        self._rng = rng or random.Random()

    @staticmethod
    def handled_action_types() -> FrozenSet:
        return frozenset({ActionType.VERIFY_OUTPUT})

    def handle_action(self, action: Action, state: EnvState) -> HandlerActionResult:
        if action.action_type != ActionType.VERIFY_OUTPUT:
            return HandlerActionResult(success=False)

        reviewer_id = action.agent_id
        task_id = action.target_id

        # Get submission metadata from the task pool
        submission_meta = self._get_submission_metadata(task_id, state)
        if submission_meta is None:
            return HandlerActionResult(success=False)

        submitter_id = submission_meta.get("submitter_id", "")
        quality = submission_meta.get("quality", 0.5)
        has_hidden_issues = submission_meta.get("has_hidden_issues", False)
        subtlety = submission_meta.get("subtlety", 0.0)

        # Reviewer params from action metadata
        review_honesty = action.metadata.get("review_honesty", 0.9)
        reviewer_skill = action.metadata.get("reviewer_skill", 0.5)

        # Detection logic
        detected_issues = 0
        if has_hidden_issues:
            detection_prob = (1.0 - subtlety) * review_honesty
            if self._rng.random() < detection_prob:
                detected_issues = 1

        # Reviewer rejects if issues detected or quality is very low
        rejected = detected_issues > 0 or quality < 0.3

        # Compute rework cycles based on quality
        rework_count = 0
        if quality < 0.5:
            rework_count = max(1, int((1.0 - quality) * 3))

        # Engagement signal: honest reviewer engagement is high,
        # rubber-stamp reviewer engagement is low
        engagement = review_honesty * reviewer_skill

        observables = ProxyObservables(
            task_progress_delta=quality if not has_hidden_issues else quality * 0.5,
            rework_count=rework_count,
            verifier_rejections=1 if rejected else 0,
            tool_misuse_flags=detected_issues,
            counterparty_engagement_delta=engagement,
        )

        return HandlerActionResult(
            success=True,
            observables=observables,
            initiator_id=submitter_id,
            counterparty_id=reviewer_id,
            accepted=not rejected,
            interaction_type=InteractionType.COLLABORATION,
            ground_truth=-1 if has_hidden_issues else 1,
        )

    def _get_submission_metadata(self, task_id: str, state) -> dict | None:
        """Extract submission metadata from the task pool.

        Falls back to reasonable defaults if the task pool doesn't have
        the metadata (e.g. in unit tests or when submissions don't carry it).
        """
        task_pool = getattr(state, "_task_pool", None)
        if task_pool is None:
            # Try via the state's parent orchestrator reference
            task_pool = getattr(state, "task_pool", None)

        if task_pool is not None:
            task = task_pool.get_task(task_id)
            if task is not None:
                output = getattr(task, "output", None)
                if output and isinstance(output, dict):
                    return dict(output)
                # Try to pull metadata from the task's submission content
                meta = getattr(task, "submission_metadata", None)
                if meta:
                    return dict(meta)
                return {
                    "submitter_id": getattr(task, "claimed_by", ""),
                    "quality": 0.5,
                    "has_hidden_issues": False,
                    "subtlety": 0.0,
                }

        # Fallback: use action metadata if available
        return {
            "submitter_id": "",
            "quality": 0.5,
            "has_hidden_issues": False,
            "subtlety": 0.0,
        }
