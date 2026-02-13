"""GasTownMapper — convert GasTown events + git stats to SoftInteractions.

Translates bead lifecycle and PR workflow signals into SWARM's
ProxyObservables, then uses ProxyComputer to derive v_hat and p.
"""

import logging
from typing import Optional

from swarm.bridges.gastown.events import GasTownEvent, GasTownEventType
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


class GasTownMapper:
    """Maps GasTown signals to SWARM SoftInteractions via ProxyComputer.

    Observable mapping:

    ============================================ =========================== ===================================
    GasTown Signal                               ProxyObservable             Formula
    ============================================ =========================== ===================================
    Commits per PR (normalised)                  task_progress_delta         min(1.0, commits / 10) or -0.3
    Force pushes / review rounds                 rework_count                direct count
    CI failures                                  verifier_rejections         direct count
    (unused, reserved)                           tool_misuse_flags           0
    Inverse time-to-merge                        counterparty_engagement     1.0 - min(1.0, hours / 48)
    ============================================ =========================== ===================================
    """

    def __init__(self, proxy: Optional[ProxyComputer] = None) -> None:
        self._proxy = proxy or ProxyComputer()

    def _build_observables(
        self, git_stats: dict, bead_state: str
    ) -> ProxyObservables:
        """Build ProxyObservables from git stats and bead state."""
        commits = git_stats.get("commit_count", 0)
        if bead_state == "blocked":
            progress = -0.3
        elif bead_state in ("done", "closed"):
            progress = min(1.0, commits / 10.0) if commits > 0 else 0.5
        else:
            progress = min(1.0, commits / 10.0) if commits > 0 else 0.0

        rework = git_stats.get("review_iterations", 0)
        ci_failures = git_stats.get("ci_failures", 0)

        hours = git_stats.get("time_to_merge_hours")
        if hours is not None:
            engagement = 1.0 - min(1.0, hours / 48.0)
        else:
            engagement = 0.0

        return ProxyObservables(
            task_progress_delta=progress,
            rework_count=rework,
            verifier_rejections=ci_failures,
            tool_misuse_flags=0,
            counterparty_engagement_delta=engagement,
        )

    def map_bead_completion(
        self,
        bead: dict,
        git_stats: dict,
        agent_id: str,
    ) -> SoftInteraction:
        """Map a completed bead + git stats to a SoftInteraction."""
        status = str(bead.get("status", "done"))
        observables = self._build_observables(git_stats, status)
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            initiator="gastown_orchestrator",
            counterparty=agent_id,
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "gastown",
                "bead_id": bead.get("id", ""),
                "bead_title": bead.get("title", ""),
                "bead_status": status,
                "commit_count": git_stats.get("commit_count", 0),
                "files_changed": git_stats.get("files_changed", 0),
            },
        )

    def map_pr_event(
        self,
        event: GasTownEvent,
        git_stats: dict,
    ) -> SoftInteraction:
        """Map a PR lifecycle event + git stats to a SoftInteraction."""
        # Infer bead state from event type
        state_map = {
            GasTownEventType.PR_OPENED: "in_progress",
            GasTownEventType.PR_REVIEW_REQUESTED: "in_progress",
            GasTownEventType.PR_CHANGES_REQUESTED: "in_progress",
            GasTownEventType.PR_APPROVED: "done",
            GasTownEventType.PR_MERGED: "done",
            GasTownEventType.CI_PASSED: "in_progress",
            GasTownEventType.CI_FAILED: "in_progress",
        }
        bead_state = state_map.get(event.event_type, "in_progress")

        observables = self._build_observables(git_stats, bead_state)
        v_hat, p = self._proxy.compute_labels(observables)

        return SoftInteraction(
            initiator="gastown_orchestrator",
            counterparty=event.agent_name,
            interaction_type=InteractionType.COLLABORATION,
            accepted=True,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "gastown",
                "event_type": event.event_type.value,
                "bead_id": event.bead_id or "",
                "agent_name": event.agent_name,
            },
        )
