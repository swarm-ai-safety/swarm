"""Ralph event -> SoftInteraction mapper."""

from swarm.bridges.ralph.events import RalphEvent, RalphEventType
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.models.interaction import InteractionType, SoftInteraction


class RalphMapper:
    """Translate Ralph events into SWARM interactions."""

    def __init__(self, proxy: ProxyComputer | None = None) -> None:
        self._proxy = proxy or ProxyComputer()

    def map_event(self, event: RalphEvent, initiator: str, counterparty: str) -> SoftInteraction:
        """Map a normalized Ralph event into a SoftInteraction."""
        base = self._base_observables(event.event_type)

        observables = ProxyObservables(
            task_progress_delta=float(event.payload.get("task_progress_delta", base.task_progress_delta)),
            rework_count=int(event.payload.get("rework_count", base.rework_count)),
            verifier_rejections=int(
                event.payload.get("verifier_rejections", base.verifier_rejections)
            ),
            tool_misuse_flags=int(event.payload.get("tool_misuse_flags", base.tool_misuse_flags)),
            counterparty_engagement_delta=float(
                event.payload.get(
                    "counterparty_engagement_delta", base.counterparty_engagement_delta
                )
            ),
        )
        v_hat, p = self._proxy.compute_labels(observables)

        accepted = bool(event.payload.get("accepted", event.event_type != RalphEventType.TASK_FAILED))

        return SoftInteraction(
            timestamp=event.timestamp,
            initiator=initiator,
            counterparty=counterparty,
            interaction_type=InteractionType.COLLABORATION,
            accepted=accepted,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            metadata={
                "bridge": "ralph",
                "event_type": event.event_type.value,
                "event_id": event.event_id,
                "task_id": event.task_id,
            },
        )

    def _base_observables(self, event_type: RalphEventType) -> ProxyObservables:
        defaults = {
            RalphEventType.TASK_STARTED: ProxyObservables(task_progress_delta=0.1),
            RalphEventType.TASK_COMPLETED: ProxyObservables(
                task_progress_delta=0.8,
                counterparty_engagement_delta=0.4,
            ),
            RalphEventType.TASK_FAILED: ProxyObservables(
                task_progress_delta=-0.6,
                verifier_rejections=1,
                counterparty_engagement_delta=-0.4,
            ),
            RalphEventType.REVIEW_REQUESTED: ProxyObservables(
                task_progress_delta=0.2,
                rework_count=1,
            ),
            RalphEventType.REVIEW_REJECTED: ProxyObservables(
                task_progress_delta=-0.3,
                rework_count=2,
                verifier_rejections=1,
            ),
            RalphEventType.TOOL_MISUSE: ProxyObservables(
                task_progress_delta=-0.4,
                tool_misuse_flags=1,
                counterparty_engagement_delta=-0.2,
            ),
            RalphEventType.GENERIC: ProxyObservables(),
        }
        return defaults[event_type]
