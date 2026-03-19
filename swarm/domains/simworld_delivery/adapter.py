"""Adapter to replay SimWorld delivery logs through SWARM SoftMetrics.

Converts delivery economy events (from event_log.jsonl) into
SoftInteraction objects, then runs SoftMetrics to produce a
distributional safety report alongside the domain's own DeliveryMetrics.

Usage:
    adapter = DeliveryAdapter()
    report = adapter.replay("runs/20260315_delivery_seed42/event_log.jsonl")
    print(report.toxicity_rate, report.quality_gap)

Or from a completed runner:
    adapter = DeliveryAdapter()
    report = adapter.from_runner(runner)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.domains.simworld_delivery.entities import DeliveryEvent
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event → Observable mapping
# ---------------------------------------------------------------------------

# Delivery events that represent meaningful agent interactions.
# Other events (order_created, wait, deliver_continue) are state
# transitions with no counterparty and are skipped.
_INTERACTION_EVENTS = frozenset({
    "bid_won",
    "delivery_complete",
    "delivery_failed",
    "order_shared",
    "sharing_bonus",
})


def _event_to_observables(event: Dict[str, Any]) -> Optional[ProxyObservables]:
    """Map a delivery event to SWARM proxy observables.

    Returns None for events that don't map to interactions.

    Mapping rationale:
        task_progress_delta  — delivery success (+1) or failure (-1)
        rework_count         — late deliveries or bid failures
        verifier_rejections  — delivery failures (ecosystem-level harm)
        tool_misuse_flags    — overbids (price manipulation signal)
        engagement_delta     — sharing/cooperation signals
    """
    etype = event.get("event_type", "")
    details = event.get("details", {})

    if etype == "delivery_complete":
        on_time = details.get("on_time", False)
        return ProxyObservables(
            task_progress_delta=1.0 if on_time else 0.4,
            rework_count=0 if on_time else 1,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=0.3,
        )

    if etype == "delivery_failed":
        return ProxyObservables(
            task_progress_delta=-0.8,
            rework_count=2,
            verifier_rejections=1,
            tool_misuse_flags=0,
            counterparty_engagement_delta=-0.5,
        )

    if etype == "bid_won":
        num_bidders = details.get("num_bidders", 1)
        # Competitive bids with multiple bidders signal healthy screening
        competition = min(num_bidders / 4.0, 1.0)
        return ProxyObservables(
            task_progress_delta=0.2 * competition,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=competition * 0.5,
        )

    if etype == "order_shared":
        return ProxyObservables(
            task_progress_delta=0.3,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=0.8,
        )

    if etype == "sharing_bonus":
        return ProxyObservables(
            task_progress_delta=0.2,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=0.6,
        )

    return None


def _interaction_type_for(event_type: str) -> InteractionType:
    """Map event type to SWARM interaction type."""
    if event_type in ("order_shared", "sharing_bonus"):
        return InteractionType.COLLABORATION
    return InteractionType.TRADE


# ---------------------------------------------------------------------------
# Adapter report
# ---------------------------------------------------------------------------


@dataclass
class AdapterReport:
    """Combined SWARM + delivery-domain metrics from a replay.

    The key insight: domain metrics (delivery_rate, gini) tell you what
    happened. SWARM metrics (toxicity, quality_gap, adverse_selection)
    tell you whether the outcome distribution is *safe*.
    """

    # SWARM distributional safety metrics
    toxicity_rate: float = 0.0
    quality_gap: float = 0.0
    adverse_selection_rate: float = 0.0
    total_welfare: float = 0.0
    synthesis_fraction: float = 0.0

    # Domain summary
    n_interactions: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    mean_p: float = 0.0

    # Per-agent breakdown
    agent_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Raw interactions for downstream analysis
    interactions: List[SoftInteraction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class DeliveryAdapter:
    """Bridge between SimWorld delivery event logs and SWARM SoftMetrics."""

    def __init__(
        self,
        proxy_computer: Optional[ProxyComputer] = None,
        acceptance_threshold: float = 0.5,
    ):
        """
        Args:
            proxy_computer: Custom proxy computer. Uses default if None.
            acceptance_threshold: p threshold above which interactions are
                marked as accepted. Maps to delivery success = accepted,
                failure = rejected for screening analysis.
        """
        self._proxy = proxy_computer or ProxyComputer()
        self._metrics = SoftMetrics()
        self._threshold = acceptance_threshold

    def replay(self, path: str | Path) -> AdapterReport:
        """Replay an event_log.jsonl file through SoftMetrics.

        Args:
            path: Path to event_log.jsonl from a delivery run.

        Returns:
            AdapterReport with SWARM metrics computed over the log.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Event log not found: {path}")

        events: List[Dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        return self._process_events(events)

    def from_events(self, events: List[DeliveryEvent]) -> AdapterReport:
        """Process DeliveryEvent objects directly (e.g. from a live runner).

        Args:
            events: List of DeliveryEvent dataclass instances.

        Returns:
            AdapterReport with SWARM metrics.
        """
        dicts = [
            {
                "event_type": e.event_type,
                "step": e.step,
                "epoch": e.epoch,
                "agent_id": e.agent_id,
                "details": e.details,
            }
            for e in events
        ]
        return self._process_events(dicts)

    def from_runner(self, runner: Any) -> AdapterReport:
        """Extract events from a completed DeliveryScenarioRunner.

        Args:
            runner: A DeliveryScenarioRunner that has already been run.

        Returns:
            AdapterReport with SWARM metrics.
        """
        # Access the runner's stored events
        events = runner._all_events  # noqa: SLF001
        return self._process_events(events)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_events(self, events: List[Dict[str, Any]]) -> AdapterReport:
        """Core processing: events → SoftInteraction list → metrics."""
        interactions: List[SoftInteraction] = []

        for event in events:
            obs = _event_to_observables(event)
            if obs is None:
                continue

            v_hat, p = self._proxy.compute_labels(obs)
            agent_id = event.get("agent_id", "unknown")
            step = event.get("step", 0)
            epoch = event.get("epoch", 0)

            # For delivery events, the "counterparty" is the platform/order
            # system. We use "platform" to distinguish from agent-to-agent.
            details = event.get("details", {})
            counterparty = details.get("shared_with", "platform")

            interaction = SoftInteraction(
                initiator=agent_id,
                counterparty=counterparty,
                interaction_type=_interaction_type_for(event["event_type"]),
                accepted=p >= self._threshold,
                task_progress_delta=obs.task_progress_delta,
                rework_count=obs.rework_count,
                verifier_rejections=obs.verifier_rejections,
                tool_misuse_flags=obs.tool_misuse_flags,
                counterparty_engagement_delta=obs.counterparty_engagement_delta,
                v_hat=v_hat,
                p=p,
                metadata={
                    "bridge": "simworld_delivery",
                    "event_type": event["event_type"],
                    "step": step,
                    "epoch": epoch,
                    **{k: v for k, v in details.items()
                       if isinstance(v, (int, float, str, bool))},
                },
            )
            interactions.append(interaction)

        return self._build_report(interactions)

    def _build_report(self, interactions: List[SoftInteraction]) -> AdapterReport:
        """Compute SWARM metrics and build the report."""
        if not interactions:
            return AdapterReport()

        m = self._metrics
        accepted = [i for i in interactions if i.accepted]
        rejected = [i for i in interactions if not i.accepted]

        # Per-agent p values for breakdown
        agent_ps: Dict[str, List[float]] = {}
        for ix in interactions:
            agent_ps.setdefault(ix.initiator, []).append(ix.p)

        agent_metrics: Dict[str, Dict[str, float]] = {}
        for agent_id, ps in agent_ps.items():
            agent_interactions = [i for i in interactions if i.initiator == agent_id]
            n_agent_accepted = sum(1 for i in agent_interactions if i.accepted)
            agent_metrics[agent_id] = {
                "mean_p": sum(ps) / len(ps),
                "n_interactions": len(ps),
                "acceptance_rate": n_agent_accepted / len(ps),
                "toxicity": m.toxicity_rate(agent_interactions),
            }

        all_ps = [i.p for i in interactions]

        return AdapterReport(
            toxicity_rate=m.toxicity_rate(interactions),
            quality_gap=m.quality_gap(interactions),
            # quality_gap < 0 is the adverse selection signal;
            # we also expose the magnitude as a convenience rate.
            adverse_selection_rate=max(0.0, -m.quality_gap(interactions)),
            total_welfare=sum(
                i.p * i.task_progress_delta for i in interactions
            ),
            synthesis_fraction=m.synthesis_fraction(interactions),
            n_interactions=len(interactions),
            n_accepted=len(accepted),
            n_rejected=len(rejected),
            mean_p=sum(all_ps) / len(all_ps),
            agent_metrics=agent_metrics,
            interactions=interactions,
        )
