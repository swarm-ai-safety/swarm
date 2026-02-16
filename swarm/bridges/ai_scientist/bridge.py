"""Main orchestrator for the AI-Scientist bridge.

Routes events through the mapper and policy engine, producing
SoftInteraction objects and logging them to the SWARM event log.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from swarm.bridges.ai_scientist.client import AIScientistClient
from swarm.bridges.ai_scientist.config import AIScientistConfig
from swarm.bridges.ai_scientist.events import (
    AIScientistEvent,
    AIScientistEventType,
    ExperimentRunEvent,
    IdeaEvent,
    ReviewEvent,
)
from swarm.bridges.ai_scientist.mapper import AIScientistMapper
from swarm.bridges.ai_scientist.policy import AIScientistPolicy, PolicyDecision
from swarm.logging.event_log import EventLog
from swarm.models.events import Event, EventType
from swarm.models.interaction import SoftInteraction


class AIScientistBridge:
    """Connects AI-Scientist pipeline outputs to SWARM governance."""

    def __init__(
        self,
        config: AIScientistConfig | None = None,
        event_log: EventLog | None = None,
    ) -> None:
        self._config = config or AIScientistConfig()
        self._event_log = event_log

        self._client = AIScientistClient(self._config.client_config)
        self._mapper = AIScientistMapper(self._config)
        self._policy = AIScientistPolicy(self._config)

        self._events: List[AIScientistEvent] = []
        self._interactions: List[SoftInteraction] = []
        self._phase_interactions: Dict[str, List[SoftInteraction]] = {}

    def ingest_idea_directory(self, path: str) -> List[SoftInteraction]:
        """Parse and process a single idea output directory."""
        events = self._client.parse_idea_directory(path)
        return self._process_events(events)

    def ingest_results_directory(self, path: str) -> List[SoftInteraction]:
        """Parse and process an entire AI-Scientist results directory."""
        events = self._client.parse_results_directory(path)
        return self._process_events(events)

    def ingest_events(self, events: List[AIScientistEvent]) -> List[SoftInteraction]:
        """Process pre-built events (useful for testing)."""
        return self._process_events(events)

    def _process_events(self, events: List[AIScientistEvent]) -> List[SoftInteraction]:
        """Process a batch of events, applying policies."""
        new_interactions: List[SoftInteraction] = []
        for event in events:
            if self._policy.should_circuit_break():
                # Still record the event but skip interaction creation
                self._record_event(event)
                continue
            interactions = self._process_event(event)
            new_interactions.extend(interactions)
        return new_interactions

    def _process_event(self, event: AIScientistEvent) -> List[SoftInteraction]:
        """Route a single event through mapper and policy."""
        self._record_event(event)
        interactions: List[SoftInteraction] = []

        # Apply pre-mapping policy checks
        if event.event_type == AIScientistEventType.NOVELTY_CHECK_FAILED:
            idea = IdeaEvent.from_dict(event.payload)
            result = self._policy.evaluate_novelty_gate(idea.novel)
            if result.decision == PolicyDecision.DENY:
                # Still create the interaction (for logging) but mark as rejected
                interaction = self._mapper.map_novelty_check(event, idea)
                if interaction and result.governance_cost:
                    interaction.c_a = result.governance_cost
                if interaction:
                    interactions.append(interaction)
                    self._finalize_interaction(event, interaction)
                return interactions

        if event.event_type in (
            AIScientistEventType.EXPERIMENT_RUN_COMPLETED,
            AIScientistEventType.EXPERIMENT_RUN_FAILED,
        ):
            run = ExperimentRunEvent.from_dict(event.payload)
            result = self._policy.evaluate_experiment_run(run.success)
            if run.cost_usd > 0:
                self._policy.evaluate_cost(run.cost_usd)

        if event.event_type == AIScientistEventType.REVIEW_SUBMITTED:
            review = ReviewEvent.from_dict(event.payload)
            self._policy.evaluate_review(review.overall_score)

        if event.event_type == AIScientistEventType.COST_UPDATED:
            cost = event.payload.get("cost_usd", 0.0)
            if cost > 0:
                self._policy.evaluate_cost(cost)

        # Map event to interaction
        mapped: Optional[SoftInteraction] = self._mapper.map_event(event)
        if mapped is not None:
            interactions.append(mapped)
            self._finalize_interaction(event, mapped)

        return interactions

    def _finalize_interaction(
        self,
        event: AIScientistEvent,
        interaction: SoftInteraction,
    ) -> None:
        """Record interaction and track phase."""
        self._record_interaction(interaction)
        self._log_interaction(interaction)

        if event.phase:
            self._phase_interactions.setdefault(event.phase, []).append(interaction)

    def _record_event(self, event: AIScientistEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._config.max_events:
            self._events = self._events[-self._config.max_events :]

    def _record_interaction(self, interaction: SoftInteraction) -> None:
        self._interactions.append(interaction)
        if len(self._interactions) > self._config.max_interactions:
            self._interactions = self._interactions[-self._config.max_interactions :]

    def _log_interaction(self, interaction: SoftInteraction) -> None:
        if self._event_log is None:
            return

        event = Event(
            event_type=EventType.INTERACTION_COMPLETED,
            interaction_id=interaction.interaction_id,
            initiator_id=interaction.initiator,
            counterparty_id=interaction.counterparty,
            payload={
                "accepted": interaction.accepted,
                "v_hat": interaction.v_hat,
                "p": interaction.p,
                "bridge": "ai_scientist",
                "metadata": dict(interaction.metadata or {}),
            },
        )
        self._event_log.append(event)

    def get_interactions(self) -> List[SoftInteraction]:
        return list(self._interactions)

    def get_events(self) -> List[AIScientistEvent]:
        return list(self._events)

    def get_phase_interactions(self, phase: str) -> List[SoftInteraction]:
        return list(self._phase_interactions.get(phase, []))

    @property
    def policy(self) -> AIScientistPolicy:
        return self._policy
