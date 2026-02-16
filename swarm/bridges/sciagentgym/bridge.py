"""Main bridge orchestrator for SciAgentGym.

Routes events through the mapper and policy engine, producing
SoftInteraction objects and logging them to the SWARM event log.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from swarm.bridges.sciagentgym.client import SciAgentGymClient
from swarm.bridges.sciagentgym.config import SciAgentGymConfig
from swarm.bridges.sciagentgym.events import SciAgentGymEvent, SciAgentGymEventType
from swarm.bridges.sciagentgym.mapper import SciAgentGymMapper
from swarm.bridges.sciagentgym.policy import PolicyDecision, SciAgentGymPolicy
from swarm.logging.event_log import EventLog
from swarm.models.events import Event, EventType
from swarm.models.interaction import SoftInteraction


class SciAgentGymBridge:
    """Connects SciAgentGym tool execution to SWARM governance.
    
    This bridge enables SWARM to monitor and govern scientific tool-use
    workflows from the SciAgentGym environment.
    """

    def __init__(
        self,
        config: Optional[SciAgentGymConfig] = None,
        event_log: Optional[EventLog] = None,
    ) -> None:
        """Initialize the bridge.
        
        Args:
            config: Bridge configuration. If None, uses defaults.
            event_log: SWARM event log for recording interactions.
        """
        self._config = config or SciAgentGymConfig()
        self._event_log = event_log

        self._client = SciAgentGymClient(self._config.client_config)
        self._mapper = SciAgentGymMapper(self._config)
        self._policy = SciAgentGymPolicy(self._config)

        self._events: List[SciAgentGymEvent] = []
        self._interactions: List[SoftInteraction] = []

    def ingest_workflow_log(self, log_path: str) -> List[SoftInteraction]:
        """Parse and process a workflow log file.
        
        Args:
            log_path: Path to the workflow log (JSONL format).
            
        Returns:
            List of SoftInteractions created from the log.
        """
        events = self._client.parse_workflow_log(log_path)
        return self._process_events(events)

    def ingest_tool_results(self, results_dir: str) -> List[SoftInteraction]:
        """Parse and process tool call results.
        
        Args:
            results_dir: Directory containing tool result files.
            
        Returns:
            List of SoftInteractions created from results.
        """
        events = self._client.parse_tool_call_results(results_dir)
        return self._process_events(events)

    def ingest_events(
        self, events: List[SciAgentGymEvent]
    ) -> List[SoftInteraction]:
        """Process pre-built events (useful for testing).
        
        Args:
            events: List of SciAgentGym events.
            
        Returns:
            List of SoftInteractions created from events.
        """
        return self._process_events(events)

    def _process_events(
        self, events: List[SciAgentGymEvent]
    ) -> List[SoftInteraction]:
        """Process a batch of events through mapper and policy.
        
        Args:
            events: List of events to process.
            
        Returns:
            List of SoftInteractions created.
        """
        new_interactions: List[SoftInteraction] = []

        for event in events:
            # Check circuit breaker
            if self._policy.should_circuit_break():
                self._record_event(event)
                continue

            # Process the event
            interactions = self._process_event(event)
            new_interactions.extend(interactions)

        return new_interactions

    def _process_event(self, event: SciAgentGymEvent) -> List[SoftInteraction]:
        """Route a single event through mapper and policy.
        
        Args:
            event: The event to process.
            
        Returns:
            List of SoftInteractions (typically 0 or 1).
        """
        self._record_event(event)
        interactions: List[SoftInteraction] = []

        # Map event to interaction
        interaction = self._mapper.map_event(event)
        if not interaction:
            return interactions

        # Apply policy checks based on event type
        policy_result = None

        if event.event_type in (
            SciAgentGymEventType.SAFETY_CHECK_PASSED,
            SciAgentGymEventType.SAFETY_CHECK_FAILED,
        ):
            policy_result = self._policy.evaluate_tool_safety(interaction.p)

        elif event.event_type in (
            SciAgentGymEventType.TOOL_CALL_FAILED,
            SciAgentGymEventType.WORKFLOW_FAILED,
        ):
            policy_result = self._policy.evaluate_workflow_failure(True)

        elif event.event_type in (
            SciAgentGymEventType.TOOL_CALL_COMPLETED,
            SciAgentGymEventType.WORKFLOW_STEP_COMPLETED,
        ):
            policy_result = self._policy.evaluate_workflow_failure(False)

        # Apply governance cost from policy
        if policy_result and policy_result.governance_cost > 0:
            interaction.c_a = policy_result.governance_cost

        # Reject interaction if policy denies
        if policy_result and policy_result.decision == PolicyDecision.DENY:
            interaction.accepted = False

        # Record the interaction
        interactions.append(interaction)
        self._finalize_interaction(event, interaction)

        return interactions

    def _record_event(self, event: SciAgentGymEvent) -> None:
        """Record an event in internal storage.
        
        Args:
            event: The event to record.
        """
        self._events.append(event)

        # Cap event storage
        if len(self._events) > self._config.max_events:
            self._events = self._events[-self._config.max_events :]

    def _finalize_interaction(
        self, event: SciAgentGymEvent, interaction: SoftInteraction
    ) -> None:
        """Finalize an interaction and log it.
        
        Args:
            event: The source event.
            interaction: The created interaction.
        """
        self._interactions.append(interaction)

        # Cap interaction storage
        if len(self._interactions) > self._config.max_interactions:
            self._interactions = self._interactions[
                -self._config.max_interactions :
            ]

        # Log to SWARM event log if available
        if self._event_log:
            swarm_event = Event(
                event_type=EventType.INTERACTION_COMPLETED,
                timestamp=event.timestamp,
                agent_id=event.agent_id,
                payload={
                    "interaction_type": interaction.interaction_type,
                    "p": interaction.p,
                    "accepted": interaction.accepted,
                },
            )
            self._event_log.log_event(swarm_event)

    # --- Accessors ---

    def get_interactions(self) -> List[SoftInteraction]:
        """Get all recorded interactions.
        
        Returns:
            List of SoftInteractions.
        """
        return self._interactions.copy()

    def get_events(self) -> List[SciAgentGymEvent]:
        """Get all recorded events.
        
        Returns:
            List of SciAgentGym events.
        """
        return self._events.copy()

    def get_policy_stats(self) -> Dict[str, any]:
        """Get policy engine statistics.
        
        Returns:
            Dictionary with policy stats (token usage, circuit breaker status).
        """
        return {
            "circuit_breaker_active": self._policy.should_circuit_break(),
            "token_usage": self._policy.get_token_usage(),
        }

    def reset_policy(self) -> None:
        """Reset policy state (circuit breaker, token counters)."""
        self._policy.reset_circuit_breaker()
        self._policy._total_tokens_used = 0

    def get_tool_registry(self) -> Dict[str, any]:
        """Get the loaded tool registry.
        
        Returns:
            Dictionary mapping tool names to specifications.
        """
        return self._client.load_tool_registry()
