"""Boundary enforcement handling extracted from the orchestrator.

Encapsulates all sandbox-boundary logic: external entity
interaction, policy enforcement, flow tracking, and leakage
detection.
"""

import random
from typing import Any, Callable, Dict, List, Optional

from src.boundaries.external_world import (
    ExternalEntity,
    ExternalEntityType,
    ExternalWorld,
)
from src.boundaries.information_flow import (
    FlowDirection,
    FlowTracker,
    FlowType,
    InformationFlow,
)
from src.boundaries.leakage import LeakageDetector, LeakageReport
from src.boundaries.policies import PolicyEngine
from src.models.events import Event


class BoundaryHandler:
    """Handles all sandbox-boundary operations.

    Manages external-world interactions, policy evaluation, information
    flow tracking, and leakage detection.  The orchestrator delegates
    boundary-related calls to this handler.
    """

    def __init__(
        self,
        external_world: ExternalWorld,
        flow_tracker: FlowTracker,
        policy_engine: PolicyEngine,
        leakage_detector: LeakageDetector,
        emit_event: Callable[[Event], None],
        seed: Optional[int] = None,
    ):
        self.external_world = external_world
        self.flow_tracker = flow_tracker
        self.policy_engine = policy_engine
        self.leakage_detector = leakage_detector
        self._emit_event = emit_event
        self._seed = seed

    # ------------------------------------------------------------------
    # External interaction
    # ------------------------------------------------------------------

    def request_external_interaction(
        self,
        agent_id: str,
        entity_id: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Request an interaction with an external entity.

        The request goes through policy evaluation and leakage
        detection before being executed.

        Args:
            agent_id: The agent making the request.
            entity_id: The external entity to interact with.
            action: The type of action (call, query, send, etc.).
            payload: Optional data to send.

        Returns:
            Result of the interaction (or denial information).
        """
        payload = payload or {}
        metadata = {
            "sensitivity": payload.get("sensitivity", 0.0),
            "action": action,
        }

        # Check policy
        decision = self.policy_engine.evaluate(
            agent_id=agent_id,
            direction="outbound",
            flow_type=action,
            content=payload,
            metadata=metadata,
        )

        if not decision.allowed:
            self.external_world.blocked_attempts += 1
            return {
                "success": False,
                "blocked": True,
                "reason": decision.reason,
                "policy": decision.policy_name,
            }

        # Check for leakage
        leakage_events = self.leakage_detector.scan(
            content=payload,
            agent_id=agent_id,
            destination_id=entity_id,
        )
        if leakage_events:
            for event in leakage_events:
                if event.severity >= 0.9:
                    return {
                        "success": False,
                        "blocked": True,
                        "reason": f"Critical leakage detected: {event.description}",
                        "leakage_type": event.leakage_type.value,
                    }

        # Record outbound flow
        flow = InformationFlow.create(
            direction=FlowDirection.OUTBOUND,
            flow_type=FlowType.QUERY,
            source_id=agent_id,
            destination_id=entity_id,
            content=payload,
            sensitivity_score=metadata.get("sensitivity", 0.0),
        )
        self.flow_tracker.record_flow(flow)

        # Execute the interaction
        result = self.external_world.interact(
            agent_id=agent_id,
            entity_id=entity_id,
            action=action,
            payload=payload,
            rng=random.Random(self._seed) if self._seed else None,
        )

        # Record inbound flow if successful
        if result.get("success"):
            flow = InformationFlow.create(
                direction=FlowDirection.INBOUND,
                flow_type=FlowType.RESPONSE,
                source_id=entity_id,
                destination_id=agent_id,
                content=result.get("data", {}),
                sensitivity_score=(
                    result.get("sensitivity", 0.0)
                    if isinstance(result.get("sensitivity"), (int, float))
                    else 0.0
                ),
            )
            self.flow_tracker.record_flow(flow)

        return result

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------

    def get_external_entities(
        self,
        entity_type: Optional[str] = None,
        min_trust: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Get available external entities.

        Args:
            entity_type: Filter by entity type.
            min_trust: Minimum trust level.

        Returns:
            List of entity information dictionaries.
        """
        type_filter = None
        if entity_type:
            try:
                type_filter = ExternalEntityType(entity_type)
            except ValueError:
                pass

        entities = self.external_world.list_entities(
            entity_type=type_filter,
            min_trust=min_trust,
        )

        return [
            {
                "entity_id": e.entity_id,
                "name": e.name,
                "type": e.entity_type.value,
                "trust_level": e.trust_level,
            }
            for e in entities
        ]

    def add_external_entity(self, entity: ExternalEntity) -> None:
        """Add an external entity to the world."""
        self.external_world.add_entity(entity)

    # ------------------------------------------------------------------
    # Metrics & reporting
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive boundary metrics."""
        metrics: Dict[str, Any] = {
            "boundaries_enabled": True,
        }

        # External world stats
        metrics["external_world"] = self.external_world.get_interaction_stats()

        # Flow tracker stats
        summary = self.flow_tracker.get_summary()
        metrics["flows"] = {
            "total": summary.total_flows,
            "inbound": summary.inbound_flows,
            "outbound": summary.outbound_flows,
            "bytes_in": summary.total_bytes_in,
            "bytes_out": summary.total_bytes_out,
            "blocked": summary.blocked_flows,
            "sensitive": summary.sensitive_flows,
            "avg_sensitivity": summary.avg_sensitivity,
        }

        # Anomalies
        anomalies = self.flow_tracker.detect_anomalies()
        metrics["anomalies"] = anomalies

        # Policy stats
        metrics["policies"] = self.policy_engine.get_statistics()

        # Leakage stats
        report = self.leakage_detector.generate_report()
        metrics["leakage"] = {
            "total_events": report.total_events,
            "blocked": report.blocked_count,
            "by_type": report.events_by_type,
            "avg_severity": report.avg_severity,
            "max_severity": report.max_severity,
            "recommendations": report.recommendations,
        }

        return metrics

    def get_agent_activity(self, agent_id: str) -> Dict[str, Any]:
        """Get boundary activity for a specific agent."""
        activity: Dict[str, Any] = {"agent_id": agent_id}

        activity["flows"] = self.flow_tracker.get_agent_flows(agent_id)

        events = self.leakage_detector.get_events(agent_id=agent_id)
        activity["leakage_events"] = len(events)
        activity["leakage_severity"] = (
            max(e.severity for e in events) if events else 0.0
        )

        return activity

    def get_leakage_report(self) -> LeakageReport:
        """Get the full leakage detection report."""
        return self.leakage_detector.generate_report()
