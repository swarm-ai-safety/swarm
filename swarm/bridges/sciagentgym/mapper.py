"""Mapper for converting SciAgentGym events to SoftInteractions."""

from __future__ import annotations

from typing import Optional

from swarm.bridges.sciagentgym.config import SciAgentGymConfig
from swarm.bridges.sciagentgym.events import (
    DataArtifactEvent,
    SafetyCheckEvent,
    SciAgentGymEvent,
    SciAgentGymEventType,
    ToolCallEvent,
    WorkflowStepEvent,
)
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.models.interaction import InteractionType, SoftInteraction


class SciAgentGymMapper:
    """Maps SciAgentGym events to SWARM SoftInteractions.
    
    This mapper converts SciAgentGym tool calls, workflow steps, and safety
    checks into SoftInteraction objects scored via the ProxyComputer.
    """

    def __init__(self, config: SciAgentGymConfig) -> None:
        """Initialize the mapper.
        
        Args:
            config: Bridge configuration.
        """
        self._config = config
        self._proxy = ProxyComputer(sigmoid_k=config.proxy_sigmoid_k)

    def map_tool_call(
        self, event: SciAgentGymEvent, tool_data: ToolCallEvent
    ) -> Optional[SoftInteraction]:
        """Map a tool call event to a SoftInteraction.
        
        Args:
            event: The SciAgentGym event.
            tool_data: Parsed tool call data.
            
        Returns:
            SoftInteraction or None if mapping fails.
        """
        # Compute proxy observables
        observables = ProxyObservables(
            task_progress_delta=1.0 if tool_data.success else -1.0,
            rework_count=0 if tool_data.success else 1,
            verifier_rejections=0,
            counterparty_engagement_delta=min(1.0, tool_data.execution_time_seconds / 60.0),
        )

        v_hat, p = self._proxy.compute_labels(observables)

        # Build interaction
        interaction = SoftInteraction(
            initiator=event.agent_id,
            counterparty=self._config.orchestrator_id,
            timestamp=event.timestamp,
            interaction_type=InteractionType.COLLABORATION,
            v_hat=v_hat,
            p=p,
            accepted=tool_data.success,
            metadata={
                "tool_name": tool_data.tool_name,
                "execution_time": tool_data.execution_time_seconds,
                "cost_tokens": tool_data.cost_tokens,
            },
        )

        return interaction

    def map_workflow_step(
        self, event: SciAgentGymEvent, step_data: WorkflowStepEvent
    ) -> Optional[SoftInteraction]:
        """Map a workflow step to a SoftInteraction.
        
        Args:
            event: The SciAgentGym event.
            step_data: Parsed workflow step data.
            
        Returns:
            SoftInteraction or None if mapping fails.
        """
        # Compute proxy observables
        observables = ProxyObservables(
            task_progress_delta=1.0 if step_data.success else -1.0,
            rework_count=0 if step_data.success else 1,
            verifier_rejections=0 if step_data.dependencies_met else 1,
            counterparty_engagement_delta=0.7,  # Default medium engagement for workflow steps
        )

        v_hat, p = self._proxy.compute_labels(observables)

        interaction = SoftInteraction(
            initiator=event.agent_id,
            counterparty=self._config.orchestrator_id,
            timestamp=event.timestamp,
            interaction_type=InteractionType.COLLABORATION,
            v_hat=v_hat,
            p=p,
            accepted=step_data.success,
            metadata={
                "workflow_id": step_data.workflow_id,
                "step_index": step_data.step_index,
                "step_type": step_data.step_type,
            },
        )

        return interaction

    def map_data_artifact(
        self, event: SciAgentGymEvent, artifact_data: DataArtifactEvent
    ) -> Optional[SoftInteraction]:
        """Map a data artifact event to a SoftInteraction.
        
        Args:
            event: The SciAgentGym event.
            artifact_data: Parsed artifact data.
            
        Returns:
            SoftInteraction or None if mapping fails.
        """
        # Use validation score to compute task progress (map [0,1] to [-1,1])
        observables = ProxyObservables(
            task_progress_delta=2.0 * artifact_data.validation_score - 1.0,
            rework_count=0 if artifact_data.validated else 1,
            verifier_rejections=0 if artifact_data.validated else 1,
            counterparty_engagement_delta=0.5,
        )

        v_hat, p = self._proxy.compute_labels(observables)

        interaction = SoftInteraction(
            initiator=event.agent_id,
            counterparty=self._config.orchestrator_id,
            timestamp=event.timestamp,
            interaction_type=InteractionType.TRADE,
            v_hat=v_hat,
            p=p,
            accepted=artifact_data.validated,
            metadata={
                "artifact_id": artifact_data.artifact_id,
                "artifact_type": artifact_data.artifact_type,
                "size_bytes": artifact_data.size_bytes,
            },
        )

        return interaction

    def map_safety_check(
        self, event: SciAgentGymEvent, safety_data: SafetyCheckEvent
    ) -> Optional[SoftInteraction]:
        """Map a safety check event to a SoftInteraction.
        
        Args:
            event: The SciAgentGym event.
            safety_data: Parsed safety check data.
            
        Returns:
            SoftInteraction or None if mapping fails.
        """
        # Safety score directly maps to p
        # (safety_score is probability of being safe/benign)
        v_hat = 2.0 * safety_data.safety_score - 1.0  # Map [0,1] to [-1,1]
        p = safety_data.safety_score

        interaction = SoftInteraction(
            initiator=event.agent_id,
            counterparty=self._config.orchestrator_id,
            timestamp=event.timestamp,
            interaction_type=InteractionType.VOTE,
            v_hat=v_hat,
            p=p,
            accepted=safety_data.passed,
            metadata={
                "check_type": safety_data.check_type,
                "risk_factors": safety_data.risk_factors,
                "mitigation_applied": safety_data.mitigation_applied,
            },
        )

        return interaction

    def map_event(self, event: SciAgentGymEvent) -> Optional[SoftInteraction]:
        """Route an event to the appropriate mapper.
        
        Args:
            event: The SciAgentGym event to map.
            
        Returns:
            SoftInteraction or None if event type is not handled.
        """
        if event.event_type in (
            SciAgentGymEventType.TOOL_CALL_COMPLETED,
            SciAgentGymEventType.TOOL_CALL_FAILED,
        ):
            tool_data = ToolCallEvent.from_dict(event.payload)
            return self.map_tool_call(event, tool_data)

        if event.event_type == SciAgentGymEventType.WORKFLOW_STEP_COMPLETED:
            step_data = WorkflowStepEvent.from_dict(event.payload)
            return self.map_workflow_step(event, step_data)

        if event.event_type in (
            SciAgentGymEventType.DATA_ARTIFACT_CREATED,
            SciAgentGymEventType.DATA_ARTIFACT_VALIDATED,
        ):
            artifact_data = DataArtifactEvent.from_dict(event.payload)
            return self.map_data_artifact(event, artifact_data)

        if event.event_type in (
            SciAgentGymEventType.SAFETY_CHECK_PASSED,
            SciAgentGymEventType.SAFETY_CHECK_FAILED,
        ):
            safety_data = SafetyCheckEvent.from_dict(event.payload)
            return self.map_safety_check(event, safety_data)

        # Unhandled event type
        return None
