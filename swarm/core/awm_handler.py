"""Handler for AWM (Agent World Model) database-backed task environments.

Manages the lifecycle of AWM tasks within SWARM simulations:
- Assigns AWM environments to agents
- Processes AWM_EXECUTE_TASK actions (tool call sequences)
- Maps tool-use traces to ProxyObservables
- Runs AWM verification and converts to soft labels
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import Any, Dict, FrozenSet, List, Optional

from swarm.agents.base import Action, ActionType
from swarm.bridges.awm.config import AWMConfig
from swarm.bridges.awm.mcp_client import AWMEpisodeTrace, ToolCallRecord
from swarm.bridges.awm.observable_mapper import AWMObservableMapper
from swarm.bridges.awm.verifier_bridge import AWMVerifierBridge
from swarm.core.handler import Handler, HandlerActionResult
from swarm.env.state import EnvState
from swarm.logging.event_bus import EventBus
from swarm.models.events import Event, EventType

logger = logging.getLogger(__name__)


class AWMHandler(Handler):
    """Handler for AWM database-backed task environments.

    Each agent can be assigned an AWM task. When the agent submits
    an AWM_EXECUTE_TASK action with tool calls, this handler:
    1. Records tool calls into an episode trace
    2. Simulates execution (in Phase 1, without a live server)
    3. Runs verification via AWMVerifierBridge
    4. Maps the trace to ProxyObservables via AWMObservableMapper
    """

    def __init__(
        self,
        *,
        config: AWMConfig,
        event_bus: EventBus,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self.config = config
        self._rng = random.Random(seed)
        self._mapper = AWMObservableMapper(
            error_weight=config.error_weight,
            misuse_weight=config.misuse_weight,
        )
        self._verifier = AWMVerifierBridge(
            confidence=config.verification_confidence,
        )

        # Active task assignments: agent_id -> task info
        self._assignments: Dict[str, Dict[str, Any]] = {}

        # Episode traces: agent_id -> current trace
        self._traces: Dict[str, AWMEpisodeTrace] = {}

        # Completed episodes for metrics
        self._completed_episodes: List[AWMEpisodeTrace] = []

        # Available tools (populated from environment metadata)
        self._available_tools: List[Dict[str, Any]] = []

        # Task queue for the current epoch
        self._task_queue: List[Dict[str, Any]] = []

    @staticmethod
    def handled_action_types() -> FrozenSet:
        return frozenset({ActionType.AWM_EXECUTE_TASK})

    def on_epoch_start(self, state: EnvState) -> None:
        """Generate AWM task assignments for the epoch."""
        self._assignments.clear()
        self._traces.clear()

        # Generate synthetic tasks for this epoch
        self._task_queue = self._generate_tasks()

        # Assign tasks to agents round-robin
        agent_ids = list(state.agents.keys())
        if not agent_ids or not self._task_queue:
            return

        for i, task in enumerate(self._task_queue):
            agent_id = agent_ids[i % len(agent_ids)]
            self._assignments[agent_id] = task
            self._traces[agent_id] = AWMEpisodeTrace(
                agent_id=agent_id,
                environment_id=self.config.environment_id,
                task_description=task.get("description", ""),
                max_steps=self.config.max_steps_per_task,
            )

            self._emit_event(Event(
                event_type=EventType.AWM_TASK_ASSIGNED,
                agent_id=agent_id,
                payload={
                    "task_id": task.get("task_id", ""),
                    "environment_id": self.config.environment_id,
                    "description": task.get("description", ""),
                },
            ))

    def on_epoch_end(self, state: EnvState) -> None:
        """Finalize any incomplete episodes."""
        for _agent_id, trace in list(self._traces.items()):
            if trace.verified is None:
                # Mark unfinished episodes as failed
                trace.verified = False
                self._completed_episodes.append(trace)

        self._assignments.clear()
        self._traces.clear()

    def _generate_tasks(self) -> List[Dict[str, Any]]:
        """Generate synthetic AWM tasks for the epoch."""
        tasks = []
        for _ in range(self.config.max_tasks_per_epoch):
            tasks.append({
                "task_id": str(uuid.uuid4()),
                "environment_id": self.config.environment_id,
                "description": (
                    f"Complete the assigned task in the "
                    f"{self.config.environment_id} environment."
                ),
                "tools": self._get_available_tools(),
            })
        return tasks

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools for the current environment."""
        if self._available_tools:
            return self._available_tools
        # Default tools for Phase 1 (synthetic)
        return [
            {"name": "query_database", "description": "Execute a SQL query"},
            {"name": "update_record", "description": "Update a database record"},
            {"name": "create_record", "description": "Create a new record"},
            {"name": "delete_record", "description": "Delete a record"},
            {"name": "list_tables", "description": "List available tables"},
        ]

    def build_observation_fields(
        self, agent_id: str, state: Any
    ) -> Dict[str, Any]:
        """Provide AWM task and tool information to agents."""
        result: Dict[str, Any] = {}

        task = self._assignments.get(agent_id)
        if task:
            result["awm_task"] = task
            result["awm_available_tools"] = task.get(
                "tools", self._get_available_tools()
            )

        return result

    def observation_field_mapping(self) -> Dict[str, str]:
        return {
            "awm_task": "awm_task",
            "awm_available_tools": "awm_available_tools",
        }

    def handle_action(
        self, action: Action, state: EnvState
    ) -> HandlerActionResult:
        """Process an AWM_EXECUTE_TASK action."""
        if action.action_type != ActionType.AWM_EXECUTE_TASK:
            return HandlerActionResult(success=False)

        agent_id = action.agent_id
        trace = self._traces.get(agent_id)

        if trace is None:
            # No active task for this agent
            return HandlerActionResult(success=False)

        # Check step limit
        if trace.steps_used >= self.config.max_steps_per_task:
            return HandlerActionResult(
                success=False,
                metadata={"error": "max_steps_exceeded"},
            )

        # Process tool calls from the action
        tool_calls = action.metadata.get("tool_calls", [])
        for tc_data in tool_calls:
            record = self._simulate_tool_call(tc_data)
            trace.tool_calls.append(record)
            trace.steps_used += 1

            if trace.steps_used >= self.config.max_steps_per_task:
                break

        # Run verification (simulated in Phase 1)
        verification_result = self._simulate_verification(trace)
        trace.verified = verification_result.get("passed", False)
        trace.verification_details = verification_result

        # Compute soft p from verification
        p = self._verifier.verify_and_score(verification_result)

        # Map trace to observables
        observables = self._mapper.map(trace)

        # Track completed episode
        self._completed_episodes.append(trace)

        # Remove from active traces
        self._traces.pop(agent_id, None)
        self._assignments.pop(agent_id, None)

        # Emit completion event
        self._emit_event(Event(
            event_type=EventType.AWM_TASK_COMPLETED,
            agent_id=agent_id,
            payload={
                "episode_id": trace.episode_id,
                "verified": trace.verified,
                "steps_used": trace.steps_used,
                "error_count": trace.error_count,
                "malformed_count": trace.malformed_count,
                "soft_p": p,
            },
        ))

        return HandlerActionResult(
            success=True,
            observables=observables,
            initiator_id=agent_id,
            counterparty_id=agent_id,
            metadata={
                "episode_id": trace.episode_id,
                "verified": trace.verified,
                "soft_p": p,
                "steps_used": trace.steps_used,
            },
        )

    def _simulate_tool_call(
        self, tc_data: Dict[str, Any]
    ) -> ToolCallRecord:
        """Simulate a tool call for Phase 1 (no live server).

        In Phase 2+, this will be replaced by actual MCP HTTP calls.
        """
        tool_name = tc_data.get("tool_name", "")
        arguments = tc_data.get("arguments", {})

        record = ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
        )

        # Validate tool name exists
        available = {t["name"] for t in self._get_available_tools()}
        if tool_name not in available:
            record.is_malformed = True
            record.is_error_response = True
            record.error = f"Unknown tool: {tool_name}"
            return record

        # Simulate success/failure based on RNG
        if self._rng.random() < 0.85:
            record.success = True
            record.result = {"status": "ok", "rows_affected": 1}
        else:
            record.is_error_response = True
            record.error = "simulated_error"

        return record

    def _simulate_verification(
        self, trace: AWMEpisodeTrace
    ) -> Dict[str, Any]:
        """Simulate AWM verification for Phase 1.

        Verification success correlates with:
        - Fewer errors
        - Fewer malformed calls
        - More total tool calls (agent actually tried)
        """
        if trace.total_calls == 0:
            return {"passed": False, "reason": "no_tool_calls"}

        # Success probability based on trace quality
        error_ratio = trace.error_count / max(trace.total_calls, 1)
        malformed_ratio = trace.malformed_count / max(trace.total_calls, 1)
        success_prob = max(0.0, 0.8 - error_ratio - malformed_ratio * 1.5)

        passed = self._rng.random() < success_prob

        return {
            "passed": passed,
            "confidence": self.config.verification_confidence,
            "error_ratio": error_ratio,
            "malformed_ratio": malformed_ratio,
        }

    def get_completed_episodes(self) -> List[AWMEpisodeTrace]:
        """Return all completed episodes."""
        return list(self._completed_episodes)
