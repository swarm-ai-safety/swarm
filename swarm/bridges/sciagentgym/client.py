"""Client for interacting with SciAgentGym environment."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.bridges.sciagentgym.config import SciAgentGymClientConfig
from swarm.bridges.sciagentgym.events import (
    DataArtifactEvent,
    SafetyCheckEvent,
    SciAgentGymEvent,
    SciAgentGymEventType,
    ToolCallEvent,
    WorkflowStepEvent,
)


class SciAgentGymClient:
    """Client for parsing and interacting with SciAgentGym environment.
    
    This client provides a typed interface for:
    - Loading tool registries
    - Parsing task specifications
    - Reading workflow execution logs
    - Extracting events from SciAgentGym outputs
    """

    def __init__(self, config: SciAgentGymClientConfig) -> None:
        """Initialize the SciAgentGym client.
        
        Args:
            config: Client configuration.
        """
        self._config = config
        self._data_dir = Path(config.data_dir)
        self._tool_registry: Optional[Dict[str, Any]] = None

    def load_tool_registry(self) -> Dict[str, Any]:
        """Load the tool registry from disk.
        
        Returns:
            Dictionary mapping tool names to tool specifications.
        """
        if self._tool_registry is not None:
            return self._tool_registry

        registry_path = self._data_dir / self._config.tool_registry_path
        if not registry_path.exists():
            return {}

        with open(registry_path) as f:
            self._tool_registry = json.load(f)

        return self._tool_registry

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load task specifications.
        
        Returns:
            List of task dictionaries.
        """
        task_path = self._data_dir / self._config.task_file
        if not task_path.exists():
            return []

        with open(task_path) as f:
            tasks = json.load(f)

        return tasks if isinstance(tasks, list) else [tasks]

    def parse_workflow_log(self, log_path: str) -> List[SciAgentGymEvent]:
        """Parse a workflow execution log into events.
        
        Args:
            log_path: Path to the workflow log file (JSONL format).
            
        Returns:
            List of SciAgentGym events extracted from the log.
        """
        events: List[SciAgentGymEvent] = []
        log_file = Path(log_path)

        if not log_file.exists():
            return events

        with open(log_file) as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    log_entry = json.loads(line)
                    event = self._parse_log_entry(log_entry)
                    if event:
                        events.append(event)
                except json.JSONDecodeError:
                    continue

        return events

    def _parse_log_entry(self, entry: Dict[str, Any]) -> Optional[SciAgentGymEvent]:
        """Parse a single log entry into an event.
        
        Args:
            entry: Log entry dictionary.
            
        Returns:
            SciAgentGymEvent if the entry is valid, None otherwise.
        """
        event_type_str = entry.get("event_type")
        if not event_type_str:
            return None

        try:
            event_type = SciAgentGymEventType(event_type_str)
        except ValueError:
            return None

        timestamp_str = entry.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_str)
            if timestamp_str
            else datetime.now()
        )

        return SciAgentGymEvent(
            event_type=event_type,
            timestamp=timestamp,
            agent_id=entry.get("agent_id", "unknown"),
            payload=entry.get("payload", {}),
        )

    def parse_tool_call_results(
        self, results_dir: str
    ) -> List[SciAgentGymEvent]:
        """Parse tool call results from a directory.
        
        Args:
            results_dir: Directory containing tool call result files.
            
        Returns:
            List of tool call events.
        """
        events: List[SciAgentGymEvent] = []
        results_path = Path(results_dir)

        if not results_path.exists():
            return events

        for result_file in results_path.glob("tool_*.json"):
            with open(result_file) as f:
                result_data = json.load(f)

            event_type = (
                SciAgentGymEventType.TOOL_CALL_COMPLETED
                if result_data.get("success")
                else SciAgentGymEventType.TOOL_CALL_FAILED
            )

            event = SciAgentGymEvent(
                event_type=event_type,
                timestamp=datetime.now(),
                agent_id=result_data.get("agent_id", "unknown"),
                payload=result_data,
            )
            events.append(event)

        return events

    def get_tool_specification(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get the specification for a specific tool.
        
        Args:
            tool_name: Name of the tool.
            
        Returns:
            Tool specification dictionary or None if not found.
        """
        registry = self.load_tool_registry()
        return registry.get(tool_name)

    def validate_workflow_structure(
        self, workflow: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Validate a workflow structure.
        
        Args:
            workflow: Workflow specification dictionary.
            
        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors: List[str] = []

        if "steps" not in workflow:
            errors.append("Workflow must have 'steps' field")

        if "workflow_id" not in workflow:
            errors.append("Workflow must have 'workflow_id' field")

        steps = workflow.get("steps", [])
        if len(steps) > self._config.max_workflow_steps:
            errors.append(
                f"Workflow has {len(steps)} steps, "
                f"exceeds max {self._config.max_workflow_steps}"
            )

        # Check for cyclic dependencies
        if steps and self._has_cyclic_dependencies(steps):
            errors.append("Workflow has cyclic dependencies")

        return (len(errors) == 0, errors)

    def _has_cyclic_dependencies(self, steps: List[Dict[str, Any]]) -> bool:
        """Check if workflow steps have cyclic dependencies.
        
        Args:
            steps: List of workflow steps.
            
        Returns:
            True if there are cycles, False otherwise.
        """
        # Build dependency graph
        graph: Dict[int, List[int]] = {}
        for i, step in enumerate(steps):
            deps = step.get("dependencies", [])
            graph[i] = deps

        # DFS cycle detection
        visited = set()
        rec_stack = set()

        def has_cycle(node: int) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True

        return False
