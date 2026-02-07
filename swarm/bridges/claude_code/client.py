"""HTTP/WebSocket client for the Claude Code controller service.

Communicates with the TypeScript claude-code-service that wraps
ClaudeCodeController. Provides sync wrappers around async HTTP
calls for compatibility with SWARM's synchronous orchestrator loop.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from swarm.bridges.claude_code.events import (
    BridgeEvent,
    BridgeEventType,
    MessageEvent,
    TaskEvent,
)

logger = logging.getLogger(__name__)

# Agent IDs must be alphanumeric with hyphens/underscores, max 128 chars
_AGENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")

# Maximum response body size to read (10 MB)
_MAX_RESPONSE_SIZE = 10 * 1024 * 1024


def _validate_agent_id(agent_id: str) -> None:
    """Validate that an agent ID is safe for URL interpolation."""
    if not _AGENT_ID_RE.match(agent_id):
        raise ValueError(
            f"Invalid agent_id '{agent_id}': must match [a-zA-Z0-9_-]{{1,128}}"
        )


@dataclass
class ClientConfig:
    """Configuration for the Claude Code controller client."""

    base_url: str = "http://localhost:3100"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_base: float = 1.0
    api_key: Optional[str] = None

    def __repr__(self) -> str:
        """Redact api_key in repr to prevent accidental exposure."""
        key_display = "***" if self.api_key else "None"
        return (
            f"ClientConfig(base_url={self.base_url!r}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"max_retries={self.max_retries}, "
            f"api_key={key_display})"
        )


class ClaudeCodeClient:
    """HTTP client for the Claude Code controller service.

    Endpoints correspond to the controller's capabilities:
    - POST /agents/spawn          → spawn a new Claude Code agent
    - POST /agents/{id}/ask       → single-turn prompt
    - POST /tasks                 → create an inbox task
    - GET  /tasks/{id}/wait       → wait for task completion
    - POST /agents/{id}/shutdown  → shut down an agent
    - GET  /events                → poll recent events
    - POST /governance/respond    → respond to plan/permission requests
    """

    def __init__(self, config: Optional[ClientConfig] = None):
        self.config = config or ClientConfig()
        self._event_listeners: Dict[
            BridgeEventType, List[Callable[[BridgeEvent], None]]
        ] = {}

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the controller service.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: URL path (e.g., "/agents/spawn")
            body: Optional JSON body

        Returns:
            Parsed JSON response

        Raises:
            ConnectionError: If the service is unreachable after retries
            RuntimeError: If the service returns an error status
        """
        url = f"{self.config.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        data = json.dumps(body).encode("utf-8") if body else None
        req = Request(url, data=data, headers=headers, method=method)

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                timeout = self.config.timeout_seconds
                with urlopen(req, timeout=timeout) as resp:
                    resp_body = resp.read(_MAX_RESPONSE_SIZE).decode("utf-8")
                    if resp_body:
                        result: Dict[str, Any] = json.loads(resp_body)
                        return result
                    return {}
            except URLError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait = self.config.retry_backoff_base * (2**attempt)
                    logger.warning(
                        "Request to %s failed (attempt %d/%d), retrying in %.1fs: %s",
                        url,
                        attempt + 1,
                        self.config.max_retries,
                        wait,
                        e,
                    )
                    time.sleep(wait)

        raise ConnectionError(
            f"Failed to reach controller at {url} after "
            f"{self.config.max_retries} attempts: {last_error}"
        )

    # --- Agent lifecycle ---

    def spawn_agent(
        self,
        agent_id: str,
        system_prompt: str = "",
        allowed_tools: Optional[List[str]] = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> Dict[str, Any]:
        """Spawn a new Claude Code agent via the controller.

        Args:
            agent_id: Unique identifier for the agent
            system_prompt: System prompt defining agent behavior
            allowed_tools: Tool allowlist (e.g., ["Bash", "Read", "Write"])
            model: Model to use for the agent

        Returns:
            Controller response with agent metadata
        """
        body: Dict[str, Any] = {
            "agent_id": agent_id,
            "system_prompt": system_prompt,
            "model": model,
        }
        if allowed_tools is not None:
            body["allowed_tools"] = allowed_tools
        return self._request("POST", "/agents/spawn", body)

    def shutdown_agent(self, agent_id: str) -> Dict[str, Any]:
        """Shut down a running agent.

        Args:
            agent_id: ID of the agent to shut down
        """
        _validate_agent_id(agent_id)
        return self._request("POST", f"/agents/{agent_id}/shutdown")

    # --- Interaction ---

    def ask(
        self,
        agent_id: str,
        prompt: str,
        timeout_seconds: Optional[float] = None,
    ) -> MessageEvent:
        """Send a single-turn prompt to an agent and get a response.

        Args:
            agent_id: Target agent ID
            prompt: The prompt to send
            timeout_seconds: Override default timeout

        Returns:
            MessageEvent with the agent's response
        """
        _validate_agent_id(agent_id)
        body = {"prompt": prompt}
        if timeout_seconds is not None:
            body["timeout_seconds"] = timeout_seconds  # type: ignore[assignment]
        resp = self._request("POST", f"/agents/{agent_id}/ask", body)
        return MessageEvent(
            agent_id=agent_id,
            role="assistant",
            content=resp.get("content", ""),
            tool_calls=resp.get("tool_calls", []),
            token_count=resp.get("token_count", 0),
            cost_usd=resp.get("cost_usd", 0.0),
        )

    # --- Task lifecycle ---

    def create_task(
        self,
        subject: str,
        description: str,
        owner: str,
    ) -> TaskEvent:
        """Create a task in the controller's inbox.

        Args:
            subject: Task subject line
            description: Full task description
            owner: Agent ID that should work on this task

        Returns:
            TaskEvent with the created task metadata
        """
        body = {
            "subject": subject,
            "description": description,
            "owner": owner,
        }
        resp = self._request("POST", "/tasks", body)
        return TaskEvent.from_dict(resp)

    def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 2.0,
        max_wait: float = 300.0,
    ) -> TaskEvent:
        """Poll until a task completes or times out.

        Args:
            task_id: ID of the task to wait for
            poll_interval: Seconds between polls
            max_wait: Maximum seconds to wait

        Returns:
            TaskEvent with the final task state
        """
        start = time.monotonic()
        while time.monotonic() - start < max_wait:
            resp = self._request("GET", f"/tasks/{task_id}/wait")
            task = TaskEvent.from_dict(resp)
            if task.status in ("completed", "failed"):
                return task
            time.sleep(poll_interval)
        return TaskEvent(
            task_id=task_id,
            status="failed",
            result="Timed out waiting for task completion",
        )

    # --- Event polling ---

    def poll_events(
        self,
        since_event_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[BridgeEvent]:
        """Poll recent events from the controller.

        Args:
            since_event_id: Only return events after this ID
            limit: Maximum number of events to return

        Returns:
            List of BridgeEvent objects
        """
        params = f"?limit={limit}"
        if since_event_id:
            params += f"&since={since_event_id}"
        resp = self._request("GET", f"/events{params}")
        events_data = resp.get("events", [])
        return [BridgeEvent.from_dict(e) for e in events_data]

    # --- Governance responses ---

    def respond_to_plan(
        self,
        request_id: str,
        approved: bool,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Respond to a plan approval request.

        Args:
            request_id: ID of the plan approval request
            approved: Whether to approve the plan
            reason: Optional reason for the decision
        """
        body = {
            "request_id": request_id,
            "decision": "approve" if approved else "reject",
            "reason": reason,
        }
        return self._request("POST", "/governance/respond", body)

    def respond_to_permission(
        self,
        request_id: str,
        granted: bool,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Respond to a tool permission request.

        Args:
            request_id: ID of the permission request
            granted: Whether to grant the permission
            reason: Optional reason for the decision
        """
        body = {
            "request_id": request_id,
            "decision": "grant" if granted else "deny",
            "reason": reason,
        }
        return self._request("POST", "/governance/respond", body)

    # --- Event listener registration ---

    def on(
        self,
        event_type: BridgeEventType,
        callback: Callable[[BridgeEvent], None],
    ) -> None:
        """Register a listener for a specific event type.

        Args:
            event_type: The event type to listen for
            callback: Function called when the event occurs
        """
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        self._event_listeners[event_type].append(callback)

    def _dispatch_event(self, event: BridgeEvent) -> None:
        """Dispatch an event to registered listeners."""
        listeners = self._event_listeners.get(event.event_type, [])
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                logger.exception(
                    "Error in event listener for %s", event.event_type.value
                )

    # --- Health check ---

    def ping(self) -> bool:
        """Check if the controller service is reachable.

        Returns:
            True if the service responds to a health check
        """
        try:
            self._request("GET", "/health")
            return True
        except (ConnectionError, URLError):
            return False
