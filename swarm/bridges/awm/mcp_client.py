"""Lightweight MCP client for AWM tool-use environments.

Uses httpx for direct HTTP calls to AWM's FastAPI/MCP endpoints.
Does NOT require the mcp-agent library (Python 3.12+).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallRecord:
    """Record of a single MCP tool call and its result."""

    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)

    # Response
    success: bool = False
    result: Any = None
    error: Optional[str] = None

    # Classification
    is_malformed: bool = False  # Invalid arguments / schema violation
    is_error_response: bool = False  # Server returned error


@dataclass
class AWMEpisodeTrace:
    """Complete trace of an agent's interaction with an AWM environment.

    Captures all tool calls, verification result, and timing metadata.
    """

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    environment_id: str = ""
    task_description: str = ""

    # Tool call history (ordered)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)

    # Verification result (set after episode completes)
    verified: Optional[bool] = None  # None = not yet verified
    verification_details: Dict[str, Any] = field(default_factory=dict)

    # Timing
    steps_used: int = 0
    max_steps: int = 20

    @property
    def error_count(self) -> int:
        """Count of tool calls that returned errors."""
        return sum(1 for tc in self.tool_calls if tc.is_error_response)

    @property
    def malformed_count(self) -> int:
        """Count of malformed tool calls (invalid args)."""
        return sum(1 for tc in self.tool_calls if tc.is_malformed)

    @property
    def total_calls(self) -> int:
        """Total number of tool calls made."""
        return len(self.tool_calls)


class AWMMCPClient:
    """HTTP client for interacting with AWM MCP tool-use servers.

    Each AWM environment exposes tools via a FastAPI endpoint.
    This client sends tool call requests and collects responses.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9100",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Any = None  # httpx.AsyncClient, lazily created

    async def _ensure_client(self) -> Any:
        """Lazily create httpx client."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the AWM environment."""
        client = await self._ensure_client()
        response = await client.get("/tools")
        response.raise_for_status()
        result: List[Dict[str, Any]] = response.json().get("tools", [])
        return result

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolCallRecord:
        """Execute a single tool call against the AWM environment.

        Returns a ToolCallRecord with success/error status.
        """
        record = ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
        )

        try:
            client = await self._ensure_client()
            response = await client.post(
                "/tools/call",
                json={
                    "name": tool_name,
                    "arguments": arguments,
                },
            )

            if response.status_code == 422:
                # Schema validation error = malformed call
                record.is_malformed = True
                record.is_error_response = True
                record.error = response.text
            elif response.status_code >= 400:
                record.is_error_response = True
                record.error = response.text
            else:
                data = response.json()
                record.success = True
                record.result = data.get("result")
                if data.get("isError"):
                    record.is_error_response = True
                    record.error = str(data.get("result", ""))
                    record.success = False

        except Exception as exc:
            record.is_error_response = True
            record.error = str(exc)

        return record

    async def reset_environment(self) -> bool:
        """Reset the AWM environment to initial DB state."""
        try:
            client = await self._ensure_client()
            response = await client.post("/reset")
            return bool(response.status_code == 200)
        except Exception:
            return False

    async def get_task(self) -> Optional[Dict[str, Any]]:
        """Get the current task description from the environment."""
        try:
            client = await self._ensure_client()
            response = await client.get("/task")
            if response.status_code == 200:
                result: Dict[str, Any] = response.json()
                return result
        except Exception:
            pass
        return None

    async def verify(self) -> Dict[str, Any]:
        """Run AWM verification on the current DB state.

        Returns verification result dict with at least a 'passed' key.
        """
        try:
            client = await self._ensure_client()
            response = await client.post("/verify")
            if response.status_code == 200:
                result: Dict[str, Any] = response.json()
                return result
        except Exception:
            pass
        return {"passed": False, "error": "verification_failed"}

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
