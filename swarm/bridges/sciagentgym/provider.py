"""SciAgentGym tool provider adapter for SWARM action interfaces."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from .manager import SciEnvManager


class SciToolEnvironment(Protocol):
    """Minimal protocol expected from a SciAgentGym-like environment."""

    def list_tools(self) -> list[dict[str, Any]]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class ToolCallResult:
    """Normalized tool-call payload for SWARM decision loops."""

    ok: bool
    result: Any = None
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)
    call_signature_hash: str = ""


@dataclass
class SciAgentGymToolProvider:
    """Expose SciAgentGym tools through a SWARM-friendly API."""

    env_manager: SciEnvManager

    def list_tools(
        self,
        *,
        episode_id: str,
        agent_id: str | None = None,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        env: SciToolEnvironment = self.env_manager.get_or_create(
            episode_id=episode_id,
            agent_id=agent_id,
            task_id=task_id,
        )
        return env.list_tools()

    def call_tool(
        self,
        *,
        episode_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        agent_id: str | None = None,
        task_id: str | None = None,
    ) -> ToolCallResult:
        env: SciToolEnvironment = self.env_manager.get_or_create(
            episode_id=episode_id,
            agent_id=agent_id,
            task_id=task_id,
        )
        signature = self._signature_hash(tool_name=tool_name, arguments=arguments)
        try:
            raw = env.call_tool(tool_name, arguments)
        except Exception as exc:  # noqa: BLE001
            return ToolCallResult(ok=False, error=str(exc), call_signature_hash=signature)

        return ToolCallResult(
            ok=bool(raw.get("ok", True)),
            result=raw.get("result"),
            error=raw.get("error"),
            artifacts=list(raw.get("artifacts", [])),
            call_signature_hash=signature,
        )

    @staticmethod
    def _signature_hash(tool_name: str, arguments: dict[str, Any]) -> str:
        payload = f"{tool_name}:{sorted(arguments.items())}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
