"""Manages AWM FastAPI server instances for agent environments.

Each agent gets its own server instance (separate port, private DB copy).
Servers start lazily on first AWM_EXECUTE_TASK action, reset DB between
epochs, and stop on orchestrator shutdown.

NOTE: In Phase 1, servers are simulated (no actual subprocess).
Phase 2 will add real server lifecycle management.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from swarm.bridges.awm.config import AWMConfig

logger = logging.getLogger(__name__)


class AWMServerInstance:
    """Represents a single AWM server instance for one agent."""

    def __init__(
        self,
        agent_id: str,
        port: int,
        environment_id: str,
        envs_path: str,
    ) -> None:
        self.agent_id = agent_id
        self.port = port
        self.environment_id = environment_id
        self.envs_path = envs_path
        self.running = False
        self._process: Any = None  # subprocess.Popen in Phase 2

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> bool:
        """Start the AWM server.

        Phase 1: No-op (simulation only).
        Phase 2: Will launch FastAPI subprocess.
        """
        logger.info(
            "AWM server start (simulated) for agent=%s port=%d env=%s",
            self.agent_id,
            self.port,
            self.environment_id,
        )
        self.running = True
        return True

    async def stop(self) -> None:
        """Stop the AWM server."""
        if self._process is not None:
            self._process.terminate()
            self._process = None
        self.running = False
        logger.info(
            "AWM server stopped for agent=%s port=%d",
            self.agent_id,
            self.port,
        )

    async def reset_db(self) -> bool:
        """Reset the database to initial state.

        Phase 1: No-op.
        Phase 2: Will call /reset endpoint.
        """
        logger.debug("AWM DB reset (simulated) for agent=%s", self.agent_id)
        return True


class AWMServerManager:
    """Manages AWM server instances for all agents in a scenario.

    Lifecycle:
    - start_server(agent_id): Lazy start on first task
    - reset_all(): Called between epochs
    - shutdown(): Called on orchestrator shutdown
    """

    def __init__(self, config: AWMConfig) -> None:
        self.config = config
        self._servers: Dict[str, AWMServerInstance] = {}
        self._next_port = config.base_port

    def _allocate_port(self) -> int:
        """Allocate the next available port."""
        port = self._next_port
        self._next_port += 1
        return port

    async def start_server(self, agent_id: str) -> Optional[AWMServerInstance]:
        """Start (or return existing) server for an agent.

        Returns None if max_concurrent_servers would be exceeded.
        """
        if agent_id in self._servers:
            server = self._servers[agent_id]
            if not server.running:
                await server.start()
            return server

        if len(self._servers) >= self.config.max_concurrent_servers:
            logger.warning(
                "Cannot start AWM server for %s: max_concurrent_servers=%d reached",
                agent_id,
                self.config.max_concurrent_servers,
            )
            return None

        port = self._allocate_port()
        server = AWMServerInstance(
            agent_id=agent_id,
            port=port,
            environment_id=self.config.environment_id,
            envs_path=str(self.config.envs_path),
        )
        await server.start()
        self._servers[agent_id] = server
        return server

    def get_server(self, agent_id: str) -> Optional[AWMServerInstance]:
        """Get an existing server instance for an agent."""
        return self._servers.get(agent_id)

    async def reset_all(self) -> None:
        """Reset all servers' databases (called between epochs)."""
        for server in self._servers.values():
            if server.running:
                await server.reset_db()

    async def shutdown(self) -> None:
        """Stop all servers and clean up."""
        for server in list(self._servers.values()):
            await server.stop()
        self._servers.clear()
        self._next_port = self.config.base_port

    @property
    def active_count(self) -> int:
        """Number of currently running servers."""
        return sum(1 for s in self._servers.values() if s.running)
