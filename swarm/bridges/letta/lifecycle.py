"""Letta lifecycle manager for orchestrator integration.

Owns the server manager, client, and memory mapper. Called by the
orchestrator at simulation start, epoch boundaries, and shutdown.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from swarm.bridges.letta.client import LettaSwarmClient
from swarm.bridges.letta.config import LettaConfig
from swarm.bridges.letta.memory_mapper import LettaMemoryMapper
from swarm.bridges.letta.response_parser import LettaResponseParser
from swarm.bridges.letta.server_manager import LettaServerManager

logger = logging.getLogger(__name__)


class LettaLifecycleManager:
    """Manages the Letta runtime lifecycle for the orchestrator."""

    def __init__(self, config: LettaConfig) -> None:
        self.config = config
        self._server_manager = LettaServerManager(config)
        self._client = LettaSwarmClient(config)
        self._memory_mapper = LettaMemoryMapper(config)
        self._response_parser = LettaResponseParser()
        self._governance_block_id: Optional[str] = None
        self._agent_ids: List[str] = []  # Track created Letta agent IDs

    @property
    def client(self) -> LettaSwarmClient:
        return self._client

    @property
    def memory_mapper(self) -> LettaMemoryMapper:
        return self._memory_mapper

    @property
    def response_parser(self) -> LettaResponseParser:
        return self._response_parser

    @property
    def governance_block_id(self) -> Optional[str]:
        return self._governance_block_id

    def start(self) -> bool:
        """Start the Letta server and create shared governance block.

        Returns:
            True if startup succeeded.
        """
        if not self._server_manager.start():
            logger.error("Failed to start Letta server")
            return False

        # Create shared governance block
        if self.config.shared_governance_block:
            try:
                self._governance_block_id = self._client.create_shared_block(
                    label="governance_state",
                    value="{}",
                )
                logger.info(
                    "Created shared governance block: %s",
                    self._governance_block_id,
                )
            except Exception:
                logger.exception("Failed to create shared governance block")
                # Non-fatal: agents can still work without shared governance

        return True

    def register_letta_agent_id(self, letta_agent_id: str) -> None:
        """Track a Letta agent ID for cleanup."""
        self._agent_ids.append(letta_agent_id)

    def update_governance_block(
        self,
        metrics: Dict[str, Any],
        proposals: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update the shared governance block content.

        Called at epoch start by the orchestrator.

        Args:
            metrics: Current ecosystem metrics.
            proposals: Active governance proposals.
        """
        if self._governance_block_id is None:
            return

        content = self._memory_mapper.governance_state_to_block(metrics, proposals)
        try:
            client = self._client._ensure_client()
            client.blocks.update(
                block_id=self._governance_block_id,
                value=content,
            )
        except Exception:
            logger.exception("Failed to update governance block")

    def shutdown(self) -> None:
        """Delete all Letta agents and stop the server."""
        # Delete agents
        for letta_agent_id in self._agent_ids:
            try:
                self._client.delete_agent(letta_agent_id)
            except Exception:
                logger.debug(
                    "Failed to delete Letta agent %s during shutdown",
                    letta_agent_id,
                )

        self._agent_ids.clear()
        self._governance_block_id = None
        self._server_manager.stop()
        logger.info("Letta lifecycle manager shut down")
