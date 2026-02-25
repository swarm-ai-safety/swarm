"""Synchronous wrapper around the Letta Python SDK.

All SDK calls are lazy-imported so the bridge can be loaded without
``letta-client`` installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swarm.bridges.letta.config import LettaConfig

logger = logging.getLogger(__name__)


@dataclass
class LettaInteractionTrace:
    """Record of a single Letta send_message exchange."""

    agent_id: str = ""
    letta_agent_id: str = ""
    message: str = ""
    response: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)


class LettaSwarmClient:
    """Synchronous wrapper around ``letta_client.Letta`` SDK."""

    def __init__(self, config: LettaConfig) -> None:
        self.config = config
        self._client: Any = None
        self._initialized = False

    def _ensure_client(self) -> Any:
        """Lazy-init the Letta SDK client."""
        if self._client is not None:
            return self._client

        from letta_client import Letta

        kwargs: Dict[str, Any] = {"base_url": self.config.base_url}
        if self.config.api_key:
            kwargs["token"] = self.config.api_key
        self._client = Letta(**kwargs)
        self._initialized = True
        logger.info("Letta SDK client initialized at %s", self.config.base_url)
        return self._client

    def create_agent(
        self,
        name: str,
        persona: str,
        human_description: str = "SWARM orchestrator",
        model: Optional[str] = None,
        memory_blocks: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Create a Letta agent and return its ID.

        Args:
            name: Agent name (must be unique).
            persona: Persona text for the core memory persona block.
            human_description: Description for the core memory human block.
            model: LLM model identifier.
            memory_blocks: Additional memory blocks ``[{"label": ..., "value": ...}]``.

        Returns:
            The Letta agent ID string.
        """
        client = self._ensure_client()
        model = model or self.config.default_model

        # Build memory block list
        blocks = []
        if memory_blocks:
            for mb in memory_blocks:
                blocks.append(
                    {"label": mb["label"], "value": mb.get("value", "")}
                )

        try:
            agent = client.agents.create(
                name=name,
                model=model,
                memory_blocks=[
                    {"label": "human", "value": human_description},
                    {"label": "persona", "value": persona},
                    *blocks,
                ],
            )
            agent_id: str = agent.id
            logger.info("Created Letta agent name=%s id=%s", name, agent_id)
            return agent_id
        except Exception as exc:
            logger.exception("Failed to create Letta agent name=%s: %s", name, exc)
            raise

    def send_message(
        self,
        letta_agent_id: str,
        message: str,
    ) -> str:
        """Send a message to a Letta agent and return the response text.

        Args:
            letta_agent_id: The Letta agent's ID.
            message: The message to send.

        Returns:
            The agent's response text.
        """
        client = self._ensure_client()
        try:
            response = client.agents.messages.create(
                agent_id=letta_agent_id,
                messages=[{"role": "user", "content": message}],
            )
            # Extract text from response messages
            texts = []
            for msg in response.messages:
                if hasattr(msg, "content") and msg.content:
                    texts.append(msg.content)
                elif hasattr(msg, "assistant_message") and msg.assistant_message:
                    texts.append(msg.assistant_message)
            return "\n".join(texts) if texts else ""
        except Exception as exc:
            logger.exception(
                "Failed to send message to Letta agent %s: %s", letta_agent_id, exc
            )
            raise

    def update_core_memory(
        self,
        letta_agent_id: str,
        label: str,
        value: str,
    ) -> None:
        """Update a core memory block for an agent.

        Args:
            letta_agent_id: The Letta agent's ID.
            label: The memory block label.
            value: The new value.
        """
        client = self._ensure_client()
        try:
            # Get all blocks for the agent, find the one with matching label
            blocks = client.agents.blocks.list(agent_id=letta_agent_id)
            for block in blocks:
                if block.label == label:
                    client.agents.blocks.update(
                        agent_id=letta_agent_id,
                        block_id=block.id,
                        value=value,
                    )
                    return
            logger.warning(
                "Core memory block '%s' not found for agent %s",
                label,
                letta_agent_id,
            )
        except Exception as exc:
            logger.exception(
                "Failed to update core memory for agent %s: %s", letta_agent_id, exc
            )

    def get_core_memory(
        self,
        letta_agent_id: str,
        label: str,
    ) -> Optional[str]:
        """Get a core memory block value.

        Args:
            letta_agent_id: The Letta agent's ID.
            label: The memory block label.

        Returns:
            The block value, or None if not found.
        """
        client = self._ensure_client()
        try:
            blocks = client.agents.blocks.list(agent_id=letta_agent_id)
            for block in blocks:
                if block.label == label:
                    result: Optional[str] = block.value
                    return result
        except Exception as exc:
            logger.exception(
                "Failed to get core memory for agent %s: %s", letta_agent_id, exc
            )
        return None

    def create_shared_block(
        self,
        label: str,
        value: str,
    ) -> str:
        """Create a shared memory block (not attached to any agent yet).

        Args:
            label: Block label.
            value: Initial block content.

        Returns:
            The block ID.
        """
        client = self._ensure_client()
        block = client.blocks.create(label=label, value=value)
        block_id: str = block.id
        logger.info("Created shared block label=%s id=%s", label, block_id)
        return block_id

    def attach_shared_block(
        self,
        letta_agent_id: str,
        block_id: str,
    ) -> None:
        """Attach a shared memory block to an agent.

        Args:
            letta_agent_id: The Letta agent's ID.
            block_id: The shared block ID.
        """
        client = self._ensure_client()
        try:
            client.agents.blocks.attach(
                agent_id=letta_agent_id,
                block_id=block_id,
            )
        except Exception as exc:
            logger.exception(
                "Failed to attach shared block %s to agent %s: %s",
                block_id,
                letta_agent_id,
                exc,
            )

    def insert_archival(
        self,
        letta_agent_id: str,
        content: str,
    ) -> None:
        """Insert a record into archival memory.

        Args:
            letta_agent_id: The Letta agent's ID.
            content: The content to store.
        """
        client = self._ensure_client()
        try:
            client.agents.archival_memory.create(
                agent_id=letta_agent_id,
                content=content,
            )
        except Exception as exc:
            logger.exception(
                "Failed to insert archival memory for agent %s: %s",
                letta_agent_id,
                exc,
            )

    def search_archival(
        self,
        letta_agent_id: str,
        query: str,
        limit: int = 10,
    ) -> List[str]:
        """Search archival memory.

        Args:
            letta_agent_id: The Letta agent's ID.
            query: Search query.
            limit: Max results.

        Returns:
            List of matching content strings.
        """
        client = self._ensure_client()
        try:
            results = client.agents.archival_memory.list(
                agent_id=letta_agent_id,
                query=query,
                limit=limit,
            )
            return [r.content for r in results if hasattr(r, "content")]
        except Exception as exc:
            logger.exception(
                "Failed to search archival memory for agent %s: %s",
                letta_agent_id,
                exc,
            )
            return []

    def delete_agent(self, letta_agent_id: str) -> None:
        """Delete a Letta agent.

        Args:
            letta_agent_id: The Letta agent's ID.
        """
        client = self._ensure_client()
        try:
            client.agents.delete(agent_id=letta_agent_id)
            logger.info("Deleted Letta agent %s", letta_agent_id)
        except Exception as exc:
            logger.exception(
                "Failed to delete Letta agent %s: %s", letta_agent_id, exc
            )
