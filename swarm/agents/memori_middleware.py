"""Memori integration middleware for LLM agents.

Wraps LLM client objects (Anthropic, OpenAI) with Memori's semantic memory
layer, enabling persistent fact recall and extraction across simulation steps.

Requires: pip install memori>=3.0.0
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoriConfig:
    """Configuration for Memori semantic memory integration.

    Attributes:
        enabled: Master toggle for the middleware.
        db_path: SQLite path for fact storage (":memory:" for ephemeral).
        entity_id: Memori entity identifier; defaults to agent_id.
        process_id: Memori process identifier; defaults to "swarm-{agent_id}".
        auto_wait: Block until fact extraction completes (for reproducibility).
        decay_on_epoch: Call new_session() at epoch boundaries for non-river
            agents (persistence < 1.0), causing recency down-weighting.
    """

    enabled: bool = False
    db_path: str = ":memory:"
    entity_id: Optional[str] = None
    process_id: Optional[str] = None
    auto_wait: bool = True
    decay_on_epoch: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoriConfig":
        """Construct from a plain dictionary (e.g. YAML config)."""
        return cls(
            enabled=data.get("enabled", False),
            db_path=data.get("db_path", ":memory:"),
            entity_id=data.get("entity_id"),
            process_id=data.get("process_id"),
            auto_wait=data.get("auto_wait", True),
            decay_on_epoch=data.get("decay_on_epoch", True),
        )


class MemoriMiddleware:
    """Lazy-loading middleware that wraps LLM clients with Memori.

    Usage::

        mw = MemoriMiddleware(config, agent_id="agent_1")
        client = mw.wrap_client(anthropic_client)
        # client now auto-recalls and extracts facts on each LLM call
    """

    def __init__(self, config: MemoriConfig, agent_id: str) -> None:
        self._config = config
        self._agent_id = agent_id
        self._memori: Any = None  # Lazy-loaded Memori instance

    def _get_memori(self) -> Any:
        """Lazy-import and initialize the Memori instance."""
        if self._memori is not None:
            return self._memori

        try:
            from memori import Memori
        except ImportError as err:
            raise ImportError(
                "memori package not installed. "
                "Install with: python -m pip install 'memori>=3.0.0'"
            ) from err

        entity_id = self._config.entity_id or self._agent_id
        process_id = self._config.process_id or f"swarm-{self._agent_id}"

        self._memori = Memori(  # type: ignore[call-arg]
            entity_id=entity_id,
            process_id=process_id,
            db_path=self._config.db_path,
        )
        # Build storage tables
        self._memori.config.storage.build()
        logger.info(
            "Memori initialized for agent %s (entity=%s, db=%s)",
            self._agent_id,
            entity_id,
            self._config.db_path,
        )
        return self._memori

    def wrap_client(self, client: Any) -> Any:
        """Register an LLM client with Memori for automatic memory injection.

        Args:
            client: An Anthropic, OpenAI, or compatible client object.

        Returns:
            The same client object (Memori patches it in-place).
        """
        memori = self._get_memori()
        try:
            memori.llm.register(client)
        except Exception:
            logger.warning(
                "Memori could not register client for agent %s "
                "(provider may be unsupported, e.g. Ollama). "
                "Memory features disabled for this client.",
                self._agent_id,
            )
        return client

    def wait_for_augmentation(self) -> None:
        """Block until any pending fact extraction completes."""
        if self._memori is None:
            return
        try:
            self._memori.augmentation.wait()
        except Exception:
            logger.debug(
                "Memori augmentation wait failed for agent %s", self._agent_id
            )

    def on_epoch_boundary(self, epoch: int, epistemic_persistence: float) -> None:
        """Handle epoch transitions for memory decay.

        For agents with persistence < 1.0 (non-river), starts a new Memori
        session so that older facts are naturally down-weighted by recency
        in semantic search.

        Args:
            epoch: The new epoch number.
            epistemic_persistence: The agent's epistemic persistence value
                (0.0 = full rain, 1.0 = full river).
        """
        if not self._config.decay_on_epoch:
            return

        if epistemic_persistence >= 1.0:
            # River agents: full persistence, no session rotation
            return

        if self._memori is None:
            return

        try:
            self._memori.new_session()
            logger.debug(
                "Memori new_session() for agent %s at epoch %d "
                "(persistence=%.2f)",
                self._agent_id,
                epoch,
                epistemic_persistence,
            )
        except Exception:
            logger.warning(
                "Memori new_session() failed for agent %s at epoch %d",
                self._agent_id,
                epoch,
            )

    def close(self) -> None:
        """Wait for pending work and clean up."""
        if self._config.auto_wait:
            self.wait_for_augmentation()
        self._memori = None
