"""Vector backend abstraction for the RAG bridge.

Provides a Protocol-based interface so the indexer and retriever can
work with either ChromaDB (default) or LEANN without knowing which
backend is active.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from swarm.bridges.rag.config import RAGConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared result type
# ------------------------------------------------------------------


@dataclass
class BackendResult:
    """A single result returned by a vector backend query."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Protocol
# ------------------------------------------------------------------


@runtime_checkable
class VectorBackend(Protocol):
    """Minimal interface that every vector backend must satisfy."""

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None: ...

    def finalize(self) -> None: ...

    def query(
        self,
        query_embedding: list[float] | None = None,
        query_text: str | None = None,
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[BackendResult]: ...

    @property
    def count(self) -> int: ...


# ------------------------------------------------------------------
# ChromaDB backend
# ------------------------------------------------------------------


class ChromaBackend:
    """Wraps ChromaDB as a VectorBackend."""

    def __init__(self, config: RAGConfig) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=config.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {"ids": ids, "documents": texts}
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        self._collection.upsert(**kwargs)

    def finalize(self) -> None:
        pass  # ChromaDB persists incrementally

    def query(
        self,
        query_embedding: list[float] | None = None,
        query_text: str | None = None,
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[BackendResult]:
        if query_embedding is None:
            raise ValueError("ChromaBackend.query requires query_embedding")

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        out: list[BackendResult] = []
        for doc_id, doc, meta, dist in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            out.append(
                BackendResult(
                    id=doc_id,
                    text=doc or "",
                    score=1.0 - dist,  # cosine distance -> similarity
                    metadata=dict(meta) if meta else {},
                )
            )
        return out

    @property
    def count(self) -> int:
        result: int = self._collection.count()
        return result


# ------------------------------------------------------------------
# LEANN backend
# ------------------------------------------------------------------


class LeannBackend:
    """Wraps LEANN builder/searcher with a JSON sidecar for metadata.

    LEANN computes embeddings internally, so ``embeddings`` passed to
    ``upsert()`` are ignored.  The ``query()`` method accepts
    ``query_text`` and delegates embedding + search to LEANN.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._index_dir = Path(config.leann_index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._sidecar_path = self._index_dir / "metadata.json"
        self._backend = config.leann_backend

        # Accumulate texts + metadata until finalize()
        self._pending_ids: list[str] = []
        self._pending_texts: list[str] = []
        self._pending_meta: list[dict[str, Any]] = []

        # Sidecar: id -> {text, metadata}
        self._sidecar: dict[str, dict[str, Any]] = {}
        if self._sidecar_path.exists():
            self._sidecar = json.loads(self._sidecar_path.read_text())

        self._searcher: Any = None

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        metas = metadatas or [{} for _ in ids]
        for chunk_id, text, meta in zip(ids, texts, metas, strict=False):
            self._pending_ids.append(chunk_id)
            self._pending_texts.append(text)
            self._pending_meta.append(meta)
            self._sidecar[chunk_id] = {"text": text, "metadata": meta}

    def finalize(self) -> None:
        """Build the LEANN index from all accumulated texts."""
        if not self._pending_texts:
            return

        from leann import LeannBuilder  # type: ignore[import-untyped]

        builder = LeannBuilder(backend=self._backend)
        builder.add_texts(self._pending_texts, ids=self._pending_ids)
        builder.build_index(str(self._index_dir))

        # Persist sidecar
        self._sidecar_path.write_text(json.dumps(self._sidecar))

        # Clear pending
        self._pending_ids.clear()
        self._pending_texts.clear()
        self._pending_meta.clear()

        # Invalidate cached searcher so next query picks up new index
        self._searcher = None

        logger.info(
            "LEANN index built at %s (%d documents)",
            self._index_dir,
            len(self._sidecar),
        )

    def query(
        self,
        query_embedding: list[float] | None = None,
        query_text: str | None = None,
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[BackendResult]:
        if query_text is None:
            raise ValueError("LeannBackend.query requires query_text")

        searcher = self._get_searcher()

        # Over-fetch when filtering to compensate for post-retrieval filtering
        fetch_k = top_k * 3 if where else top_k
        raw_results = searcher.search(query_text, k=fetch_k)

        out: list[BackendResult] = []
        for hit in raw_results:
            hit_id = hit.id
            sidecar_entry = self._sidecar.get(hit_id, {})
            meta = sidecar_entry.get("metadata", {})
            text = sidecar_entry.get("text", "")

            if where and not _apply_where_filter(meta, where):
                continue

            out.append(
                BackendResult(
                    id=hit_id,
                    text=text,
                    score=hit.score,
                    metadata=meta,
                )
            )
            if len(out) >= top_k:
                break

        return out

    @property
    def count(self) -> int:
        return len(self._sidecar)

    def _get_searcher(self) -> Any:
        if self._searcher is None:
            from leann import LeannSearcher  # type: ignore[import-untyped]

            self._searcher = LeannSearcher(str(self._index_dir))
        return self._searcher


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _apply_where_filter(
    metadata: dict[str, Any], where: dict[str, Any]
) -> bool:
    """Evaluate a ChromaDB-style ``where`` clause against metadata.

    Supports:
      - ``{"key": value}``  — equality
      - ``{"key": {"$in": [...]}}`` — membership
      - ``{"key": {"$ne": value}}`` — inequality
    """
    for key, condition in where.items():
        val = metadata.get(key)
        if isinstance(condition, dict):
            if "$in" in condition:
                if val not in condition["$in"]:
                    return False
            if "$ne" in condition:
                if val == condition["$ne"]:
                    return False
        else:
            if val != condition:
                return False
    return True


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def build_backend(config: RAGConfig) -> VectorBackend:
    """Construct the appropriate vector backend from config."""
    backend_name = config.vector_backend.lower()
    if backend_name == "chromadb":
        return ChromaBackend(config)
    elif backend_name == "leann":
        return LeannBackend(config)
    else:
        raise ValueError(f"Unknown vector_backend: {config.vector_backend!r}")
