"""Run retriever — queries the vector store and synthesizes answers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from swarm.bridges.rag.config import RAGConfig

try:
    import chromadb

    _HAS_RAG = True
except ImportError:
    _HAS_RAG = False

logger = logging.getLogger(__name__)


def _require_rag() -> None:
    if not _HAS_RAG:
        raise ImportError(
            "RAG dependencies not installed. "
            "Install with: python -m pip install 'swarm-safety[rag]'"
        )


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its metadata."""

    text: str
    run_id: str
    doc_type: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Response from a RAG query."""

    answer: str
    sources: list[RetrievedChunk]
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


class RunRetriever:
    """Queries run history via vector search and LLM synthesis.

    Uses ChromaDB for retrieval and the Anthropic/OpenAI SDK for
    answer synthesis (following the existing LLMAgent dispatch pattern).
    """

    def __init__(self, config: RAGConfig | None = None) -> None:
        _require_rag()
        self.config = config or RAGConfig()
        self._client = chromadb.PersistentClient(path=self.config.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_fn = self._build_embedding_fn()

    def _build_embedding_fn(self) -> Any:
        """Build the embedding function based on config."""
        provider = self.config.embedding_provider.lower()
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=self.config.embedding_model)
        elif provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(
                model=self.config.embedding_model,
                base_url=self.config.ollama_base_url,
            )
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        doc_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks without LLM synthesis.

        Args:
            question: Natural language query.
            top_k: Override default top_k.
            doc_types: Filter by document types.

        Returns:
            List of retrieved chunks sorted by relevance.
        """
        k = top_k or self.config.top_k
        query_embedding = self._embedding_fn.embed_query(question)

        where_filter = None
        if doc_types:
            where_filter = {"doc_type": {"$in": doc_types}}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances, strict=False):
            chunks.append(
                RetrievedChunk(
                    text=doc or "",
                    run_id=meta.get("run_id", "") if meta else "",
                    doc_type=meta.get("doc_type", "") if meta else "",
                    score=1.0 - dist,  # cosine distance → similarity
                    metadata=dict(meta) if meta else {},
                )
            )

        return chunks

    def query(
        self,
        question: str,
        top_k: int | None = None,
        doc_types: list[str] | None = None,
    ) -> RAGResponse:
        """End-to-end RAG: retrieve chunks then synthesize an answer.

        Args:
            question: Natural language query.
            top_k: Override default top_k.
            doc_types: Filter by document types.

        Returns:
            RAGResponse with answer and source chunks.
        """
        chunks = self.retrieve(question, top_k=top_k, doc_types=doc_types)

        if not chunks:
            return RAGResponse(
                answer="No relevant run data found. Try indexing some runs first.",
                sources=[],
                model=self.config.llm_model,
            )

        prompt = self._build_synthesis_prompt(question, chunks)
        answer, usage = self._call_llm(prompt)

        return RAGResponse(
            answer=answer,
            sources=chunks,
            model=self.config.llm_model,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_synthesis_prompt(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> str:
        """Build the synthesis prompt with retrieved context."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] (run={chunk.run_id}, type={chunk.doc_type}, "
                f"score={chunk.score:.3f})\n{chunk.text}"
            )
        context = "\n\n".join(context_parts)

        return (
            "You are a research assistant analyzing SWARM simulation runs. "
            "Answer the question based ONLY on the retrieved context below. "
            "Cite specific runs, epochs, and metrics. If the context doesn't "
            "contain enough information, say so.\n\n"
            f"## Retrieved Context\n\n{context}\n\n"
            f"## Question\n\n{question}\n\n"
            "## Answer\n\n"
        )

    def _call_llm(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Call the LLM for synthesis. Returns (answer, usage_dict)."""
        provider = self.config.llm_provider.lower()

        if provider == "anthropic":
            return self._call_anthropic(prompt)
        elif provider == "openai":
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def _call_anthropic(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Call Anthropic API for synthesis."""
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.config.llm_model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = ""
        if response.content:
            block = response.content[0]
            answer = block.text if hasattr(block, "text") else ""
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return answer, usage

    def _call_openai(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Call OpenAI API for synthesis."""
        import openai

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.config.llm_model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = response.choices[0] if response.choices else None
        answer = choice.message.content or "" if choice else ""
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return answer, usage
