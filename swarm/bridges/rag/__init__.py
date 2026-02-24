"""SWARM RAG Bridge — semantic search over run history.

Indexes run artifacts (JSONL event logs, JSON history snapshots, CSV
metrics, scenario YAMLs) into a vector store and enables natural
language queries grounded in actual run data.

Supports two vector backends (selected via ``RAGConfig.vector_backend``):

- **chromadb** (default) — ChromaDB with LangChain embeddings.
- **leann** — LEANN graph-based index with ~97% storage savings,
  local embedding models, and JSON sidecar metadata.

Architecture::

    Run artifacts (JSONL, JSON, CSV, YAML)
      -> Document chunking & metadata extraction
      -> Embeddings (OpenAI/Ollama for ChromaDB; internal for LEANN)
      -> Vector backend (ChromaDB or LEANN, local, persistent)
      -> Natural language query -> retrieve relevant chunks
      -> LLM synthesizes answer

Usage::

    from swarm.bridges.rag import RAGConfig, RunIndexer, RunRetriever

    config = RAGConfig()
    indexer = RunIndexer(config)
    indexer.index_all(Path("runs/"))

    retriever = RunRetriever(config)
    response = retriever.query("Which governance configs kept toxicity below 0.1?")
    print(response.answer)

CLI::

    python -m swarm.bridges.rag index runs/
    python -m swarm.bridges.rag query "Compare runs with and without circuit breakers"
    python -m swarm.bridges.rag status
"""

from swarm.bridges.rag.config import RAGConfig

try:
    from swarm.bridges.rag.indexer import RunIndexer
    from swarm.bridges.rag.retriever import RAGResponse, RetrievedChunk, RunRetriever

    _HAS_RAG = True
except ImportError:
    _HAS_RAG = False

__all__ = [
    "RAGConfig",
    "RunIndexer",
    "RunRetriever",
    "RAGResponse",
    "RetrievedChunk",
    "_HAS_RAG",
]


def index_run(run_dir: str, config: RAGConfig | None = None) -> int:
    """Convenience: index a single run directory.

    Args:
        run_dir: Path to the run directory.
        config: Optional RAG configuration.

    Returns:
        Number of chunks indexed.
    """
    from pathlib import Path

    indexer = RunIndexer(config)
    return indexer.index_run(Path(run_dir))


def query_runs(question: str, config: RAGConfig | None = None) -> "RAGResponse":
    """Convenience: query indexed runs.

    Args:
        question: Natural language question.
        config: Optional RAG configuration.

    Returns:
        RAGResponse with answer and sources.
    """
    retriever = RunRetriever(config)
    return retriever.query(question)
