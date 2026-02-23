"""Configuration for the RAG bridge."""

from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    """Configuration for RAG over run history.

    Attributes:
        persist_dir: ChromaDB storage path.
        collection_name: Name of the ChromaDB collection.
        embedding_provider: Embedding backend ("openai" or "ollama").
        embedding_model: Model name for embeddings.
        ollama_base_url: Base URL for Ollama server.
        chunk_size: Characters per text chunk.
        chunk_overlap: Overlap between chunks.
        top_k: Number of chunks to retrieve per query.
        llm_provider: LLM backend for synthesis ("anthropic" or "openai").
        llm_model: Model name for synthesis.
        max_tokens: Max tokens for synthesis response.
    """

    # Vector store
    persist_dir: str = ".rag_store"
    collection_name: str = "swarm_runs"

    # Embeddings
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    ollama_base_url: str = "http://localhost:11434"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 8

    # LLM for synthesis
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024

    # Metadata filter defaults
    default_doc_types: list[str] = field(
        default_factory=lambda: [
            "epoch_summary",
            "scenario_config",
            "agent_state",
            "event_summary",
        ]
    )
