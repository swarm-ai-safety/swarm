"""CLI entry point for the RAG bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Run the RAG CLI."""
    parser = argparse.ArgumentParser(
        prog="swarm.bridges.rag",
        description="RAG over SWARM run history",
    )
    sub = parser.add_subparsers(dest="command")

    # --- index ---
    idx = sub.add_parser("index", help="Index run artifacts")
    idx.add_argument("path", type=Path, help="Run directory or parent runs/ dir")
    idx.add_argument("--persist-dir", default=".rag_store", help="ChromaDB path")
    idx.add_argument(
        "--embedding-provider",
        default="openai",
        choices=["openai", "ollama"],
    )
    idx.add_argument("--embedding-model", default=None)
    idx.add_argument(
        "--vector-backend",
        default="chromadb",
        choices=["chromadb", "leann"],
        help="Vector store backend",
    )

    # --- query ---
    q = sub.add_parser("query", help="Query indexed runs")
    q.add_argument("question", type=str, help="Natural language question")
    q.add_argument("--persist-dir", default=".rag_store", help="ChromaDB path")
    q.add_argument("--top-k", type=int, default=8)
    q.add_argument("--retrieve-only", action="store_true", help="Skip LLM synthesis")
    q.add_argument("--llm-provider", default="anthropic", choices=["anthropic", "openai"])
    q.add_argument("--llm-model", default=None)
    q.add_argument(
        "--embedding-provider",
        default="openai",
        choices=["openai", "ollama"],
    )
    q.add_argument("--embedding-model", default=None)
    q.add_argument(
        "--vector-backend",
        default="chromadb",
        choices=["chromadb", "leann"],
        help="Vector store backend",
    )

    # --- status ---
    st = sub.add_parser("status", help="Show index status")
    st.add_argument("--persist-dir", default=".rag_store", help="Store path")
    st.add_argument(
        "--vector-backend",
        default="chromadb",
        choices=["chromadb", "leann"],
        help="Vector store backend",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    from swarm.bridges.rag.config import RAGConfig

    if args.command == "index":
        config = RAGConfig(
            persist_dir=args.persist_dir,
            embedding_provider=args.embedding_provider,
            vector_backend=args.vector_backend,
        )
        if args.embedding_model:
            config.embedding_model = args.embedding_model

        from swarm.bridges.rag.indexer import RunIndexer

        indexer = RunIndexer(config)
        path = args.path.resolve()

        if (path / "history.json").exists() or list(path.glob("*.jsonl")):
            count = indexer.index_run(path)
        else:
            count = indexer.index_all(path)

        print(f"Indexed {count} chunks into {config.persist_dir}")

    elif args.command == "query":
        config = RAGConfig(
            persist_dir=args.persist_dir,
            top_k=args.top_k,
            llm_provider=args.llm_provider,
            embedding_provider=args.embedding_provider,
            vector_backend=args.vector_backend,
        )
        if args.llm_model:
            config.llm_model = args.llm_model
        if args.embedding_model:
            config.embedding_model = args.embedding_model

        from swarm.bridges.rag.retriever import RunRetriever

        retriever = RunRetriever(config)

        if args.retrieve_only:
            chunks = retriever.retrieve(args.question)
            for i, chunk in enumerate(chunks, 1):
                print(f"\n--- Source {i} (run={chunk.run_id}, "
                      f"type={chunk.doc_type}, score={chunk.score:.3f}) ---")
                print(chunk.text)
        else:
            response = retriever.query(args.question)
            print(response.answer)
            print(f"\n--- {len(response.sources)} sources used, "
                  f"model={response.model} ---")
            if response.usage:
                print(f"Tokens: {response.usage}")

    elif args.command == "status":
        _show_status(args)


def _show_status(args: argparse.Namespace) -> None:
    """Show index status."""
    persist_dir = getattr(args, "persist_dir", ".rag_store")
    vector_backend = getattr(args, "vector_backend", "chromadb")
    store_path = Path(persist_dir)

    if not store_path.exists():
        print(f"No RAG store found at {store_path}")
        return

    from swarm.bridges.rag.backend import build_backend
    from swarm.bridges.rag.config import RAGConfig

    config = RAGConfig(persist_dir=persist_dir, vector_backend=vector_backend)
    backend = build_backend(config)
    count = backend.count

    # Approximate store size
    total_bytes = sum(f.stat().st_size for f in store_path.rglob("*") if f.is_file())
    size_mb = total_bytes / (1024 * 1024)

    print(f"RAG store: {store_path.resolve()}")
    print(f"  Backend: {vector_backend}")
    print(f"  Chunks indexed: {count}")
    print(f"  Store size: {size_mb:.1f} MB")
