#!/usr/bin/env python
"""Demo: RAG over SWARM run history.

Usage::

    # First, generate some runs:
    python -m swarm run scenarios/baseline.yaml --seed 42 --epochs 10 --steps 10

    # Then run this demo:
    python examples/rag_demo.py runs/
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python examples/rag_demo.py <runs_dir> [question]")
        sys.exit(1)

    runs_dir = Path(sys.argv[1])
    question = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "Which runs had the lowest toxicity rates and what governance settings did they use?"
    )

    from swarm.bridges.rag import RAGConfig, RunIndexer, RunRetriever

    config = RAGConfig()

    # Index
    print(f"Indexing runs from {runs_dir}...")
    indexer = RunIndexer(config)
    count = indexer.index_all(runs_dir)
    print(f"  Indexed {count} chunks.\n")

    # Query
    print(f"Question: {question}\n")
    retriever = RunRetriever(config)
    response = retriever.query(question)
    print(response.answer)
    print(f"\n--- {len(response.sources)} sources, model={response.model} ---")
    for i, src in enumerate(response.sources, 1):
        print(f"  [{i}] run={src.run_id} type={src.doc_type} score={src.score:.3f}")


if __name__ == "__main__":
    main()
