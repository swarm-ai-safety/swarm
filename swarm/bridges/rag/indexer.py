"""Run indexer — loads run artifacts and indexes them into a vector backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from swarm.bridges.rag.config import RAGConfig

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

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


class RunIndexer:
    """Indexes run artifacts into a vector backend.

    Each chunk is stored with rich metadata (run_id, scenario_id, seed,
    epoch, doc_type) so retrieval can be filtered.
    """

    def __init__(self, config: RAGConfig | None = None) -> None:
        _require_rag()
        self.config = config or RAGConfig()

        from swarm.bridges.rag.backend import build_backend

        self._backend = build_backend(self.config)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        # Embedding fn only needed for chromadb (LEANN handles its own)
        self._embedding_fn: Any = None
        if self.config.vector_backend == "chromadb":
            self._embedding_fn = self._build_embedding_fn()

    # ------------------------------------------------------------------
    # Embedding setup
    # ------------------------------------------------------------------

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

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        result: list[list[float]] = self._embedding_fn.embed_documents(texts)
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_run(self, run_dir: Path) -> int:
        """Index a single run directory.

        Args:
            run_dir: Path to a run directory containing history.json,
                     scenario YAML, and/or event logs.

        Returns:
            Number of chunks indexed.
        """
        count = self._index_run_no_finalize(run_dir)
        self._backend.finalize()
        return count

    def _index_run_no_finalize(self, run_dir: Path) -> int:
        """Index a run without calling finalize (for batch use)."""
        run_dir = Path(run_dir)
        run_id = run_dir.name
        chunks: list[dict[str, Any]] = []

        # 1. History JSON (epoch summaries + agent states)
        history_path = run_dir / "history.json"
        if history_path.exists():
            chunks.extend(self._index_history(history_path, run_id))

        # 2. Scenario YAML
        for yaml_path in run_dir.glob("*.yaml"):
            chunks.extend(self._index_scenario(yaml_path, run_id))
        for yaml_path in run_dir.glob("*.yml"):
            chunks.extend(self._index_scenario(yaml_path, run_id))

        # 3. Event log
        for jsonl_path in run_dir.glob("*.jsonl"):
            chunks.extend(self._index_events(jsonl_path, run_id))

        # 4. CSV metrics (brief summary only)
        csv_dir = run_dir / "csv"
        if csv_dir.exists():
            for csv_path in csv_dir.glob("*_epochs.csv"):
                chunks.extend(self._index_csv_summary(csv_path, run_id))

        if not chunks:
            logger.warning("No indexable artifacts found in %s", run_dir)
            return 0

        # Batch embed and upsert
        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        embeddings: list[list[float]] | None = None
        if self._embedding_fn is not None:
            embeddings = self._embed(texts)

        self._backend.upsert(
            ids=ids,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info("Indexed %d chunks from %s", len(chunks), run_dir)
        return len(chunks)

    def index_all(self, runs_dir: Path) -> int:
        """Discover and index all runs in a directory.

        Accumulates all chunks across runs, then calls ``finalize()``
        once at the end.  This is more efficient for backends like LEANN
        that rebuild the entire index on finalize.

        Args:
            runs_dir: Parent directory containing run subdirectories.

        Returns:
            Total number of chunks indexed.
        """
        runs_dir = Path(runs_dir)
        total = 0
        for child in sorted(runs_dir.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                total += self._index_run_no_finalize(child)
        self._backend.finalize()
        return total

    @property
    def count(self) -> int:
        """Number of chunks in the collection."""
        return self._backend.count

    # ------------------------------------------------------------------
    # Internal indexing helpers
    # ------------------------------------------------------------------

    def _index_history(
        self, history_path: Path, run_id: str
    ) -> list[dict[str, Any]]:
        """Index epoch summaries and agent final states from history.json."""
        from swarm.analysis.export import load_from_json

        chunks: list[dict[str, Any]] = []

        try:
            history = load_from_json(history_path)
        except Exception:
            logger.warning("Failed to load %s, falling back to raw JSON", history_path)
            return self._index_raw_json(history_path, run_id)

        scenario_id = history.simulation_id or run_id
        seed = history.seed

        # Epoch summaries
        for snap in history.epoch_snapshots:
            text = (
                f"Epoch {snap.epoch} of run {run_id} "
                f"(scenario={scenario_id}, seed={seed}):\n"
                f"  interactions: {snap.total_interactions} "
                f"(accepted={snap.accepted_interactions}, "
                f"rejected={snap.rejected_interactions})\n"
                f"  toxicity_rate={snap.toxicity_rate:.4f}, "
                f"quality_gap={snap.quality_gap:.4f}\n"
                f"  avg_p={snap.avg_p:.4f}, avg_payoff={snap.avg_payoff:.4f}\n"
                f"  total_welfare={snap.total_welfare:.2f}, "
                f"gini={snap.gini_coefficient:.4f}\n"
                f"  agents: {snap.n_agents} total, "
                f"{snap.n_frozen} frozen, {snap.n_quarantined} quarantined\n"
                f"  avg_reputation={snap.avg_reputation:.4f}\n"
                f"  threat_level={snap.ecosystem_threat_level:.4f}, "
                f"collusion_risk={snap.ecosystem_collusion_risk:.4f}"
            )
            chunks.append(
                {
                    "id": f"{run_id}__epoch_{snap.epoch}",
                    "text": text,
                    "metadata": {
                        "run_id": run_id,
                        "scenario_id": scenario_id,
                        "seed": str(seed) if seed is not None else "",
                        "epoch": snap.epoch,
                        "doc_type": "epoch_summary",
                        "toxicity_rate": snap.toxicity_rate,
                        "quality_gap": snap.quality_gap,
                        "avg_payoff": snap.avg_payoff,
                    },
                }
            )

        # Agent final states
        final_states = history.get_final_agent_states()
        for agent_id, state in final_states.items():
            status = (
                "frozen"
                if state.is_frozen
                else ("quarantined" if state.is_quarantined else "active")
            )
            text = (
                f"Agent {agent_id} final state in run {run_id} "
                f"(scenario={scenario_id}):\n"
                f"  type={state.agent_type}, status={status}\n"
                f"  reputation={state.reputation:.4f}, "
                f"total_payoff={state.total_payoff:.4f}\n"
                f"  interactions: initiated={state.interactions_initiated}, "
                f"received={state.interactions_received}\n"
                f"  avg_p: initiated={state.avg_p_initiated:.4f}, "
                f"received={state.avg_p_received:.4f}"
            )
            chunks.append(
                {
                    "id": f"{run_id}__agent_{agent_id}",
                    "text": text,
                    "metadata": {
                        "run_id": run_id,
                        "scenario_id": scenario_id,
                        "seed": str(seed) if seed is not None else "",
                        "doc_type": "agent_state",
                        "agent_id": agent_id,
                        "agent_type": state.agent_type or "",
                        "reputation": state.reputation,
                    },
                }
            )

        return chunks

    def _index_raw_json(
        self, json_path: Path, run_id: str
    ) -> list[dict[str, Any]]:
        """Fallback: index raw JSON as text chunks."""
        text = json_path.read_text()
        splits = self._splitter.split_text(text)
        return [
            {
                "id": f"{run_id}__json_chunk_{i}",
                "text": chunk,
                "metadata": {
                    "run_id": run_id,
                    "scenario_id": run_id,
                    "doc_type": "epoch_summary",
                    "source_file": json_path.name,
                },
            }
            for i, chunk in enumerate(splits)
        ]

    def _index_scenario(
        self, yaml_path: Path, run_id: str
    ) -> list[dict[str, Any]]:
        """Index a scenario YAML config."""
        import yaml

        text = yaml_path.read_text()
        try:
            data = yaml.safe_load(text)
        except Exception:
            data = {}

        scenario_id = data.get("scenario_id", data.get("name", yaml_path.stem))

        # Extract governance settings for metadata
        governance = data.get("governance", {})
        gov_summary = ", ".join(f"{k}={v}" for k, v in governance.items()) if governance else "none"

        doc_text = (
            f"Scenario configuration '{scenario_id}' from {yaml_path.name}:\n"
            f"Governance settings: {gov_summary}\n\n"
            f"{text}"
        )

        splits = self._splitter.split_text(doc_text)
        return [
            {
                "id": f"{run_id}__scenario_{yaml_path.stem}_{i}",
                "text": chunk,
                "metadata": {
                    "run_id": run_id,
                    "scenario_id": str(scenario_id),
                    "doc_type": "scenario_config",
                    "source_file": yaml_path.name,
                },
            }
            for i, chunk in enumerate(splits)
        ]

    def _index_events(
        self, jsonl_path: Path, run_id: str
    ) -> list[dict[str, Any]]:
        """Index event log by sampling and summarizing events."""
        chunks: list[dict[str, Any]] = []

        # Read events and group by epoch
        epoch_events: dict[int, list[dict[str, Any]]] = {}
        line_count = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                epoch = event.get("epoch", 0) or 0
                epoch_events.setdefault(epoch, []).append(event)

        if line_count == 0:
            return chunks

        # Summarize each epoch's events
        for epoch, events in sorted(epoch_events.items()):
            type_counts: dict[str, int] = {}
            for e in events:
                etype = e.get("event_type", "unknown")
                type_counts[etype] = type_counts.get(etype, 0) + 1

            summary_lines = [
                f"Event summary for epoch {epoch} of run {run_id} "
                f"({len(events)} events from {jsonl_path.name}):"
            ]
            for etype, count in sorted(type_counts.items()):
                summary_lines.append(f"  {etype}: {count}")

            # Include a few sample events for context
            samples = events[:3]
            if samples:
                summary_lines.append("Sample events:")
                for s in samples:
                    summary_lines.append(
                        f"  - {s.get('event_type', '?')}: "
                        f"agent={s.get('agent_id', '?')}, "
                        f"payload_keys={list(s.get('payload', {}).keys())}"
                    )

            text = "\n".join(summary_lines)
            chunks.append(
                {
                    "id": f"{run_id}__events_epoch_{epoch}",
                    "text": text,
                    "metadata": {
                        "run_id": run_id,
                        "scenario_id": run_id,
                        "epoch": epoch,
                        "doc_type": "event_summary",
                        "event_count": len(events),
                        "source_file": jsonl_path.name,
                    },
                }
            )

        return chunks

    def _index_csv_summary(
        self, csv_path: Path, run_id: str
    ) -> list[dict[str, Any]]:
        """Index a brief summary of a CSV metrics file."""
        import csv as csv_mod

        rows: list[dict[str, str]] = []
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                rows.append(dict(row))

        if not rows:
            return []

        # Summarize first and last rows
        first = rows[0]
        last = rows[-1]
        text = (
            f"CSV metrics summary for run {run_id} ({csv_path.name}, "
            f"{len(rows)} rows):\n"
            f"First epoch: {json.dumps(first, default=str)}\n"
            f"Last epoch: {json.dumps(last, default=str)}"
        )

        return [
            {
                "id": f"{run_id}__csv_{csv_path.stem}",
                "text": text,
                "metadata": {
                    "run_id": run_id,
                    "scenario_id": run_id,
                    "doc_type": "epoch_summary",
                    "source_file": csv_path.name,
                },
            }
        ]
