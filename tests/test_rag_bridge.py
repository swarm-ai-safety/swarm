"""Tests for the RAG bridge."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from swarm.bridges.rag.backend import BackendResult, LeannBackend, _apply_where_filter
from swarm.bridges.rag.config import RAGConfig


class TestRAGConfig:
    """Test RAGConfig defaults and validation."""

    def test_defaults(self) -> None:
        config = RAGConfig()
        assert config.persist_dir == ".rag_store"
        assert config.collection_name == "swarm_runs"
        assert config.embedding_provider == "openai"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.top_k == 8
        assert config.llm_provider == "anthropic"
        assert config.vector_backend == "chromadb"
        assert config.leann_backend == "hnsw"
        assert config.leann_index_dir == ".rag_store/leann"

    def test_custom_values(self) -> None:
        config = RAGConfig(
            persist_dir="/tmp/test_rag",
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            top_k=4,
        )
        assert config.persist_dir == "/tmp/test_rag"
        assert config.embedding_provider == "ollama"
        assert config.embedding_model == "nomic-embed-text"
        assert config.top_k == 4

    def test_default_doc_types(self) -> None:
        config = RAGConfig()
        assert "epoch_summary" in config.default_doc_types
        assert "scenario_config" in config.default_doc_types
        assert "agent_state" in config.default_doc_types
        assert "event_summary" in config.default_doc_types


# Only run indexer/retriever tests if RAG deps are available
try:
    import importlib.util

    _HAS_RAG = (
        importlib.util.find_spec("chromadb") is not None
        and importlib.util.find_spec("langchain_text_splitters") is not None
    )
except Exception:
    _HAS_RAG = False

requires_rag = pytest.mark.skipif(not _HAS_RAG, reason="RAG dependencies not installed")


def _make_history_json(run_dir: Path, seed: int = 42) -> None:
    """Create a minimal history.json for testing."""
    data = {
        "simulation_id": "test_sim",
        "started_at": "2026-01-01T00:00:00",
        "ended_at": "2026-01-01T01:00:00",
        "n_epochs": 2,
        "steps_per_epoch": 5,
        "n_agents": 3,
        "seed": seed,
        "epoch_snapshots": [
            {
                "epoch": 0,
                "total_interactions": 10,
                "accepted_interactions": 8,
                "rejected_interactions": 2,
                "toxicity_rate": 0.15,
                "quality_gap": 0.3,
                "avg_p": 0.7,
                "total_welfare": 50.0,
                "avg_payoff": 5.0,
                "payoff_std": 1.0,
                "gini_coefficient": 0.2,
                "n_agents": 3,
                "n_frozen": 0,
                "n_quarantined": 0,
                "avg_reputation": 0.8,
                "reputation_std": 0.1,
                "ecosystem_threat_level": 0.1,
                "ecosystem_collusion_risk": 0.05,
            },
            {
                "epoch": 1,
                "total_interactions": 12,
                "accepted_interactions": 10,
                "rejected_interactions": 2,
                "toxicity_rate": 0.08,
                "quality_gap": 0.4,
                "avg_p": 0.75,
                "total_welfare": 60.0,
                "avg_payoff": 6.0,
                "payoff_std": 0.8,
                "gini_coefficient": 0.18,
                "n_agents": 3,
                "n_frozen": 0,
                "n_quarantined": 0,
                "avg_reputation": 0.85,
                "reputation_std": 0.08,
                "ecosystem_threat_level": 0.05,
                "ecosystem_collusion_risk": 0.03,
            },
        ],
        "agent_snapshots": [
            {
                "agent_id": "agent_0",
                "epoch": 1,
                "name": "Alice",
                "agent_type": "cooperative",
                "reputation": 0.9,
                "resources": 120.0,
                "interactions_initiated": 5,
                "interactions_received": 3,
                "avg_p_initiated": 0.8,
                "avg_p_received": 0.7,
                "total_payoff": 15.0,
                "is_frozen": False,
                "is_quarantined": False,
            },
            {
                "agent_id": "agent_1",
                "epoch": 1,
                "name": "Bob",
                "agent_type": "selfish",
                "reputation": 0.5,
                "resources": 80.0,
                "interactions_initiated": 4,
                "interactions_received": 6,
                "avg_p_initiated": 0.4,
                "avg_p_received": 0.6,
                "total_payoff": 8.0,
                "is_frozen": False,
                "is_quarantined": False,
            },
        ],
    }
    (run_dir / "history.json").write_text(json.dumps(data))


def _make_scenario_yaml(run_dir: Path) -> None:
    """Create a minimal scenario YAML."""
    yaml_content = (
        "scenario_id: test_baseline\n"
        "name: Test Baseline\n"
        "governance:\n"
        "  circuit_breaker: true\n"
        "  toxicity_threshold: 0.2\n"
        "agents:\n"
        "  - type: cooperative\n"
        "    count: 2\n"
        "  - type: selfish\n"
        "    count: 1\n"
    )
    (run_dir / "scenario.yaml").write_text(yaml_content)


def _make_event_log(run_dir: Path) -> None:
    """Create a minimal event log."""
    events = [
        {
            "event_type": "interaction_proposed",
            "epoch": 0,
            "agent_id": "agent_0",
            "payload": {"p": 0.8, "v_hat": 0.6},
        },
        {
            "event_type": "interaction_accepted",
            "epoch": 0,
            "agent_id": "agent_1",
            "payload": {},
        },
        {
            "event_type": "payoff_computed",
            "epoch": 0,
            "agent_id": "agent_0",
            "payload": {"components": {"tau": 0.1}},
        },
    ]
    with open(run_dir / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


@requires_rag
class TestRunIndexer:
    """Test RunIndexer with mock embeddings."""

    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "test_run_seed42"
        d.mkdir()
        _make_history_json(d)
        _make_scenario_yaml(d)
        _make_event_log(d)
        return d

    @pytest.fixture()
    def config(self, tmp_path: Path) -> RAGConfig:
        return RAGConfig(persist_dir=str(tmp_path / "rag_store"))

    def _mock_embedding(self, n: int = 1) -> list[list[float]]:
        """Return deterministic mock embeddings."""
        import hashlib

        results = []
        for i in range(n):
            seed_bytes = hashlib.md5(str(i).encode()).digest()
            vec = [b / 255.0 for b in seed_bytes] + [0.0] * (384 - 16)
            results.append(vec)
        return results

    @patch("swarm.bridges.rag.indexer.RunIndexer._build_embedding_fn")
    def test_index_run(
        self, mock_build: MagicMock, run_dir: Path, config: RAGConfig
    ) -> None:
        mock_fn = MagicMock()
        mock_fn.embed_documents = lambda texts: self._mock_embedding(len(texts))
        mock_build.return_value = mock_fn

        from swarm.bridges.rag.indexer import RunIndexer

        indexer = RunIndexer(config)
        indexer._embedding_fn = mock_fn

        count = indexer.index_run(run_dir)
        assert count > 0
        assert indexer.count == count

    @patch("swarm.bridges.rag.indexer.RunIndexer._build_embedding_fn")
    def test_index_all(
        self, mock_build: MagicMock, tmp_path: Path, config: RAGConfig
    ) -> None:
        mock_fn = MagicMock()
        mock_fn.embed_documents = lambda texts: self._mock_embedding(len(texts))
        mock_build.return_value = mock_fn

        # Create two run dirs
        for seed in [1, 2]:
            d = tmp_path / f"run_seed{seed}"
            d.mkdir()
            _make_history_json(d, seed=seed)

        from swarm.bridges.rag.indexer import RunIndexer

        indexer = RunIndexer(config)
        indexer._embedding_fn = mock_fn

        # index_all should skip the rag_store dir (starts with .)
        # but index the two run dirs
        total = indexer.index_all(tmp_path)
        assert total > 0

    @patch("swarm.bridges.rag.indexer.RunIndexer._build_embedding_fn")
    def test_metadata_on_chunks(
        self, mock_build: MagicMock, run_dir: Path, config: RAGConfig
    ) -> None:
        mock_fn = MagicMock()
        mock_fn.embed_documents = lambda texts: self._mock_embedding(len(texts))
        mock_build.return_value = mock_fn

        from swarm.bridges.rag.indexer import RunIndexer

        indexer = RunIndexer(config)
        indexer._embedding_fn = mock_fn
        indexer.index_run(run_dir)

        # Verify metadata via direct chromadb query on the backend's collection
        results = indexer._backend._collection.get(include=["metadatas"])
        metadatas = results["metadatas"]
        assert metadatas is not None
        assert len(metadatas) > 0

        doc_types = {m["doc_type"] for m in metadatas if m}
        assert "epoch_summary" in doc_types
        assert "agent_state" in doc_types


@requires_rag
class TestRunRetriever:
    """Test RunRetriever with mock embeddings and LLM."""

    @pytest.fixture()
    def populated_store(self, tmp_path: Path) -> RAGConfig:
        """Create and populate a test store."""
        config = RAGConfig(persist_dir=str(tmp_path / "rag_store"))

        mock_fn = MagicMock()
        import hashlib

        def _mock_embed(texts: list[str]) -> list[list[float]]:
            results = []
            for t in texts:
                seed_bytes = hashlib.md5(t.encode()).digest()
                vec = [b / 255.0 for b in seed_bytes] + [0.0] * (384 - 16)
                results.append(vec)
            return results

        mock_fn.embed_documents = _mock_embed
        mock_fn.embed_query = lambda t: _mock_embed([t])[0]

        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        _make_history_json(run_dir)
        _make_scenario_yaml(run_dir)

        with patch("swarm.bridges.rag.indexer.RunIndexer._build_embedding_fn") as mock_build:
            mock_build.return_value = mock_fn
            from swarm.bridges.rag.indexer import RunIndexer

            indexer = RunIndexer(config)
            indexer._embedding_fn = mock_fn
            indexer.index_run(run_dir)

        # Stash mock for retriever use
        config._test_mock_fn = mock_fn  # type: ignore[attr-defined]
        return config

    @patch("swarm.bridges.rag.retriever.RunRetriever._build_embedding_fn")
    def test_retrieve(
        self, mock_build: MagicMock, populated_store: RAGConfig
    ) -> None:
        mock_build.return_value = populated_store._test_mock_fn  # type: ignore[attr-defined]

        from swarm.bridges.rag.retriever import RunRetriever

        retriever = RunRetriever(populated_store)
        retriever._embedding_fn = populated_store._test_mock_fn  # type: ignore[attr-defined]

        chunks = retriever.retrieve("toxicity rate")
        assert len(chunks) > 0
        assert all(hasattr(c, "run_id") for c in chunks)
        assert all(hasattr(c, "score") for c in chunks)

    @patch("swarm.bridges.rag.retriever.RunRetriever._call_llm")
    @patch("swarm.bridges.rag.retriever.RunRetriever._build_embedding_fn")
    def test_query_with_mock_llm(
        self,
        mock_build: MagicMock,
        mock_llm: MagicMock,
        populated_store: RAGConfig,
    ) -> None:
        mock_build.return_value = populated_store._test_mock_fn  # type: ignore[attr-defined]
        mock_llm.return_value = (
            "The toxicity rate decreased from 0.15 to 0.08 across epochs.",
            {"input_tokens": 100, "output_tokens": 20},
        )

        from swarm.bridges.rag.retriever import RunRetriever

        retriever = RunRetriever(populated_store)
        retriever._embedding_fn = populated_store._test_mock_fn  # type: ignore[attr-defined]

        response = retriever.query("What was the toxicity trend?")
        assert "toxicity" in response.answer.lower()
        assert len(response.sources) > 0
        assert response.model == populated_store.llm_model
        assert response.usage.get("input_tokens") == 100

    @patch("swarm.bridges.rag.retriever.RunRetriever._build_embedding_fn")
    def test_retrieve_with_doc_type_filter(
        self, mock_build: MagicMock, populated_store: RAGConfig
    ) -> None:
        mock_build.return_value = populated_store._test_mock_fn  # type: ignore[attr-defined]

        from swarm.bridges.rag.retriever import RunRetriever

        retriever = RunRetriever(populated_store)
        retriever._embedding_fn = populated_store._test_mock_fn  # type: ignore[attr-defined]

        chunks = retriever.retrieve("agent", doc_types=["agent_state"])
        # All returned chunks should be agent_state type
        for chunk in chunks:
            assert chunk.doc_type == "agent_state"


# ------------------------------------------------------------------
# LEANN backend tests (mock LEANN builder/searcher — no real dep)
# ------------------------------------------------------------------


@dataclass
class _MockHit:
    """Mimics a LEANN search hit."""

    id: str
    score: float


class MockLeannBuilder:
    """In-memory fake for leann.LeannBuilder."""

    def __init__(self, backend: str = "hnsw") -> None:
        self.texts: list[str] = []
        self.ids: list[str] = []
        self.backend = backend
        self._built_dir: str | None = None

    def add_texts(self, texts: list[str], ids: list[str] | None = None) -> None:
        self.texts.extend(texts)
        if ids:
            self.ids.extend(ids)
        else:
            self.ids.extend(str(i) for i in range(len(texts)))

    def build_index(self, path: str) -> None:
        self._built_dir = path


class MockLeannSearcher:
    """In-memory fake for leann.LeannSearcher.

    Returns results in insertion order with decreasing scores.
    Uses the sidecar's ids to simulate realistic search results.
    """

    def __init__(self, index_dir: str, sidecar_ids: list[str] | None = None) -> None:
        self.index_dir = index_dir
        self._ids = sidecar_ids or []

    def search(self, query_text: str, k: int = 8) -> list[_MockHit]:
        results = []
        for i, doc_id in enumerate(self._ids[:k]):
            results.append(_MockHit(id=doc_id, score=1.0 - i * 0.1))
        return results


class TestLeannBackend:
    """Test LeannBackend with mocked LEANN builder/searcher."""

    @pytest.fixture(autouse=True)
    def _mock_leann_module(self) -> Any:
        """Inject a fake ``leann`` module into sys.modules."""
        mod = ModuleType("leann")
        mod.LeannBuilder = MockLeannBuilder  # type: ignore[attr-defined]
        mod.LeannSearcher = MockLeannSearcher  # type: ignore[attr-defined]
        old = sys.modules.get("leann")
        sys.modules["leann"] = mod
        yield
        if old is None:
            sys.modules.pop("leann", None)
        else:
            sys.modules["leann"] = old

    @pytest.fixture()
    def config(self, tmp_path: Path) -> RAGConfig:
        return RAGConfig(
            vector_backend="leann",
            leann_index_dir=str(tmp_path / "leann_idx"),
        )

    def _make_backend_with_data(
        self, config: RAGConfig
    ) -> LeannBackend:
        """Create a LeannBackend and upsert test data."""
        backend = LeannBackend(config)
        ids = ["chunk_0", "chunk_1", "chunk_2"]
        texts = ["hello world", "toxicity rate", "agent state"]
        metadatas = [
            {"doc_type": "epoch_summary", "run_id": "run1"},
            {"doc_type": "epoch_summary", "run_id": "run1"},
            {"doc_type": "agent_state", "run_id": "run1"},
        ]
        backend.upsert(ids=ids, texts=texts, metadatas=metadatas)
        return backend

    def test_upsert_and_finalize(self, config: RAGConfig) -> None:
        backend = self._make_backend_with_data(config)
        backend.finalize()
        assert backend.count == 3

    def test_sidecar_persistence(self, config: RAGConfig) -> None:
        backend = self._make_backend_with_data(config)
        backend.finalize()

        sidecar_path = Path(config.leann_index_dir) / "metadata.json"
        assert sidecar_path.exists()

        sidecar = json.loads(sidecar_path.read_text())
        assert "chunk_0" in sidecar
        assert sidecar["chunk_0"]["text"] == "hello world"
        assert sidecar["chunk_0"]["metadata"]["doc_type"] == "epoch_summary"

    def test_query_returns_results(self, config: RAGConfig) -> None:
        backend = self._make_backend_with_data(config)
        backend.finalize()

        # Inject mock searcher
        mock_searcher = MockLeannSearcher(
            config.leann_index_dir,
            sidecar_ids=list(backend._sidecar.keys()),
        )
        backend._searcher = mock_searcher

        results = backend.query(query_text="toxicity", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, BackendResult) for r in results)
        assert results[0].score > results[1].score

    def test_post_retrieval_filtering(self, config: RAGConfig) -> None:
        backend = self._make_backend_with_data(config)
        backend.finalize()

        # Mock searcher returns all 3 items
        mock_searcher = MockLeannSearcher(
            config.leann_index_dir,
            sidecar_ids=list(backend._sidecar.keys()),
        )
        backend._searcher = mock_searcher

        # Filter to agent_state only
        results = backend.query(
            query_text="agent",
            top_k=10,
            where={"doc_type": "agent_state"},
        )
        assert len(results) == 1
        assert results[0].metadata["doc_type"] == "agent_state"

    def test_overfetch_for_filtered_queries(self, config: RAGConfig) -> None:
        """When a where filter is given, LEANN fetches 3x top_k."""
        backend = self._make_backend_with_data(config)
        backend.finalize()

        call_log: list[int] = []

        class TrackingSearcher:
            def __init__(self, ids: list[str]) -> None:
                self._ids = ids

            def search(self, query_text: str, k: int = 8) -> list[_MockHit]:
                call_log.append(k)
                return [_MockHit(id=i, score=0.9) for i in self._ids[:k]]

        backend._searcher = TrackingSearcher(list(backend._sidecar.keys()))

        backend.query(query_text="test", top_k=2, where={"doc_type": "epoch_summary"})
        assert call_log[0] == 6  # 2 * 3 = 6

    def test_finalize_no_op_when_empty(self, config: RAGConfig) -> None:
        backend = LeannBackend(config)
        backend.finalize()  # Should not raise
        assert backend.count == 0


class TestApplyWhereFilter:
    """Test the post-retrieval filter helper."""

    def test_equality(self) -> None:
        assert _apply_where_filter({"a": 1}, {"a": 1}) is True
        assert _apply_where_filter({"a": 1}, {"a": 2}) is False

    def test_in_operator(self) -> None:
        assert _apply_where_filter({"t": "x"}, {"t": {"$in": ["x", "y"]}}) is True
        assert _apply_where_filter({"t": "z"}, {"t": {"$in": ["x", "y"]}}) is False

    def test_ne_operator(self) -> None:
        assert _apply_where_filter({"a": 1}, {"a": {"$ne": 2}}) is True
        assert _apply_where_filter({"a": 1}, {"a": {"$ne": 1}}) is False

    def test_missing_key(self) -> None:
        assert _apply_where_filter({}, {"a": 1}) is False
