"""Tests for the RAG bridge."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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

        # Verify metadata via direct chromadb query
        results = indexer._collection.get(include=["metadatas"])
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
