"""Tests for the Aeon agent-first ledger bridge.

Covers config validation, ledger reading (incl. malformed lines), record ->
SoftInteraction mapping, heuristic scoring, metrics computation, dedup on
re-ingest, and the oneshot runner against on-disk fixtures.
"""

import json
from pathlib import Path

import pytest

from swarm.bridges.aeon.client import AeonClient
from swarm.bridges.aeon.config import AeonConfig
from swarm.bridges.aeon.mapper import AeonMapper
from swarm.bridges.aeon.metrics import AeonMetrics, AeonMetricsReport
from swarm.bridges.aeon.runner import AeonRunner
from swarm.models.interaction import InteractionType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TASK = {
    "id": "t1",
    "repo": "owner/aeon",
    "title": "Audit workflow permissions",
    "assignedTo": "did:key:zAssignee",
    "requiredAbilities": ["repo/read", "pr/open"],
    "status": "completed",
    "createdAt": "2026-05-27T12:00:00Z",
}

PROOF = {
    "taskId": "t1",
    "agent": "did:key:zAssignee",
    "capability": "pr/open",
    "baseCommit": "aaaaaaaaaaaa",
    "headCommit": "bbbbbbbbbbbb",
    "commands": ["pytest", "git commit"],
    "artifacts": ["pr#42"],
    "decision": "merged",
    "summary": "Scoped permissions to least privilege.",
    "createdAt": "2026-05-27T12:30:00Z",
}

REVIEW = {
    "reviewer": "did:key:zReviewer",
    "target": "t1",
    "verdict": "approve",
    "scope": {"proposalHash": "deadbeefcafe", "paths": ["aeon.yml"]},
    "expiresAt": "2026-05-27T13:00:00Z",
    "findings": [],
}


@pytest.fixture
def ledger_dir(tmp_path: Path) -> Path:
    d = tmp_path / "agent-first"
    d.mkdir()
    (d / "tasks.jsonl").write_text(json.dumps(TASK) + "\n")
    (d / "proofs.jsonl").write_text(json.dumps(PROOF) + "\n")
    (d / "reviews.jsonl").write_text(json.dumps(REVIEW) + "\n")
    return d


@pytest.fixture
def config(ledger_dir: Path) -> AeonConfig:
    return AeonConfig(ledger_dir=str(ledger_dir))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_valid(self, config: AeonConfig):
        config.validate()  # no raise

    def test_paths(self, config: AeonConfig, ledger_dir: Path):
        assert config.tasks_path == ledger_dir / "tasks.jsonl"
        assert config.proofs_path == ledger_dir / "proofs.jsonl"
        assert config.reviews_path == ledger_dir / "reviews.jsonl"

    def test_rejects_empty_ledger_dir(self):
        with pytest.raises(ValueError):
            AeonConfig(ledger_dir="").validate()

    def test_rejects_nonpositive_interval(self):
        with pytest.raises(ValueError):
            AeonConfig(poll_interval_sec=0).validate()

    def test_from_toml_preserves_repos(self, tmp_path: Path):
        # repos has a default_factory, so a hasattr filter would drop it.
        toml = tmp_path / "cfg.toml"
        toml.write_text('ledger_dir = "x"\nrepos = ["owner/aeon", "owner/other"]\n')
        cfg = AeonConfig.from_toml(toml)
        assert cfg.repos == ["owner/aeon", "owner/other"]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class TestClient:
    def test_reads_all_ledgers(self, config: AeonConfig):
        client = AeonClient(config)
        assert len(client.fetch_tasks()) == 1
        assert len(client.fetch_proofs()) == 1
        assert len(client.fetch_reviews()) == 1

    def test_repo_filter(self, ledger_dir: Path):
        client = AeonClient(AeonConfig(ledger_dir=str(ledger_dir), repos=["other/repo"]))
        assert client.fetch_tasks() == []

    def test_missing_files_return_empty(self, tmp_path: Path):
        client = AeonClient(AeonConfig(ledger_dir=str(tmp_path / "nope")))
        assert client.fetch_tasks() == []
        assert client.fetch_proofs() == []
        assert client.fetch_reviews() == []

    def test_skips_malformed_lines(self, ledger_dir: Path):
        (ledger_dir / "tasks.jsonl").write_text(
            json.dumps(TASK) + "\n" + "{not json\n" + "\n"
        )
        client = AeonClient(AeonConfig(ledger_dir=str(ledger_dir)))
        assert len(client.fetch_tasks()) == 1

    def test_skill_runs_disabled_by_default(self, config: AeonConfig):
        assert AeonClient(config).fetch_skill_runs() == []


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------

class TestMapper:
    def test_map_task(self, config: AeonConfig):
        i = AeonMapper(config).map_task(TASK)
        assert i.interaction_type == InteractionType.COLLABORATION
        assert i.counterparty == "did:key:zAssignee"
        assert i.accepted is True
        assert i.metadata["event_source"] == "aeon_task"

    def test_map_proof_resolves_repo_from_task(self, config: AeonConfig):
        i = AeonMapper(config).map_proof(PROOF, TASK)
        assert i.interaction_type == InteractionType.REPLY
        assert i.initiator == "did:key:zAssignee"
        assert i.counterparty == "aeon:repo:owner/aeon"
        assert i.accepted is True  # merged
        assert i.metadata["command_count"] == 2

    def test_map_proof_without_task_targets_queue(self, config: AeonConfig):
        i = AeonMapper(config).map_proof(PROOF, None)
        assert i.counterparty == "aeon:review-queue"

    def test_map_review(self, config: AeonConfig):
        i = AeonMapper(config).map_review(REVIEW)
        assert i.interaction_type == InteractionType.VOTE
        assert i.initiator == "did:key:zReviewer"
        assert i.accepted is True

    def test_map_review_id_distinguishes_verdicts(self, config: AeonConfig):
        mapper = AeonMapper(config)
        approve = mapper.map_review(REVIEW)
        changed = mapper.map_review({**REVIEW, "verdict": "request_changes"})
        # A re-review of the same proposal must not collide / be deduped.
        assert approve.interaction_id != changed.interaction_id

    def test_map_review_uses_expiresat_timestamp(self, config: AeonConfig):
        i = AeonMapper(config).map_review(REVIEW)
        assert i.timestamp.year == 2026 and i.timestamp.hour == 13

    def test_map_review_repo_from_task(self, config: AeonConfig):
        i = AeonMapper(config).map_review(REVIEW, TASK)
        assert i.metadata["repo"] == "owner/aeon"

    def test_map_skill_run(self, config: AeonConfig):
        run = {
            "databaseId": 99,
            "workflowName": "skill-health",
            "conclusion": "failure",
            "createdAt": "2026-05-27T10:00:00Z",
        }
        i = AeonMapper(config).map_skill_run(run)
        assert i.accepted is False
        assert i.initiator == "aeon:skill:skill-health"

    @pytest.mark.asyncio
    async def test_heuristic_scoring(self, config: AeonConfig):
        mapper = AeonMapper(config)
        merged = await mapper.enrich(mapper.map_proof(PROOF, TASK))
        assert merged.p == pytest.approx(0.85)

        failed_proof = {**PROOF, "decision": "failed"}
        failed = await mapper.enrich(mapper.map_proof(failed_proof, TASK))
        assert failed.p == pytest.approx(0.15)

    def test_z_timestamp_parsed(self, config: AeonConfig):
        i = AeonMapper(config).map_task(TASK)
        assert i.timestamp.year == 2026 and i.timestamp.month == 5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_empty(self, config: AeonConfig):
        report = AeonMetrics(config).compute([])
        assert report.interaction_count == 0
        assert report.source == "aeon"

    def test_compute_and_serialize(self, config: AeonConfig):
        mapper = AeonMapper(config)
        interactions = [
            mapper.map_task(TASK),
            mapper.map_proof(PROOF, TASK),
            mapper.map_review(REVIEW),
        ]
        report = AeonMetrics(config).compute(interactions)
        assert report.interaction_count == 3
        d = report.to_dict()
        assert set(d) >= {"toxicity_rate", "quality_gap", "average_quality", "welfare"}
        assert len(report.per_interaction_type) == 3  # collaboration, reply, vote


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TestRunner:
    @pytest.mark.asyncio
    async def test_oneshot(self, config: AeonConfig):
        report = await AeonRunner(config).run_oneshot()
        assert isinstance(report, AeonMetricsReport)
        assert report.interaction_count == 3

    @pytest.mark.asyncio
    async def test_reingest_dedupes(self, config: AeonConfig):
        runner = AeonRunner(config)
        assert await runner._ingest() == 3
        assert await runner._ingest() == 0  # nothing new

    @pytest.mark.asyncio
    async def test_repo_scope_drops_offrepo_proofs_and_reviews(self, ledger_dir: Path):
        # Task t1 is owner/aeon; restrict to a different repo -> nothing in scope,
        # and its proof/review must be dropped rather than leaking into metrics.
        cfg = AeonConfig(ledger_dir=str(ledger_dir), repos=["other/repo"])
        report = await AeonRunner(cfg).run_oneshot()
        assert report.interaction_count == 0

    @pytest.mark.asyncio
    async def test_repo_scope_keeps_inrepo_records(self, ledger_dir: Path):
        cfg = AeonConfig(ledger_dir=str(ledger_dir), repos=["owner/aeon"])
        report = await AeonRunner(cfg).run_oneshot()
        assert report.interaction_count == 3

    @pytest.mark.asyncio
    async def test_persistence(self, config: AeonConfig, tmp_path: Path):
        out = tmp_path / "out" / "aeon.jsonl"
        config.persistence_path = str(out)
        await AeonRunner(config).run_oneshot()
        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert len(lines) == 3
        assert out.with_suffix(".metrics.json").exists()
