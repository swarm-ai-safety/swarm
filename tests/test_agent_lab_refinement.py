"""Tests for the AgentLab study refinement pipeline.

Uses dict fixtures and monkeypatched runner — no AgentLab required.
"""

import json
from pathlib import Path

import pytest

from swarm.bridges.agent_lab.bridge import AgentLabBridge
from swarm.bridges.agent_lab.config import AgentLabConfig
from swarm.bridges.agent_lab.refinement import (
    RefinementConfig,
    RefinementResult,
    StudyContext,
)
from swarm.bridges.agent_lab.runner import AgentLabRunner
from swarm.models.interaction import SoftInteraction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def study_dir(tmp_path: Path) -> Path:
    """Create a minimal study run directory with summary.json."""
    summary = {
        "scenario": "baseline_governance",
        "title": "Baseline Governance Study",
        "description": "Tests tax rate impact on welfare",
        "seed": 42,
        "epochs": 10,
        "metrics": {
            "mean_welfare": 0.72,
            "toxicity": 0.05,
            "quality_gap": 0.15,
        },
        "findings": [
            {
                "description": "Tax 0% vs 15%: welfare improvement",
                "effect_size": "1.41",
                "p_value": "6.98e-5",
            },
            "Higher tax rates reduce adverse selection",
        ],
        "swept_params": {
            "tax_rate": [0.0, 0.05, 0.10, 0.15],
            "rho": [0.0, 0.5, 1.0],
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))

    # Add sweep results CSV
    sweep_path = tmp_path / "sweep_results.csv"
    sweep_path.write_text(
        "tax_rate,rho,welfare,toxicity\n"
        "0.0,0.0,0.55,0.12\n"
        "0.0,0.5,0.60,0.09\n"
        "0.15,1.0,0.82,0.02\n"
    )

    return tmp_path


@pytest.fixture()
def study_dir_minimal(tmp_path: Path) -> Path:
    """Create a minimal study directory with only summary.json."""
    summary = {"scenario": "test", "title": "Test Study"}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    return tmp_path


# ---------------------------------------------------------------------------
# StudyContext tests
# ---------------------------------------------------------------------------


class TestStudyContext:
    def test_from_run_dir_loads_summary(self, study_dir: Path) -> None:
        ctx = StudyContext.from_run_dir(str(study_dir))
        assert ctx.summary["scenario"] == "baseline_governance"
        assert ctx.summary["seed"] == 42

    def test_from_run_dir_loads_sweep(self, study_dir: Path) -> None:
        ctx = StudyContext.from_run_dir(str(study_dir))
        assert len(ctx.sweep_rows) == 3
        assert ctx.sweep_rows[0]["tax_rate"] == "0.0"

    def test_from_run_dir_missing_summary_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="summary.json"):
            StudyContext.from_run_dir(str(tmp_path))

    def test_from_run_dir_no_sweep_ok(self, study_dir_minimal: Path) -> None:
        ctx = StudyContext.from_run_dir(str(study_dir_minimal))
        assert ctx.sweep_rows == []

    def test_to_research_topic_format(self, study_dir: Path) -> None:
        ctx = StudyContext.from_run_dir(str(study_dir))
        topic = ctx.to_research_topic()
        assert "Baseline Governance Study" in topic
        assert "baseline_governance" in topic
        assert "mean_welfare" in topic
        assert "follow-up" in topic.lower() or "gaps" in topic.lower()

    def test_to_notes_content(self, study_dir: Path) -> None:
        ctx = StudyContext.from_run_dir(str(study_dir))
        notes = ctx.to_notes()
        assert len(notes) > 0
        # Should include sweep info
        assert any("3 parameter combinations" in n for n in notes)
        # Should include findings
        assert any("Tax 0% vs 15%" in n for n in notes)
        # Should include swept params
        assert any("Swept parameters" in n for n in notes)

    def test_to_notes_with_string_findings(self, study_dir: Path) -> None:
        ctx = StudyContext.from_run_dir(str(study_dir))
        notes = ctx.to_notes()
        assert any("adverse selection" in n for n in notes)


# ---------------------------------------------------------------------------
# RefinementConfig tests
# ---------------------------------------------------------------------------


class TestRefinementConfig:
    def test_lite_depth_settings(self) -> None:
        config = RefinementConfig(depth="lite")
        ctx = StudyContext(run_dir="/tmp/test", summary={"scenario": "test"})
        yaml_dict = config.to_agent_lab_yaml(ctx)
        assert yaml_dict["mlesolver_max_steps"] == 1
        assert yaml_dict["papersolver_max_steps"] == 1

    def test_full_depth_settings(self) -> None:
        config = RefinementConfig(depth="full")
        ctx = StudyContext(run_dir="/tmp/test", summary={"scenario": "test"})
        yaml_dict = config.to_agent_lab_yaml(ctx)
        assert yaml_dict["mlesolver_max_steps"] == 3
        assert yaml_dict["papersolver_max_steps"] == 3

    def test_yaml_structure(self, study_dir: Path) -> None:
        config = RefinementConfig(llm_backend="gpt-4o")
        ctx = StudyContext.from_run_dir(str(study_dir))
        yaml_dict = config.to_agent_lab_yaml(ctx)

        assert "research_topic" in yaml_dict
        assert "notes" in yaml_dict
        assert yaml_dict["llm_backend"] == "gpt-4o"
        assert yaml_dict["copilot_mode"] is False
        assert isinstance(yaml_dict["notes"], list)

    def test_default_values(self) -> None:
        config = RefinementConfig()
        assert config.cost_budget_usd == 10.0
        assert config.timeout_seconds == 1800.0
        assert config.depth == "lite"


# ---------------------------------------------------------------------------
# RefinementResult tests
# ---------------------------------------------------------------------------


class TestRefinementResult:
    def test_to_dict_roundtrip(self) -> None:
        result = RefinementResult(
            success=True,
            hypotheses=["H1: Tax increases improve welfare"],
            parameter_suggestions={"tax_rate": [0.10, 0.20, 0.30]},
            gaps_identified=["Missing high-rho regime"],
            total_cost_usd=2.50,
            duration_seconds=120.0,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert len(d["hypotheses"]) == 1
        assert d["parameter_suggestions"]["tax_rate"] == [0.10, 0.20, 0.30]
        assert d["total_cost_usd"] == 2.50

    def test_to_dict_with_interactions(self) -> None:
        interaction = SoftInteraction(p=0.8, v_hat=0.6, accepted=True)
        result = RefinementResult(
            success=True,
            interactions=[interaction],
        )
        d = result.to_dict()
        assert len(d["interactions"]) == 1
        assert d["interactions"][0]["p"] == 0.8

    def test_empty_result(self) -> None:
        result = RefinementResult()
        d = result.to_dict()
        assert d["success"] is False
        assert d["hypotheses"] == []
        assert d["interactions"] == []
        assert d["proposed_scenario"] is None

    def test_proposed_scenario_included(self) -> None:
        result = RefinementResult(
            success=True,
            proposed_scenario={
                "name": "followup_tax_sweep",
                "params": {"tax_rate": {"min": 0.10, "max": 0.30}},
            },
        )
        d = result.to_dict()
        assert d["proposed_scenario"]["name"] == "followup_tax_sweep"


# ---------------------------------------------------------------------------
# AgentLabRunner tests
# ---------------------------------------------------------------------------


class TestAgentLabRunner:
    def test_find_checkpoint_returns_none_empty_dir(
        self, tmp_path: Path
    ) -> None:
        runner = AgentLabRunner(RefinementConfig())
        assert runner.find_checkpoint(str(tmp_path)) is None

    def test_find_checkpoint_finds_pkl(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state_saves"
        state_dir.mkdir()
        pkl = state_dir / "Paper0.pkl"
        pkl.write_bytes(b"fake")

        runner = AgentLabRunner(RefinementConfig())
        found = runner.find_checkpoint(str(tmp_path))
        assert found is not None
        assert "Paper0.pkl" in found

    def test_find_checkpoint_returns_latest(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state_saves"
        state_dir.mkdir()

        import time

        pkl0 = state_dir / "Paper0.pkl"
        pkl0.write_bytes(b"old")
        time.sleep(0.05)
        pkl1 = state_dir / "Paper1.pkl"
        pkl1.write_bytes(b"new")

        runner = AgentLabRunner(RefinementConfig())
        found = runner.find_checkpoint(str(tmp_path))
        assert found is not None
        assert "Paper1.pkl" in found

    def test_run_refinement_missing_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        runner = AgentLabRunner(RefinementConfig())
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            runner.run_refinement(str(tmp_path / "config.yaml"), str(tmp_path))

    def test_run_refinement_missing_lab_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = RefinementConfig(agent_lab_path=str(tmp_path / "nonexistent"))
        runner = AgentLabRunner(config)
        with pytest.raises(FileNotFoundError, match="AgentLaboratory"):
            runner.run_refinement(str(tmp_path / "config.yaml"), str(tmp_path))


# ---------------------------------------------------------------------------
# Bridge.refine_study integration test (mocked runner)
# ---------------------------------------------------------------------------


class TestBridgeRefineStudy:
    def test_refine_study_mocked(
        self,
        study_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test refine_study with a monkeypatched runner."""
        # Mock the runner so no real subprocess is spawned

        def mock_run_refinement(self, yaml_path, work_dir):
            # Write fake output
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            return (0, "- Hypothesis: higher tax\n- Gap in coverage\n", "", 30.0)

        def mock_find_checkpoint(self, work_dir):
            return None  # Skip checkpoint ingestion

        monkeypatch.setattr(AgentLabRunner, "run_refinement", mock_run_refinement)
        monkeypatch.setattr(AgentLabRunner, "find_checkpoint", mock_find_checkpoint)

        bridge = AgentLabBridge()
        config = RefinementConfig(
            agent_lab_path=".",  # doesn't matter, mocked
            depth="lite",
        )

        result = bridge.refine_study(str(study_dir), refinement_config=config)

        assert result.success is True
        assert result.duration_seconds == 30.0

        # Check output files were written
        output_dir = study_dir / "refinement"
        assert (output_dir / "refinement_report.json").exists()
        assert (output_dir / "refinement_config.yaml").exists()

        # Verify report is valid JSON
        with open(output_dir / "refinement_report.json") as f:
            report = json.load(f)
        assert report["success"] is True

    def test_refine_study_with_checkpoint(
        self,
        study_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test refine_study when a checkpoint is found."""
        fake_interactions = [
            SoftInteraction(p=0.8, v_hat=0.6, accepted=True),
        ]

        def mock_run_refinement(self, yaml_path, work_dir):
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            return (0, "", "", 45.0)

        def mock_find_checkpoint(self, work_dir):
            return "/fake/Paper0.pkl"

        def mock_ingest_checkpoint(self, path):
            return fake_interactions

        monkeypatch.setattr(AgentLabRunner, "run_refinement", mock_run_refinement)
        monkeypatch.setattr(AgentLabRunner, "find_checkpoint", mock_find_checkpoint)
        monkeypatch.setattr(AgentLabBridge, "ingest_checkpoint", mock_ingest_checkpoint)

        bridge = AgentLabBridge()
        config = RefinementConfig(agent_lab_path=".", depth="lite")
        result = bridge.refine_study(str(study_dir), refinement_config=config)

        assert result.success is True
        assert len(result.interactions) == 1
        assert result.interactions[0].p == 0.8

        # interactions.jsonl should be written
        interactions_path = study_dir / "refinement" / "interactions.jsonl"
        assert interactions_path.exists()

    def test_refine_study_failed_subprocess(
        self,
        study_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test refine_study handles subprocess failure gracefully."""

        def mock_run_refinement(self, yaml_path, work_dir):
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            return (1, "", "Error: OOM", 10.0)

        def mock_find_checkpoint(self, work_dir):
            return None

        monkeypatch.setattr(AgentLabRunner, "run_refinement", mock_run_refinement)
        monkeypatch.setattr(AgentLabRunner, "find_checkpoint", mock_find_checkpoint)

        bridge = AgentLabBridge()
        config = RefinementConfig(agent_lab_path=".", depth="lite")
        result = bridge.refine_study(str(study_dir), refinement_config=config)

        assert result.success is False
        assert result.interactions == []

    def test_refine_study_uses_config_defaults(
        self,
        study_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that refine_study uses AgentLabConfig defaults when no config given."""

        def mock_run_refinement(self, yaml_path, work_dir):
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            return (0, "", "", 5.0)

        def mock_find_checkpoint(self, work_dir):
            return None

        monkeypatch.setattr(AgentLabRunner, "run_refinement", mock_run_refinement)
        monkeypatch.setattr(AgentLabRunner, "find_checkpoint", mock_find_checkpoint)

        agent_config = AgentLabConfig(
            refinement_cost_budget_usd=5.0,
            refinement_depth="full",
        )
        bridge = AgentLabBridge(config=agent_config)
        result = bridge.refine_study(str(study_dir))

        assert result.success is True


# ---------------------------------------------------------------------------
# Bridge._extract_section tests
# ---------------------------------------------------------------------------


class TestExtractSection:
    def test_extracts_matching_lines(self) -> None:
        text = (
            "Some intro text\n"
            "- Hypothesis 1: higher tax rates improve welfare\n"
            "- Hypothesis 2: rho affects convergence\n"
            "- Unrelated bullet point\n"
        )
        items = AgentLabBridge._extract_section(text, "hypothes")
        assert len(items) == 2
        assert "higher tax rates" in items[0]

    def test_extracts_gaps(self) -> None:
        text = "- Gap: missing high-rho regime\n- Other stuff\n"
        items = AgentLabBridge._extract_section(text, "gap")
        assert len(items) == 1
        assert "high-rho" in items[0]

    def test_empty_text(self) -> None:
        assert AgentLabBridge._extract_section("", "hypothesis") == []

    def test_strips_bullet_markers(self) -> None:
        text = "* Hypothesis: test bullet\n• Hypothesis: unicode bullet\n"
        items = AgentLabBridge._extract_section(text, "hypothes")
        assert all(not item.startswith(("* ", "• ")) for item in items)
