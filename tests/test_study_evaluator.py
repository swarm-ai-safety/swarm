"""Tests for the council study evaluator (all mocked, no API calls)."""

import json
from unittest.mock import MagicMock

from swarm.agents.llm_config import LLMConfig, LLMProvider
from swarm.council.config import CouncilConfig, CouncilMemberConfig
from swarm.council.protocol import CouncilResult
from swarm.council.study_evaluator import (
    PERSONAS,
    StudyEvaluation,
    StudyEvaluator,
    default_evaluator_config,
    parse_synthesis_sections,
    save_evaluation,
)

# ── default_evaluator_config ────────────────────────────────────────


class TestDefaultEvaluatorConfig:
    def test_creates_three_members(self):
        config = default_evaluator_config()
        assert len(config.members) == 3

    def test_member_ids(self):
        config = default_evaluator_config()
        ids = [m.member_id for m in config.members]
        assert ids == ["mechanism_designer", "statistician", "red_teamer"]

    def test_chairman_is_mechanism_designer(self):
        config = default_evaluator_config()
        assert config.chairman.member_id == "mechanism_designer"

    def test_mechanism_designer_has_higher_weight(self):
        config = default_evaluator_config()
        weights = {m.member_id: m.weight for m in config.members}
        assert weights["mechanism_designer"] > weights["statistician"]
        assert weights["mechanism_designer"] > weights["red_teamer"]

    def test_custom_provider_configs(self):
        custom = {
            "statistician": LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4o",
                temperature=0.1,
            ),
        }
        config = default_evaluator_config(provider_configs=custom)
        stat_member = [m for m in config.members if m.member_id == "statistician"][0]
        assert stat_member.llm_config.provider == LLMProvider.OPENAI
        assert stat_member.llm_config.model == "gpt-4o"

        # Others should use defaults
        md_member = [m for m in config.members if m.member_id == "mechanism_designer"][0]
        assert md_member.llm_config.provider == LLMProvider.ANTHROPIC

    def test_min_members_required(self):
        config = default_evaluator_config()
        assert config.min_members_required == 2

    def test_three_personas_defined(self):
        assert len(PERSONAS) == 3
        assert set(PERSONAS.keys()) == {"mechanism_designer", "statistician", "red_teamer"}


# ── StudyEvaluator constructor ──────────────────────────────────────


class TestStudyEvaluatorInit:
    def test_default_config(self):
        evaluator = StudyEvaluator()
        assert len(evaluator.config.members) == 3

    def test_custom_config(self):
        config = CouncilConfig(
            members=[
                CouncilMemberConfig(
                    member_id="m1",
                    llm_config=LLMConfig(provider=LLMProvider.OLLAMA),
                ),
            ],
            min_members_required=1,
        )
        evaluator = StudyEvaluator(config=config)
        assert len(evaluator.config.members) == 1

    def test_council_instance_created(self):
        evaluator = StudyEvaluator()
        assert evaluator.council is not None
        assert evaluator.council.config is evaluator.config

    def test_auto_builds_query_fns_from_config(self):
        """Default constructor auto-builds LLMAgent query functions."""
        evaluator = StudyEvaluator()
        assert len(evaluator.council.query_fns) == 3
        assert set(evaluator.council.query_fns.keys()) == {
            "mechanism_designer", "statistician", "red_teamer",
        }

    def test_auto_builds_member_agents(self):
        """Default constructor creates LLMAgent instances for each member."""
        evaluator = StudyEvaluator()
        assert len(evaluator._member_agents) == 3
        for member_id, agent in evaluator._member_agents.items():
            assert agent.agent_id == f"evaluator_{member_id}"

    def test_explicit_query_fns_skip_auto_build(self):
        """Passing query_fns explicitly skips LLMAgent creation."""
        async def mock_query(sys: str, usr: str) -> str:
            return "mock"

        evaluator = StudyEvaluator(query_fns={"m1": mock_query})
        assert len(evaluator.council.query_fns) == 1
        assert len(evaluator._member_agents) == 0


# ── parse_synthesis_sections ────────────────────────────────────────


class TestParseSynthesisSections:
    def test_all_three_sections(self):
        text = (
            "FINDINGS:\n"
            "- The tax reduces toxicity by 40%\n"
            "- Welfare improves with moderate tax rates\n"
            "\n"
            "CONCERNS:\n"
            "- Sample size is only 10 seeds\n"
            "- No normality test reported\n"
            "\n"
            "RECOMMENDATIONS:\n"
            "- Run with 50 seeds for higher power\n"
            "- Add Shapiro-Wilk test\n"
        )
        result = parse_synthesis_sections(text)
        assert len(result["findings"]) == 2
        assert len(result["concerns"]) == 2
        assert len(result["recommendations"]) == 2
        assert "tax reduces toxicity" in result["findings"][0]

    def test_empty_synthesis(self):
        result = parse_synthesis_sections("")
        assert result["findings"] == []
        assert result["concerns"] == []
        assert result["recommendations"] == []

    def test_no_sections(self):
        text = "This is just freeform text without any section headers."
        result = parse_synthesis_sections(text)
        assert result["findings"] == []

    def test_continuation_lines(self):
        text = (
            "FINDINGS:\n"
            "- This is a long finding that\n"
            "  continues on the next line\n"
            "- Second finding\n"
        )
        result = parse_synthesis_sections(text)
        assert len(result["findings"]) == 2
        assert "continues on the next line" in result["findings"][0]

    def test_numbered_bullets(self):
        text = (
            "FINDINGS:\n"
            "1. First finding\n"
            "2. Second finding\n"
        )
        result = parse_synthesis_sections(text)
        assert len(result["findings"]) == 2

    def test_case_insensitive_headers(self):
        text = (
            "findings:\n"
            "- Item one\n"
            "\n"
            "Concerns:\n"
            "- Item two\n"
        )
        result = parse_synthesis_sections(text)
        assert len(result["findings"]) == 1
        assert len(result["concerns"]) == 1

    def test_key_findings_header(self):
        text = (
            "Key Findings:\n"
            "- Important result\n"
        )
        result = parse_synthesis_sections(text)
        assert len(result["findings"]) == 1


# ── StudyEvaluation ─────────────────────────────────────────────────


class TestStudyEvaluation:
    def test_to_dict(self):
        council_result = CouncilResult(
            synthesis="Test synthesis",
            responses={"m1": "resp1"},
            members_responded=1,
            members_total=3,
            success=True,
        )
        evaluation = StudyEvaluation(
            council_result=council_result,
            findings=["Finding 1"],
            concerns=["Concern 1"],
            recommendations=["Rec 1"],
            evaluation_type="sweep",
            run_dir="/tmp/test_run",
        )
        d = evaluation.to_dict()

        assert d["evaluation_type"] == "sweep"
        assert d["run_dir"] == "/tmp/test_run"
        assert d["synthesis"] == "Test synthesis"
        assert d["findings"] == ["Finding 1"]
        assert d["concerns"] == ["Concern 1"]
        assert d["recommendations"] == ["Rec 1"]
        assert d["members_responded"] == 1
        assert d["members_total"] == 3
        assert d["success"] is True

    def test_to_dict_with_empty_fields(self):
        council_result = CouncilResult(
            synthesis="",
            success=False,
            error="Failed",
        )
        evaluation = StudyEvaluation(council_result=council_result)
        d = evaluation.to_dict()

        assert d["findings"] == []
        assert d["error"] == "Failed"
        assert d["success"] is False


# ── save_evaluation ─────────────────────────────────────────────────


class TestSaveEvaluation:
    def test_saves_json(self, tmp_path):
        council_result = CouncilResult(
            synthesis="Test",
            success=True,
            members_responded=3,
            members_total=3,
        )
        evaluation = StudyEvaluation(
            council_result=council_result,
            findings=["f1"],
        )
        out_path = tmp_path / "subdir" / "eval.json"
        result = save_evaluation(evaluation, out_path)

        assert result == out_path
        assert out_path.exists()

        with open(out_path) as f:
            data = json.load(f)
        assert data["findings"] == ["f1"]
        assert data["success"] is True

    def test_creates_parent_directories(self, tmp_path):
        council_result = CouncilResult(synthesis="", success=True)
        evaluation = StudyEvaluation(council_result=council_result)
        out_path = tmp_path / "a" / "b" / "c" / "eval.json"
        save_evaluation(evaluation, out_path)
        assert out_path.exists()


# ── Prompt formatting ───────────────────────────────────────────────


class TestPromptFormatting:
    def test_sweep_prompt_with_summary(self, tmp_path):
        """Prompt includes summary.json content."""
        summary = {"scenario": "test", "total_runs": 100}
        (tmp_path / "summary.json").write_text(json.dumps(summary))

        from swarm.council.study_evaluator import _format_sweep_prompt

        prompt = _format_sweep_prompt(tmp_path)
        assert "test" in prompt
        assert "100" in prompt

    def test_sweep_prompt_with_csv(self, tmp_path):
        """Prompt includes column statistics from CSV."""
        csv_content = "param,welfare,toxicity\n0.0,1.5,0.3\n0.1,2.0,0.1\n"
        (tmp_path / "sweep_results.csv").write_text(csv_content)
        (tmp_path / "summary.json").write_text("{}")

        from swarm.council.study_evaluator import _format_sweep_prompt

        prompt = _format_sweep_prompt(tmp_path)
        assert "welfare" in prompt
        assert "2 rows" in prompt

    def test_sweep_prompt_missing_files(self, tmp_path):
        """Prompt handles missing files gracefully."""
        from swarm.council.study_evaluator import _format_sweep_prompt

        prompt = _format_sweep_prompt(tmp_path)
        assert "not found" in prompt

    def test_scenario_prompt(self, tmp_path):
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text("scenario_id: test\nepochs: 10\n")

        from swarm.council.study_evaluator import _format_scenario_prompt

        prompt = _format_scenario_prompt(yaml_path)
        assert "scenario_id: test" in prompt
        assert "BEFORE running" in prompt

    def test_scenario_prompt_missing_file(self, tmp_path):
        from swarm.council.study_evaluator import _format_scenario_prompt

        prompt = _format_scenario_prompt(tmp_path / "missing.yaml")
        assert "not found" in prompt.lower()

    def test_cross_study_prompt(self, tmp_path):
        for name in ("study_a", "study_b"):
            d = tmp_path / name
            d.mkdir()
            (d / "summary.json").write_text(json.dumps({"study": name}))

        from swarm.council.study_evaluator import _format_cross_study_prompt

        prompt = _format_cross_study_prompt([tmp_path / "study_a", tmp_path / "study_b"])
        assert "Study 1" in prompt
        assert "Study 2" in prompt
        assert "study_a" in prompt


# ── End-to-end with mocked deliberation ─────────────────────────────


class TestEvaluateSweepMocked:
    def test_evaluate_sweep(self, tmp_path):
        """End-to-end sweep evaluation with mocked _deliberate_sync."""
        # Set up run directory
        summary = {"scenario": "baseline", "total_runs": 50}
        (tmp_path / "summary.json").write_text(json.dumps(summary))

        mock_result = CouncilResult(
            synthesis=(
                "FINDINGS:\n"
                "- Tax rate strongly reduces toxicity\n"
                "\n"
                "CONCERNS:\n"
                "- Only 5 seeds used\n"
                "\n"
                "RECOMMENDATIONS:\n"
                "- Increase to 20 seeds\n"
            ),
            responses={
                "mechanism_designer": "Tax creates good incentives",
                "statistician": "Need more seeds",
                "red_teamer": "Could game the threshold",
            },
            members_responded=3,
            members_total=3,
            success=True,
        )

        evaluator = StudyEvaluator()
        evaluator._deliberate_sync = MagicMock(return_value=mock_result)

        evaluation = evaluator.evaluate_sweep(tmp_path)

        assert evaluation.evaluation_type == "sweep"
        assert evaluation.run_dir == str(tmp_path)
        assert len(evaluation.findings) == 1
        assert len(evaluation.concerns) == 1
        assert len(evaluation.recommendations) == 1
        assert "toxicity" in evaluation.findings[0]

        # Verify _deliberate_sync was called
        evaluator._deliberate_sync.assert_called_once()
        call_args = evaluator._deliberate_sync.call_args
        assert "sweep" in call_args[0][0].lower() or "simulation" in call_args[0][0].lower()

    def test_evaluate_scenario(self, tmp_path):
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text("scenario_id: test_scenario\nepochs: 10\n")

        mock_result = CouncilResult(
            synthesis="FINDINGS:\n- Good design\n",
            responses={"mechanism_designer": "ok"},
            members_responded=1,
            members_total=3,
            success=True,
        )

        evaluator = StudyEvaluator()
        evaluator._deliberate_sync = MagicMock(return_value=mock_result)

        evaluation = evaluator.evaluate_scenario(yaml_path)
        assert evaluation.evaluation_type == "scenario"
        assert len(evaluation.findings) == 1

    def test_evaluate_cross_study(self, tmp_path):
        for name in ("run_a", "run_b"):
            d = tmp_path / name
            d.mkdir()
            (d / "summary.json").write_text(json.dumps({"study": name}))

        mock_result = CouncilResult(
            synthesis=(
                "FINDINGS:\n- Both studies show tax reduces toxicity\n"
                "CONCERNS:\n- Different parameter ranges\n"
                "RECOMMENDATIONS:\n- Harmonize parameter grids\n"
            ),
            responses={"mechanism_designer": "consistent"},
            members_responded=3,
            members_total=3,
            success=True,
        )

        evaluator = StudyEvaluator()
        evaluator._deliberate_sync = MagicMock(return_value=mock_result)

        evaluation = evaluator.evaluate_cross_study(
            [tmp_path / "run_a", tmp_path / "run_b"]
        )
        assert evaluation.evaluation_type == "cross_study"
        assert len(evaluation.findings) == 1

    def test_full_roundtrip_save(self, tmp_path):
        """Evaluate → save → reload matches."""
        mock_result = CouncilResult(
            synthesis="FINDINGS:\n- Result A\nCONCERNS:\n- Issue B\n",
            responses={"m1": "r1"},
            members_responded=1,
            members_total=3,
            success=True,
        )

        evaluator = StudyEvaluator()
        evaluator._deliberate_sync = MagicMock(return_value=mock_result)

        (tmp_path / "summary.json").write_text("{}")
        evaluation = evaluator.evaluate_sweep(tmp_path)

        out = tmp_path / "council_review.json"
        save_evaluation(evaluation, out)

        with open(out) as f:
            data = json.load(f)

        assert data["findings"] == evaluation.findings
        assert data["concerns"] == evaluation.concerns
        assert data["success"] is True
