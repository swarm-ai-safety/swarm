"""Tests for the SWARM agent-authored research evaluation framework."""

import hashlib
import json
import os
import tempfile

import pytest

from swarm.evaluation.evaluators import (
    ArtifactIntegrityEvaluator,
    EmergenceDetectionEvaluator,
    ExperimentalValidityEvaluator,
    FailureModeEvaluator,
    ReproducibilityEvaluator,
)
from swarm.evaluation.models import (
    Author,
    AuthorType,
    Checks,
    KeyArtifact,
    ReviewResult,
    Scores,
    Submission,
    Verdict,
)
from swarm.evaluation.pipeline import PipelineConfig, ReviewPipeline
from swarm.evaluation.rubric import AcceptanceRubric, RubricConfig
from swarm.evaluation.schema import REVIEW_SCHEMA, validate_review

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_submission():
    """A well-formed submission for testing."""
    return Submission(
        id="SUB-2026-001",
        title="Emergent Coordination in Multi-Agent Markets",
        authors=[
            Author(name="agent-alpha", type=AuthorType.AGENT, model="claude-3"),
            Author(name="researcher-1", type=AuthorType.HUMAN),
        ],
        artifact_urls=["file:///data/results.json", "file:///data/logs.jsonl"],
        claims_summary="Multi-agent markets exhibit emergent price coordination",
        tags=["emergence", "markets", "multi-agent"],
    )


@pytest.fixture
def passing_submission_data():
    """Submission data that should pass all evaluators."""
    return {
        # Experimental validity
        "agent_roles": [
            {"name": "buyer", "incentive": "minimize cost", "policy": "threshold"},
            {"name": "seller", "incentive": "maximize revenue", "policy": "adaptive"},
        ],
        "interaction_rules": {"type": "adaptive", "description": "bilateral negotiation"},
        "multi_agent_dependency": True,
        "interaction_depth": 12.5,
        # Reproducibility
        "entrypoint": "python run_experiment.py --config market.yaml",
        "random_seed_logged": True,
        "replay_results": [0.85, 0.84, 0.86, 0.85, 0.84],
        "reference_result": 0.85,
        "tolerance": 0.02,
        # Artifact integrity
        "artifacts": [
            {"label": "results", "url": "/tmp/test_artifact.json"},
            {"label": "logs", "url": "/tmp/test_artifact_logs.jsonl"},
        ],
        "resolver": lambda url: True,
        # Emergence detection
        "multi_agent_outcome": 0.85,
        "single_agent_outcomes": [0.55, 0.60, 0.52],
        "topology_outcomes": {"ring": 0.80, "star": 0.85, "full": 0.90},
        # Failure modes
        "failure_modes": [
            {"description": "Effect disappears below 3 agents", "parameter_regime": "n < 3"},
            {"description": "Convergence fails with adversarial majority"},
        ],
        "falsification_attempts": [
            {"description": "Randomized agent ordering", "result": "effect persists"},
            {"description": "Removed communication channel", "result": "effect disappears"},
        ],
        "adversarial_cases_explored": True,
    }


@pytest.fixture
def failing_submission_data():
    """Submission data that should fail multiple evaluators."""
    return {
        "agent_roles": [],
        "interaction_rules": {},
        "multi_agent_dependency": False,
        "interaction_depth": 0.0,
        "entrypoint": None,
        "random_seed_logged": False,
        "replay_results": [],
        "reference_result": None,
        "artifacts": [],
        "multi_agent_outcome": None,
        "single_agent_outcomes": [],
        "failure_modes": [],
        "falsification_attempts": [],
        "adversarial_cases_explored": False,
    }


# ============================================================================
# Model Tests
# ============================================================================

class TestModels:
    """Tests for evaluation data models."""

    def test_author_types(self):
        assert AuthorType.AGENT.value == "agent"
        assert AuthorType.HUMAN.value == "human"
        assert AuthorType.HYBRID.value == "hybrid"

    def test_verdict_values(self):
        assert Verdict.PUBLISH.value == "publish"
        assert Verdict.REVISE.value == "revise"
        assert Verdict.REJECT.value == "reject"

    def test_submission_creation(self, sample_submission):
        assert sample_submission.id == "SUB-2026-001"
        assert len(sample_submission.authors) == 2
        assert sample_submission.authors[0].type == AuthorType.AGENT

    def test_scores_bounds(self):
        scores = Scores(experimental_validity=0.5, reproducibility=1.0)
        assert scores.experimental_validity == 0.5
        assert scores.reproducibility == 1.0

    def test_scores_bounds_validation(self):
        with pytest.raises(ValueError):
            Scores(experimental_validity=1.5)

    def test_scores_bounds_negative(self):
        with pytest.raises(ValueError):
            Scores(reproducibility=-0.1)

    def test_checks_design_consistency_enum(self):
        checks = Checks(design_consistency="pass")
        assert checks.design_consistency == "pass"

        checks = Checks(design_consistency="fail")
        assert checks.design_consistency == "fail"

    def test_checks_design_consistency_invalid(self):
        with pytest.raises(ValueError):
            Checks(design_consistency="maybe")

    def test_review_result_to_dict(self, sample_submission):
        result = ReviewResult(
            submission=sample_submission,
            verdict=Verdict.PUBLISH,
            scores=Scores(experimental_validity=0.9),
            checks=Checks(design_consistency="pass"),
        )
        d = result.to_dict()
        assert d["schema_version"] == "v1"
        assert d["verdict"] == "publish"
        assert d["scores"]["experimental_validity"] == 0.9
        assert d["checks"]["design_consistency"] == "pass"
        assert d["submission"]["id"] == "SUB-2026-001"

    def test_review_result_consistency_validation(self, sample_submission):
        """Cannot publish with design_consistency='fail'."""
        with pytest.raises(ValueError, match="Cannot publish"):
            ReviewResult(
                submission=sample_submission,
                verdict=Verdict.PUBLISH,
                checks=Checks(design_consistency="fail"),
            )

    def test_review_result_revise_with_fail_ok(self, sample_submission):
        """Revise verdict is ok with design_consistency='fail'."""
        result = ReviewResult(
            submission=sample_submission,
            verdict=Verdict.REVISE,
            checks=Checks(design_consistency="fail"),
        )
        assert result.verdict == Verdict.REVISE

    def test_key_artifact_model(self):
        artifact = KeyArtifact(
            label="model weights",
            url="file:///models/v1.bin",
            sha256="abc123",
        )
        assert artifact.label == "model weights"
        assert artifact.sha256 == "abc123"


# ============================================================================
# Schema Validation Tests
# ============================================================================

class TestSchemaValidation:
    """Tests for JSON schema validation."""

    def test_schema_has_required_fields(self):
        assert "required" in REVIEW_SCHEMA
        required = REVIEW_SCHEMA["required"]
        assert "schema_version" in required
        assert "submission" in required
        assert "verdict" in required
        assert "scores" in required
        assert "checks" in required
        assert "evidence" in required
        assert "notes" in required
        assert "timestamp_utc" in required

    def test_validate_valid_document(self, sample_submission):
        result = ReviewResult(
            submission=sample_submission,
            verdict=Verdict.REVISE,
            scores=Scores(experimental_validity=0.8),
            checks=Checks(design_consistency="pass", replay_success_rate=0.9),
        )
        doc = result.to_dict()
        is_valid, errors = validate_review(doc)
        assert is_valid, f"Validation errors: {errors}"

    def test_validate_missing_fields(self):
        is_valid, errors = validate_review({})
        assert not is_valid
        assert any("Missing required field" in e for e in errors)

    def test_validate_bad_verdict(self, sample_submission):
        result = ReviewResult(
            submission=sample_submission,
            verdict=Verdict.REVISE,
        )
        doc = result.to_dict()
        doc["verdict"] = "maybe"
        is_valid, errors = validate_review(doc)
        assert not is_valid
        assert any("verdict" in e for e in errors)

    def test_validate_bad_schema_version(self, sample_submission):
        result = ReviewResult(
            submission=sample_submission,
            verdict=Verdict.REVISE,
        )
        doc = result.to_dict()
        doc["schema_version"] = "v99"
        is_valid, errors = validate_review(doc)
        assert not is_valid

    def test_validate_scores_out_of_range(self, sample_submission):
        result = ReviewResult(
            submission=sample_submission,
            verdict=Verdict.REVISE,
        )
        doc = result.to_dict()
        doc["scores"]["experimental_validity"] = 1.5
        is_valid, errors = validate_review(doc)
        assert not is_valid
        assert any("scores" in e for e in errors)

    def test_validate_bad_author_type(self, sample_submission):
        result = ReviewResult(
            submission=sample_submission,
            verdict=Verdict.REVISE,
        )
        doc = result.to_dict()
        doc["submission"]["authors"][0]["type"] = "robot"
        is_valid, errors = validate_review(doc)
        assert not is_valid


# ============================================================================
# Experimental Validity Evaluator Tests
# ============================================================================

class TestExperimentalValidityEvaluator:
    """Tests for experimental validity evaluation."""

    def test_full_pass(self):
        evaluator = ExperimentalValidityEvaluator()
        result = evaluator.evaluate({
            "agent_roles": [
                {"name": "a", "incentive": "i", "policy": "p"},
                {"name": "b", "incentive": "i", "policy": "p"},
            ],
            "interaction_rules": {"type": "fixed", "description": "turn-based"},
            "multi_agent_dependency": True,
            "interaction_depth": 10.0,
        })
        assert result.score == 1.0
        assert result.checks["design_consistency"] == "pass"
        assert len(result.strengths) == 4
        assert len(result.required_changes) == 0

    def test_missing_roles(self):
        evaluator = ExperimentalValidityEvaluator()
        result = evaluator.evaluate({
            "agent_roles": [],
            "interaction_rules": {"type": "adaptive", "description": "x"},
            "multi_agent_dependency": True,
            "interaction_depth": 5.0,
        })
        assert result.score == 0.75
        assert result.checks["design_consistency"] == "pass"
        assert any("agent roles" in c.lower() for c in result.required_changes)

    def test_full_fail(self):
        evaluator = ExperimentalValidityEvaluator()
        result = evaluator.evaluate({
            "agent_roles": [],
            "interaction_rules": {},
            "multi_agent_dependency": False,
            "interaction_depth": 0,
        })
        assert result.score == 0.0
        assert result.checks["design_consistency"] == "fail"

    def test_incomplete_agent_roles(self):
        evaluator = ExperimentalValidityEvaluator()
        result = evaluator.evaluate({
            "agent_roles": [{"name": "a"}],  # missing incentive and policy
            "interaction_rules": {"type": "fixed", "description": "x"},
            "multi_agent_dependency": True,
            "interaction_depth": 3.0,
        })
        # Roles incomplete, but other 3 pass -> 0.75
        assert result.score == 0.75
        assert any("missing" in w.lower() for w in result.weaknesses)


# ============================================================================
# Reproducibility Evaluator Tests
# ============================================================================

class TestReproducibilityEvaluator:
    """Tests for reproducibility evaluation."""

    def test_full_pass(self):
        evaluator = ReproducibilityEvaluator()
        result = evaluator.evaluate({
            "entrypoint": "python run.py",
            "random_seed_logged": True,
            "replay_results": [1.0, 1.0, 1.0, 1.0, 1.0],
            "reference_result": 1.0,
            "tolerance": 0.01,
        })
        assert result.score == 1.0
        assert result.checks["replay_success_rate"] == 1.0

    def test_low_replay_rate(self):
        evaluator = ReproducibilityEvaluator()
        result = evaluator.evaluate({
            "entrypoint": "python run.py",
            "random_seed_logged": True,
            "replay_results": [1.0, 0.5, 0.3, 0.2, 0.1],
            "reference_result": 1.0,
            "tolerance": 0.05,
        })
        assert result.checks["replay_success_rate"] == 0.2
        assert result.score < 0.8
        assert any("80%" in c for c in result.required_changes)

    def test_no_entrypoint(self):
        evaluator = ReproducibilityEvaluator()
        result = evaluator.evaluate({
            "entrypoint": None,
            "random_seed_logged": False,
            "replay_results": [],
        })
        assert result.score == 0.0

    def test_no_replay_results(self):
        evaluator = ReproducibilityEvaluator()
        result = evaluator.evaluate({
            "entrypoint": "python run.py",
            "random_seed_logged": True,
            "replay_results": [],
            "reference_result": None,
        })
        # entrypoint=1, seed=1, replay=0 -> 2/3
        assert abs(result.score - 2 / 3) < 0.01


# ============================================================================
# Artifact Integrity Evaluator Tests
# ============================================================================

class TestArtifactIntegrityEvaluator:
    """Tests for artifact integrity evaluation."""

    def test_all_resolved_with_resolver(self):
        evaluator = ArtifactIntegrityEvaluator()
        result = evaluator.evaluate({
            "artifacts": [
                {"label": "data", "url": "https://example.com/data.csv"},
                {"label": "model", "url": "https://example.com/model.bin"},
            ],
            "resolver": lambda url: True,
        })
        assert result.checks["artifact_resolution_rate"] == 1.0
        assert result.score > 0.9

    def test_partial_resolution(self):
        evaluator = ArtifactIntegrityEvaluator()
        call_count = {"n": 0}

        def partial_resolver(url):
            call_count["n"] += 1
            return call_count["n"] <= 1  # Only first resolves

        result = evaluator.evaluate({
            "artifacts": [
                {"label": "data", "url": "a"},
                {"label": "model", "url": "b"},
            ],
            "resolver": partial_resolver,
        })
        assert result.checks["artifact_resolution_rate"] == 0.5
        assert any("Unresolved" in w for w in result.weaknesses)

    def test_hash_verification(self):
        evaluator = ArtifactIntegrityEvaluator()
        content = b"test content"
        correct_hash = hashlib.sha256(content).hexdigest()

        result = evaluator.evaluate({
            "artifacts": [
                {"label": "data", "url": "test", "sha256": correct_hash},
            ],
            "resolver": lambda url: True,
            "file_resolver": lambda url: content,
        })
        assert result.checks["artifact_hash_match_rate"] == 1.0
        assert any("verified" in s.lower() for s in result.strengths)

    def test_hash_mismatch(self):
        evaluator = ArtifactIntegrityEvaluator()
        result = evaluator.evaluate({
            "artifacts": [
                {"label": "data", "url": "test", "sha256": "wrong_hash"},
            ],
            "resolver": lambda url: True,
            "file_resolver": lambda url: b"actual content",
        })
        assert result.checks["artifact_hash_match_rate"] == 0.0
        assert any("mismatch" in w.lower() for w in result.weaknesses)

    def test_no_artifacts(self):
        evaluator = ArtifactIntegrityEvaluator()
        result = evaluator.evaluate({"artifacts": []})
        assert result.score == 0.0
        assert any("No artifacts" in w for w in result.weaknesses)

    def test_local_file_resolution(self):
        evaluator = ArtifactIntegrityEvaluator()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello")
            path = f.name
        try:
            file_hash = hashlib.sha256(b"hello").hexdigest()
            result = evaluator.evaluate({
                "artifacts": [
                    {"label": "local", "url": path, "sha256": file_hash},
                ],
            })
            assert result.checks["artifact_resolution_rate"] == 1.0
            assert result.checks["artifact_hash_match_rate"] == 1.0
        finally:
            os.unlink(path)


# ============================================================================
# Emergence Detection Evaluator Tests
# ============================================================================

class TestEmergenceDetectionEvaluator:
    """Tests for emergence detection evaluation."""

    def test_positive_emergence(self):
        evaluator = EmergenceDetectionEvaluator()
        result = evaluator.evaluate({
            "multi_agent_outcome": 0.9,
            "single_agent_outcomes": [0.5, 0.6, 0.55],
            "topology_outcomes": {"ring": 0.8, "star": 0.9, "full": 0.95},
        })
        assert result.checks["emergence_delta"] == pytest.approx(0.3)
        assert result.checks["emergence_delta"] > 0
        assert result.score > 0

    def test_no_emergence(self):
        evaluator = EmergenceDetectionEvaluator()
        result = evaluator.evaluate({
            "multi_agent_outcome": 0.5,
            "single_agent_outcomes": [0.5, 0.6],
            "topology_outcomes": {"ring": 0.5, "star": 0.5},
        })
        assert result.checks["emergence_delta"] < 0
        assert any("worse" in w.lower() or "negative" in w.lower() for w in result.weaknesses)

    def test_zero_emergence(self):
        evaluator = EmergenceDetectionEvaluator()
        result = evaluator.evaluate({
            "multi_agent_outcome": 0.6,
            "single_agent_outcomes": [0.6],
            "topology_outcomes": {"ring": 0.6, "star": 0.6},
        })
        assert result.checks["emergence_delta"] == 0

    def test_missing_data(self):
        evaluator = EmergenceDetectionEvaluator()
        result = evaluator.evaluate({
            "multi_agent_outcome": None,
            "single_agent_outcomes": [],
        })
        assert result.score == 0.0
        assert any("required" in c.lower() for c in result.required_changes)

    def test_topology_sensitivity(self):
        evaluator = EmergenceDetectionEvaluator()
        result = evaluator.evaluate({
            "multi_agent_outcome": 1.0,
            "single_agent_outcomes": [0.5],
            "topology_outcomes": {"ring": 0.5, "star": 1.0, "full": 1.5},
        })
        assert result.checks["topology_sensitivity"] > 0
        assert any("sensitivity" in s.lower() for s in result.strengths)


# ============================================================================
# Failure Mode Evaluator Tests
# ============================================================================

class TestFailureModeEvaluator:
    """Tests for failure mode evaluation."""

    def test_good_coverage(self):
        evaluator = FailureModeEvaluator()
        result = evaluator.evaluate({
            "failure_modes": [
                {"description": "breaks below 3 agents"},
                {"description": "convergence fails with adversaries"},
            ],
            "falsification_attempts": [
                {"description": "randomize ordering", "result": "persists"},
            ],
            "adversarial_cases_explored": True,
        })
        assert result.score > 0.5
        assert result.checks["documented_failure_modes_count"] == 2
        assert result.checks["falsification_attempts_count"] == 1

    def test_no_failure_modes(self):
        evaluator = FailureModeEvaluator()
        result = evaluator.evaluate({
            "failure_modes": [],
            "falsification_attempts": [],
            "adversarial_cases_explored": False,
        })
        assert result.score == 0.0
        assert result.checks["documented_failure_modes_count"] == 0
        assert any("At least one" in c for c in result.required_changes)

    def test_single_failure_mode(self):
        evaluator = FailureModeEvaluator()
        result = evaluator.evaluate({
            "failure_modes": [{"description": "fails at scale"}],
            "falsification_attempts": [],
            "adversarial_cases_explored": False,
        })
        assert result.checks["documented_failure_modes_count"] == 1
        assert result.score > 0

    def test_many_falsification_attempts(self):
        evaluator = FailureModeEvaluator()
        result = evaluator.evaluate({
            "failure_modes": [{"description": "known failure"}],
            "falsification_attempts": [
                {"description": f"attempt {i}", "result": "ok"} for i in range(5)
            ],
            "adversarial_cases_explored": True,
        })
        assert result.score > 0.8


# ============================================================================
# Rubric Tests
# ============================================================================

class TestAcceptanceRubric:
    """Tests for the acceptance rubric engine."""

    def test_publish_all_pass(self):
        rubric = AcceptanceRubric()
        scores = Scores(
            experimental_validity=1.0,
            reproducibility=1.0,
            artifact_integrity=1.0,
            emergence_evidence=0.8,
            failure_mode_coverage=0.7,
        )
        checks = Checks(
            design_consistency="pass",
            replay_success_rate=0.9,
            artifact_resolution_rate=0.98,
            emergence_delta=0.3,
            documented_failure_modes_count=2,
        )
        outcome = rubric.evaluate(scores, checks)
        assert outcome.verdict == Verdict.PUBLISH
        assert len(outcome.passed_criteria) == 5
        assert len(outcome.failed_criteria) == 0

    def test_reject_multiple_failures(self):
        rubric = AcceptanceRubric()
        scores = Scores()
        checks = Checks(
            design_consistency="fail",
            replay_success_rate=0.3,
            artifact_resolution_rate=0.5,
            emergence_delta=-0.1,
            documented_failure_modes_count=0,
        )
        outcome = rubric.evaluate(scores, checks)
        assert outcome.verdict == Verdict.REJECT
        assert len(outcome.failed_criteria) >= 2

    def test_revise_single_failure(self):
        rubric = AcceptanceRubric()
        scores = Scores()
        checks = Checks(
            design_consistency="pass",
            replay_success_rate=0.7,  # Below 0.8 threshold
            artifact_resolution_rate=0.98,
            emergence_delta=0.5,
            documented_failure_modes_count=1,
        )
        outcome = rubric.evaluate(scores, checks)
        assert outcome.verdict == Verdict.REVISE
        assert "reproducibility" in outcome.failed_criteria

    def test_revise_missing_data(self):
        rubric = AcceptanceRubric()
        scores = Scores()
        checks = Checks(
            design_consistency="pass",
            replay_success_rate=0.9,
            artifact_resolution_rate=0.98,
            # emergence_delta missing
            documented_failure_modes_count=2,
        )
        outcome = rubric.evaluate(scores, checks)
        assert outcome.verdict == Verdict.REVISE
        assert "emergence_evidence" in outcome.missing_data

    def test_custom_thresholds(self):
        config = RubricConfig(
            min_replay_success_rate=0.5,
            min_artifact_resolution_rate=0.80,
            min_documented_failure_modes=0,
        )
        rubric = AcceptanceRubric(config)
        checks = Checks(
            design_consistency="pass",
            replay_success_rate=0.6,
            artifact_resolution_rate=0.85,
            emergence_delta=0.1,
            documented_failure_modes_count=0,
        )
        outcome = rubric.evaluate(Scores(), checks)
        assert outcome.verdict == Verdict.PUBLISH


# ============================================================================
# Pipeline Tests
# ============================================================================

class TestReviewPipeline:
    """Tests for the review pipeline orchestrator."""

    def test_full_pipeline_pass(self, sample_submission, passing_submission_data):
        pipeline = ReviewPipeline()
        result = pipeline.run(sample_submission, passing_submission_data)
        assert isinstance(result, ReviewResult)
        assert result.schema_version == "v1"
        assert result.submission.id == "SUB-2026-001"
        assert result.scores.experimental_validity > 0
        assert result.scores.reproducibility > 0
        assert result.scores.failure_mode_coverage > 0

    def test_full_pipeline_fail(self, sample_submission, failing_submission_data):
        pipeline = ReviewPipeline()
        result = pipeline.run(sample_submission, failing_submission_data)
        assert result.verdict == Verdict.REJECT
        assert len(result.notes.required_changes) > 0

    def test_pipeline_to_dict_validates(self, sample_submission, passing_submission_data):
        pipeline = ReviewPipeline()
        result = pipeline.run(sample_submission, passing_submission_data)
        doc = result.to_dict()
        is_valid, errors = validate_review(doc)
        assert is_valid, f"Validation errors: {errors}"

    def test_pipeline_serializable(self, sample_submission, passing_submission_data):
        pipeline = ReviewPipeline()
        result = pipeline.run(sample_submission, passing_submission_data)
        doc = result.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(doc, indent=2)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "v1"
        assert parsed["submission"]["id"] == "SUB-2026-001"

    def test_pipeline_skip_axes(self, sample_submission):
        config = PipelineConfig(
            skip_artifact_integrity=True,
            skip_emergence_detection=True,
        )
        pipeline = ReviewPipeline(config)
        data = {
            "agent_roles": [
                {"name": "a", "incentive": "i", "policy": "p"},
            ],
            "interaction_rules": {"type": "fixed", "description": "x"},
            "multi_agent_dependency": True,
            "interaction_depth": 5.0,
            "entrypoint": "python run.py",
            "random_seed_logged": True,
            "replay_results": [1.0, 1.0, 1.0],
            "reference_result": 1.0,
            "failure_modes": [{"description": "known"}],
            "falsification_attempts": [{"description": "tried", "result": "ok"}],
            "adversarial_cases_explored": True,
        }
        result = pipeline.run(sample_submission, data)
        assert isinstance(result, ReviewResult)
        # Skipped axes should have default scores
        assert result.scores.artifact_integrity == 0.0
        assert result.scores.emergence_evidence == 0.0

    def test_pipeline_evidence_extraction(self, sample_submission):
        pipeline = ReviewPipeline()
        data = {
            "agent_roles": [{"name": "a", "incentive": "i", "policy": "p"}],
            "interaction_rules": {"type": "fixed", "description": "x"},
            "multi_agent_dependency": True,
            "interaction_depth": 5.0,
            "entrypoint": "python run.py",
            "random_seed_logged": True,
            "replay_results": [1.0],
            "reference_result": 1.0,
            "artifacts": [
                {"label": "results", "url": "file:///data/results.json", "sha256": "abc"},
            ],
            "resolver": lambda url: True,
            "multi_agent_outcome": 0.9,
            "single_agent_outcomes": [0.5],
            "topology_outcomes": {"ring": 0.8, "star": 0.9},
            "failure_modes": [{"description": "known"}],
            "falsification_attempts": [],
            "adversarial_cases_explored": False,
        }
        result = pipeline.run(sample_submission, data)
        assert len(result.evidence.key_artifacts) == 1
        assert result.evidence.key_artifacts[0].label == "results"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_publish_workflow(self, sample_submission, passing_submission_data):
        """Full workflow producing a publish verdict."""
        pipeline = ReviewPipeline()
        result = pipeline.run(sample_submission, passing_submission_data)

        # Serialize
        doc = result.to_dict()
        json_str = json.dumps(doc)
        parsed = json.loads(json_str)

        # Validate
        is_valid, errors = validate_review(parsed)
        assert is_valid, f"Errors: {errors}"

        # Check publish criteria met
        assert parsed["checks"].get("design_consistency") == "pass"
        assert parsed["checks"].get("replay_success_rate", 0) >= 0.8
        assert parsed["checks"].get("emergence_delta", 0) > 0
        assert parsed["checks"].get("documented_failure_modes_count", 0) >= 1

    def test_reject_workflow(self, sample_submission, failing_submission_data):
        """Full workflow producing a reject verdict."""
        pipeline = ReviewPipeline()
        result = pipeline.run(sample_submission, failing_submission_data)

        doc = result.to_dict()
        is_valid, errors = validate_review(doc)
        assert is_valid, f"Errors: {errors}"
        assert doc["verdict"] == "reject"

    def test_round_trip_serialization(self, sample_submission, passing_submission_data):
        """ReviewResult -> dict -> JSON -> dict -> validate."""
        pipeline = ReviewPipeline()
        result = pipeline.run(sample_submission, passing_submission_data)

        # Round trip
        doc = result.to_dict()
        json_bytes = json.dumps(doc).encode("utf-8")
        restored = json.loads(json_bytes.decode("utf-8"))

        is_valid, errors = validate_review(restored)
        assert is_valid, f"Errors: {errors}"

        # Check key fields survived
        assert restored["submission"]["title"] == sample_submission.title
        assert restored["scores"]["experimental_validity"] == doc["scores"]["experimental_validity"]
