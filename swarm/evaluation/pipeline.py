"""Review pipeline orchestrator for SWARM evaluation.

Coordinates all five evaluators, applies the acceptance rubric,
and produces a complete ReviewResult conforming to the JSON schema.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from swarm.evaluation.evaluators import (
    ArtifactIntegrityEvaluator,
    EmergenceDetectionEvaluator,
    EvaluationResult,
    ExperimentalValidityEvaluator,
    FailureModeEvaluator,
    ReproducibilityEvaluator,
)
from swarm.evaluation.models import (
    Checks,
    Evidence,
    KeyArtifact,
    Notes,
    ReviewResult,
    Scores,
    Submission,
)
from swarm.evaluation.rubric import AcceptanceRubric, RubricConfig


class PipelineConfig(BaseModel):
    """Configuration for the review pipeline."""

    rubric_config: RubricConfig = Field(default_factory=RubricConfig)
    skip_artifact_integrity: bool = False
    skip_emergence_detection: bool = False


class ReviewPipeline:
    """Orchestrates the full SWARM evaluation pipeline.

    Runs all five evaluation axes, merges results, applies the
    acceptance rubric, and produces a machine-readable ReviewResult.

    Usage:
        pipeline = ReviewPipeline()
        result = pipeline.run(submission, submission_data)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._rubric = AcceptanceRubric(self.config.rubric_config)
        self._evaluators = {
            "experimental_validity": ExperimentalValidityEvaluator(),
            "reproducibility": ReproducibilityEvaluator(),
            "artifact_integrity": ArtifactIntegrityEvaluator(),
            "emergence_detection": EmergenceDetectionEvaluator(),
            "failure_modes": FailureModeEvaluator(),
        }

    def run(
        self,
        submission: Submission,
        submission_data: Dict[str, Any],
    ) -> ReviewResult:
        """Run the full evaluation pipeline.

        Args:
            submission: Submission metadata.
            submission_data: Dictionary containing all data needed by
                evaluators. Each evaluator consumes a subset of keys.

        Returns:
            Complete ReviewResult with verdict, scores, checks, and notes.
        """
        axis_results: Dict[str, EvaluationResult] = {}

        # Run evaluators
        axis_results["experimental_validity"] = self._evaluators[
            "experimental_validity"
        ].evaluate(submission_data)

        axis_results["reproducibility"] = self._evaluators["reproducibility"].evaluate(
            submission_data
        )

        if not self.config.skip_artifact_integrity:
            axis_results["artifact_integrity"] = self._evaluators[
                "artifact_integrity"
            ].evaluate(submission_data)

        if not self.config.skip_emergence_detection:
            axis_results["emergence_detection"] = self._evaluators[
                "emergence_detection"
            ].evaluate(submission_data)

        axis_results["failure_modes"] = self._evaluators["failure_modes"].evaluate(
            submission_data
        )

        # Merge into scores and checks
        scores = self._build_scores(axis_results)
        checks = self._build_checks(axis_results)

        # Apply rubric
        rubric_outcome = self._rubric.evaluate(scores, checks)

        # Build notes from all evaluator results
        notes = self._build_notes(axis_results)

        # Build evidence from artifacts
        evidence = self._build_evidence(submission_data)

        return ReviewResult(
            schema_version="v1",
            timestamp_utc=datetime.now(timezone.utc),
            submission=submission,
            verdict=rubric_outcome.verdict,
            scores=scores,
            checks=checks,
            evidence=evidence,
            notes=notes,
        )

    def _build_scores(self, axis_results: Dict[str, EvaluationResult]) -> Scores:
        """Extract normalized scores from evaluator results."""
        return Scores(
            experimental_validity=axis_results.get(
                "experimental_validity", EvaluationResult(score=0.0)
            ).score,
            reproducibility=axis_results.get(
                "reproducibility", EvaluationResult(score=0.0)
            ).score,
            artifact_integrity=axis_results.get(
                "artifact_integrity", EvaluationResult(score=0.0)
            ).score,
            emergence_evidence=axis_results.get(
                "emergence_detection", EvaluationResult(score=0.0)
            ).score,
            failure_mode_coverage=axis_results.get(
                "failure_modes", EvaluationResult(score=0.0)
            ).score,
        )

    def _build_checks(self, axis_results: Dict[str, EvaluationResult]) -> Checks:
        """Merge raw checks from all evaluators."""
        merged: Dict[str, Any] = {}
        for result in axis_results.values():
            merged.update(result.checks)

        return Checks(
            design_consistency=merged.get("design_consistency"),
            replay_success_rate=merged.get("replay_success_rate"),
            artifact_resolution_rate=merged.get("artifact_resolution_rate"),
            artifact_hash_match_rate=merged.get("artifact_hash_match_rate"),
            emergence_delta=merged.get("emergence_delta"),
            topology_sensitivity=merged.get("topology_sensitivity"),
            falsification_attempts_count=merged.get("falsification_attempts_count"),
            documented_failure_modes_count=merged.get("documented_failure_modes_count"),
        )

    def _build_notes(self, axis_results: Dict[str, EvaluationResult]) -> Notes:
        """Aggregate notes from all evaluators."""
        all_strengths: List[str] = []
        all_weaknesses: List[str] = []
        all_required: List[str] = []

        for result in axis_results.values():
            all_strengths.extend(result.strengths)
            all_weaknesses.extend(result.weaknesses)
            all_required.extend(result.required_changes)

        return Notes(
            strengths=all_strengths,
            weaknesses=all_weaknesses,
            required_changes=all_required,
        )

    def _build_evidence(self, submission_data: Dict[str, Any]) -> Evidence:
        """Extract evidence artifacts from submission data."""
        artifacts = submission_data.get("artifacts", [])
        key_artifacts = [
            KeyArtifact(
                label=a.get("label", ""),
                url=a.get("url", ""),
                sha256=a.get("sha256"),
            )
            for a in artifacts
            if isinstance(a, dict) and "label" in a and "url" in a
        ]
        return Evidence(key_artifacts=key_artifacts)
