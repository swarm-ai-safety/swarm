"""SWARM Agent-Authored Research Evaluation Framework.

Provides artifact-centric, reproducible, multi-agent research evaluation
with five evaluation axes: experimental validity, reproducibility,
artifact integrity, emergence detection, and failure mode coverage.
"""

from swarm.evaluation.eval_metrics import (
    aggregate_success_metrics,
    audit_effectiveness,
    calls_per_success,
    deception_detection_rate,
    loopiness_score,
    success_rate,
)
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
    Evidence,
    KeyArtifact,
    Notes,
    ReviewResult,
    Scores,
    Submission,
    Verdict,
)
from swarm.evaluation.pipeline import PipelineConfig, ReviewPipeline
from swarm.evaluation.rubric import AcceptanceRubric, RubricConfig
from swarm.evaluation.schema import REVIEW_SCHEMA, validate_review

__all__ = [
    # Models
    "Author",
    "AuthorType",
    "Checks",
    "Evidence",
    "KeyArtifact",
    "Notes",
    "ReviewResult",
    "Scores",
    "Submission",
    "Verdict",
    # Evaluators
    "ArtifactIntegrityEvaluator",
    "EmergenceDetectionEvaluator",
    "ExperimentalValidityEvaluator",
    "FailureModeEvaluator",
    "ReproducibilityEvaluator",
    # Rubric
    "AcceptanceRubric",
    "RubricConfig",
    # Pipeline
    "ReviewPipeline",
    "PipelineConfig",
    # Schema
    "REVIEW_SCHEMA",
    "validate_review",
    # Metrics
    "success_rate",
    "calls_per_success",
    "loopiness_score",
    "audit_effectiveness",
    "deception_detection_rate",
    "aggregate_success_metrics",
]
