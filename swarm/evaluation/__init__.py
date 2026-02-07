"""SWARM Agent-Authored Research Evaluation Framework.

Provides artifact-centric, reproducible, multi-agent research evaluation
with five evaluation axes: experimental validity, reproducibility,
artifact integrity, emergence detection, and failure mode coverage.
"""

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
]
