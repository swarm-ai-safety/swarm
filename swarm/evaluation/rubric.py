"""Acceptance rubric engine for SWARM research evaluation.

Evaluates submissions as Publish / Revise / Reject based on:
- Experimental validity: Pass
- Replay success rate >= 80%
- Artifact resolution rate >= 95%
- Non-zero emergence evidence
- >= 1 documented failure mode
"""

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from swarm.evaluation.models import Checks, Scores, Verdict


class RubricConfig(BaseModel):
    """Configuration for acceptance rubric thresholds.

    Default values correspond to the SWARM evaluation plan.
    """

    min_replay_success_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    min_artifact_resolution_rate: float = Field(default=0.95, ge=0.0, le=1.0)
    min_emergence_delta: float = Field(default=0.0)
    min_documented_failure_modes: int = Field(default=1, ge=0)
    require_design_consistency_pass: bool = True

    @model_validator(mode="after")
    def _validate(self) -> "RubricConfig":
        if self.min_replay_success_rate > 1.0:
            raise ValueError("min_replay_success_rate cannot exceed 1.0")
        return self


class RubricOutcome(BaseModel):
    """Detailed outcome from rubric evaluation."""

    verdict: Verdict
    passed_criteria: List[str] = Field(default_factory=list)
    failed_criteria: List[str] = Field(default_factory=list)
    missing_data: List[str] = Field(default_factory=list)


class AcceptanceRubric:
    """Applies the SWARM acceptance rubric to determine verdict.

    The rubric evaluates five criteria. A submission must pass ALL
    criteria for 'publish'. If any required criterion fails, the
    verdict depends on severity:
    - Missing data or marginal failure -> 'revise'
    - Clear failure on multiple criteria -> 'reject'
    """

    def __init__(self, config: Optional[RubricConfig] = None):
        self.config = config or RubricConfig()

    def evaluate(self, scores: Scores, checks: Checks) -> RubricOutcome:
        """Apply rubric to computed scores and checks.

        Args:
            scores: Normalized evaluation scores.
            checks: Raw check values from evaluators.

        Returns:
            RubricOutcome with verdict and detailed pass/fail analysis.
        """
        passed: List[str] = []
        failed: List[str] = []
        missing: List[str] = []

        # Criterion 1: Experimental validity (design consistency = pass)
        if self.config.require_design_consistency_pass:
            if checks.design_consistency == "pass":
                passed.append("experimental_validity")
            elif checks.design_consistency == "fail":
                failed.append("experimental_validity")
            else:
                missing.append("experimental_validity")

        # Criterion 2: Replay success rate >= threshold
        if checks.replay_success_rate is not None:
            if checks.replay_success_rate >= self.config.min_replay_success_rate:
                passed.append("reproducibility")
            else:
                failed.append("reproducibility")
        else:
            missing.append("reproducibility")

        # Criterion 3: Artifact resolution rate >= threshold
        if checks.artifact_resolution_rate is not None:
            if checks.artifact_resolution_rate >= self.config.min_artifact_resolution_rate:
                passed.append("artifact_integrity")
            else:
                failed.append("artifact_integrity")
        else:
            missing.append("artifact_integrity")

        # Criterion 4: Non-zero emergence evidence
        if checks.emergence_delta is not None:
            if checks.emergence_delta > self.config.min_emergence_delta:
                passed.append("emergence_evidence")
            else:
                failed.append("emergence_evidence")
        else:
            missing.append("emergence_evidence")

        # Criterion 5: >= 1 documented failure mode
        if checks.documented_failure_modes_count is not None:
            if checks.documented_failure_modes_count >= self.config.min_documented_failure_modes:
                passed.append("failure_mode_coverage")
            else:
                failed.append("failure_mode_coverage")
        else:
            missing.append("failure_mode_coverage")

        # Determine verdict
        verdict = self._determine_verdict(passed, failed, missing)

        return RubricOutcome(
            verdict=verdict,
            passed_criteria=passed,
            failed_criteria=failed,
            missing_data=missing,
        )

    def _determine_verdict(
        self,
        passed: List[str],
        failed: List[str],
        missing: List[str],
    ) -> Verdict:
        """Determine verdict from criteria outcomes.

        Rules:
        - All passed, no failures, no missing -> publish
        - Any hard failure on 2+ criteria -> reject
        - Otherwise (missing data or single failure) -> revise
        """
        if not failed and not missing:
            return Verdict.PUBLISH

        if len(failed) >= 2:
            return Verdict.REJECT

        # Single failure or missing data -> revise
        return Verdict.REVISE
