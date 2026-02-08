"""Evaluation axis implementations for SWARM agent-authored research review.

Each evaluator checks a specific axis of the SWARM evaluation plan:
1. Experimental Validity (Agent Design Review)
2. Reproducibility (Executable, Not Verbal)
3. Artifact Integrity (Agent-Era Citation Checking)
4. Emergence Detection (SWARM-Specific)
5. Limits & Failure Modes
"""

import hashlib
import os
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Base evaluator
# ---------------------------------------------------------------------------

class BaseEvaluator(ABC):
    """Abstract base for all evaluation axis evaluators."""

    @abstractmethod
    def evaluate(self, submission_data: Dict[str, Any]) -> "EvaluationResult":
        """Run evaluation and return structured result.

        Args:
            submission_data: Dictionary containing submission artifacts
                and metadata needed for this evaluation axis.

        Returns:
            EvaluationResult with score, checks, and notes.
        """


@dataclass
class EvaluationResult:
    """Result from a single evaluation axis."""

    score: float  # Normalized 0-1
    checks: Dict[str, Any] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    required_changes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Experimental Validity
# ---------------------------------------------------------------------------

class ExperimentalValidityEvaluator(BaseEvaluator):
    """Evaluates whether agent interactions actually test the stated claim.

    Checks:
    - Agent roles, incentives, and policies are explicitly specified.
    - Interaction rules are clearly defined (fixed vs adaptive).
    - The claim depends on multi-agent interaction, not single-agent behavior.

    Signals:
    - design_consistency in {pass, fail}
    - interaction_depth (actions per agent x coordination steps)

    Expected keys in submission_data:
        agent_roles: list[dict] - Each with 'name', 'incentive', 'policy'.
        interaction_rules: dict - With 'type' ('fixed'|'adaptive') and 'description'.
        claims: list[str] - Stated research claims.
        multi_agent_dependency: bool - Whether claims require multi-agent behavior.
        interaction_depth: float - actions_per_agent * coordination_steps.
    """

    def evaluate(self, submission_data: Dict[str, Any]) -> EvaluationResult:
        strengths: List[str] = []
        weaknesses: List[str] = []
        required_changes: List[str] = []
        passed_checks = 0
        total_checks = 4

        # Check 1: Agent roles specified
        agent_roles = submission_data.get("agent_roles", [])
        if agent_roles and all(
            _has_keys(r, ("name", "incentive", "policy")) for r in agent_roles
        ):
            passed_checks += 1
            strengths.append(
                f"Agent roles fully specified ({len(agent_roles)} agents)"
            )
        else:
            if not agent_roles:
                required_changes.append("Agent roles must be explicitly specified")
            else:
                weaknesses.append(
                    "Some agent roles missing incentive or policy specification"
                )

        # Check 2: Interaction rules defined
        interaction_rules = submission_data.get("interaction_rules", {})
        if isinstance(interaction_rules, dict) and interaction_rules.get("type") in (
            "fixed",
            "adaptive",
        ):
            passed_checks += 1
            strengths.append(
                f"Interaction rules clearly defined as {interaction_rules['type']}"
            )
        else:
            required_changes.append(
                "Interaction rules must specify type (fixed or adaptive)"
            )

        # Check 3: Multi-agent dependency
        multi_agent = submission_data.get("multi_agent_dependency", False)
        if multi_agent:
            passed_checks += 1
            strengths.append("Claims depend on multi-agent interaction")
        else:
            weaknesses.append(
                "Claims may be reducible to single-agent behavior"
            )

        # Check 4: Interaction depth
        depth = submission_data.get("interaction_depth", 0.0)
        if depth > 0:
            passed_checks += 1
            strengths.append(f"Interaction depth = {depth:.1f}")
        else:
            weaknesses.append("Interaction depth not measured or is zero")

        design_consistency = "pass" if passed_checks >= 3 else "fail"
        score = passed_checks / total_checks

        return EvaluationResult(
            score=score,
            checks={
                "design_consistency": design_consistency,
                "interaction_depth": depth,
            },
            strengths=strengths,
            weaknesses=weaknesses,
            required_changes=required_changes,
        )


# ---------------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------------

class ReproducibilityEvaluator(BaseEvaluator):
    """Evaluates whether another agent can rerun and recover the result.

    Checks:
    - Runnable configuration or entrypoint exists.
    - Randomness is parameterized and logged.
    - Auditor agents can regenerate results within tolerance.

    Metrics:
    - replay_success_rate (0-1)
    - result_variance across replays

    Minimum bar: >= 80% replay success.

    Expected keys in submission_data:
        entrypoint: str | None - Path or command to run.
        random_seed_logged: bool - Whether randomness is parameterized.
        replay_results: list[float] - Outcome values from replay runs.
        reference_result: float - The original reported result.
        tolerance: float - Acceptable deviation (default 0.05).
    """

    def evaluate(self, submission_data: Dict[str, Any]) -> EvaluationResult:
        strengths: List[str] = []
        weaknesses: List[str] = []
        required_changes: List[str] = []
        score_components: List[float] = []

        # Check 1: Entrypoint exists
        entrypoint = submission_data.get("entrypoint")
        if entrypoint:
            strengths.append(f"Runnable entrypoint provided: {entrypoint}")
            score_components.append(1.0)
        else:
            required_changes.append("Runnable configuration or entrypoint required")
            score_components.append(0.0)

        # Check 2: Random seed logging
        seed_logged = submission_data.get("random_seed_logged", False)
        if seed_logged:
            strengths.append("Randomness is parameterized and logged")
            score_components.append(1.0)
        else:
            weaknesses.append("Random seeds not logged; replays may diverge")
            score_components.append(0.0)

        # Check 3: Replay success rate
        replay_results = submission_data.get("replay_results", [])
        reference = submission_data.get("reference_result")
        tolerance = submission_data.get("tolerance", 0.05)

        replay_success_rate: Optional[float] = None
        result_variance: Optional[float] = None

        if replay_results and reference is not None:
            successes = sum(
                1 for r in replay_results if abs(r - reference) <= tolerance
            )
            replay_success_rate = successes / len(replay_results)
            result_variance = (
                statistics.variance(replay_results)
                if len(replay_results) >= 2
                else 0.0
            )

            score_components.append(replay_success_rate)

            if replay_success_rate >= 0.8:
                strengths.append(
                    f"Replay success rate: {replay_success_rate:.0%} "
                    f"({successes}/{len(replay_results)} within tolerance)"
                )
            else:
                required_changes.append(
                    f"Replay success rate {replay_success_rate:.0%} "
                    f"below minimum bar of 80%"
                )

            if result_variance is not None and result_variance > 0:
                weaknesses.append(
                    f"Result variance across replays: {result_variance:.4f}"
                ) if result_variance > tolerance else None
        else:
            weaknesses.append("No replay results provided for verification")
            score_components.append(0.0)

        score = sum(score_components) / len(score_components) if score_components else 0.0

        checks: Dict[str, Any] = {}
        if replay_success_rate is not None:
            checks["replay_success_rate"] = replay_success_rate
        if result_variance is not None:
            checks["result_variance"] = result_variance

        return EvaluationResult(
            score=score,
            checks=checks,
            strengths=strengths,
            weaknesses=weaknesses,
            required_changes=required_changes,
        )


# ---------------------------------------------------------------------------
# 3. Artifact Integrity
# ---------------------------------------------------------------------------

class ArtifactIntegrityEvaluator(BaseEvaluator):
    """Evaluates artifact integrity (agent-era citation checking).

    Replaces traditional citation verification. Checks that referenced
    datasets, logs, models, or tools exist and match expected hashes.

    Metrics:
    - artifact_resolution_rate
    - artifact_hash_match_rate

    Minimum bar: >= 95% resolution.

    Expected keys in submission_data:
        artifacts: list[dict] - Each with 'label', 'url', and optional 'sha256'.
        resolver: callable | None - Function(url) -> bool that checks URL existence.
        file_resolver: callable | None - Function(path) -> bytes|None for local files.
    """

    def evaluate(self, submission_data: Dict[str, Any]) -> EvaluationResult:
        strengths: List[str] = []
        weaknesses: List[str] = []
        required_changes: List[str] = []

        artifacts = submission_data.get("artifacts", [])
        resolver: Optional[Callable] = submission_data.get("resolver")
        file_resolver: Optional[Callable] = submission_data.get("file_resolver")

        if not artifacts:
            return EvaluationResult(
                score=0.0,
                checks={
                    "artifact_resolution_rate": 0.0,
                    "artifact_hash_match_rate": 0.0,
                },
                weaknesses=["No artifacts declared"],
                required_changes=["At least one artifact must be referenced"],
            )

        resolved_count = 0
        hash_checked_count = 0
        hash_matched_count = 0
        unresolved: List[str] = []
        mismatched: List[str] = []

        for artifact in artifacts:
            label = artifact.get("label", "unknown")
            url = artifact.get("url", "")
            expected_hash = artifact.get("sha256")

            # Resolution check
            is_resolved = False
            if resolver is not None:
                is_resolved = resolver(url)
            elif url and (url.startswith("file://") or os.path.exists(url)):
                is_resolved = True
            elif url:
                # Without a resolver, we can only check local files
                is_resolved = os.path.exists(url)

            if is_resolved:
                resolved_count += 1
            else:
                unresolved.append(label)

            # Hash check
            if expected_hash:
                hash_checked_count += 1
                actual_hash = None

                if file_resolver is not None:
                    content = file_resolver(url)
                    if content is not None:
                        actual_hash = hashlib.sha256(content).hexdigest()
                elif os.path.isfile(url):
                    with open(url, "rb") as f:
                        actual_hash = hashlib.sha256(f.read()).hexdigest()

                if actual_hash == expected_hash:
                    hash_matched_count += 1
                elif actual_hash is not None:
                    mismatched.append(label)

        resolution_rate = resolved_count / len(artifacts)
        hash_match_rate = (
            hash_matched_count / hash_checked_count
            if hash_checked_count > 0
            else 1.0  # No hashes to check = no hash failures
        )

        # Score: weighted combination
        score = 0.7 * resolution_rate + 0.3 * hash_match_rate

        if resolution_rate >= 0.95:
            strengths.append(
                f"Artifact resolution rate: {resolution_rate:.0%} "
                f"({resolved_count}/{len(artifacts)})"
            )
        else:
            required_changes.append(
                f"Artifact resolution rate {resolution_rate:.0%} "
                f"below minimum bar of 95%"
            )

        if unresolved:
            weaknesses.append(
                f"Unresolved artifacts: {', '.join(unresolved)}"
            )

        if mismatched:
            weaknesses.append(
                f"Hash mismatches: {', '.join(mismatched)}"
            )

        if hash_checked_count > 0 and hash_match_rate == 1.0:
            strengths.append(
                f"All {hash_checked_count} artifact hashes verified"
            )

        return EvaluationResult(
            score=score,
            checks={
                "artifact_resolution_rate": resolution_rate,
                "artifact_hash_match_rate": hash_match_rate,
            },
            strengths=strengths,
            weaknesses=weaknesses,
            required_changes=required_changes,
        )


# ---------------------------------------------------------------------------
# 4. Emergence Detection
# ---------------------------------------------------------------------------

class EmergenceDetectionEvaluator(BaseEvaluator):
    """Evaluates whether results are irreducible to single-agent behavior.

    Checks:
    - Compare outcomes to single-agent baselines.
    - Perturb agent count or network topology.
    - Look for phase transitions or nonlinear effects.

    Metrics:
    - emergence_delta = outcome_multi - max(outcome_single)
    - topology_sensitivity

    A non-zero emergence signal is required.

    Expected keys in submission_data:
        multi_agent_outcome: float - Outcome from the multi-agent experiment.
        single_agent_outcomes: list[float] - Outcomes from single-agent baselines.
        topology_outcomes: dict[str, float] - {topology_name: outcome} for perturbation analysis.
        baseline_topology: str | None - Name of the default topology.
    """

    def evaluate(self, submission_data: Dict[str, Any]) -> EvaluationResult:
        strengths: List[str] = []
        weaknesses: List[str] = []
        required_changes: List[str] = []

        multi_outcome = submission_data.get("multi_agent_outcome")
        single_outcomes = submission_data.get("single_agent_outcomes", [])
        topology_outcomes = submission_data.get("topology_outcomes", {})

        emergence_delta: Optional[float] = None
        topology_sensitivity: Optional[float] = None

        score_components: List[float] = []

        # Emergence delta
        if multi_outcome is not None and single_outcomes:
            multi_outcome_val = float(multi_outcome)
            single_outcomes_vals = [float(x) for x in single_outcomes]
            max_single = max(single_outcomes_vals)
            emergence_delta_val = multi_outcome_val - max_single
            emergence_delta = emergence_delta_val

            if emergence_delta_val > 0:
                strengths.append(
                    f"Positive emergence delta: {emergence_delta_val:.4f} "
                    f"(multi={multi_outcome_val:.4f}, max_single={max_single:.4f})"
                )
                # Normalize: sigmoid-like scaling capped at 1.0
                score_components.append(
                    min(1.0, emergence_delta_val / max(abs(max_single), 1e-9))
                )
            elif emergence_delta_val == 0:
                weaknesses.append(
                    "Zero emergence delta; result matches single-agent baseline"
                )
                score_components.append(0.0)
            else:
                weaknesses.append(
                    f"Negative emergence delta: {emergence_delta_val:.4f}; "
                    f"multi-agent outcome worse than single-agent"
                )
                score_components.append(0.0)
        else:
            required_changes.append(
                "Multi-agent and single-agent baseline outcomes required "
                "for emergence detection"
            )
            score_components.append(0.0)

        # Topology sensitivity
        if topology_outcomes and len(topology_outcomes) >= 2:
            values = [float(v) for v in topology_outcomes.values()]
            mean_val = statistics.mean(values)
            if mean_val != 0:
                std_val = statistics.stdev(values) if len(values) >= 2 else 0.0
                topology_sensitivity_val = std_val / abs(mean_val)
            else:
                topology_sensitivity_val = 0.0

            topology_sensitivity = topology_sensitivity_val

            if topology_sensitivity_val > 0:
                strengths.append(
                    f"Topology sensitivity: {topology_sensitivity_val:.4f} "
                    f"across {len(topology_outcomes)} topologies"
                )
                score_components.append(min(1.0, topology_sensitivity_val))
            else:
                weaknesses.append(
                    "Results invariant to topology perturbation"
                )
                score_components.append(0.0)
        else:
            weaknesses.append(
                "Insufficient topology perturbation data "
                "(need >= 2 topologies)"
            )
            score_components.append(0.0)

        score = sum(score_components) / len(score_components) if score_components else 0.0

        checks: Dict[str, Any] = {}
        if emergence_delta is not None:
            checks["emergence_delta"] = emergence_delta
        if topology_sensitivity is not None:
            checks["topology_sensitivity"] = topology_sensitivity

        return EvaluationResult(
            score=score,
            checks=checks,
            strengths=strengths,
            weaknesses=weaknesses,
            required_changes=required_changes,
        )


# ---------------------------------------------------------------------------
# 5. Limits & Failure Modes
# ---------------------------------------------------------------------------

class FailureModeEvaluator(BaseEvaluator):
    """Evaluates whether the submission identifies where and how it fails.

    Checks:
    - Parameter regimes where effects disappear.
    - Adversarial or degenerate cases explored.
    - Explicit falsification attempts by auditor agents.

    Metrics:
    - falsification_attempts_count
    - documented_failure_modes_count

    At least one documented failure mode is required.

    Expected keys in submission_data:
        failure_modes: list[dict] - Each with 'description' and optional 'parameter_regime'.
        falsification_attempts: list[dict] - Each with 'description' and 'result'.
        adversarial_cases_explored: bool
    """

    def evaluate(self, submission_data: Dict[str, Any]) -> EvaluationResult:
        strengths: List[str] = []
        weaknesses: List[str] = []
        required_changes: List[str] = []

        failure_modes = submission_data.get("failure_modes", [])
        falsification_attempts = submission_data.get("falsification_attempts", [])
        adversarial_explored = submission_data.get(
            "adversarial_cases_explored", False
        )

        failure_count = len(failure_modes)
        falsification_count = len(falsification_attempts)

        score_components: List[float] = []

        # Check 1: At least one documented failure mode (required)
        if failure_count >= 1:
            strengths.append(
                f"{failure_count} failure mode(s) documented"
            )
            # Diminishing returns: 1 mode = 0.5, 2 = 0.75, 3+ = ~1.0
            score_components.append(min(1.0, 0.5 + 0.25 * (failure_count - 1)))
        else:
            required_changes.append(
                "At least one failure mode must be documented"
            )
            score_components.append(0.0)

        # Check 2: Falsification attempts
        if falsification_count > 0:
            strengths.append(
                f"{falsification_count} falsification attempt(s) conducted"
            )
            score_components.append(min(1.0, falsification_count / 3.0))
        else:
            weaknesses.append("No explicit falsification attempts")
            score_components.append(0.0)

        # Check 3: Adversarial / degenerate cases
        if adversarial_explored:
            strengths.append("Adversarial or degenerate cases explored")
            score_components.append(1.0)
        else:
            weaknesses.append(
                "No adversarial or degenerate cases explored"
            )
            score_components.append(0.0)

        score = sum(score_components) / len(score_components) if score_components else 0.0

        return EvaluationResult(
            score=score,
            checks={
                "falsification_attempts_count": falsification_count,
                "documented_failure_modes_count": failure_count,
            },
            strengths=strengths,
            weaknesses=weaknesses,
            required_changes=required_changes,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_keys(d: Any, keys: Tuple[str, ...]) -> bool:
    """Check that a dict-like object has all specified keys."""
    if not isinstance(d, dict):
        return False
    return all(k in d for k in keys)
