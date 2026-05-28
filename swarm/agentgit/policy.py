"""Task-scoped policy evaluation for agent-authored git changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from swarm.agentgit.git import GitSnapshot


@dataclass(frozen=True)
class PolicyDecision:
    """A single pass/fail result emitted into provenance bundles."""

    policy_id: str
    passed: bool
    reason: str
    severity: str = "error"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentGitPolicy:
    """Declarative limits for one delegated agent task."""

    allowed_paths: List[str] = field(default_factory=list)
    denied_paths: List[str] = field(default_factory=list)
    max_changed_files: int | None = None
    max_added_lines: int | None = None
    required_checks: List[str] = field(default_factory=list)
    require_human_review: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "AgentGitPolicy":
        data = yaml.safe_load(path.read_text()) or {}
        return cls(
            allowed_paths=list(data.get("allowed_paths", [])),
            denied_paths=list(data.get("denied_paths", [])),
            max_changed_files=data.get("max_changed_files"),
            max_added_lines=data.get("max_added_lines"),
            required_checks=list(data.get("required_checks", [])),
            require_human_review=bool(data.get("require_human_review", False)),
        )

    def evaluate(
        self,
        snapshot: GitSnapshot,
        check_results: Dict[str, bool] | None = None,
    ) -> List[PolicyDecision]:
        """Evaluate repository state against this task policy."""

        check_results = check_results or {}
        changed_paths = [file.path for file in snapshot.changed_files]
        decisions = [
            self._evaluate_allowed_paths(changed_paths),
            self._evaluate_denied_paths(changed_paths),
            self._evaluate_max_changed_files(snapshot),
            self._evaluate_max_added_lines(snapshot),
            self._evaluate_required_checks(check_results),
        ]
        if self.require_human_review:
            decisions.append(
                PolicyDecision(
                    policy_id="human-review",
                    passed=False,
                    reason="Policy requires human review before merge",
                    severity="warning",
                )
            )
        return decisions

    def _evaluate_allowed_paths(self, paths: List[str]) -> PolicyDecision:
        if not self.allowed_paths:
            return PolicyDecision(
                policy_id="allowed-paths",
                passed=True,
                reason="No allowed_paths constraint configured",
            )

        violations = [
            path for path in paths if not _matches_any(path, self.allowed_paths)
        ]
        return PolicyDecision(
            policy_id="allowed-paths",
            passed=not violations,
            reason=(
                "All changed files are within allowed paths"
                if not violations
                else "Changed files outside allowed paths"
            ),
            details={"violations": violations, "allowed_paths": self.allowed_paths},
        )

    def _evaluate_denied_paths(self, paths: List[str]) -> PolicyDecision:
        violations = [path for path in paths if _matches_any(path, self.denied_paths)]
        return PolicyDecision(
            policy_id="denied-paths",
            passed=not violations,
            reason=(
                "No changed files match denied paths"
                if not violations
                else "Changed files match denied paths"
            ),
            details={"violations": violations, "denied_paths": self.denied_paths},
        )

    def _evaluate_max_changed_files(self, snapshot: GitSnapshot) -> PolicyDecision:
        count = len(snapshot.changed_files)
        if self.max_changed_files is None:
            return PolicyDecision(
                policy_id="max-changed-files",
                passed=True,
                reason="No max_changed_files constraint configured",
                details={"changed_files": count},
            )
        return PolicyDecision(
            policy_id="max-changed-files",
            passed=count <= self.max_changed_files,
            reason=f"Changed file count is {count}/{self.max_changed_files}",
            details={"changed_files": count, "limit": self.max_changed_files},
        )

    def _evaluate_max_added_lines(self, snapshot: GitSnapshot) -> PolicyDecision:
        added = snapshot.total_additions
        if self.max_added_lines is None:
            return PolicyDecision(
                policy_id="max-added-lines",
                passed=True,
                reason="No max_added_lines constraint configured",
                details={"added_lines": added},
            )
        return PolicyDecision(
            policy_id="max-added-lines",
            passed=added <= self.max_added_lines,
            reason=f"Added line count is {added}/{self.max_added_lines}",
            details={"added_lines": added, "limit": self.max_added_lines},
        )

    def _evaluate_required_checks(
        self, check_results: Dict[str, bool]
    ) -> PolicyDecision:
        missing = [
            check
            for check in self.required_checks
            if check not in check_results or not check_results[check]
        ]
        return PolicyDecision(
            policy_id="required-checks",
            passed=not missing,
            reason=(
                "All required checks passed"
                if not missing
                else "Required checks are missing or failed"
            ),
            details={"missing_or_failed": missing, "checks": check_results},
        )


def decisions_passed(decisions: Iterable[PolicyDecision]) -> bool:
    return all(decision.passed for decision in decisions if decision.severity != "warning")


def _matches_any(path: str, patterns: List[str]) -> bool:
    return any(fnmatch(path, pattern) for pattern in patterns)
