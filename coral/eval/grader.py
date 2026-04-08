"""CORAL grader for the SWARM simulation framework.

Scores agent contributions by running the test suite, linter, and type checker.
Each check contributes to a composite score (0.0 - 1.0):
  - pytest pass rate:  60% weight
  - ruff lint:         20% weight
  - mypy type check:   20% weight
"""

from __future__ import annotations

import subprocess


class Grader:
    """CORAL-compatible grader that evaluates swarm codebase quality."""

    def __init__(self, codebase_path: str, **kwargs):
        self.codebase_path = codebase_path

    # ------------------------------------------------------------------
    # Public API (called by CORAL)
    # ------------------------------------------------------------------

    def evaluate(self) -> float:
        """Run all checks and return a composite score in [0.0, 1.0]."""
        test_score = self._run_tests()
        lint_score = self._run_lint()
        type_score = self._run_typecheck()

        composite = 0.60 * test_score + 0.20 * lint_score + 0.20 * type_score
        return round(composite, 4)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _run_tests(self) -> float:
        """Run pytest and return fraction of tests passed."""
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            cwd=self.codebase_path,
            timeout=240,
        )
        return self._parse_pytest_output(result.stdout, result.returncode)

    def _run_lint(self) -> float:
        """Run ruff and return 1.0 if clean, else penalize per error."""
        result = subprocess.run(
            ["python", "-m", "ruff", "check", "swarm/", "tests/", "--quiet"],
            capture_output=True,
            text=True,
            cwd=self.codebase_path,
            timeout=60,
        )
        if result.returncode == 0:
            return 1.0
        error_count = len(
            [line for line in result.stdout.splitlines() if line.strip()]
        )
        # Each lint error reduces score; cap at 0.0
        return max(0.0, 1.0 - error_count * 0.02)

    def _run_typecheck(self) -> float:
        """Run mypy and return 1.0 if clean, else penalize per error."""
        result = subprocess.run(
            ["python", "-m", "mypy", "swarm/", "--no-error-summary"],
            capture_output=True,
            text=True,
            cwd=self.codebase_path,
            timeout=120,
        )
        if result.returncode == 0:
            return 1.0
        error_lines = [
            line for line in result.stdout.splitlines() if ": error:" in line
        ]
        return max(0.0, 1.0 - len(error_lines) * 0.01)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pytest_output(stdout: str, returncode: int) -> float:
        """Extract pass/fail counts from pytest output."""
        if returncode == 0:
            return 1.0

        # Look for summary line like "120 passed, 3 failed"
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if "passed" in line or "failed" in line:
                passed = 0
                failed = 0
                num = 0
                for token in line.replace(",", "").split():
                    if token.isdigit():
                        num = int(token)
                    elif token == "passed":
                        passed = num
                    elif token == "failed":
                        failed = num
                total = passed + failed
                if total > 0:
                    return passed / total
        # If we can't parse, returncode != 0 means something broke
        return 0.0
