"""CORAL grader for the SWARM simulation framework.

Scores agent contributions by running the test suite, linter, and type checker.
Each check contributes to a composite score (0.0 - 1.0):
  - pytest pass rate:  60% weight
  - ruff lint:         20% weight
  - mypy type check:   20% weight
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class _ProcResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    spawn_error: bool = False


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
        result = self._safe_run(
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
            timeout=240,
        )
        if result.timed_out or result.spawn_error:
            return 0.0
        return self._parse_pytest_output(result.stdout, result.returncode)

    def _run_lint(self) -> float:
        """Run ruff and return 1.0 if clean, else penalize per error."""
        result = self._safe_run(
            ["python", "-m", "ruff", "check", "swarm/", "tests/", "--quiet"],
            timeout=60,
        )
        if result.timed_out or result.spawn_error:
            return 0.0
        if result.returncode == 0:
            return 1.0
        # Count diagnostics from both streams; ruff may write to stderr on
        # invocation errors, which should not be scored as a clean run.
        combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
        error_count = len([line for line in combined.splitlines() if line.strip()])
        if error_count == 0:
            # Non-zero exit with no parseable output = treat as failure.
            return 0.0
        return max(0.0, 1.0 - error_count * 0.02)

    def _run_typecheck(self) -> float:
        """Run mypy and return 1.0 if clean, else penalize per error."""
        result = self._safe_run(
            ["python", "-m", "mypy", "swarm/", "--no-error-summary"],
            timeout=120,
        )
        if result.timed_out or result.spawn_error:
            return 0.0
        if result.returncode == 0:
            return 1.0
        combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
        error_lines = [line for line in combined.splitlines() if ": error:" in line]
        if not error_lines:
            # Non-zero exit with no parseable "error:" lines = treat as failure.
            return 0.0
        return max(0.0, 1.0 - len(error_lines) * 0.01)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_run(self, cmd: list[str], timeout: int) -> _ProcResult:
        """Run a subprocess without letting timeouts/OS errors crash grading."""
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.codebase_path,
                timeout=timeout,
            )
            return _ProcResult(
                returncode=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
            )
        except subprocess.TimeoutExpired as exc:
            return _ProcResult(
                returncode=-1,
                stdout=(exc.stdout or b"").decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or ""),
                stderr=(exc.stderr or b"").decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or ""),
                timed_out=True,
            )
        except (OSError, FileNotFoundError):
            return _ProcResult(returncode=-1, stdout="", stderr="", spawn_error=True)

    @staticmethod
    def _parse_pytest_output(stdout: str, returncode: int) -> float:
        """Extract pass/fail/error counts from pytest output.

        Includes ``errors`` (collection/runtime failures) in the denominator so
        broken runs cannot score as all-passed.
        """
        if returncode == 0:
            return 1.0

        # Look for summary line like "120 passed, 3 failed, 1 error"
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not any(tok in line for tok in ("passed", "failed", "error")):
                continue
            passed = 0
            failed = 0
            errors = 0
            num = 0
            for token in line.replace(",", "").split():
                if token.isdigit():
                    num = int(token)
                elif token == "passed":
                    passed = num
                elif token == "failed":
                    failed = num
                elif token in ("error", "errors"):
                    errors = num
            total = passed + failed + errors
            if total > 0:
                return passed / total
        # If we can't parse, returncode != 0 means something broke.
        return 0.0
