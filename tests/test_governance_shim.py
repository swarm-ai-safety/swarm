"""Pytest wrapper for the governance_shim.sh bash tests.

Ensures the shell-based hook tests run as part of the standard pytest suite.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIM_TEST = REPO_ROOT / "tests" / "test_governance_shim.sh"


@pytest.mark.skipif(not SHIM_TEST.exists(), reason="test_governance_shim.sh not found")
def test_governance_shim_bash():
    """Run the bash test suite for governance_shim.sh."""
    result = subprocess.run(
        ["bash", str(SHIM_TEST)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        # Show full output on failure for easy debugging
        pytest.fail(
            f"governance_shim.sh tests failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
