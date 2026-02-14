from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_illusion_delta_minimal_example_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "illusion_delta_minimal.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "illusion_delta=" in result.stdout
    assert "seed,accepted_interactions" in result.stdout
