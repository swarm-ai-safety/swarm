"""Run metadata envelope (run.yaml) for autoresearch outputs.

Emits a compliant run.yaml that can be consumed by the swarm-artifacts
synthesis pipeline (generate-note.py, claim-lifecycle.py).
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"], capture_output=True
        )
        return result.returncode != 0
    except Exception:
        return True


@dataclass
class RunEnvelope:
    """Metadata envelope for an autoresearch run."""

    run_id: str
    scenario_ref: str
    hypothesis: str
    experiment_type: str = "autoresearch"
    seeds: list[int] = field(default_factory=list)
    total_iterations: int = 0
    accepted_iterations: int = 0
    primary_metric: str = ""
    primary_result: str = ""
    baseline_value: float = 0.0
    best_value: float = 0.0
    tags: list[str] = field(default_factory=list)
    significant_findings: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "run_id": self.run_id,
            "slug": self.run_id,
            "created_utc": now,
            "provenance": {
                "swarm_version": "0.1.0",
                "git_sha": _git_sha(),
                "git_dirty": _git_dirty(),
                "python_version": platform.python_version(),
                "platform": platform.system(),
            },
            "experiment": {
                "type": self.experiment_type,
                "hypothesis": self.hypothesis,
                "scenario_ref": self.scenario_ref,
                "seeds": self.seeds,
                "total_iterations": self.total_iterations,
                "accepted_iterations": self.accepted_iterations,
            },
            "results": {
                "status": "completed",
                "primary_metric": self.primary_metric,
                "primary_result": self.primary_result,
                "baseline_value": self.baseline_value,
                "best_value": self.best_value,
                "significant_findings": self.significant_findings,
            },
            "artifacts": self.artifacts,
            "tags": self.tags,
        }


def write_run_yaml(envelope: RunEnvelope, output_dir: str | Path) -> Path:
    """Write a run.yaml envelope to the output directory.

    Returns the path to the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run.yaml"
    path.write_text(
        yaml.dump(envelope.to_dict(), default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return path
