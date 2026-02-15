"""Subprocess executor for AgentLaboratory refinement runs.

Mirrors the LiveSWE bridge pattern: spawns AgentLab as a subprocess,
captures output, and locates the resulting checkpoint for ingestion.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from swarm.bridges.agent_lab.refinement import RefinementConfig

logger = logging.getLogger(__name__)


class AgentLabRunner:
    """Runs AgentLaboratory as a subprocess for study refinement.

    Example::

        runner = AgentLabRunner(RefinementConfig(depth="lite"))
        returncode, stdout, stderr, duration = runner.run_refinement(
            yaml_path="/tmp/refinement.yaml",
            work_dir="/tmp/agent_lab_work",
        )
    """

    def __init__(self, config: RefinementConfig) -> None:
        self._config = config

    def run_refinement(
        self,
        yaml_path: str,
        work_dir: str,
    ) -> Tuple[int, str, str, float]:
        """Spawn AgentLab as a subprocess to run a refinement.

        Args:
            yaml_path: Path to the generated AgentLab YAML config.
            work_dir: Working directory for AgentLab output.

        Returns:
            Tuple of (returncode, stdout, stderr, duration_seconds).

        Raises:
            FileNotFoundError: If agent_lab_path or yaml_path doesn't exist.
            EnvironmentError: If OPENAI_API_KEY is not set.
        """
        # Validate prerequisites
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is required for "
                "AgentLab refinement runs."
            )

        lab_path = Path(self._config.agent_lab_path).resolve()
        if not lab_path.exists():
            raise FileNotFoundError(
                f"AgentLaboratory not found at {lab_path}. "
                "Set agent_lab_path in RefinementConfig."
            )

        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"YAML config not found: {yaml_path}")

        # Ensure work directory exists
        Path(work_dir).mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "ai_lab_repo.py",
            "--yaml-location",
            str(yaml_file),
        ]

        logger.info(
            "Launching AgentLab refinement: %s (cwd=%s, timeout=%ds)",
            " ".join(cmd),
            lab_path,
            int(self._config.timeout_seconds),
        )

        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(lab_path),
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
                env={**os.environ, "AGENT_LAB_WORK_DIR": work_dir},
            )
            duration = time.monotonic() - start
            return (result.returncode, result.stdout, result.stderr, duration)
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            logger.warning(
                "AgentLab refinement timed out after %.1fs", duration
            )
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return (-1, stdout, stderr, duration)

    def find_checkpoint(self, work_dir: str) -> Optional[str]:
        """Locate the most recent Paper*.pkl checkpoint in work_dir.

        Args:
            work_dir: Directory to search for checkpoints.

        Returns:
            Path to the most recent checkpoint, or None if not found.
        """
        work_path = Path(work_dir)
        # AgentLab writes checkpoints to state_saves/ within its work dir
        search_dirs = [
            work_path / "state_saves",
            work_path,
        ]

        candidates: list[Path] = []
        for search_dir in search_dirs:
            if search_dir.exists():
                candidates.extend(search_dir.glob("Paper*.pkl"))

        if not candidates:
            return None

        # Return most recently modified
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(latest)
