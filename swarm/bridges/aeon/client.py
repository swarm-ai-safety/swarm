"""Aeon ledger client.

Reads Aeon's append-only agent-first JSONL ledgers from a local checkout and,
optionally, GitHub Actions skill-run conclusions via the ``gh`` CLI. Pure
filesystem / subprocess access — no network transport, no async runtime.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from swarm.bridges.aeon.config import AeonConfig

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank/corrupt lines."""
    if not path.exists():
        logger.debug("Ledger not found: %s", path)
        return []
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


class AeonClient:
    """Reads Aeon agent-first ledgers and (optionally) GH Actions skill runs."""

    def __init__(self, config: AeonConfig) -> None:
        self._config = config

    # -- ledger sources -----------------------------------------------------

    def fetch_tasks(self) -> list[dict[str, Any]]:
        """Read all task records, applying the repo filter if configured."""
        tasks = _read_jsonl(self._config.tasks_path)
        return [t for t in tasks if self._repo_ok(t.get("repo", ""))]

    def fetch_proofs(self) -> list[dict[str, Any]]:
        """Read all proof-bundle records."""
        return _read_jsonl(self._config.proofs_path)

    def fetch_reviews(self) -> list[dict[str, Any]]:
        """Read all review-decision records."""
        return _read_jsonl(self._config.reviews_path)

    def _repo_ok(self, repo: str) -> bool:
        return not self._config.repos or repo in self._config.repos

    # -- optional GitHub Actions skill runs ---------------------------------

    def fetch_skill_runs(self) -> list[dict[str, Any]]:
        """Fetch completed GitHub Actions runs via the `gh` CLI.

        Returns an empty list (and logs a warning) if `gh` is unavailable or
        fails, so the bridge degrades gracefully when run off-box.
        """
        if not self._config.include_skill_runs:
            return []
        cmd = [
            "gh", "run", "list",
            "--limit", str(self._config.skill_runs_limit),
            "--json", "databaseId,name,workflowName,status,conclusion,headSha,createdAt,event",
        ]
        if self._config.skill_runs_repo:
            cmd += ["--repo", self._config.skill_runs_repo]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, check=True
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            logger.warning("Skill-run fetch via gh failed: %s", exc)
            return []
        try:
            runs = json.loads(out.stdout or "[]")
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse gh run list output: %s", exc)
            return []
        return [r for r in runs if r.get("status") == "completed"]
