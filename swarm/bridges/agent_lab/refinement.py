"""Data classes for the AgentLab study refinement pipeline.

Packages completed SWARM study artifacts (summary.json, sweep_results.csv)
into AgentLab's research_topic + notes inputs, and structures the output
as a RefinementResult with hypotheses, parameter suggestions, and gaps.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


@dataclass
class StudyContext:
    """Loaded context from a completed SWARM study run directory.

    Expects:
      - ``<run_dir>/summary.json``
      - ``<run_dir>/sweep_results.csv`` (optional)
    """

    run_dir: str
    summary: Dict[str, Any] = field(default_factory=dict)
    sweep_rows: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_run_dir(cls, path: str) -> StudyContext:
        """Load study context from a run directory.

        Args:
            path: Path to the run directory containing summary.json.

        Returns:
            StudyContext populated from the run directory.

        Raises:
            FileNotFoundError: If summary.json is missing.
        """
        run_path = Path(path)
        summary_path = run_path / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"summary.json not found in {path}. "
                "Expected a completed SWARM study run directory."
            )

        with open(summary_path) as f:
            summary = json.load(f)

        sweep_rows: List[Dict[str, str]] = []
        sweep_path = run_path / "sweep_results.csv"
        if sweep_path.exists():
            with open(sweep_path, newline="") as f:
                reader = csv.DictReader(f)
                sweep_rows = list(reader)

        return cls(run_dir=str(run_path), summary=summary, sweep_rows=sweep_rows)

    def to_research_topic(self) -> str:
        """Render the study as an AgentLab research topic prompt.

        Returns:
            A string describing the study for AgentLab's research pipeline.
        """
        scenario = self.summary.get("scenario", "unknown")
        title = self.summary.get("title", scenario)
        description = self.summary.get("description", "")
        seed = self.summary.get("seed", "N/A")
        epochs = self.summary.get("epochs", "N/A")

        topic = (
            f"Analyze this completed SWARM study and propose follow-up experiments.\n\n"
            f"Study: {title}\n"
            f"Scenario: {scenario}\n"
            f"Description: {description}\n"
            f"Seed: {seed}, Epochs: {epochs}\n"
        )

        # Add key metrics if present
        metrics = self.summary.get("metrics", {})
        if metrics:
            topic += "\nKey metrics:\n"
            for k, v in metrics.items():
                topic += f"  - {k}: {v}\n"

        topic += (
            "\nIdentify gaps in the experimental design, propose hypotheses "
            "for follow-up experiments, and suggest refined parameter sweeps "
            "that would strengthen the conclusions."
        )
        return topic

    def to_notes(self) -> List[str]:
        """Render significant findings as bullet notes for AgentLab.

        Returns:
            List of note strings summarizing sweep results and findings.
        """
        notes: List[str] = []

        # Add sweep result highlights
        if self.sweep_rows:
            notes.append(f"Sweep contains {len(self.sweep_rows)} parameter combinations")
            for row in self.sweep_rows[:10]:  # Cap to avoid overwhelming
                parts = []
                for k, v in row.items():
                    if k not in ("run_id", "timestamp", "seed"):
                        parts.append(f"{k}={v}")
                if parts:
                    notes.append(", ".join(parts))

        # Add findings from summary
        findings = self.summary.get("findings", [])
        if isinstance(findings, list):
            for finding in findings:
                if isinstance(finding, str):
                    notes.append(finding)
                elif isinstance(finding, dict):
                    desc = finding.get("description", finding.get("text", ""))
                    effect = finding.get("effect_size", "")
                    p_val = finding.get("p_value", "")
                    parts = [desc]
                    if effect:
                        parts.append(f"d={effect}")
                    if p_val:
                        parts.append(f"p={p_val}")
                    notes.append(", ".join(p for p in parts if p))

        # Add parameter ranges
        params = self.summary.get("parameters", self.summary.get("swept_params", {}))
        if isinstance(params, dict) and params:
            notes.append(f"Swept parameters: {', '.join(params.keys())}")

        return notes


@dataclass
class RefinementConfig:
    """Configuration for the AgentLab refinement subprocess.

    Attributes:
        agent_lab_path: Path to AgentLaboratory installation.
        depth: Refinement depth — "lite" uses minimal solver steps.
        llm_backend: LLM backend for AgentLab to use.
        cost_budget_usd: Maximum spend for this refinement run.
        timeout_seconds: Subprocess timeout.
    """

    agent_lab_path: str = "external/AgentLaboratory"
    depth: str = "lite"  # "lite" or "full"
    llm_backend: str = "o3-mini"
    cost_budget_usd: float = 10.0
    timeout_seconds: float = 1800.0

    def to_agent_lab_yaml(self, context: StudyContext) -> Dict[str, Any]:
        """Generate an AgentLab YAML config dict for this refinement.

        Args:
            context: The study context to base the refinement on.

        Returns:
            Dictionary suitable for writing as AgentLab YAML config.
        """
        if self.depth == "lite":
            mle_steps = 1
            paper_steps = 1
        else:
            mle_steps = 3
            paper_steps = 3

        return {
            "research_topic": context.to_research_topic(),
            "notes": context.to_notes(),
            "llm_backend": self.llm_backend,
            "mlesolver_max_steps": mle_steps,
            "papersolver_max_steps": paper_steps,
            "copilot_mode": False,
        }


@dataclass
class RefinementResult:
    """Structured output from an AgentLab refinement run.

    Attributes:
        success: Whether the refinement completed successfully.
        hypotheses: Follow-up hypotheses proposed by AgentLab.
        parameter_suggestions: Suggested parameter ranges for follow-up.
        gaps_identified: Gaps in the original study design.
        proposed_scenario: Suggested follow-up scenario YAML dict.
        interactions: SoftInteractions from the AgentLab bridge pipeline.
        total_cost_usd: Total LLM spend for this refinement.
        duration_seconds: Wall-clock time for the refinement subprocess.
    """

    success: bool = False
    hypotheses: List[str] = field(default_factory=list)
    parameter_suggestions: Dict[str, Any] = field(default_factory=dict)
    gaps_identified: List[str] = field(default_factory=list)
    proposed_scenario: Optional[Dict[str, Any]] = None
    interactions: List[SoftInteraction] = field(default_factory=list)
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "success": self.success,
            "hypotheses": self.hypotheses,
            "parameter_suggestions": self.parameter_suggestions,
            "gaps_identified": self.gaps_identified,
            "proposed_scenario": self.proposed_scenario,
            "interactions": [i.to_dict() for i in self.interactions],
            "total_cost_usd": self.total_cost_usd,
            "duration_seconds": self.duration_seconds,
        }
