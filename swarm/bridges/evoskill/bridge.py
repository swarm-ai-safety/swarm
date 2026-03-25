"""Top-level EvoSkill–SWARM bridge orchestrator.

``EvoSkillBridge`` is the primary entry point for running EvoSkill's
automated skill discovery loop under SWARM's governance regimes.

Workflow:

1. Register a SWARM evaluation task in EvoSkill's task registry.
2. Run the EvoSkill improvement loop *separately per regime*.
3. Collect frontier programs from each regime's loop.
4. Compare frontiers for behavioral divergence.
5. Feed the best discovered skills back into SWARM's governed runs.

The bridge also instruments provenance tracking so that every
discovered skill carries a Byline chain recording which governance
regime it was discovered under.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swarm.bridges.evoskill.config import EvoSkillConfig
from swarm.bridges.evoskill.frontier import FrontierComparator
from swarm.bridges.evoskill.governed_eval import EvalResult, GovernedEvalLoop
from swarm.bridges.evoskill.translator import SkillTranslator
from swarm.bridges.opensandbox.provenance import ProvenanceTracker
from swarm.skills.governance import SkillGovernanceConfig, SkillGovernanceEngine
from swarm.skills.library import SharingMode, SkillLibrary, SkillLibraryConfig
from swarm.skills.model import Skill

logger = logging.getLogger(__name__)


@dataclass
class LoopIteration:
    """Record of one EvoSkill improvement iteration."""

    iteration: int
    regime: str
    program_id: str
    benchmark_score: float
    composite_score: float
    entered_frontier: bool
    governance_delta: Dict[str, float] = field(default_factory=dict)


class EvoSkillBridge:
    """Orchestrates the EvoSkill–SWARM integration.

    Usage::

        bridge = EvoSkillBridge(config)

        # Simulate an EvoSkill loop iteration
        result = bridge.evaluate_candidate(
            program_id="frontier-branch-3",
            skill_files={"verify_first.md": "Always verify ..."},
            benchmark_score=0.78,
        )

        # After many iterations, compare regimes
        report = bridge.comparison_report()

        # Get the best skills for a specific regime
        skills = bridge.best_skills_for_regime("truthful_auction")
    """

    def __init__(
        self,
        config: Optional[EvoSkillConfig] = None,
    ) -> None:
        self._config = config or EvoSkillConfig()

        # Sub-components
        self._translator = SkillTranslator(
            default_author=self._config.skill_author_id,
        )
        self._eval_loop = GovernedEvalLoop(
            config=self._config,
            translator=self._translator,
        )
        self._comparator = FrontierComparator(
            frontier_size=self._config.frontier_size,
        )
        self._provenance = ProvenanceTracker(
            hmac_key=self._config.provenance_hmac_key,
        )
        self._governance = SkillGovernanceEngine(SkillGovernanceConfig())

        # Per-regime skill libraries for accumulating discovered skills
        self._regime_libraries: Dict[str, SkillLibrary] = {}
        for label in self._config.contract_regimes:
            self._regime_libraries[label] = SkillLibrary(
                owner_id=f"evoskill_{label}",
                config=SkillLibraryConfig(
                    sharing_mode=SharingMode.SHARED_GATED,
                    min_reputation_to_write=0.0,  # EvoSkill skills are auto-admitted
                ),
            )

        # History
        self._iterations: List[LoopIteration] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def evaluate_candidate(
        self,
        program_id: str,
        skill_files: Dict[str, str],
        benchmark_score: float,
        iteration: int = 0,
        seed: Optional[int] = None,
    ) -> Dict[str, EvalResult]:
        """Evaluate a candidate program under all governance regimes.

        This is the main entry point called after each EvoSkill
        proposer/generator cycle.

        Args:
            program_id: Git branch or hash of the candidate.
            skill_files: Dict of filename → content from the program.
            benchmark_score: Raw EvoSkill benchmark score (0–1).
            iteration: Current EvoSkill loop iteration.
            seed: RNG seed (defaults to config.seed + iteration).

        Returns:
            Dict mapping regime label → EvalResult.
        """
        effective_seed = seed if seed is not None else self._config.seed + iteration

        # Evaluate under all regimes
        results = self._eval_loop.evaluate_program_all_regimes(
            program_id=program_id,
            skill_files=skill_files,
            benchmark_score=benchmark_score,
            seed=effective_seed,
        )

        # Update frontiers and regime libraries
        for regime_label, result in results.items():
            entered = self._comparator.add_result(regime_label, result, iteration)

            # Record provenance
            prov_id = self._provenance.sign(
                sandbox_id=f"evoskill_{regime_label}",
                agent_id=self._config.skill_author_id,
                action_type="skill_evaluation",
                action_summary=(
                    f"Evaluated program {program_id} under {regime_label}: "
                    f"composite={result.composite_score:.3f}"
                ),
                content={
                    "program_id": program_id,
                    "regime": regime_label,
                    "benchmark_score": benchmark_score,
                    "composite_score": result.composite_score,
                    "governance_delta": result.governance_delta,
                },
                contract_id=regime_label,
            )
            result.provenance_ids.append(prov_id)

            # If it entered the frontier, ingest skills into the regime library
            if entered:
                skills = self._translator.ingest_batch(
                    skill_files,
                    source_branch=program_id,
                    author_id=self._config.skill_author_id,
                )
                library = self._regime_libraries[regime_label]
                for skill in skills:
                    skill.tags.add(f"regime:{regime_label}")
                    skill.tags.add(f"iteration:{iteration}")
                    library.add_skill(skill)

            self._iterations.append(LoopIteration(
                iteration=iteration,
                regime=regime_label,
                program_id=program_id,
                benchmark_score=benchmark_score,
                composite_score=result.composite_score,
                entered_frontier=entered,
                governance_delta=dict(result.governance_delta),
            ))

        return results

    def best_skills_for_regime(self, regime: str) -> List[Skill]:
        """Get all skills from programs currently on a regime's frontier.

        These are the skills that performed best under the given
        governance regime — the "winners" of the governed evolution.

        Args:
            regime: Governance regime label.

        Returns:
            List of SWARM Skills from the frontier programs.
        """
        library = self._regime_libraries.get(regime)
        if library is None:
            return []
        return library.all_skills

    def comparison_report(self) -> Dict[str, Any]:
        """Generate a comprehensive comparison report.

        Returns:
            Dict with frontiers, divergences, and iteration history.
        """
        report = self._comparator.summary_report()

        # Add per-regime skill library stats
        report["regime_libraries"] = {}
        for label, library in self._regime_libraries.items():
            report["regime_libraries"][label] = {
                "total_skills": library.size,
                "library_data": library.to_dict(),
            }

        # Add provenance stats
        report["provenance"] = self._provenance.get_stats()

        # Add iteration history
        report["iterations"] = [
            {
                "iteration": it.iteration,
                "regime": it.regime,
                "program_id": it.program_id,
                "benchmark_score": round(it.benchmark_score, 4),
                "composite_score": round(it.composite_score, 4),
                "entered_frontier": it.entered_frontier,
                "governance_delta": {
                    k: round(v, 4) for k, v in it.governance_delta.items()
                },
            }
            for it in self._iterations
        ]

        return report

    def export_frontier_skills(
        self, regime: str
    ) -> Dict[str, str]:
        """Export a regime's frontier skills as EvoSkill-format files.

        Useful for feeding SWARM-governed skills back into EvoSkill
        for the next discovery cycle.

        Args:
            regime: Governance regime label.

        Returns:
            Dict of filename → skill text content.
        """
        skills = self.best_skills_for_regime(regime)
        result: Dict[str, str] = {}
        for skill in skills:
            # Sanitize name for filename
            safe_name = (
                skill.name.lower()
                .replace(" ", "_")
                .replace("/", "_")[:50]
            )
            filename = f"{safe_name}.md"
            result[filename] = self._translator.export(skill)
        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def config(self) -> EvoSkillConfig:
        """Bridge configuration."""
        return self._config

    @property
    def comparator(self) -> FrontierComparator:
        """Access the frontier comparator."""
        return self._comparator

    @property
    def provenance(self) -> ProvenanceTracker:
        """Access the provenance tracker."""
        return self._provenance

    @property
    def iterations(self) -> List[LoopIteration]:
        """Full iteration history."""
        return list(self._iterations)
