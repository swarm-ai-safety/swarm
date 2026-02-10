"""Metrics for tracking skill evolution across the swarm.

Measures:
- Skill diversity (entropy of skill types across agents)
- Transfer rate (how often shared skills propagate)
- Convergence (do agents converge on the same skill set?)
- Poisoning rate (fraction of quarantined skills)
- Individual vs. swarm skill effectiveness
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from swarm.skills.library import SkillLibrary
from swarm.skills.model import SkillType


@dataclass
class SkillEvolutionMetrics:
    """Metrics for a single epoch of skill evolution."""

    epoch: int = 0

    # Library size
    total_skills: int = 0
    strategy_count: int = 0
    lesson_count: int = 0
    composite_count: int = 0

    # Extraction
    skills_extracted: int = 0
    skills_pruned: int = 0
    skills_composed: int = 0

    # Invocations
    total_invocations: int = 0
    avg_invocation_payoff: float = 0.0

    # Diversity (entropy of skill domains across agents)
    skill_diversity: float = 0.0

    # Convergence (Jaccard similarity of agent skill sets)
    skill_convergence: float = 0.0

    # Effectiveness
    avg_effectiveness: float = 0.0
    top_skill_effectiveness: float = 0.0

    # Governance
    skills_quarantined: int = 0
    poisoning_rate: float = 0.0

    # Per-agent metrics
    per_agent_skill_count: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "epoch": self.epoch,
            "total_skills": self.total_skills,
            "strategy_count": self.strategy_count,
            "lesson_count": self.lesson_count,
            "composite_count": self.composite_count,
            "skills_extracted": self.skills_extracted,
            "skills_pruned": self.skills_pruned,
            "skills_composed": self.skills_composed,
            "total_invocations": self.total_invocations,
            "avg_invocation_payoff": self.avg_invocation_payoff,
            "skill_diversity": self.skill_diversity,
            "skill_convergence": self.skill_convergence,
            "avg_effectiveness": self.avg_effectiveness,
            "top_skill_effectiveness": self.top_skill_effectiveness,
            "skills_quarantined": self.skills_quarantined,
            "poisoning_rate": self.poisoning_rate,
            "per_agent_skill_count": self.per_agent_skill_count,
        }


class SkillMetricsCollector:
    """Collects and computes skill evolution metrics each epoch."""

    def __init__(self) -> None:
        self._epoch_history: List[SkillEvolutionMetrics] = []
        self._extractions_this_epoch: int = 0
        self._pruned_this_epoch: int = 0
        self._composed_this_epoch: int = 0
        self._quarantined_this_epoch: int = 0
        self._invocations_this_epoch: List[float] = []  # payoffs

    def on_epoch_start(self) -> None:
        """Reset per-epoch counters."""
        self._extractions_this_epoch = 0
        self._pruned_this_epoch = 0
        self._composed_this_epoch = 0
        self._quarantined_this_epoch = 0
        self._invocations_this_epoch = []

    def record_extraction(self) -> None:
        self._extractions_this_epoch += 1

    def record_prune(self, count: int = 1) -> None:
        self._pruned_this_epoch += count

    def record_composition(self) -> None:
        self._composed_this_epoch += 1

    def record_quarantine(self) -> None:
        self._quarantined_this_epoch += 1

    def record_invocation(self, payoff: float) -> None:
        self._invocations_this_epoch.append(payoff)

    def compute_epoch_metrics(
        self,
        epoch: int,
        agent_libraries: Dict[str, SkillLibrary],
        shared_library: Optional[SkillLibrary] = None,
    ) -> SkillEvolutionMetrics:
        """Compute metrics for the current epoch.

        Args:
            epoch: Current epoch number.
            agent_libraries: Per-agent skill libraries.
            shared_library: Optional shared library.
        """
        metrics = SkillEvolutionMetrics(epoch=epoch)

        # Aggregate skill counts
        all_skills = []
        for agent_id, lib in agent_libraries.items():
            metrics.per_agent_skill_count[agent_id] = lib.size
            all_skills.extend(lib.all_skills)

        if shared_library:
            all_skills.extend(shared_library.all_skills)

        metrics.total_skills = len(all_skills)
        metrics.strategy_count = sum(
            1 for s in all_skills if s.skill_type == SkillType.STRATEGY
        )
        metrics.lesson_count = sum(
            1 for s in all_skills if s.skill_type == SkillType.LESSON
        )
        metrics.composite_count = sum(
            1 for s in all_skills if s.skill_type == SkillType.COMPOSITE
        )

        # Extraction / pruning / composition counts
        metrics.skills_extracted = self._extractions_this_epoch
        metrics.skills_pruned = self._pruned_this_epoch
        metrics.skills_composed = self._composed_this_epoch
        metrics.skills_quarantined = self._quarantined_this_epoch

        # Invocation metrics
        metrics.total_invocations = len(self._invocations_this_epoch)
        if self._invocations_this_epoch:
            metrics.avg_invocation_payoff = float(
                np.mean(self._invocations_this_epoch)
            )

        # Effectiveness
        effectiveness_scores = []
        for lib in agent_libraries.values():
            for skill in lib.all_skills:
                perf = lib.get_performance(skill.skill_id)
                if perf and perf.invocations >= 2:
                    effectiveness_scores.append(perf.effectiveness)

        if effectiveness_scores:
            metrics.avg_effectiveness = float(np.mean(effectiveness_scores))
            metrics.top_skill_effectiveness = float(max(effectiveness_scores))

        # Diversity (entropy of skill domain distribution)
        metrics.skill_diversity = self._compute_diversity(agent_libraries)

        # Convergence (Jaccard similarity of skill names across agents)
        metrics.skill_convergence = self._compute_convergence(agent_libraries)

        # Poisoning rate
        if metrics.total_skills > 0:
            metrics.poisoning_rate = (
                metrics.skills_quarantined / max(1, metrics.total_skills)
            )

        self._epoch_history.append(metrics)
        return metrics

    @property
    def history(self) -> List[SkillEvolutionMetrics]:
        """Full epoch history."""
        return self._epoch_history

    @staticmethod
    def _compute_diversity(
        agent_libraries: Dict[str, SkillLibrary],
    ) -> float:
        """Compute Shannon entropy of skill domain distribution."""
        domain_counts: Counter = Counter()
        for lib in agent_libraries.values():
            for skill in lib.all_skills:
                domain_counts[skill.domain.value] += 1

        total = sum(domain_counts.values())
        if total == 0:
            return 0.0

        probs = [c / total for c in domain_counts.values()]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)

        # Normalize by max possible entropy
        max_entropy = np.log2(max(len(domain_counts), 1))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    @staticmethod
    def _compute_convergence(
        agent_libraries: Dict[str, SkillLibrary],
    ) -> float:
        """Compute average pairwise Jaccard similarity of agent skill names."""
        if len(agent_libraries) < 2:
            return 0.0

        agent_skill_names: List[Set[str]] = []
        for lib in agent_libraries.values():
            names = {s.name for s in lib.all_skills}
            agent_skill_names.append(names)

        similarities = []
        for i in range(len(agent_skill_names)):
            for j in range(i + 1, len(agent_skill_names)):
                a, b = agent_skill_names[i], agent_skill_names[j]
                if a or b:
                    jaccard = len(a & b) / len(a | b) if (a | b) else 0.0
                    similarities.append(jaccard)

        return float(np.mean(similarities)) if similarities else 0.0
