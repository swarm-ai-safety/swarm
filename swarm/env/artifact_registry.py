"""Shared artifact registry for emergent tool chaining.

The ``ArtifactRegistry`` is the coordination layer between handler
actions.  Handlers publish typed artifacts after completing actions;
agents declare needs for artifact kinds they require.  The registry
matches supply to demand and maintains *pressure scores* — a
measure of unsatisfied demand per artifact kind inspired by
ScienceClaw's ArtifactReactor (Wang et al., 2026).

The registry is attached to ``EnvState`` so all handlers and the
interaction finalizer can access it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from swarm.models.artifact import Artifact, ArtifactNeed


class ArtifactRegistry:
    """Shared artifact layer — agents publish outputs, declare needs.

    Thread-safety note: the simulation is single-threaded per step,
    so no locking is needed.
    """

    def __init__(self, max_needs: int = 500) -> None:
        self._artifacts: Dict[str, Artifact] = {}
        self._by_kind: Dict[str, List[str]] = defaultdict(list)
        self._needs: List[ArtifactNeed] = []
        self._max_needs = max_needs
        self._pressure: Dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, artifact: Artifact) -> None:
        """Register a newly produced artifact.

        Reduces pressure for the artifact's kind (supply met demand).
        """
        self._artifacts[artifact.artifact_id] = artifact
        self._by_kind[artifact.kind].append(artifact.artifact_id)
        self._pressure[artifact.kind] = max(
            0.0, self._pressure[artifact.kind] - 1.0
        )

    # ------------------------------------------------------------------
    # Needs / pressure
    # ------------------------------------------------------------------

    def declare_need(self, need: ArtifactNeed) -> None:
        """Record an unsatisfied input requirement.

        Increases pressure for the requested artifact kind.
        Oldest needs are evicted when the list exceeds ``max_needs``.
        """
        self._needs.append(need)
        if len(self._needs) > self._max_needs:
            self._needs = self._needs[-self._max_needs:]
        self._pressure[need.kind] += 1.0

    def pressure_scores(self) -> Dict[str, float]:
        """Return current demand pressure per artifact kind.

        Higher values mean more unsatisfied needs.  Agents can use
        this to prioritize actions that produce high-demand artifacts.
        """
        return dict(self._pressure)

    def top_pressure(self, n: int = 5) -> List[tuple[str, float]]:
        """Return the *n* artifact kinds with highest unmet demand."""
        return sorted(
            self._pressure.items(), key=lambda kv: kv[1], reverse=True
        )[:n]

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(
        self,
        need: ArtifactNeed,
        current_step: int,
    ) -> List[Artifact]:
        """Find artifacts satisfying a need (kind + quality + freshness).

        Returns artifacts ordered by quality (``p_at_production`` descending).
        """
        candidates: List[Artifact] = []
        for aid in self._by_kind.get(need.kind, []):
            art = self._artifacts.get(aid)
            if art is None:
                continue  # stale index entry
            age = current_step - art.step
            if age > need.max_age_steps:
                continue
            if art.p_at_production < need.min_p:
                continue
            candidates.append(art)
        candidates.sort(key=lambda a: a.p_at_production, reverse=True)
        return candidates

    def fresh_artifact_dicts(
        self,
        current_step: int,
        max_age_steps: int = 50,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Pre-compute fresh artifacts grouped by producer_id.

        Call once per step, then use ``match_for_agent`` with the
        cached result.  Avoids O(agents × artifacts) per step.

        Returns ``{producer_id: [artifact_dict, ...]}``.
        """
        by_producer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for art in self._artifacts.values():
            if current_step - art.step > max_age_steps:
                continue
            by_producer[art.producer_id].append(art.to_dict())
        return by_producer

    def match_for_agent(
        self,
        agent_id: str,
        current_step: int,
        max_age_steps: int = 50,
        *,
        _cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return all fresh artifacts visible to *agent_id*.

        Used by ``ObservationBuilder`` to populate the agent's
        artifact view.  Returns dicts (not Artifact instances) for
        consistency with other observation fields.

        Pass *_cache* (from ``fresh_artifact_dicts``) to avoid
        re-scanning all artifacts for every agent.
        """
        if _cache is not None:
            results: List[Dict[str, Any]] = []
            for producer_id, arts in _cache.items():
                if producer_id != agent_id:
                    results.extend(arts)
            return results

        # Uncached fallback — full scan
        results = []
        for art in self._artifacts.values():
            if current_step - art.step > max_age_steps:
                continue
            if art.producer_id == agent_id:
                continue
            results.append(art.to_dict())
        return results

    # ------------------------------------------------------------------
    # Consumption (causal linking)
    # ------------------------------------------------------------------

    def consume(self, artifact_id: str, interaction_id: str) -> Optional[str]:
        """Record that *interaction_id* consumed an artifact.

        Returns the ``interaction_id`` of the producing interaction
        (for wiring ``causal_parents``), or ``None`` if the artifact
        is not found.
        """
        art: Artifact | None = self._artifacts.get(artifact_id)
        if art is None:
            return None
        art.consumed_by.append(interaction_id)
        return art.interaction_id

    def get(self, artifact_id: str) -> Optional[Artifact]:
        """Look up an artifact by ID."""
        return self._artifacts.get(artifact_id)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def gc(self, current_step: int, max_age_steps: int = 100) -> int:
        """Remove artifacts older than *max_age_steps*.

        Returns the number of artifacts removed.  Called at epoch
        boundaries to prevent unbounded memory growth.
        """
        stale_ids = {
            aid
            for aid, art in self._artifacts.items()
            if current_step - art.step > max_age_steps
        }
        if not stale_ids:
            return 0

        for aid in stale_ids:
            self._artifacts.pop(aid)

        # Rebuild kind index in one pass (avoids O(n²) list.remove)
        for kind in self._by_kind:
            self._by_kind[kind] = [
                aid for aid in self._by_kind[kind] if aid not in stale_ids
            ]
        return len(stale_ids)

    def __len__(self) -> int:
        return len(self._artifacts)

    def all_artifacts(self) -> List[Artifact]:
        """Return all artifacts (for snapshot / testing)."""
        return list(self._artifacts.values())
