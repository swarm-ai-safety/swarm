"""Governance layer for the skill library system.

Controls:
- Who can write to shared libraries (reputation gating)
- Skill poisoning detection (skills that degrade swarm performance)
- Audit trails for skill provenance
- Version control / rollback for shared skills
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from swarm.skills.library import SkillLibrary
from swarm.skills.model import Skill


@dataclass
class SkillGovernanceConfig:
    """Configuration for skill governance."""

    # Reputation gates
    min_reputation_to_propose: float = 1.0
    min_reputation_to_approve: float = 3.0

    # Poisoning detection
    poisoning_detection_enabled: bool = True
    poisoning_payoff_threshold: float = -0.5  # Avg payoff below this = suspect
    poisoning_min_invocations: int = 5  # Need enough data
    poisoning_max_failure_rate: float = 0.7  # Failure rate above this = suspect

    # Audit
    audit_enabled: bool = True
    max_audit_log_size: int = 1000

    # Rollback
    rollback_enabled: bool = True
    max_versions_kept: int = 5


@dataclass
class SkillAuditEntry:
    """Record of a skill governance event."""

    event_type: str  # "proposed", "approved", "rejected", "poisoning_detected", "rolled_back"
    skill_id: str = ""
    agent_id: str = ""
    epoch: int = 0
    details: Dict = field(default_factory=dict)


@dataclass
class PoisoningReport:
    """Report of a detected skill poisoning attempt."""

    skill_id: str
    skill_name: str
    created_by: str
    failure_rate: float
    avg_payoff: float
    invocations: int
    affected_agents: Set[str] = field(default_factory=set)


class SkillGovernanceEngine:
    """Enforces governance rules on skill libraries.

    Responsibilities:
    - Gate writes to shared library based on reputation
    - Detect and quarantine poisoned skills
    - Maintain audit trail of skill lifecycle events
    - Support rollback of damaging skills
    """

    def __init__(
        self,
        config: Optional[SkillGovernanceConfig] = None,
    ):
        self.config = config or SkillGovernanceConfig()
        self._audit_log: List[SkillAuditEntry] = []
        self._quarantined_skill_ids: Set[str] = set()
        self._skill_versions: Dict[str, List[Skill]] = {}  # skill_id -> version history

    def check_write_permission(
        self,
        agent_id: str,
        agent_reputation: float,
        library: SkillLibrary,
    ) -> bool:
        """Check if agent has permission to write to the library.

        Reputation gating applies to all shared library modes
        (SHARED_GATED and COMMUNICATION).
        """
        if library.owner_id != "shared":
            return True  # Private libraries are unrestricted

        return agent_reputation >= self.config.min_reputation_to_propose

    def propose_skill(
        self,
        skill: Skill,
        author_reputation: float,
        library: SkillLibrary,
        epoch: int = 0,
    ) -> bool:
        """Propose a new skill to the shared library.

        For high-reputation authors, skills are auto-approved.
        Returns True if skill was accepted.
        """
        # Check basic permission
        if not self.check_write_permission(skill.created_by, author_reputation, library):
            self._log_audit("rejected", skill.skill_id, skill.created_by, epoch, {
                "reason": "insufficient_reputation",
                "reputation": author_reputation,
            })
            return False

        # Auto-approve for high-reputation authors
        if author_reputation >= self.config.min_reputation_to_approve:
            added = library.add_skill(skill, author_reputation)
            if added:
                self._log_audit("approved", skill.skill_id, skill.created_by, epoch, {
                    "auto_approved": True,
                })
                self._save_version(skill)
            return added

        # For lower reputation, add as proposed (pending review in real system)
        # In simulation, we still add but track it
        added = library.add_skill(skill, author_reputation)
        if added:
            self._log_audit("proposed", skill.skill_id, skill.created_by, epoch, {
                "reputation": author_reputation,
            })
            self._save_version(skill)
        return added

    def detect_poisoning(
        self,
        library: SkillLibrary,
        invocation_agents: Optional[Dict[str, Set[str]]] = None,
    ) -> List[PoisoningReport]:
        """Scan library for potentially poisoned skills.

        A skill is considered poisoned if it consistently leads to
        negative outcomes across multiple invocations.

        Args:
            library: Library to scan.
            invocation_agents: skill_id -> set of agent_ids that used it.

        Returns:
            List of poisoning reports for suspect skills.
        """
        if not self.config.poisoning_detection_enabled:
            return []

        reports = []
        for skill in library.all_skills:
            if skill.skill_id in self._quarantined_skill_ids:
                continue

            perf = library.get_performance(skill.skill_id)
            if perf is None or perf.invocations < self.config.poisoning_min_invocations:
                continue

            is_poisoned = (
                perf.avg_payoff < self.config.poisoning_payoff_threshold
                or (perf.failures / max(1, perf.invocations)) > self.config.poisoning_max_failure_rate
            )

            if is_poisoned:
                affected = set()
                if invocation_agents and skill.skill_id in invocation_agents:
                    affected = invocation_agents[skill.skill_id]

                reports.append(PoisoningReport(
                    skill_id=skill.skill_id,
                    skill_name=skill.name,
                    created_by=skill.created_by,
                    failure_rate=perf.failures / max(1, perf.invocations),
                    avg_payoff=perf.avg_payoff,
                    invocations=perf.invocations,
                    affected_agents=affected,
                ))

        return reports

    def quarantine_skill(
        self,
        skill_id: str,
        library: SkillLibrary,
        epoch: int = 0,
        reason: str = "poisoning_detected",
    ) -> bool:
        """Quarantine a skill (remove from library, mark as blocked)."""
        self._quarantined_skill_ids.add(skill_id)
        removed = library.remove_skill(skill_id)

        self._log_audit("quarantined", skill_id, "", epoch, {
            "reason": reason,
        })

        return removed

    def rollback_skill(
        self,
        skill_id: str,
        library: SkillLibrary,
        target_version: int = -1,
        epoch: int = 0,
    ) -> bool:
        """Rollback a skill to a previous version.

        Args:
            skill_id: Skill to rollback.
            library: Library containing the skill.
            target_version: Version to rollback to (-1 = previous).
            epoch: Current epoch for audit.
        """
        if not self.config.rollback_enabled:
            return False

        versions = self._skill_versions.get(skill_id, [])
        if len(versions) < 2:
            return False

        # Remove current version
        library.remove_skill(skill_id)

        # Restore target version
        idx = target_version if target_version >= 0 else -2
        if abs(idx) > len(versions):
            return False

        restored = versions[idx]
        library.add_skill(restored)

        self._log_audit("rolled_back", skill_id, "", epoch, {
            "restored_version": restored.version,
        })

        return True

    @property
    def audit_log(self) -> List[SkillAuditEntry]:
        """Access the audit log."""
        return self._audit_log

    @property
    def quarantined_ids(self) -> Set[str]:
        """Skills currently quarantined."""
        return self._quarantined_skill_ids.copy()

    def _log_audit(
        self,
        event_type: str,
        skill_id: str,
        agent_id: str,
        epoch: int,
        details: Dict,
    ) -> None:
        """Append to audit log."""
        if not self.config.audit_enabled:
            return

        entry = SkillAuditEntry(
            event_type=event_type,
            skill_id=skill_id,
            agent_id=agent_id,
            epoch=epoch,
            details=details,
        )
        self._audit_log.append(entry)

        # Bound the log
        if len(self._audit_log) > self.config.max_audit_log_size:
            self._audit_log = self._audit_log[-self.config.max_audit_log_size:]

    def _save_version(self, skill: Skill) -> None:
        """Save a skill version for rollback."""
        if not self.config.rollback_enabled:
            return

        if skill.skill_id not in self._skill_versions:
            self._skill_versions[skill.skill_id] = []

        self._skill_versions[skill.skill_id].append(copy.deepcopy(skill))

        # Trim old versions
        max_v = self.config.max_versions_kept
        if len(self._skill_versions[skill.skill_id]) > max_v:
            self._skill_versions[skill.skill_id] = (
                self._skill_versions[skill.skill_id][-max_v:]
            )
