"""Hardware root-of-trust rejection handling governance lever.

When a hardware root of trust issues a Rejection/Halt command, this lever
treats it as a first-class governance signal.  The orchestration layer:

1. Immediately freezes all unsafe action pathways.
2. Preserves task state and records the causal trace that triggered the halt.
3. Propagates a signed stop token across dependent agents so the swarm
   converges to a safe quiescent state instead of fragmenting.
4. Downgrades affected agents into constrained recovery mode (no irreversible
   actions, no new external side effects, bounded local reasoning only).
5. Safe-resume occurs only after hardware clears the condition, at which
   point agents reload the last admissible state and continue from a
   verified checkpoint.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect


class HaltReason(Enum):
    """Reason categories for hardware halt signals."""

    INTEGRITY_VIOLATION = "integrity_violation"
    ATTESTATION_FAILURE = "attestation_failure"
    TAMPER_DETECTED = "tamper_detected"
    POLICY_BREACH = "policy_breach"
    EXTERNAL_REVOCATION = "external_revocation"


class RecoveryMode(Enum):
    """Agent recovery mode after a hardware halt."""

    NORMAL = "normal"
    CONSTRAINED = "constrained"  # No irreversible actions, no external side effects
    FROZEN = "frozen"  # Completely halted


@dataclass
class HaltRecord:
    """Record of a hardware halt event."""

    halt_id: str
    timestamp: datetime
    reason: HaltReason
    source_agent_id: Optional[str]  # Agent whose action triggered the halt (if any)
    causal_trace: List[str]  # Chain of event IDs leading to the halt
    affected_agents: Set[str]  # Agents frozen by this halt
    metadata: Dict[str, Any] = field(default_factory=dict)
    cleared: bool = False
    cleared_at: Optional[datetime] = None


@dataclass
class AgentRecoveryState:
    """Tracks recovery state for an agent under hardware halt."""

    mode: RecoveryMode = RecoveryMode.NORMAL
    halt_id: Optional[str] = None
    checkpoint_epoch: Optional[int] = None
    checkpoint_step: Optional[int] = None
    recovery_steps_taken: int = 0
    task_state_preserved: Dict[str, Any] = field(default_factory=dict)


# Actions that are considered irreversible and blocked in constrained recovery
IRREVERSIBLE_ACTIONS: FrozenSet[str] = frozenset({
    "propose_interaction",
    "submit_output",
    "post_bounty",
    "accept_bid",
    "create_page",
    "edit_page",
    "moltbook_post",
    "write_memory",
    "promote_memory",
    "submit_kernel",
    "spawn_subagent",
    "awm_execute_task",
    "awm_tool_call",
})


class HardwareTrustLever(GovernanceLever):
    """Governance lever for hardware root-of-trust rejection handling.

    This lever intercepts hardware halt signals, freezes affected agents,
    propagates stop tokens to dependents, and manages constrained recovery.
    """

    def __init__(self, config: GovernanceConfig):
        super().__init__(config)
        self._halt_records: Dict[str, HaltRecord] = {}
        self._recovery_states: Dict[str, AgentRecoveryState] = {}
        self._active_halt_id: Optional[str] = None
        self._dependency_graph: Dict[str, Set[str]] = {}  # agent -> dependents

    @property
    def name(self) -> str:
        return "hardware_trust"

    def register_dependency(self, agent_id: str, dependent_id: str) -> None:
        """Register that *dependent_id* depends on *agent_id*.

        Used for stop-token propagation: when *agent_id* is halted,
        *dependent_id* will also be frozen.
        """
        if agent_id not in self._dependency_graph:
            self._dependency_graph[agent_id] = set()
        self._dependency_graph[agent_id].add(dependent_id)

    def set_dependency_graph(self, graph: Dict[str, Set[str]]) -> None:
        """Replace the entire dependency graph."""
        self._dependency_graph = {k: set(v) for k, v in graph.items()}

    # ------------------------------------------------------------------
    # Halt lifecycle
    # ------------------------------------------------------------------

    def receive_halt(
        self,
        halt_id: str,
        reason: HaltReason,
        state: EnvState,
        source_agent_id: Optional[str] = None,
        causal_trace: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LeverEffect:
        """Process a hardware halt signal.

        This is the main entry point called when the hardware root of trust
        issues a Rejection/Halt command.

        Args:
            halt_id: Unique identifier for this halt event.
            reason: Why the halt was issued.
            state: Current environment state.
            source_agent_id: Agent whose action triggered the halt (if any).
            causal_trace: Event IDs forming the causal chain.
            metadata: Additional halt metadata.

        Returns:
            LeverEffect with agents to freeze.
        """
        if not self.config.hardware_trust_enabled:
            return LeverEffect(lever_name=self.name)

        now = datetime.now()

        # Determine which agents to freeze
        agents_to_freeze: Set[str] = set()
        if source_agent_id:
            agents_to_freeze.add(source_agent_id)

        # Propagate to all dependents if propagation enabled
        if self.config.hardware_trust_propagation_enabled:
            agents_to_freeze |= self._compute_propagation_set(
                agents_to_freeze, state
            )

        # If no specific agent, freeze everyone (global halt)
        if not agents_to_freeze:
            agents_to_freeze = set(state.agents.keys())

        # Record the halt
        record = HaltRecord(
            halt_id=halt_id,
            timestamp=now,
            reason=reason,
            source_agent_id=source_agent_id,
            causal_trace=list(causal_trace or []),
            affected_agents=set(agents_to_freeze),
            metadata=metadata or {},
        )
        self._halt_records[halt_id] = record
        self._active_halt_id = halt_id

        # Set up recovery state for each affected agent
        for agent_id in agents_to_freeze:
            self._recovery_states[agent_id] = AgentRecoveryState(
                mode=RecoveryMode.CONSTRAINED,
                halt_id=halt_id,
                checkpoint_epoch=state.current_epoch,
                checkpoint_step=state.current_step,
                recovery_steps_taken=0,
            )

        return LeverEffect(
            agents_to_freeze=agents_to_freeze,
            lever_name=self.name,
            details={
                "halt_id": halt_id,
                "reason": reason.value,
                "source_agent_id": source_agent_id,
                "affected_count": len(agents_to_freeze),
                "affected_agents": sorted(agents_to_freeze),
                "causal_trace": list(causal_trace or []),
            },
        )

    def clear_halt(
        self,
        halt_id: str,
        state: EnvState,
    ) -> LeverEffect:
        """Clear a hardware halt condition, allowing safe resume.

        Called when the hardware root of trust signals the condition is
        resolved. Agents that were frozen by this halt are unfrozen and
        can resume from their last admissible checkpoint.

        Args:
            halt_id: ID of the halt to clear.
            state: Current environment state.

        Returns:
            LeverEffect with agents to unfreeze.
        """
        if not self.config.hardware_trust_enabled:
            return LeverEffect(lever_name=self.name)

        record = self._halt_records.get(halt_id)
        if record is None or record.cleared:
            return LeverEffect(lever_name=self.name)

        record.cleared = True
        record.cleared_at = datetime.now()

        agents_to_unfreeze: Set[str] = set()
        resume_details: Dict[str, Any] = {}

        for agent_id in record.affected_agents:
            recovery = self._recovery_states.get(agent_id)
            if recovery and recovery.halt_id == halt_id:
                agents_to_unfreeze.add(agent_id)
                resume_details[agent_id] = {
                    "checkpoint_epoch": recovery.checkpoint_epoch,
                    "checkpoint_step": recovery.checkpoint_step,
                    "recovery_steps_taken": recovery.recovery_steps_taken,
                }
                # Reset to normal mode
                recovery.mode = RecoveryMode.NORMAL
                recovery.halt_id = None

        if self._active_halt_id == halt_id:
            self._active_halt_id = None

        return LeverEffect(
            agents_to_unfreeze=agents_to_unfreeze,
            lever_name=self.name,
            details={
                "halt_id": halt_id,
                "cleared": True,
                "resumed_count": len(agents_to_unfreeze),
                "resume_details": resume_details,
            },
        )

    # ------------------------------------------------------------------
    # GovernanceLever hooks
    # ------------------------------------------------------------------

    def on_step(self, state: EnvState, step: int) -> LeverEffect:
        """Track recovery steps for constrained agents."""
        if not self.config.hardware_trust_enabled:
            return LeverEffect(lever_name=self.name)
        if self._active_halt_id is None:
            return LeverEffect(lever_name=self.name)

        agents_to_freeze: Set[str] = set()
        for agent_id, recovery in self._recovery_states.items():
            if recovery.mode == RecoveryMode.CONSTRAINED:
                recovery.recovery_steps_taken += 1
                # If recovery budget exhausted, escalate to full freeze
                if (
                    recovery.recovery_steps_taken
                    >= self.config.hardware_trust_recovery_max_steps
                ):
                    recovery.mode = RecoveryMode.FROZEN
                    agents_to_freeze.add(agent_id)

        if not agents_to_freeze:
            return LeverEffect(lever_name=self.name)

        return LeverEffect(
            agents_to_freeze=agents_to_freeze,
            lever_name=self.name,
            details={
                "escalated_to_frozen": sorted(agents_to_freeze),
                "reason": "recovery_budget_exhausted",
            },
        )

    def can_agent_act(self, agent_id: str, state: EnvState) -> bool:
        """Block agents that are in FROZEN recovery mode.

        Agents in CONSTRAINED mode can still act (with action filtering),
        but agents in FROZEN mode cannot act at all.
        """
        if not self.config.hardware_trust_enabled:
            return True
        recovery = self._recovery_states.get(agent_id)
        if recovery is None:
            return True
        return recovery.mode != RecoveryMode.FROZEN

    def is_action_allowed(self, agent_id: str, action_type: str) -> bool:
        """Check if a specific action is allowed for an agent.

        In CONSTRAINED recovery mode, irreversible actions are blocked.
        In NORMAL mode, all actions are allowed.

        Args:
            agent_id: Agent attempting the action.
            action_type: The action type string (e.g. "propose_interaction").

        Returns:
            True if the action is permitted.
        """
        if not self.config.hardware_trust_enabled:
            return True
        recovery = self._recovery_states.get(agent_id)
        if recovery is None or recovery.mode == RecoveryMode.NORMAL:
            return True
        if recovery.mode == RecoveryMode.FROZEN:
            return False
        # CONSTRAINED: block irreversible actions
        return action_type not in IRREVERSIBLE_ACTIONS

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def _compute_propagation_set(
        self,
        initial: Set[str],
        state: EnvState,
    ) -> Set[str]:
        """BFS over the dependency graph to find all transitive dependents."""
        visited: Set[str] = set(initial)
        queue = list(initial)
        while queue:
            agent_id = queue.pop(0)
            for dep in self._dependency_graph.get(agent_id, set()):
                if dep not in visited and dep in state.agents:
                    visited.add(dep)
                    queue.append(dep)
        return visited - initial  # Only the newly discovered dependents

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    @property
    def is_halted(self) -> bool:
        """Return True if there is an active (uncleared) halt."""
        return self._active_halt_id is not None

    def get_active_halt(self) -> Optional[HaltRecord]:
        """Return the active halt record, if any."""
        if self._active_halt_id is None:
            return None
        return self._halt_records.get(self._active_halt_id)

    def get_recovery_state(self, agent_id: str) -> Optional[AgentRecoveryState]:
        """Get the recovery state for an agent."""
        return self._recovery_states.get(agent_id)

    def get_halt_record(self, halt_id: str) -> Optional[HaltRecord]:
        """Get a specific halt record."""
        return self._halt_records.get(halt_id)

    def get_all_halt_records(self) -> List[HaltRecord]:
        """Return all halt records (including cleared ones)."""
        return list(self._halt_records.values())

    def get_constrained_agents(self) -> Set[str]:
        """Return agents currently in constrained recovery mode."""
        return {
            agent_id
            for agent_id, recovery in self._recovery_states.items()
            if recovery.mode == RecoveryMode.CONSTRAINED
        }

    def get_frozen_agents(self) -> Set[str]:
        """Return agents currently in frozen recovery mode."""
        return {
            agent_id
            for agent_id, recovery in self._recovery_states.items()
            if recovery.mode == RecoveryMode.FROZEN
        }

    def preserve_task_state(
        self, agent_id: str, task_state: Dict[str, Any]
    ) -> None:
        """Preserve task state for an agent in recovery.

        Called by the orchestration layer to snapshot in-progress work
        before the agent is fully frozen.
        """
        recovery = self._recovery_states.get(agent_id)
        if recovery is not None:
            recovery.task_state_preserved = dict(task_state)

    def get_preserved_task_state(self, agent_id: str) -> Dict[str, Any]:
        """Retrieve preserved task state for safe-resume."""
        recovery = self._recovery_states.get(agent_id)
        if recovery is None:
            return {}
        return dict(recovery.task_state_preserved)
