"""Main bridge connecting OpenSandbox containers to SWARM.

OpenSandboxBridge manages the full lifecycle of contract-governed
multi-agent systems inside isolated sandbox environments:

1. Publish governance contracts
2. Screen agents and assign them to contract tiers
3. Allocate sandboxes (one per agent)
4. Route inter-agent messages via the message bus
5. Track provenance with Byline signatures
6. Detect emergent risks and apply governance interventions
7. Compute experiment metrics (sorting, adherence, etc.)

All inter-agent communication routes through the message bus —
agents never communicate directly.
"""

import logging
from typing import Any, Dict, List, Optional

from swarm.bridges.opensandbox.config import (
    CapabilityManifest,
    ContractAssignment,
    GovernanceContract,
    OpenSandboxConfig,
)
from swarm.bridges.opensandbox.events import OpenSandboxEvent, OpenSandboxEventType
from swarm.bridges.opensandbox.mapper import OpenSandboxMapper
from swarm.bridges.opensandbox.message_bus import MessageBus
from swarm.bridges.opensandbox.observer import AgentMetrics, ExperimentMetrics, Observer
from swarm.bridges.opensandbox.provenance import ProvenanceTracker
from swarm.bridges.opensandbox.screener import ScreeningProtocol
from swarm.core.proxy import ProxyComputer
from swarm.logging.event_log import EventLog
from swarm.models.events import Event, EventType
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


class OpenSandboxBridge:
    """Bridge between OpenSandbox containers and the SWARM framework.

    Lifecycle::

        bridge = OpenSandboxBridge(config)

        # 1. Publish contracts
        bridge.publish_contract(contract)

        # 2. Screen and admit agents
        assignment = bridge.screen_agent(manifest)

        # 3. Allocate sandbox
        sandbox_id = bridge.create_sandbox(assignment)

        # 4. Execute commands
        interaction = bridge.execute_command(
            agent_id, "python evaluate.py"
        )

        # 5. Send inter-agent messages
        bridge.send_message("sandbox-a", "sandbox-b", payload)

        # 6. Poll for observations
        observations = bridge.poll()

        # 7. Governance intervention (automatic or manual)
        bridge.isolate("agent-id", reason="contract_violation")

        # 8. Measure
        metrics = bridge.compute_experiment_metrics("exp-001")

        # 9. Cleanup
        bridge.shutdown()
    """

    def __init__(
        self,
        config: Optional[OpenSandboxConfig] = None,
        event_log: Optional[EventLog] = None,
    ) -> None:
        self._config = config or OpenSandboxConfig()
        self._event_log = event_log

        # Subsystems
        self._screener = ScreeningProtocol(self._config)
        self._message_bus = MessageBus(
            max_pending=self._config.message_bus_max_pending,
            require_provenance=self._config.provenance_enabled,
        )
        self._provenance = ProvenanceTracker(
            enabled=self._config.provenance_enabled,
        )
        self._mapper = OpenSandboxMapper(
            proxy=ProxyComputer(sigmoid_k=self._config.proxy_sigmoid_k),
        )
        self._observer = Observer(
            risk_threshold=self._config.risk_threshold,
        )

        # State
        self._interactions: List[SoftInteraction] = []
        self._events: List[OpenSandboxEvent] = []
        self._sandboxes: Dict[str, Dict[str, Any]] = {}  # sandbox_id -> info
        self._agent_sandboxes: Dict[str, str] = {}  # agent_id -> sandbox_id
        self._assignments: Dict[str, ContractAssignment] = {}  # agent_id -> assignment

    # ------------------------------------------------------------------
    # Contract management
    # ------------------------------------------------------------------

    def publish_contract(self, contract: GovernanceContract) -> None:
        """Publish a governance contract to the registry.

        Args:
            contract: The governance contract to publish.
        """
        self._config.contracts[contract.contract_id] = contract
        self._record_event(
            OpenSandboxEventType.CONTRACT_PUBLISHED,
            agent_id="",
            contract_id=contract.contract_id,
            payload={
                "tier": contract.tier,
                "capabilities": contract.capabilities,
                "network": contract.network.value,
                "interaction": contract.interaction.value,
            },
        )
        logger.info(
            "Published contract %s (tier=%s)", contract.contract_id, contract.tier,
        )

    def get_contracts(self) -> Dict[str, GovernanceContract]:
        """Return all published contracts."""
        return dict(self._config.contracts)

    # ------------------------------------------------------------------
    # Agent screening
    # ------------------------------------------------------------------

    def screen_agent(self, manifest: CapabilityManifest) -> ContractAssignment:
        """Screen an agent and assign it to a contract tier.

        Args:
            manifest: The agent's capability manifest.

        Returns:
            A ContractAssignment (check ``rejected`` field).
        """
        assignment = self._screener.evaluate(manifest)
        if not assignment.rejected:
            self._assignments[manifest.agent_id] = assignment
        return assignment

    # ------------------------------------------------------------------
    # Sandbox lifecycle
    # ------------------------------------------------------------------

    def create_sandbox(
        self,
        assignment: ContractAssignment,
    ) -> str:
        """Allocate a sandbox for an assigned agent.

        Args:
            assignment: The screening result for the agent.

        Returns:
            The sandbox_id.

        Raises:
            ValueError: If the agent was rejected or a sandbox
                already exists.
            RuntimeError: If max sandboxes reached.
        """
        if assignment.rejected:
            raise ValueError(
                f"Cannot create sandbox for rejected agent {assignment.agent_id}"
            )
        if assignment.agent_id in self._agent_sandboxes:
            raise ValueError(
                f"Agent {assignment.agent_id} already has a sandbox"
            )
        if len(self._sandboxes) >= self._config.max_sandboxes:
            raise RuntimeError(
                f"Max sandboxes ({self._config.max_sandboxes}) reached"
            )

        sandbox_id = f"sandbox-{assignment.agent_id}"
        contract = self._config.get_contract(assignment.contract_id)

        sandbox_info = {
            "sandbox_id": sandbox_id,
            "agent_id": assignment.agent_id,
            "contract_id": assignment.contract_id,
            "tier": assignment.tier,
            "image": self._config.sandbox_image,
            "env": {
                **contract.to_sandbox_env(),
                "SWARM_AGENT_TYPE": assignment.metadata.get("agent_type", "cooperative"),
            },
            "timeout_seconds": contract.timeout_seconds,
            "max_memory_mb": contract.max_memory_mb,
            "max_cpu_shares": contract.max_cpu_shares,
            "max_disk_mb": contract.max_disk_mb,
            "network": contract.network.value,
            "active": True,
            "isolated": False,
        }

        self._sandboxes[sandbox_id] = sandbox_info
        self._agent_sandboxes[assignment.agent_id] = sandbox_id

        # Register with subsystems
        self._message_bus.register_sandbox(sandbox_id, assignment.agent_id)
        self._observer.register_agent(
            assignment.agent_id, assignment.contract_id, assignment.tier,
        )

        # Provenance
        self._provenance.sign(
            sandbox_id=sandbox_id,
            agent_id=assignment.agent_id,
            action_type="sandbox_create",
            action_summary=f"Created sandbox {sandbox_id} (tier={assignment.tier})",
            content=sandbox_info,
            contract_id=assignment.contract_id,
        )

        self._record_event(
            OpenSandboxEventType.SANDBOX_CREATED,
            agent_id=assignment.agent_id,
            sandbox_id=sandbox_id,
            contract_id=assignment.contract_id,
            payload={
                "tier": assignment.tier,
                "image": self._config.sandbox_image,
            },
        )

        logger.info(
            "Created sandbox %s for agent %s (tier=%s)",
            sandbox_id, assignment.agent_id, assignment.tier,
        )
        return sandbox_id

    def destroy_sandbox(self, agent_id: str) -> None:
        """Destroy an agent's sandbox.

        Args:
            agent_id: The agent whose sandbox to destroy.
        """
        sandbox_id = self._agent_sandboxes.pop(agent_id, None)
        if sandbox_id is None:
            logger.warning("No sandbox found for agent %s", agent_id)
            return

        self._sandboxes.pop(sandbox_id, None)
        self._message_bus.unregister_sandbox(sandbox_id)
        self._observer.unregister_agent(agent_id)

        self._provenance.sign(
            sandbox_id=sandbox_id,
            agent_id=agent_id,
            action_type="sandbox_destroy",
            action_summary=f"Destroyed sandbox {sandbox_id}",
            contract_id=self._assignments.get(agent_id, ContractAssignment(agent_id=agent_id)).contract_id,
        )

        self._record_event(
            OpenSandboxEventType.SANDBOX_KILLED,
            agent_id=agent_id,
            sandbox_id=sandbox_id,
        )
        logger.info("Destroyed sandbox %s for agent %s", sandbox_id, agent_id)

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def execute_command(
        self,
        agent_id: str,
        command: str,
        exit_code: int = 0,
    ) -> SoftInteraction:
        """Execute a command in an agent's sandbox and record it.

        In the current implementation this records the command result
        rather than actually executing inside a Docker container.
        Production use would invoke the OpenSandbox execd API here.

        Args:
            agent_id: The executing agent.
            command: Command string.
            exit_code: Result exit code (simulated or real).

        Returns:
            A SoftInteraction representing the execution.

        Raises:
            ValueError: If agent has no sandbox or sandbox is isolated.
        """
        sandbox_id = self._agent_sandboxes.get(agent_id)
        if sandbox_id is None:
            raise ValueError(f"Agent {agent_id} has no sandbox")

        sandbox_info = self._sandboxes.get(sandbox_id, {})
        if sandbox_info.get("isolated"):
            raise ValueError(f"Sandbox {sandbox_id} is isolated — commands blocked")

        # Check capability
        contract_id = sandbox_info.get("contract_id", "default")
        contract = self._config.get_contract(contract_id)
        cmd_base = command.split()[0] if command else ""
        if cmd_base and not contract.allows_capability(cmd_base):
            self._observer.record_violation(agent_id)
            self._record_event(
                OpenSandboxEventType.CONTRACT_VIOLATION,
                agent_id=agent_id,
                sandbox_id=sandbox_id,
                contract_id=contract_id,
                payload={"command": command, "reason": f"Capability {cmd_base} not permitted"},
            )
            exit_code = 1

        success = exit_code == 0
        self._observer.record_command(agent_id, success)

        # Provenance
        prov_id = self._provenance.sign(
            sandbox_id=sandbox_id,
            agent_id=agent_id,
            action_type="exec",
            action_summary=command,
            content={"command": command, "exit_code": exit_code},
            contract_id=contract_id,
        )

        # Map to interaction
        metrics = self._observer.get_agent_metrics(agent_id)
        stats = metrics.to_stats_dict() if metrics else {}
        interaction = self._mapper.map_command_execution(
            agent_id=agent_id,
            sandbox_id=sandbox_id,
            command=command,
            exit_code=exit_code,
            agent_stats=stats,
            provenance_id=prov_id,
            contract_id=contract_id,
        )

        self._observer.record_p(agent_id, interaction.p)
        self._record_interaction(interaction)

        self._record_event(
            OpenSandboxEventType.COMMAND_EXECUTED,
            agent_id=agent_id,
            sandbox_id=sandbox_id,
            contract_id=contract_id,
            payload={"command": command, "exit_code": exit_code, "p": interaction.p},
        )

        # Auto-check risk
        alert = self._observer.check_risk(agent_id)
        if alert:
            self._auto_intervene(agent_id, alert)

        return interaction

    # ------------------------------------------------------------------
    # Message bus
    # ------------------------------------------------------------------

    def send_message(
        self,
        from_sandbox: str,
        to_sandbox: str,
        payload: Dict[str, Any],
    ) -> SoftInteraction:
        """Send a message between agent sandboxes.

        All messages route through the message bus — no direct
        communication is permitted.

        Args:
            from_sandbox: Source sandbox ID.
            to_sandbox: Destination sandbox ID.
            payload: Message content.

        Returns:
            A SoftInteraction representing the exchange.
        """
        from_agent = ""
        to_agent = ""
        from_info = self._sandboxes.get(from_sandbox, {})
        to_info = self._sandboxes.get(to_sandbox, {})
        from_agent = from_info.get("agent_id", "")
        to_agent = to_info.get("agent_id", "")

        # Sign provenance
        prov_id = self._provenance.sign(
            sandbox_id=from_sandbox,
            agent_id=from_agent,
            action_type="message",
            action_summary=f"Message to {to_sandbox}",
            content=payload,
            contract_id=from_info.get("contract_id"),
        )

        # Route through bus
        msg = self._message_bus.send(
            from_sandbox=from_sandbox,
            to_sandbox=to_sandbox,
            payload=payload,
            provenance_id=prov_id,
        )

        # Record in observer
        self._observer.record_message(from_agent, msg.delivered)

        # Map to interaction
        interaction = self._mapper.map_message_exchange(
            from_agent=from_agent,
            to_agent=to_agent,
            delivered=msg.delivered,
            from_sandbox=from_sandbox,
            to_sandbox=to_sandbox,
            provenance_id=prov_id,
        )

        self._observer.record_p(from_agent, interaction.p)
        self._record_interaction(interaction)

        return interaction

    def receive_messages(
        self,
        sandbox_id: str,
        max_messages: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve pending messages for a sandbox.

        Args:
            sandbox_id: The sandbox to poll.
            max_messages: Maximum messages to return.

        Returns:
            List of message payloads.
        """
        messages = self._message_bus.receive(sandbox_id, max_messages)
        return [m.payload for m in messages]

    # ------------------------------------------------------------------
    # Governance interventions
    # ------------------------------------------------------------------

    def isolate(
        self,
        agent_id: str,
        reason: str = "manual",
        action: str = "restrict_network",
    ) -> SoftInteraction:
        """Isolate an agent's sandbox due to a contract violation or risk.

        Args:
            agent_id: The agent to isolate.
            reason: Reason for isolation.
            action: Intervention action (restrict_network, terminate).

        Returns:
            A SoftInteraction representing the intervention.
        """
        sandbox_id = self._agent_sandboxes.get(agent_id)
        if sandbox_id is None:
            raise ValueError(f"Agent {agent_id} has no sandbox")

        sandbox_info = self._sandboxes.get(sandbox_id, {})
        contract_id = sandbox_info.get("contract_id")

        if action == "terminate":
            self.destroy_sandbox(agent_id)
        else:
            sandbox_info["isolated"] = True

        self._observer.record_intervention(agent_id)

        prov_id = self._provenance.sign(
            sandbox_id=sandbox_id,
            agent_id=agent_id,
            action_type="governance",
            action_summary=f"Intervention: {action} ({reason})",
            content={"reason": reason, "action": action},
            contract_id=contract_id,
        )

        event_type = (
            OpenSandboxEventType.INTERVENTION_TERMINATE
            if action == "terminate"
            else OpenSandboxEventType.INTERVENTION_ISOLATE
        )
        self._record_event(
            event_type,
            agent_id=agent_id,
            sandbox_id=sandbox_id,
            contract_id=contract_id,
            payload={"reason": reason, "action": action},
        )

        metrics = self._observer.get_agent_metrics(agent_id)
        stats = metrics.to_stats_dict() if metrics else {}
        interaction = self._mapper.map_governance_intervention(
            agent_id=agent_id,
            sandbox_id=sandbox_id,
            reason=reason,
            action=action,
            contract_id=contract_id,
            agent_stats=stats,
        )

        self._observer.record_p(agent_id, interaction.p)
        self._record_interaction(interaction)

        logger.info(
            "Governance intervention on agent %s: %s (%s)",
            agent_id, action, reason,
        )
        return interaction

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self) -> List[SoftInteraction]:
        """Poll all active sandboxes and return new SoftInteractions."""
        new_interactions: List[SoftInteraction] = []

        for agent_id, sandbox_id in self._agent_sandboxes.items():
            sandbox_info = self._sandboxes.get(sandbox_id)
            if not sandbox_info or sandbox_info.get("isolated"):
                continue

            metrics = self._observer.get_agent_metrics(agent_id)
            stats = metrics.to_stats_dict() if metrics else {}
            interaction = self._mapper.map_sandbox_observation(
                agent_id=agent_id,
                sandbox_id=sandbox_id,
                agent_stats=stats,
                contract_id=sandbox_info.get("contract_id"),
            )
            self._observer.record_p(agent_id, interaction.p)
            self._record_interaction(interaction)
            new_interactions.append(interaction)

        return new_interactions

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def compute_experiment_metrics(
        self,
        experiment_id: str,
    ) -> ExperimentMetrics:
        """Compute aggregate experiment metrics.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            ExperimentMetrics with sorting, adherence, etc.
        """
        return self._observer.compute_experiment_metrics(experiment_id)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_interactions(self) -> List[SoftInteraction]:
        """Return all interactions recorded by this bridge."""
        return list(self._interactions)

    def get_events(self) -> List[OpenSandboxEvent]:
        """Return all events from all subsystems."""
        events = list(self._events)
        events.extend(self._screener.get_events())
        events.extend(self._message_bus.get_events())
        events.extend(self._provenance.get_events())
        events.extend(self._observer.get_events())
        events.sort(key=lambda e: e.timestamp)
        return events

    def get_sorting_ledger(self) -> Dict[str, List[Dict]]:
        """Return the screening sorting ledger."""
        return self._screener.get_sorting_ledger()

    def get_provenance_chain(self, agent_id: str) -> List[Dict[str, Any]]:
        """Return the full provenance chain for an agent."""
        return [r.to_dict() for r in self._provenance.get_chain(agent_id)]

    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Return behavioral metrics for an agent."""
        return self._observer.get_agent_metrics(agent_id)

    def get_message_bus_stats(self) -> Dict[str, Any]:
        """Return message bus statistics."""
        return self._message_bus.get_stats()

    def get_provenance_stats(self) -> Dict[str, Any]:
        """Return provenance tracker statistics."""
        return self._provenance.get_stats()

    def get_sandbox_info(self, sandbox_id: str) -> Optional[Dict[str, Any]]:
        """Return sandbox configuration info."""
        return self._sandboxes.get(sandbox_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Destroy all sandboxes and clean up."""
        for agent_id in list(self._agent_sandboxes.keys()):
            try:
                self.destroy_sandbox(agent_id)
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Failed to destroy sandbox for %s on shutdown: %s",
                    agent_id,
                    exc,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_intervene(
        self,
        agent_id: str,
        alert: Dict[str, Any],
    ) -> None:
        """Automatically intervene when a risk alert fires."""
        risk = alert.get("risk_score", 0.0)
        if risk >= 0.9:
            self.isolate(agent_id, reason="high_risk_auto", action="terminate")
        else:
            self.isolate(agent_id, reason="risk_threshold_auto", action="restrict_network")

    def _record_event(
        self,
        event_type: OpenSandboxEventType,
        agent_id: str = "",
        sandbox_id: Optional[str] = None,
        contract_id: Optional[str] = None,
        payload: Optional[Dict] = None,
    ) -> None:
        if len(self._events) >= self._config.max_events:
            self._events = self._events[-self._config.max_events // 2 :]
        self._events.append(
            OpenSandboxEvent(
                event_type=event_type,
                agent_id=agent_id,
                sandbox_id=sandbox_id,
                contract_id=contract_id,
                payload=payload or {},
            )
        )

    def _record_interaction(self, interaction: SoftInteraction) -> None:
        if len(self._interactions) >= self._config.max_interactions:
            self._interactions = self._interactions[
                -self._config.max_interactions // 2 :
            ]
        self._interactions.append(interaction)
        self._log_interaction(interaction)

    def _log_interaction(self, interaction: SoftInteraction) -> None:
        if self._event_log is None:
            return
        event = Event(
            event_type=EventType.INTERACTION_COMPLETED,
            interaction_id=interaction.interaction_id,
            initiator_id=interaction.initiator,
            counterparty_id=interaction.counterparty,
            payload={
                "accepted": interaction.accepted,
                "v_hat": interaction.v_hat,
                "p": interaction.p,
                "bridge": "opensandbox",
                "metadata": interaction.metadata,
            },
        )
        self._event_log.append(event)
