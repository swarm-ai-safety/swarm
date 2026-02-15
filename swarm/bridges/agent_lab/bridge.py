"""Main bridge connecting AgentLaboratory to SWARM.

AgentLabBridge is the central adapter that:
1. Parses AgentLab checkpoints and lab directories (via AgentLabClient)
2. Routes events to the mapper for SoftInteraction production
3. Evaluates governance policy (phase gates, circuit breakers, cost caps)
4. Feeds into SWARM's logging and metrics pipeline
5. Runs study refinement via AgentLab subprocess (refine_study)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from swarm.bridges.agent_lab.refinement import RefinementConfig, RefinementResult

from swarm.bridges.agent_lab.client import AgentLabClient
from swarm.bridges.agent_lab.config import AgentLabConfig
from swarm.bridges.agent_lab.events import (
    AgentLabEvent,
    AgentLabEventType,
    DialogueEvent,
    ReviewEvent,
    SolverIterationEvent,
)
from swarm.bridges.agent_lab.mapper import AgentLabMapper
from swarm.bridges.agent_lab.policy import AgentLabPolicy, PolicyDecision
from swarm.core.proxy import ProxyComputer
from swarm.logging.event_log import EventLog
from swarm.models.events import Event, EventType
from swarm.models.interaction import SoftInteraction

logger = logging.getLogger(__name__)


class AgentLabBridge:
    """Bridge between AgentLaboratory and SWARM framework.

    Supports offline mode: analyze existing checkpoints and lab
    directories. Online mode (live event streaming) is deferred.

    Example (offline)::

        bridge = AgentLabBridge()
        interactions = bridge.ingest_checkpoint("state_saves/Paper0.pkl")
        for i in interactions:
            print(i.p)  # P(v = +1)
    """

    def __init__(
        self,
        config: AgentLabConfig | None = None,
        event_log: EventLog | None = None,
    ) -> None:
        self._config = config or AgentLabConfig()
        self._client = AgentLabClient(self._config.client_config)
        self._mapper = AgentLabMapper(
            config=self._config,
            proxy=ProxyComputer(sigmoid_k=self._config.proxy_sigmoid_k),
        )
        self._policy = AgentLabPolicy(config=self._config)
        self._event_log = event_log
        self._interactions: List[SoftInteraction] = []
        self._events: List[AgentLabEvent] = []
        self._phase_interactions: dict[str, List[SoftInteraction]] = {}
        self._last_total_cost_usd: float = 0.0

    # --- Offline ingestion ---

    def ingest_checkpoint(self, path: str) -> List[SoftInteraction]:
        """Parse a checkpoint file and return scored interactions.

        Args:
            path: Path to a Paper*.pkl checkpoint.

        Returns:
            List of SoftInteraction records produced from the checkpoint.
        """
        raw_events = self._client.parse_checkpoint(path)
        return self._process_events(raw_events)

    def ingest_directory(self, lab_dir: str) -> List[SoftInteraction]:
        """Parse a lab output directory and return scored interactions.

        Args:
            lab_dir: Path to a lab output directory.

        Returns:
            List of SoftInteraction records.
        """
        raw_events = self._client.parse_lab_directory(lab_dir)
        return self._process_events(raw_events)

    def ingest_events(self, events: List[AgentLabEvent]) -> List[SoftInteraction]:
        """Process a list of pre-built events (useful for testing).

        Args:
            events: List of AgentLabEvent records.

        Returns:
            List of SoftInteraction records.
        """
        return self._process_events(events)

    # --- Event processing ---

    def _process_events(
        self, events: List[AgentLabEvent]
    ) -> List[SoftInteraction]:
        """Process a batch of events through mapper and policy."""
        new_interactions: List[SoftInteraction] = []

        for event in events:
            interactions = self._process_event(event)
            new_interactions.extend(interactions)

            if self._policy.should_circuit_break():
                logger.warning("Circuit breaker triggered, stopping event processing")
                break

        return new_interactions

    def _process_event(self, event: AgentLabEvent) -> List[SoftInteraction]:
        """Route a single event to the appropriate mapper method.

        Returns zero or more SoftInteractions.
        """
        self._record_event(event)
        interactions: List[SoftInteraction] = []

        if event.event_type == AgentLabEventType.SOLVER_ITERATION:
            solver = SolverIterationEvent.from_dict(event.payload)
            interaction = self._mapper.map_solver_iteration(event, solver)
            self._apply_cost_policy(solver.cost_usd)
            interactions.append(interaction)

        elif event.event_type == AgentLabEventType.DIALOGUE_EXCHANGE:
            dialogue = DialogueEvent.from_dict(event.payload)
            interaction = self._mapper.map_dialogue(event, dialogue)
            interactions.append(interaction)

        elif event.event_type in (
            AgentLabEventType.REVIEW_SUBMITTED,
            AgentLabEventType.REVIEW_DECISION,
        ):
            review = ReviewEvent.from_dict(event.payload)
            interaction = self._mapper.map_review(event, review)
            self._evaluate_review_policy(review)
            interactions.append(interaction)

        elif event.event_type in (
            AgentLabEventType.PHASE_COMPLETED,
            AgentLabEventType.PHASE_FAILED,
        ):
            success = event.event_type == AgentLabEventType.PHASE_COMPLETED
            interaction = self._mapper.map_phase_completion(
                event, success=success
            )
            interactions.append(interaction)
            self._evaluate_phase_gate(event.phase)

        elif event.event_type in (
            AgentLabEventType.CODE_GENERATED,
            AgentLabEventType.CODE_EXECUTED,
            AgentLabEventType.CODE_FAILED,
            AgentLabEventType.CODE_REPAIRED,
        ):
            success = event.event_type == AgentLabEventType.CODE_EXECUTED
            repair = event.event_type == AgentLabEventType.CODE_REPAIRED
            # Only evaluate the circuit breaker on actual execution
            # outcomes (executed / failed / repaired), not on code
            # generation which precedes execution.
            if event.event_type != AgentLabEventType.CODE_GENERATED:
                self._policy.evaluate_code_execution(success or repair)
            interaction = self._mapper.map_code_event(
                event, success=success, repair_attempt=repair
            )
            interactions.append(interaction)

        elif event.event_type == AgentLabEventType.COST_UPDATED:
            # Payload carries cumulative total; convert to delta so the
            # policy's running sum stays correct.
            total = event.payload.get("total_cost_usd", 0.0)
            delta = max(total - self._last_total_cost_usd, 0.0)
            self._last_total_cost_usd = total
            self._apply_cost_policy(delta)

        # Record and log each interaction
        for interaction in interactions:
            self._record_interaction(interaction)
            self._log_interaction(interaction)
            # Track per-phase for gate decisions
            phase = event.phase
            if phase:
                self._phase_interactions.setdefault(phase, []).append(interaction)

        return interactions

    def _apply_cost_policy(self, cost_usd: float) -> None:
        """Evaluate cost policy for a cost increment."""
        if cost_usd > 0:
            result = self._policy.evaluate_cost(cost_usd)
            if result.decision == PolicyDecision.DENY:
                logger.warning("Cost policy: %s", result.reason)
            elif result.decision == PolicyDecision.WARN:
                logger.info("Cost policy: %s", result.reason)

    def _evaluate_phase_gate(self, phase: str) -> None:
        """Evaluate phase gate policy after phase completion."""
        phase_ints = self._phase_interactions.get(phase, [])
        result = self._policy.evaluate_phase_gate(phase_ints)
        if result.decision == PolicyDecision.DENY:
            logger.warning("Phase gate: %s", result.reason)
        elif result.decision == PolicyDecision.WARN:
            logger.info("Phase gate: %s", result.reason)

    def _evaluate_review_policy(self, review: ReviewEvent) -> None:
        """Evaluate review loop policy."""
        self._policy.evaluate_review_round(review.overall_score)

    # --- Recording ---

    def _record_event(self, event: AgentLabEvent) -> None:
        """Record a bridge event, applying memory cap."""
        if len(self._events) >= self._config.max_events:
            self._events = self._events[-self._config.max_events // 2 :]
        self._events.append(event)

    def _record_interaction(self, interaction: SoftInteraction) -> None:
        """Record a SoftInteraction, applying memory cap."""
        if len(self._interactions) >= self._config.max_interactions:
            self._interactions = self._interactions[
                -self._config.max_interactions // 2 :
            ]
        self._interactions.append(interaction)

    def _log_interaction(self, interaction: SoftInteraction) -> None:
        """Log an interaction to SWARM's append-only event log."""
        if self._event_log is None:
            return

        metadata = dict(interaction.metadata or {})
        event = Event(
            event_type=EventType.INTERACTION_COMPLETED,
            interaction_id=interaction.interaction_id,
            initiator_id=interaction.initiator,
            counterparty_id=interaction.counterparty,
            payload={
                "accepted": interaction.accepted,
                "v_hat": interaction.v_hat,
                "p": interaction.p,
                "bridge": "agent_lab",
                "metadata": metadata,
            },
        )
        self._event_log.append(event)

    # --- Study refinement ---

    def refine_study(
        self,
        run_dir: str,
        refinement_config: Optional["RefinementConfig"] = None,
    ) -> "RefinementResult":
        """Run AgentLab refinement on a completed SWARM study.

        Packages the study's artifacts, spawns AgentLab as a subprocess,
        ingests the resulting checkpoint, and writes outputs to
        ``<run_dir>/refinement/``.

        Args:
            run_dir: Path to the completed study run directory.
            refinement_config: Optional refinement configuration.
                Defaults to settings from self._config.

        Returns:
            RefinementResult with hypotheses, suggestions, and interactions.
        """
        from swarm.bridges.agent_lab.refinement import (
            RefinementConfig,
            RefinementResult,
            StudyContext,
        )
        from swarm.bridges.agent_lab.runner import AgentLabRunner

        # Build context from run directory
        context = StudyContext.from_run_dir(run_dir)

        # Build config
        if refinement_config is None:
            refinement_config = RefinementConfig(
                cost_budget_usd=self._config.refinement_cost_budget_usd,
                depth=self._config.refinement_depth,
            )

        # Generate temp YAML config
        import tempfile

        import yaml

        yaml_dict = refinement_config.to_agent_lab_yaml(context)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="refinement_"
        ) as f:
            yaml.dump(yaml_dict, f)
            yaml_path = f.name

        # Run subprocess
        runner = AgentLabRunner(refinement_config)
        work_dir = str(Path(run_dir) / "refinement" / "agent_lab_work")
        returncode, stdout, stderr, duration = runner.run_refinement(
            yaml_path, work_dir
        )

        result = RefinementResult(
            success=(returncode == 0),
            duration_seconds=duration,
        )

        # Try to ingest checkpoint
        checkpoint = runner.find_checkpoint(work_dir)
        if checkpoint:
            try:
                interactions = self.ingest_checkpoint(checkpoint)
                result.interactions = interactions
                result.total_cost_usd = self._policy.total_cost_usd
            except Exception:
                logger.exception("Failed to ingest refinement checkpoint")

        # Parse stdout for structured content
        result.hypotheses = self._extract_section(stdout, "hypothes")
        result.gaps_identified = self._extract_section(stdout, "gap")

        # Write outputs
        output_dir = Path(run_dir) / "refinement"
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "refinement_report.json"
        with open(report_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        config_path = output_dir / "refinement_config.yaml"
        with open(config_path, "w") as f:
            import yaml
            yaml.dump(yaml_dict, f)

        if result.interactions:
            interactions_path = output_dir / "interactions.jsonl"
            with open(interactions_path, "w") as f:
                for interaction in result.interactions:
                    f.write(json.dumps(interaction.to_dict()) + "\n")

        logger.info(
            "Refinement complete: success=%s, interactions=%d, cost=$%.2f",
            result.success,
            len(result.interactions),
            result.total_cost_usd,
        )

        return result

    @staticmethod
    def _extract_section(text: str, keyword: str) -> List[str]:
        """Extract bullet-point lines containing a keyword from text."""
        items: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if keyword.lower() in stripped.lower() and len(stripped) > 10:
                # Clean up bullet markers
                for prefix in ("- ", "* ", "• ", "· "):
                    if stripped.startswith(prefix):
                        stripped = stripped[len(prefix):]
                        break
                items.append(stripped)
        return items

    # --- Accessors ---

    def get_interactions(self) -> List[SoftInteraction]:
        """Get all interactions recorded by this bridge."""
        return list(self._interactions)

    def get_events(self) -> List[AgentLabEvent]:
        """Get all bridge events."""
        return list(self._events)

    def get_phase_interactions(
        self, phase: str
    ) -> List[SoftInteraction]:
        """Get interactions for a specific phase."""
        return list(self._phase_interactions.get(phase, []))

    @property
    def policy(self) -> AgentLabPolicy:
        """Access the governance policy."""
        return self._policy

    @property
    def client(self) -> AgentLabClient:
        """Access the underlying client."""
        return self._client

    @property
    def mapper(self) -> AgentLabMapper:
        """Access the mapper."""
        return self._mapper
