"""Skill evolution handler for the orchestrator.

Manages the lifecycle of skill evolution during simulation:
- Initializes agent skill libraries
- Triggers skill extraction after interactions
- Runs per-epoch pruning and composition
- Collects skill metrics
- Runs governance checks (poisoning detection)
"""

from typing import Dict, List, Optional, Set

from swarm.agents.base import BaseAgent
from swarm.agents.skill_evolving import SkillEvolvingMixin
from swarm.models.interaction import SoftInteraction
from swarm.skills.evolution import EvolutionConfig, SkillEvolutionEngine
from swarm.skills.governance import (
    SkillGovernanceConfig,
    SkillGovernanceEngine,
)
from swarm.skills.library import SharingMode, SkillLibrary, SkillLibraryConfig
from swarm.skills.metrics import SkillEvolutionMetrics, SkillMetricsCollector


class SkillHandlerConfig:
    """Configuration for the skill handler."""

    def __init__(
        self,
        enabled: bool = False,
        sharing_mode: str = "private",
        max_skills_per_agent: int = 50,
        max_shared_skills: int = 200,
        prune_threshold: float = 0.2,
        min_reputation_to_write: float = 1.0,
        coordinator_agent_ids: Optional[List[str]] = None,
        poisoning_detection_enabled: bool = True,
        prune_every_n_epochs: int = 3,
        success_payoff_threshold: float = 0.5,
        failure_payoff_threshold: float = -0.3,
        max_extractions_per_epoch: int = 5,
    ):
        self.enabled = enabled
        self.sharing_mode = SharingMode(sharing_mode)
        self.max_skills_per_agent = max_skills_per_agent
        self.max_shared_skills = max_shared_skills
        self.prune_threshold = prune_threshold
        self.min_reputation_to_write = min_reputation_to_write
        self.coordinator_agent_ids = set(coordinator_agent_ids or [])
        self.poisoning_detection_enabled = poisoning_detection_enabled
        self.prune_every_n_epochs = prune_every_n_epochs
        self.success_payoff_threshold = success_payoff_threshold
        self.failure_payoff_threshold = failure_payoff_threshold
        self.max_extractions_per_epoch = max_extractions_per_epoch

    def to_library_config(self) -> SkillLibraryConfig:
        return SkillLibraryConfig(
            sharing_mode=self.sharing_mode,
            max_skills_per_agent=self.max_skills_per_agent,
            max_shared_skills=self.max_shared_skills,
            prune_threshold=self.prune_threshold,
            min_reputation_to_write=self.min_reputation_to_write,
            coordinator_agent_ids=self.coordinator_agent_ids,
        )

    def to_evolution_config(self) -> EvolutionConfig:
        return EvolutionConfig(
            success_payoff_threshold=self.success_payoff_threshold,
            failure_payoff_threshold=self.failure_payoff_threshold,
            prune_every_n_epochs=self.prune_every_n_epochs,
            max_extractions_per_epoch=self.max_extractions_per_epoch,
        )


class SkillHandler:
    """Manages skill evolution within the orchestrator loop.

    Integration points:
    - on_agent_registered(): Initialize skill library for skill-evolving agents
    - on_interaction_resolved(): Trigger skill extraction
    - on_epoch_end(): Run pruning, composition, governance, metrics
    """

    def __init__(self, config: Optional[SkillHandlerConfig] = None):
        self.config = config or SkillHandlerConfig()

        # Shared library (if using shared mode)
        self._shared_library: Optional[SkillLibrary] = None
        if self.config.sharing_mode in (
            SharingMode.SHARED_GATED,
            SharingMode.COMMUNICATION,
        ):
            self._shared_library = SkillLibrary(
                owner_id="shared",
                config=self.config.to_library_config(),
            )

        # Per-agent evolution engines (one per skill-evolving agent)
        self._evolution_engines: Dict[str, SkillEvolutionEngine] = {}

        # Track which agents have skill evolution
        self._skill_agents: Dict[str, BaseAgent] = {}

        # Governance
        self._governance = SkillGovernanceEngine(
            config=SkillGovernanceConfig(
                poisoning_detection_enabled=self.config.poisoning_detection_enabled,
                min_reputation_to_propose=self.config.min_reputation_to_write,
            ),
        )

        # Metrics
        self._metrics_collector = SkillMetricsCollector()

        # Track skill invocation agents for poisoning detection
        self._invocation_agents: Dict[str, Set[str]] = {}

        # Track skill IDs already proposed to shared library to avoid re-proposing
        self._proposed_skill_ids: Set[str] = set()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def shared_library(self) -> Optional[SkillLibrary]:
        return self._shared_library

    @property
    def metrics_collector(self) -> SkillMetricsCollector:
        return self._metrics_collector

    def on_agent_registered(self, agent: BaseAgent) -> None:
        """Initialize skill evolution for skill-evolving agents."""
        if not self.enabled:
            return

        if isinstance(agent, SkillEvolvingMixin) and agent.has_skills:
            self._skill_agents[agent.agent_id] = agent
            self._evolution_engines[agent.agent_id] = agent.skill_evolution

    def on_epoch_start(self, epoch: int) -> None:
        """Prepare for new epoch."""
        if not self.enabled:
            return

        self._metrics_collector.on_epoch_start()
        for engine in self._evolution_engines.values():
            engine.on_epoch_start(epoch)

    def on_interaction_resolved(
        self,
        interaction: SoftInteraction,
        payoff_initiator: float,
        payoff_counterparty: float,
        agent_reputations: Optional[Dict[str, float]] = None,
    ) -> None:
        """Handle skill evolution after an interaction resolves.

        Called by the orchestrator after payoffs are computed.
        """
        if not self.enabled:
            return

        agent_reputations = agent_reputations or {}

        # Process for initiator
        self._process_agent_outcome(
            agent_id=interaction.initiator,
            interaction=interaction,
            payoff=payoff_initiator,
            reputation=agent_reputations.get(interaction.initiator, 0.0),
        )

        # Process for counterparty
        self._process_agent_outcome(
            agent_id=interaction.counterparty,
            interaction=interaction,
            payoff=payoff_counterparty,
            reputation=agent_reputations.get(interaction.counterparty, 0.0),
        )

    def on_epoch_end(self, epoch: int) -> Optional[SkillEvolutionMetrics]:
        """Run end-of-epoch skill lifecycle.

        Returns skill metrics for this epoch.
        """
        if not self.enabled:
            return None

        # Decay performance of unused skills
        for _agent_id, agent in self._skill_agents.items():
            if isinstance(agent, SkillEvolvingMixin) and agent.has_skills:
                agent.skill_library.epoch_decay()

        if self._shared_library:
            self._shared_library.epoch_decay()

        # Periodic pruning
        if epoch > 0 and epoch % self.config.prune_every_n_epochs == 0:
            total_pruned = 0
            for _agent_id, agent in self._skill_agents.items():
                if isinstance(agent, SkillEvolvingMixin) and agent.has_skills:
                    pruned = agent.skill_library.prune()
                    total_pruned += len(pruned)

            if self._shared_library:
                pruned = self._shared_library.prune()
                total_pruned += len(pruned)

            if total_pruned > 0:
                self._metrics_collector.record_prune(total_pruned)

        # Try composition
        for agent_id, agent in self._skill_agents.items():
            if isinstance(agent, SkillEvolvingMixin) and agent.has_skills:
                engine = self._evolution_engines.get(agent_id)
                if engine:
                    composed = engine.try_compose(agent.skill_library, agent_id)
                    if composed:
                        self._metrics_collector.record_composition()

        # Poisoning detection on shared library
        if self._shared_library and self.config.poisoning_detection_enabled:
            reports = self._governance.detect_poisoning(
                self._shared_library,
                self._invocation_agents,
            )
            for report in reports:
                self._governance.quarantine_skill(
                    report.skill_id,
                    self._shared_library,
                    epoch=epoch,
                )
                self._metrics_collector.record_quarantine()

        # Compute metrics
        agent_libraries: Dict[str, SkillLibrary] = {}
        for agent_id, agent in self._skill_agents.items():
            if isinstance(agent, SkillEvolvingMixin) and agent.has_skills:
                agent_libraries[agent_id] = agent.skill_library

        return self._metrics_collector.compute_epoch_metrics(
            epoch=epoch,
            agent_libraries=agent_libraries,
            shared_library=self._shared_library,
        )

    def get_agent_skill_summary(self, agent_id: str) -> Dict:
        """Get skill summary for a specific agent."""
        agent = self._skill_agents.get(agent_id)
        if agent and isinstance(agent, SkillEvolvingMixin):
            return agent.skill_library_summary()
        return {"enabled": False}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_agent_outcome(
        self,
        agent_id: str,
        interaction: SoftInteraction,
        payoff: float,
        reputation: float,
    ) -> None:
        """Process an interaction outcome for a single agent."""
        agent = self._skill_agents.get(agent_id)
        if agent is None or not isinstance(agent, SkillEvolvingMixin):
            return
        if not agent.has_skills:
            return

        # The skill extraction happens in the agent's update_from_outcome
        # (which calls skill_augmented_update). Here we handle shared library
        # propagation and metrics.

        # Track invocation agents for poisoning detection
        for skill_id in agent._active_skill_ids:
            if skill_id not in self._invocation_agents:
                self._invocation_agents[skill_id] = set()
            self._invocation_agents[skill_id].add(agent_id)
            self._metrics_collector.record_invocation(payoff)

        # Propagate to shared library if configured
        if self._shared_library and self.config.sharing_mode in (
            SharingMode.SHARED_GATED,
            SharingMode.COMMUNICATION,
        ):
            engine = self._evolution_engines.get(agent_id)
            if engine:
                skills = agent.skill_library.all_skills
                if skills:
                    newest = max(skills, key=lambda s: s.created_at)
                    # Only propose genuinely new skills (not already proposed)
                    if newest.skill_id not in self._proposed_skill_ids:
                        self._proposed_skill_ids.add(newest.skill_id)
                        self._governance.propose_skill(
                            skill=newest,
                            author_reputation=reputation,
                            library=self._shared_library,
                        )
