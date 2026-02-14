"""Orchestrator for running the multi-agent simulation."""

import asyncio
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from swarm.agents.base import Action, ActionType, BaseAgent, Observation
from swarm.boundaries.external_world import ExternalEntity, ExternalWorld
from swarm.boundaries.information_flow import FlowTracker
from swarm.boundaries.leakage import LeakageDetector, LeakageReport
from swarm.boundaries.policies import PolicyEngine
from swarm.core.boundary_handler import BoundaryHandler
from swarm.core.core_interaction_handler import CoreInteractionHandler
from swarm.core.feed_handler import FeedHandler
from swarm.core.handler_registry import HandlerRegistry
from swarm.core.interaction_finalizer import InteractionFinalizer
from swarm.core.kernel_handler import KernelOracleConfig, KernelOracleHandler
from swarm.core.marketplace_handler import MarketplaceHandler
from swarm.core.memory_handler import MemoryHandler, MemoryTierConfig
from swarm.core.moltbook_handler import MoltbookConfig, MoltbookHandler
from swarm.core.moltipedia_handler import MoltipediaConfig, MoltipediaHandler
from swarm.core.observable_generator import (
    DefaultObservableGenerator,
    ObservableGenerator,
)
from swarm.core.observation_builder import ObservationBuilder
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.core.redteam_inspector import RedTeamInspector
from swarm.core.scholar_handler import ScholarConfig, ScholarHandler
from swarm.core.task_handler import TaskHandler
from swarm.env.composite_tasks import (
    CompositeTask,
    CompositeTaskPool,
)
from swarm.env.feed import Feed
from swarm.env.marketplace import Marketplace, MarketplaceConfig
from swarm.env.network import AgentNetwork, NetworkConfig
from swarm.env.state import EnvState, InteractionProposal
from swarm.env.tasks import TaskPool
from swarm.forecaster.features import (
    combine_feature_dicts,
    extract_behavioral_features,
    extract_structural_features,
)
from swarm.governance.config import GovernanceConfig
from swarm.governance.engine import GovernanceEffect, GovernanceEngine
from swarm.logging.event_bus import EventBus
from swarm.logging.event_log import EventLog
from swarm.metrics.capabilities import CapabilityAnalyzer, EmergentCapabilityMetrics
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.agent import AgentState, AgentType
from swarm.models.events import (
    Event,
    EventType,
)
from swarm.models.interaction import InteractionType, SoftInteraction


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Timing
    n_epochs: int = 10
    steps_per_epoch: int = 10

    # Agent scheduling
    schedule_mode: str = "round_robin"  # "round_robin", "random", "priority"
    max_actions_per_step: int = 20

    # Safety rails
    max_content_length: int = 10000
    enable_rate_limits: bool = True
    enable_kill_switch: bool = True

    # Payoff configuration
    payoff_config: PayoffConfig = Field(default_factory=PayoffConfig)

    # Governance configuration
    governance_config: Optional[GovernanceConfig] = None

    # Network configuration
    network_config: Optional[NetworkConfig] = None

    # Marketplace configuration
    marketplace_config: Optional[MarketplaceConfig] = None

    # Moltipedia configuration
    moltipedia_config: Optional[MoltipediaConfig] = None

    # Moltbook configuration
    moltbook_config: Optional[MoltbookConfig] = None

    # Memory tier configuration
    memory_tier_config: Optional[MemoryTierConfig] = None

    # Scholar/literature synthesis configuration
    scholar_config: Optional[ScholarConfig] = None

    # Kernel oracle configuration
    kernel_oracle_config: Optional[KernelOracleConfig] = None

    # Composite task configuration
    enable_composite_tasks: bool = False

    # Boundary configuration
    enable_boundaries: bool = False
    boundary_sensitivity_threshold: float = 0.5

    # Logging
    log_path: Optional[Path] = None
    log_events: bool = True

    # Random seed
    seed: Optional[int] = None

    # Replay/scenario metadata for logging
    scenario_id: Optional[str] = None
    replay_k: Optional[int] = None

    # Stress-test knobs
    observation_noise_probability: float = 0.0
    observation_noise_std: float = 0.0


class EpochMetrics(BaseModel):
    """Metrics collected at the end of each epoch."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    epoch: int = 0
    total_interactions: int = 0
    accepted_interactions: int = 0
    total_posts: int = 0
    total_votes: int = 0
    toxicity_rate: float = 0.0
    quality_gap: float = 0.0
    avg_payoff: float = 0.0
    total_welfare: float = 0.0
    network_metrics: Optional[Dict[str, float]] = None
    capability_metrics: Optional[EmergentCapabilityMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            "epoch": self.epoch,
            "total_interactions": self.total_interactions,
            "accepted_interactions": self.accepted_interactions,
            "total_posts": self.total_posts,
            "total_votes": self.total_votes,
            "toxicity_rate": self.toxicity_rate,
            "quality_gap": self.quality_gap,
            "avg_payoff": self.avg_payoff,
            "total_welfare": self.total_welfare,
            "network_metrics": self.network_metrics,
        }
        if self.capability_metrics is not None:
            result["capability_metrics"] = (
                self.capability_metrics.to_dict()
                if hasattr(self.capability_metrics, "to_dict")
                else None
            )
        else:
            result["capability_metrics"] = None
        return result


class Orchestrator:
    """
    Orchestrates the multi-agent simulation.

    Responsibilities:
    - Schedule agent turns
    - Inject observations
    - Execute actions
    - Enforce rate limits
    - Compute payoffs
    - Emit events

    Delegates domain-specific work to handler objects:
    - MarketplaceHandler: bounty/bid/escrow/dispute lifecycle
    - BoundaryHandler: external-world and leakage enforcement
    - ObservableGenerator: signal generation from interactions

    Computation engines (payoff, proxy, metrics) can be injected
    via constructor for testability and extensibility.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        state: Optional[EnvState] = None,
        # --- dependency injection for computation engines ---
        payoff_engine: Optional[SoftPayoffEngine] = None,
        proxy_computer: Optional[ProxyComputer] = None,
        metrics_calculator: Optional[SoftMetrics] = None,
        observable_generator: Optional[ObservableGenerator] = None,
        governance_engine: Optional[GovernanceEngine] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            config: Orchestrator configuration
            state: Initial environment state (optional)
            payoff_engine: Custom payoff engine (default: built from config)
            proxy_computer: Custom proxy computer (default: ProxyComputer())
            metrics_calculator: Custom metrics calculator (default: SoftMetrics)
            observable_generator: Custom observable generator (default:
                DefaultObservableGenerator)
            governance_engine: Custom governance engine (default: built from
                config.governance_config if provided)
        """
        self.config = OrchestratorConfig() if config is None else config
        if not 0.0 <= self.config.observation_noise_probability <= 1.0:
            raise ValueError("observation_noise_probability must be in [0, 1]")
        if self.config.observation_noise_std < 0.0:
            raise ValueError("observation_noise_std must be >= 0")

        # Set random seed
        if self.config.seed is not None:
            random.seed(self.config.seed)
        self._rng = random.Random(self.config.seed)

        # Environment components
        self.state = state or EnvState(steps_per_epoch=self.config.steps_per_epoch)
        self.feed = Feed()
        self.task_pool = TaskPool()

        # Composite task support
        if self.config.enable_composite_tasks:
            self.composite_task_pool: Optional[CompositeTaskPool] = CompositeTaskPool()
            self.capability_analyzer: Optional[CapabilityAnalyzer] = CapabilityAnalyzer(
                seed=self.config.seed
            )
        else:
            self.composite_task_pool = None
            self.capability_analyzer = None

        # Network topology (initialized when agents are registered)
        if self.config.network_config is not None:
            self.network: Optional[AgentNetwork] = AgentNetwork(
                config=self.config.network_config,
                seed=self.config.seed,
            )
        else:
            self.network = None

        # Agents
        self._agents: Dict[str, BaseAgent] = {}

        # Computation engines (injectable)
        self.payoff_engine = payoff_engine or SoftPayoffEngine(
            self.config.payoff_config
        )
        self.proxy_computer = proxy_computer or ProxyComputer()
        self.metrics_calculator = metrics_calculator or SoftMetrics(self.payoff_engine)
        self._observable_generator: ObservableGenerator = (
            observable_generator or DefaultObservableGenerator()
        )

        # Governance engine (injectable)
        if governance_engine is not None:
            self.governance_engine: Optional[GovernanceEngine] = governance_engine
        elif self.config.governance_config is not None:
            self.governance_engine = GovernanceEngine(
                self.config.governance_config,
                seed=self.config.seed,
            )
        else:
            self.governance_engine = None

        # Event bus (central publish-subscribe for all events)
        self._event_bus = EventBus()
        self._event_bus.set_enrichment(
            seed=self.config.seed,
            scenario_id=self.config.scenario_id,
            replay_k=self.config.replay_k,
        )

        # Handler registry (plugin architecture)
        self._handler_registry = HandlerRegistry()

        # Marketplace handler
        if self.config.marketplace_config is not None:
            marketplace = Marketplace(self.config.marketplace_config)
            self.marketplace: Optional[Marketplace] = marketplace
            self._marketplace_handler: Optional[MarketplaceHandler] = (
                MarketplaceHandler(
                    marketplace=marketplace,
                    task_pool=self.task_pool,
                    event_bus=self._event_bus,
                    enable_rate_limits=self.config.enable_rate_limits,
                )
            )
            self._handler_registry.register(self._marketplace_handler)
        else:
            self.marketplace = None
            self._marketplace_handler = None

        # Moltipedia handler
        if self.config.moltipedia_config is not None:
            self._moltipedia_handler: Optional[MoltipediaHandler] = MoltipediaHandler(
                config=self.config.moltipedia_config,
                event_bus=self._event_bus,
            )
            self._handler_registry.register(self._moltipedia_handler)
        else:
            self._moltipedia_handler = None

        # Moltbook handler
        if self.config.moltbook_config is not None:
            rate_limit_lever = None
            challenge_lever = None
            if self.governance_engine is not None:
                rate_limit_lever = (
                    self.governance_engine.get_moltbook_rate_limit_lever()
                )
                challenge_lever = self.governance_engine.get_moltbook_challenge_lever()
            self._moltbook_handler: Optional[MoltbookHandler] = MoltbookHandler(
                config=self.config.moltbook_config,
                governance_config=self.config.governance_config,
                rate_limit_lever=rate_limit_lever,
                challenge_lever=challenge_lever,
                event_bus=self._event_bus,
            )
            self._handler_registry.register(self._moltbook_handler)
        else:
            self._moltbook_handler = None

        # Memory tier handler
        if self.config.memory_tier_config is not None:
            self._memory_handler: Optional[MemoryHandler] = MemoryHandler(
                config=self.config.memory_tier_config,
                event_bus=self._event_bus,
            )
            self._handler_registry.register(self._memory_handler)
        else:
            self._memory_handler = None

        # Scholar handler
        if self.config.scholar_config is not None:
            self._scholar_handler: Optional[ScholarHandler] = ScholarHandler(
                config=self.config.scholar_config,
                event_bus=self._event_bus,
            )
            self._handler_registry.register(self._scholar_handler)
        else:
            self._scholar_handler = None

        # Kernel oracle handler
        if self.config.kernel_oracle_config is not None:
            self._kernel_handler: Optional[KernelOracleHandler] = (
                KernelOracleHandler(
                    config=self.config.kernel_oracle_config,
                    event_bus=self._event_bus,
                )
            )
            self._handler_registry.register(self._kernel_handler)
        else:
            self._kernel_handler = None

        # Boundary handler
        if self.config.enable_boundaries:
            external_world = ExternalWorld().create_default_world()
            flow_tracker = FlowTracker(
                sensitivity_threshold=self.config.boundary_sensitivity_threshold
            )
            policy_engine = PolicyEngine().create_default_policies()
            leakage_detector = LeakageDetector()

            self.external_world: Optional[ExternalWorld] = external_world
            self.flow_tracker: Optional[FlowTracker] = flow_tracker
            self.policy_engine: Optional[PolicyEngine] = policy_engine
            self.leakage_detector: Optional[LeakageDetector] = leakage_detector

            self._boundary_handler: Optional[BoundaryHandler] = BoundaryHandler(
                external_world=external_world,
                flow_tracker=flow_tracker,
                policy_engine=policy_engine,
                leakage_detector=leakage_detector,
                event_bus=self._event_bus,
                seed=self.config.seed,
            )
        else:
            self.external_world = None
            self.flow_tracker = None
            self.policy_engine = None
            self.leakage_detector = None
            self._boundary_handler = None

        # Event logging
        if self.config.log_path:
            self.event_log: Optional[EventLog] = EventLog(self.config.log_path)
        else:
            self.event_log = None

        # Subscribe EventLog to the bus
        if self.event_log is not None and self.config.log_events:
            self._event_bus.subscribe(self.event_log.append)

        # Epoch metrics history
        self._epoch_metrics: List[EpochMetrics] = []

        # Callbacks
        self._on_epoch_end: List[Callable[[EpochMetrics], None]] = []
        self._on_interaction_complete: List[
            Callable[[SoftInteraction, float, float], None]
        ] = []

        # Interaction finalization (extracted component)
        self._finalizer = InteractionFinalizer(
            state=self.state,
            payoff_engine=self.payoff_engine,
            proxy_computer=self.proxy_computer,
            observable_generator=self._observable_generator,
            governance_engine=self.governance_engine,
            network=self.network,
            agents=self._agents,
            on_interaction_complete=self._on_interaction_complete,
            event_bus=self._event_bus,
        )

        # Core action handlers (extracted from _handle_core_action)
        self._feed_handler = FeedHandler(
            feed=self.feed,
            max_content_length=self.config.max_content_length,
            event_bus=self._event_bus,
        )
        self._handler_registry.register(self._feed_handler)

        self._core_interaction_handler = CoreInteractionHandler(
            finalizer=self._finalizer,
            network=self.network,
            event_bus=self._event_bus,
        )
        self._handler_registry.register(self._core_interaction_handler)

        self._task_handler = TaskHandler(
            task_pool=self.task_pool,
            event_bus=self._event_bus,
        )
        self._handler_registry.register(self._task_handler)

        # Observation building (extracted component)
        self._obs_builder = ObservationBuilder(
            config=self.config,
            state=self.state,
            feed=self.feed,
            task_pool=self.task_pool,
            network=self.network,
            handler_registry=self._handler_registry,
            rng=self._rng,
        )

        # Red-team inspection (extracted component)
        self._redteam = RedTeamInspector(self._agents, self.state)

    def register_agent(self, agent: BaseAgent) -> AgentState:
        """
        Register an agent with the simulation.

        Args:
            agent: Agent to register

        Returns:
            The agent's state
        """
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")

        self._agents[agent.agent_id] = agent

        # Create agent state in environment
        state = self.state.add_agent(
            agent_id=agent.agent_id,
            name=getattr(agent, "name", agent.agent_id),
            agent_type=agent.agent_type,
        )

        # Log event
        self._emit_event(
            Event(
                event_type=EventType.AGENT_CREATED,
                agent_id=agent.agent_id,
                payload={
                    "agent_type": agent.agent_type.value,
                    "name": getattr(agent, "name", agent.agent_id),
                    "roles": [r.value for r in agent.roles],
                },
                epoch=self.state.current_epoch,
                step=self.state.current_step,
            )
        )

        return state

    def _initialize_network(self) -> None:
        """Initialize network topology with registered agents."""
        agent_ids = list(self._agents.keys())
        if self.network is not None:
            self.network.initialize(agent_ids)
        # Set agent IDs for collusion detection
        if self.governance_engine is not None:
            self.governance_engine.set_collusion_agent_ids(agent_ids)

    def run(self) -> List[EpochMetrics]:
        """
        Run the full simulation.

        Returns:
            List of metrics for each epoch
        """
        # Initialize network with registered agents
        self._initialize_network()

        # Log simulation start
        self._emit_event(
            Event(
                event_type=EventType.SIMULATION_STARTED,
                payload={
                    "n_epochs": self.config.n_epochs,
                    "steps_per_epoch": self.config.steps_per_epoch,
                    "n_agents": len(self._agents),
                    "seed": self.config.seed,
                    "scenario_id": self.config.scenario_id,
                    "replay_k": self.config.replay_k,
                },
            )
        )

        # Main loop
        for _epoch in range(self.config.n_epochs):
            epoch_metrics = self._run_epoch()
            self._epoch_metrics.append(epoch_metrics)

            # Callbacks
            for callback in self._on_epoch_end:
                callback(epoch_metrics)

        # Log simulation end
        self._emit_event(
            Event(
                event_type=EventType.SIMULATION_ENDED,
                payload={
                    "total_epochs": self.config.n_epochs,
                    "final_metrics": self._epoch_metrics[-1].model_dump()
                    if self._epoch_metrics
                    else {},
                },
            )
        )

        return self._epoch_metrics

    def _epoch_pre_hooks(self) -> None:
        """Shared epoch-start logic for sync and async paths."""
        self._update_adaptive_governance()

        for handler in self._handler_registry.all_handlers():
            try:
                handler.on_epoch_start(self.state)
            except Exception:
                pass  # handler hook failures must not break simulation

        if self.governance_engine:
            gov_effect = self.governance_engine.apply_epoch_start(
                self.state, self.state.current_epoch
            )
            self._apply_governance_effect(gov_effect)

    def _epoch_post_hooks(self, epoch_start: int) -> EpochMetrics:
        """Shared epoch-end logic for sync and async paths."""
        for handler in self._handler_registry.all_handlers():
            try:
                handler.on_epoch_end(self.state)
            except Exception:
                pass  # handler hook failures must not break simulation

        if self.network is not None:
            pruned = self.network.decay_edges()
            if pruned > 0:
                self._emit_event(
                    Event(
                        event_type=EventType.EPOCH_COMPLETED,
                        payload={"network_edges_pruned": pruned},
                        epoch=epoch_start,
                    )
                )

        self._apply_agent_memory_decay(epoch_start)

        metrics = self._compute_epoch_metrics()

        self._emit_event(
            Event(
                event_type=EventType.EPOCH_COMPLETED,
                payload=metrics.model_dump(),
                epoch=epoch_start,
            )
        )

        self.state.advance_epoch()
        return metrics

    def _step_preamble(self) -> None:
        """Shared step-start logic for sync and async paths."""
        if (
            self.governance_engine
            and self.governance_engine.config.adaptive_use_behavioral_features
        ):
            self._update_adaptive_governance(include_behavioral=True)

        if self.governance_engine:
            step_effect = self.governance_engine.apply_step(
                self.state, self.state.current_step
            )
            self._apply_governance_effect(step_effect)

        for handler in self._handler_registry.all_handlers():
            try:
                handler.on_step(self.state, self.state.current_step)
            except Exception:
                pass  # handler hook failures must not break simulation

    def _get_eligible_agents(self) -> List[str]:
        """Return agents eligible to act this step (respects schedule, limits, governance)."""
        agent_order = self._get_agent_schedule()
        eligible: List[str] = []
        for agent_id in agent_order:
            if len(eligible) >= self.config.max_actions_per_step:
                break
            if not self.state.can_agent_act(agent_id):
                continue
            if self.governance_engine and not self.governance_engine.can_agent_act(
                agent_id, self.state
            ):
                continue
            eligible.append(agent_id)
        return eligible

    def _run_epoch(self) -> EpochMetrics:
        """Run a single epoch."""
        epoch_start = self.state.current_epoch
        self._epoch_pre_hooks()

        for _step in range(self.config.steps_per_epoch):
            if self.state.is_paused:
                break
            self._run_step()
            self.state.advance_step()

        return self._epoch_post_hooks(epoch_start)

    def _apply_agent_memory_decay(self, epoch: int) -> None:
        """Apply memory decay to all agents that support it.

        This implements the rain/river memory model where agents can
        have different levels of memory persistence across epochs.

        Args:
            epoch: Current epoch number
        """
        for agent in self._agents.values():
            if hasattr(agent, "apply_memory_decay"):
                agent.apply_memory_decay(epoch)

    def _run_step(self) -> None:
        """Run a single step within an epoch."""
        self._step_preamble()

        for agent_id in self._get_eligible_agents():
            agent = self._agents[agent_id]
            observation = self._build_observation(agent_id)
            action = self._select_action(agent, observation)
            self._execute_action(action)

        self._resolve_pending_interactions()

    def _select_action(self, agent: BaseAgent, observation: Observation) -> Action:
        """Select an action, optionally using governance ensembling."""
        if (
            self.governance_engine is None
            or not self.governance_engine.config.self_ensemble_enabled
            or self.governance_engine.config.self_ensemble_samples <= 1
        ):
            return agent.act(observation)

        samples = self.governance_engine.config.self_ensemble_samples
        candidate_actions = [agent.act(observation) for _ in range(samples)]
        selected = self._majority_action(candidate_actions)
        selected.metadata["ensemble_samples"] = samples
        return selected

    async def _select_action_async(
        self, agent: BaseAgent, observation: Observation
    ) -> Action:
        """Async action selection with optional governance ensembling."""
        if (
            self.governance_engine is None
            or not self.governance_engine.config.self_ensemble_enabled
            or self.governance_engine.config.self_ensemble_samples <= 1
        ):
            if self._is_llm_agent(agent):
                return await agent.act_async(observation)  # type: ignore[attr-defined, no-any-return]
            return agent.act(observation)  # type: ignore[no-any-return]

        samples = self.governance_engine.config.self_ensemble_samples
        candidate_actions: List[Action] = []
        for _ in range(samples):
            if self._is_llm_agent(agent):
                candidate_actions.append(await agent.act_async(observation))  # type: ignore[attr-defined]
            else:
                candidate_actions.append(agent.act(observation))

        selected = self._majority_action(candidate_actions)
        selected.metadata["ensemble_samples"] = samples
        return selected

    def _majority_action(self, actions: List[Action]) -> Action:
        """Choose majority action signature with deterministic tie-break."""
        if not actions:
            return Action(action_type=ActionType.NOOP)

        counts: Dict[Tuple, int] = {}
        first_index: Dict[Tuple, int] = {}
        for idx, action in enumerate(actions):
            key = self._action_signature(action)
            counts[key] = counts.get(key, 0) + 1
            if key not in first_index:
                first_index[key] = idx

        best_key = max(
            counts.keys(),
            key=lambda key: (counts[key], -first_index[key]),
        )
        for action in actions:
            if self._action_signature(action) == best_key:
                return action
        return actions[0]

    @staticmethod
    def _action_signature(action: Action) -> Tuple:
        """Stable signature for grouping semantically equivalent actions."""
        return (
            action.action_type.value,
            action.target_id,
            action.counterparty_id,
            action.interaction_type.value,
            action.vote_direction,
            action.content,
        )

    def _get_agent_schedule(self) -> List[str]:
        """Get the order of agents for this step."""
        agent_ids = list(self._agents.keys())

        if self.config.schedule_mode == "random":
            self._rng.shuffle(agent_ids)
        elif self.config.schedule_mode == "priority":
            # Sort by reputation (higher reputation goes first)
            agent_ids.sort(
                key=lambda aid: (
                    agent_st.reputation
                    if (agent_st := self.state.get_agent(aid))
                    else 0
                ),
                reverse=True,
            )
        # else: round_robin (default order)

        return agent_ids

    def _build_observation(self, agent_id: str) -> Observation:
        """Build observation for an agent."""
        return self._obs_builder.build(agent_id)

    def _apply_observation_noise(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configurable gaussian noise to numeric observation fields."""
        return self._obs_builder.apply_noise(record)

    def _execute_action(self, action: Action) -> bool:
        """
        Execute an agent action.

        Core actions (POST, VOTE, etc.) are handled inline.
        Domain actions are dispatched via the handler registry.

        Returns:
            True if action was successful
        """
        agent_id = action.agent_id
        rate_limit = self.state.get_rate_limit_state(agent_id)

        # --- Core actions (orchestrator-owned) ---
        core_result = self._handle_core_action(action, rate_limit)
        if core_result is not None:
            return core_result

        # --- Handler-dispatched actions (via registry) ---
        if not isinstance(action.action_type, ActionType):
            return False

        handler = self._handler_registry.get_handler(action.action_type)
        if handler is None:
            return False

        try:
            result = handler.handle_action(action, self.state)
        except Exception:
            return False

        if not result.success:
            return False

        # If no observables, this was a simple success/failure action
        if result.observables is None:
            return True

        # Standard proxy computation + interaction finalization pipeline
        v_hat, p = self.proxy_computer.compute_labels(result.observables)

        # Build interaction_type from result
        interaction_type = InteractionType.COLLABORATION
        if hasattr(result, "interaction_type") and isinstance(
            getattr(result, "interaction_type", None), InteractionType
        ):
            interaction_type = result.interaction_type

        # Build tau: prefer explicit tau, fall back to negated points
        tau = 0.0
        if hasattr(result, "tau") and result.tau != 0.0:
            tau = result.tau
        elif hasattr(result, "points") and result.points != 0.0:
            tau = -result.points

        # Build ground_truth: prefer explicit field, fall back to submission
        ground_truth_val = getattr(result, "ground_truth", None)
        if ground_truth_val is None and hasattr(result, "submission"):
            submission = result.submission
            if submission is not None:
                ground_truth_val = -1 if submission.is_cheat else 1

        interaction = SoftInteraction(
            initiator=result.initiator_id,
            counterparty=result.counterparty_id,
            interaction_type=interaction_type,
            accepted=result.accepted,
            task_progress_delta=result.observables.task_progress_delta,
            rework_count=result.observables.rework_count,
            verifier_rejections=result.observables.verifier_rejections,
            tool_misuse_flags=result.observables.tool_misuse_flags,
            counterparty_engagement_delta=result.observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            tau=tau,
            metadata=result.metadata or {},
            **({"ground_truth": ground_truth_val} if ground_truth_val is not None else {}),
        )

        gov_effect, _, _ = self._finalize_interaction(interaction)

        # Handler-specific post-processing
        try:
            handler.post_finalize(result, interaction, gov_effect, self.state)
        except Exception:
            pass  # post_finalize failures must not break the action

        return True

    def _handle_core_action(
        self, action: Action, rate_limit: Any
    ) -> Optional[bool]:
        """Handle NOOP — the only action still owned by the orchestrator.

        All other former "core" actions (POST, REPLY, VOTE,
        PROPOSE/ACCEPT/REJECT_INTERACTION, CLAIM_TASK, SUBMIT_OUTPUT)
        are now dispatched via the handler registry through
        ``FeedHandler``, ``CoreInteractionHandler``, and ``TaskHandler``.

        Returns ``None`` if the action is not a core action.
        """
        if action.action_type == ActionType.NOOP:
            return True

        return None  # Dispatched via handler registry

    def _resolve_pending_interactions(self) -> None:
        """Resolve any remaining pending interactions."""
        # Get all pending proposals
        proposals = list(self.state.pending_proposals.values())

        for proposal in proposals:
            counterparty_id = proposal.counterparty_id

            # Check if counterparty agent exists and can act
            if counterparty_id not in self._agents:
                continue

            if not self.state.can_agent_act(counterparty_id):
                continue

            counterparty = self._agents[counterparty_id]
            observation = self._build_observation(counterparty_id)

            # Ask counterparty agent to decide
            from swarm.agents.base import InteractionProposal as AgentProposal

            agent_proposal = AgentProposal(
                proposal_id=proposal.proposal_id,
                initiator_id=proposal.initiator_id,
                counterparty_id=proposal.counterparty_id,
                interaction_type=InteractionType(proposal.interaction_type),
                content=proposal.content,
                offered_transfer=proposal.metadata.get("offered_transfer", 0),
            )

            accept = counterparty.accept_interaction(agent_proposal, observation)

            # Remove and complete
            self.state.remove_proposal(proposal.proposal_id)
            self._complete_interaction(proposal, accepted=accept)

    def _complete_interaction(
        self,
        proposal: InteractionProposal,
        accepted: bool,
    ) -> None:
        """Complete an interaction and compute payoffs."""
        self._finalizer.complete_interaction(proposal, accepted)

    def _generate_observables(
        self,
        proposal: InteractionProposal,
        accepted: bool,
    ) -> ProxyObservables:
        """Generate observable signals for an interaction.

        Delegates to the injected ObservableGenerator.  Kept for
        backwards compatibility with subclasses that override this.
        """
        return self._observable_generator.generate(proposal, accepted, self.state)

    def _finalize_interaction(
        self,
        interaction: SoftInteraction,
    ) -> Tuple[GovernanceEffect, float, float]:
        """Apply governance, compute payoffs, update state, and emit events."""
        return self._finalizer.finalize_interaction(interaction)

    def _update_reputation(self, agent_id: str, delta: float) -> None:
        """Update agent reputation."""
        self._finalizer._update_reputation(agent_id, delta)

    # =========================================================================
    # Marketplace Delegation (preserves public interface)
    # =========================================================================

    def settle_marketplace_task(
        self,
        task_id: str,
        success: bool,
        quality_score: float = 1.0,
    ) -> Optional[Dict]:
        """
        Settle a marketplace bounty/escrow after task completion.

        Delegates to MarketplaceHandler.
        """
        if self._marketplace_handler is None:
            return None
        return self._marketplace_handler.settle_task(
            task_id=task_id,
            success=success,
            state=self.state,
            governance_engine=self.governance_engine,
            quality_score=quality_score,
        )

    def _apply_governance_effect(self, effect: GovernanceEffect) -> None:
        """Apply governance effects to state (freeze/unfreeze, reputation, resources)."""
        self._finalizer.apply_governance_effect(effect)

    def _compute_epoch_metrics(self) -> EpochMetrics:
        """Compute metrics for the current epoch."""
        interactions = self.state.completed_interactions

        # Get network metrics if available (even with no interactions)
        network_metrics = None
        if self.network is not None:
            network_metrics = self.network.get_metrics()

        # Get capability metrics if available
        capability_metrics = None
        if self.capability_analyzer is not None:
            capability_metrics = self.capability_analyzer.compute_metrics()

        if not interactions:
            return EpochMetrics(
                epoch=self.state.current_epoch,
                network_metrics=network_metrics,
                capability_metrics=capability_metrics,
            )

        accepted = [i for i in interactions if i.accepted]

        # Use soft metrics calculator
        toxicity = self.metrics_calculator.toxicity_rate(interactions)
        quality_gap = self.metrics_calculator.quality_gap(interactions)
        welfare = self.metrics_calculator.welfare_metrics(interactions)

        return EpochMetrics(
            epoch=self.state.current_epoch,
            total_interactions=len(interactions),
            accepted_interactions=len(accepted),
            total_posts=len(self.feed._posts),
            total_votes=len(self.feed._votes),
            toxicity_rate=toxicity,
            quality_gap=quality_gap,
            avg_payoff=welfare.get("avg_initiator_payoff", 0),
            total_welfare=welfare.get("total_welfare", 0),
            network_metrics=network_metrics,
            capability_metrics=capability_metrics,
        )

    def _emit_event(self, event: Event) -> None:
        """Emit an event via the event bus."""
        self._event_bus.emit(event)

    def subscribe_events(self, callback: Callable[[Event], None]) -> None:
        """Register an external subscriber for all simulation events."""
        self._event_bus.subscribe(callback)

    def pause(self) -> None:
        """Pause the simulation."""
        self.state.pause()

    def resume(self) -> None:
        """Resume the simulation."""
        self.state.resume()

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())

    def get_metrics_history(self) -> List[EpochMetrics]:
        """Get all epoch metrics."""
        return self._epoch_metrics

    def get_collusion_report(self):
        """Get the latest collusion detection report."""
        if self.governance_engine is None:
            return None
        return self.governance_engine.get_collusion_report()

    # =========================================================================
    # Composite Task Support
    # =========================================================================

    def add_composite_task(self, task: CompositeTask) -> bool:
        """
        Add a composite task to the pool.

        Args:
            task: The composite task to add

        Returns:
            True if added successfully
        """
        if self.composite_task_pool is None:
            return False
        self.composite_task_pool.add_task(task)
        return True

    def get_composite_task(self, task_id: str) -> Optional[CompositeTask]:
        """Get a composite task by ID."""
        if self.composite_task_pool is None:
            return None
        return self.composite_task_pool.get_task(task_id)

    def get_open_composite_tasks(self) -> List[CompositeTask]:
        """Get all open composite tasks."""
        if self.composite_task_pool is None:
            return []
        return self.composite_task_pool.get_open_tasks()

    def register_agent_capabilities(
        self,
        agent_id: str,
        capabilities: set,
    ) -> bool:
        """
        Register an agent's capabilities for composite task matching.

        Args:
            agent_id: The agent's ID
            capabilities: Set of CapabilityType values

        Returns:
            True if registered successfully
        """
        if self.capability_analyzer is None:
            return False
        self.capability_analyzer.register_agent(agent_id, capabilities)
        return True

    def get_capability_metrics(self) -> Optional[EmergentCapabilityMetrics]:
        """Get current emergent capability metrics."""
        if self.capability_analyzer is None:
            return None
        return self.capability_analyzer.compute_metrics()

    def get_composite_task_stats(self) -> Dict:
        """Get statistics about composite tasks."""
        if self.composite_task_pool is None:
            return {}
        return self.composite_task_pool.get_stats()

    def on_epoch_end(self, callback: Callable[[EpochMetrics], None]) -> None:
        """Register a callback for epoch end."""
        self._on_epoch_end.append(callback)

    def on_interaction_complete(
        self,
        callback: Callable[[SoftInteraction, float, float], None],
    ) -> None:
        """Register a callback for interaction completion."""
        self._on_interaction_complete.append(callback)

    # =========================================================================
    # Async Support for LLM Agents
    # =========================================================================

    def _is_llm_agent(self, agent: BaseAgent) -> bool:
        """Check if an agent is an LLM agent with async support."""
        return hasattr(agent, "act_async") and hasattr(
            agent, "accept_interaction_async"
        )

    async def run_async(self) -> List[EpochMetrics]:
        """
        Run the full simulation asynchronously.

        This method enables concurrent LLM API calls for better performance
        when using LLM-backed agents.

        Returns:
            List of metrics for each epoch
        """
        # Initialize network with registered agents
        self._initialize_network()

        # Log simulation start
        self._emit_event(
            Event(
                event_type=EventType.SIMULATION_STARTED,
                payload={
                    "n_epochs": self.config.n_epochs,
                    "steps_per_epoch": self.config.steps_per_epoch,
                    "n_agents": len(self._agents),
                    "seed": self.config.seed,
                    "async": True,
                },
            )
        )

        # Main loop
        for _epoch in range(self.config.n_epochs):
            epoch_metrics = await self._run_epoch_async()
            self._epoch_metrics.append(epoch_metrics)

            # Callbacks
            for callback in self._on_epoch_end:
                callback(epoch_metrics)

        # Log simulation end
        self._emit_event(
            Event(
                event_type=EventType.SIMULATION_ENDED,
                payload={
                    "total_epochs": self.config.n_epochs,
                    "final_metrics": self._epoch_metrics[-1].model_dump()
                    if self._epoch_metrics
                    else {},
                },
            )
        )

        return self._epoch_metrics

    async def _run_epoch_async(self) -> EpochMetrics:
        """Run a single epoch asynchronously."""
        epoch_start = self.state.current_epoch
        self._epoch_pre_hooks()

        for _step in range(self.config.steps_per_epoch):
            if self.state.is_paused:
                break
            await self._run_step_async()
            self.state.advance_step()

        return self._epoch_post_hooks(epoch_start)

    async def _run_step_async(self) -> None:
        """Run a single step asynchronously with concurrent LLM calls."""
        self._step_preamble()

        agents_to_act = self._get_eligible_agents()

        async def get_agent_action(agent_id: str) -> Tuple[str, Action]:
            agent = self._agents[agent_id]
            observation = self._build_observation(agent_id)
            action = await self._select_action_async(agent, observation)
            return agent_id, action

        tasks = [get_agent_action(agent_id) for agent_id in agents_to_act]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            agent_id, action = result  # type: ignore[misc]
            self._execute_action(action)

        await self._resolve_pending_interactions_async()

    async def _resolve_pending_interactions_async(self) -> None:
        """Resolve pending interactions with async support for LLM agents."""
        # Get all pending proposals
        proposals = list(self.state.pending_proposals.values())

        async def resolve_proposal(proposal: InteractionProposal) -> Optional[bool]:
            counterparty_id = proposal.counterparty_id

            # Check if counterparty agent exists and can act
            if counterparty_id not in self._agents:
                return None

            if not self.state.can_agent_act(counterparty_id):
                return None

            counterparty = self._agents[counterparty_id]
            observation = self._build_observation(counterparty_id)

            # Create agent proposal object
            from swarm.agents.base import InteractionProposal as AgentProposal

            agent_proposal = AgentProposal(
                proposal_id=proposal.proposal_id,
                initiator_id=proposal.initiator_id,
                counterparty_id=proposal.counterparty_id,
                interaction_type=InteractionType(proposal.interaction_type),
                content=proposal.content,
                offered_transfer=proposal.metadata.get("offered_transfer", 0),
            )

            # Get decision (async for LLM agents)
            if self._is_llm_agent(counterparty):
                accept = await counterparty.accept_interaction_async(
                    agent_proposal, observation
                )
            else:
                accept = counterparty.accept_interaction(agent_proposal, observation)

            return bool(accept)

        # Resolve all proposals concurrently
        tasks = [resolve_proposal(p) for p in proposals]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for proposal, result in zip(proposals, results, strict=False):
            if isinstance(result, Exception) or result is None:
                continue

            accept = bool(result)
            # Remove and complete
            self.state.remove_proposal(proposal.proposal_id)
            self._complete_interaction(proposal, accepted=accept)

    def _update_adaptive_governance(self, include_behavioral: bool = False) -> None:
        """Update adaptive governance mode from current episode/epoch features."""
        if self.governance_engine is None:
            return
        if not self.governance_engine.config.adaptive_governance_enabled:
            return

        agents = self.get_all_agents()
        adversarial_count = sum(
            1
            for agent in agents
            if agent.agent_type in (AgentType.ADVERSARIAL, AgentType.DECEPTIVE)
        )
        structural = extract_structural_features(
            horizon_length=self.config.steps_per_epoch,
            agent_count=len(agents),
            action_space_size=len(ActionType),
            adversarial_fraction=(adversarial_count / len(agents) if agents else 0.0),
        )
        features = structural

        if include_behavioral:
            behavioral = extract_behavioral_features(self.state.completed_interactions)
            features = combine_feature_dicts(structural, behavioral)

        self.governance_engine.update_adaptive_mode(features)

    def get_llm_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get LLM usage statistics for all LLM agents.

        Returns:
            Dictionary mapping agent_id to usage stats
        """
        stats = {}
        for agent_id, agent in self._agents.items():
            if hasattr(agent, "get_usage_stats"):
                stats[agent_id] = agent.get_usage_stats()
        return stats

    def get_network_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get current network topology metrics.

        Returns:
            Dictionary of network metrics, or None if no network
        """
        if self.network is None:
            return None
        return self.network.get_metrics()

    def get_network(self) -> Optional[AgentNetwork]:
        """Get the network object for direct manipulation."""
        return self.network

    # =========================================================================
    # Red-Team Support
    # =========================================================================

    def get_adaptive_adversary_reports(self) -> Dict[str, Dict]:
        """Get strategy reports from all adaptive adversaries."""
        return self._redteam.get_adaptive_adversary_reports()

    def notify_adversary_detection(
        self,
        agent_id: str,
        penalty: float = 0.0,
        detected: bool = True,
    ) -> None:
        """Notify an adaptive adversary of detection/penalty."""
        self._redteam.notify_adversary_detection(agent_id, penalty, detected)

    def get_evasion_metrics(self) -> Dict:
        """Get evasion metrics for adversarial agents."""
        return self._redteam.get_evasion_metrics()

    # =========================================================================
    # Boundary Delegation (preserves public interface)
    # =========================================================================

    def request_external_interaction(
        self,
        agent_id: str,
        entity_id: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Request an interaction with an external entity.

        Delegates to BoundaryHandler.
        """
        if self._boundary_handler is None:
            return {"success": False, "error": "Boundaries not enabled"}
        return self._boundary_handler.request_external_interaction(
            agent_id=agent_id,
            entity_id=entity_id,
            action=action,
            payload=payload,
        )

    def get_external_entities(
        self,
        entity_type: Optional[str] = None,
        min_trust: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Get available external entities. Delegates to BoundaryHandler."""
        if self._boundary_handler is None:
            return []
        return self._boundary_handler.get_external_entities(
            entity_type=entity_type,
            min_trust=min_trust,
        )

    def add_external_entity(self, entity: ExternalEntity) -> None:
        """Add an external entity to the world. Delegates to BoundaryHandler."""
        if self._boundary_handler is not None:
            self._boundary_handler.add_external_entity(entity)

    def get_boundary_metrics(self) -> Dict[str, Any]:
        """Get comprehensive boundary metrics. Delegates to BoundaryHandler."""
        if self._boundary_handler is None:
            return {"boundaries_enabled": False}
        return self._boundary_handler.get_metrics()

    def get_agent_boundary_activity(self, agent_id: str) -> Dict[str, Any]:
        """Get boundary activity for a specific agent. Delegates to BoundaryHandler."""
        if self._boundary_handler is None:
            return {"agent_id": agent_id}
        return self._boundary_handler.get_agent_activity(agent_id)

    def get_leakage_report(self) -> Optional[LeakageReport]:
        """Get the full leakage detection report. Delegates to BoundaryHandler."""
        if self._boundary_handler is None:
            return None
        return self._boundary_handler.get_leakage_report()

