"""Orchestrator for running the multi-agent simulation."""

import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from swarm.agents.base import Action, ActionType, BaseAgent, Observation
from swarm.boundaries.external_world import ExternalEntity, ExternalWorld
from swarm.boundaries.information_flow import FlowTracker
from swarm.boundaries.leakage import LeakageDetector, LeakageReport
from swarm.boundaries.policies import PolicyEngine
from swarm.core.boundary_handler import BoundaryHandler
from swarm.core.marketplace_handler import MarketplaceHandler
from swarm.core.observable_generator import (
    DefaultObservableGenerator,
    ObservableGenerator,
)
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.env.composite_tasks import (
    CompositeTask,
    CompositeTaskPool,
)
from swarm.env.feed import Feed, VoteType
from swarm.env.marketplace import Marketplace, MarketplaceConfig
from swarm.env.network import AgentNetwork, NetworkConfig
from swarm.env.state import EnvState, InteractionProposal
from swarm.env.tasks import TaskPool, TaskStatus
from swarm.forecaster.features import (
    combine_feature_dicts,
    extract_behavioral_features,
    extract_structural_features,
)
from swarm.governance.config import GovernanceConfig
from swarm.governance.engine import GovernanceEffect, GovernanceEngine
from swarm.logging.event_log import EventLog
from swarm.metrics.capabilities import CapabilityAnalyzer, EmergentCapabilityMetrics
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.agent import AgentState, AgentType
from swarm.models.events import (
    Event,
    EventType,
    interaction_completed_event,
    interaction_proposed_event,
    payoff_computed_event,
    reputation_updated_event,
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

@dataclass
class EpochMetrics:
    """Metrics collected at the end of each epoch."""

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
        self.payoff_engine = payoff_engine or SoftPayoffEngine(self.config.payoff_config)
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

        # Marketplace handler
        if self.config.marketplace_config is not None:
            marketplace = Marketplace(self.config.marketplace_config)
            self.marketplace: Optional[Marketplace] = marketplace
            self._marketplace_handler: Optional[MarketplaceHandler] = MarketplaceHandler(
                marketplace=marketplace,
                task_pool=self.task_pool,
                emit_event=self._emit_event,
            )
        else:
            self.marketplace = None
            self._marketplace_handler = None

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
                emit_event=self._emit_event,
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

        # Epoch metrics history
        self._epoch_metrics: List[EpochMetrics] = []

        # Callbacks
        self._on_epoch_end: List[Callable[[EpochMetrics], None]] = []
        self._on_interaction_complete: List[Callable[[SoftInteraction, float, float], None]] = []

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
        self._emit_event(Event(
            event_type=EventType.AGENT_CREATED,
            agent_id=agent.agent_id,
            payload={
                "agent_type": agent.agent_type.value,
                "name": getattr(agent, "name", agent.agent_id),
                "roles": [r.value for r in agent.roles],
            },
            epoch=self.state.current_epoch,
            step=self.state.current_step,
        ))

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
        self._emit_event(Event(
            event_type=EventType.SIMULATION_STARTED,
            payload={
                "n_epochs": self.config.n_epochs,
                "steps_per_epoch": self.config.steps_per_epoch,
                "n_agents": len(self._agents),
                "seed": self.config.seed,
                "scenario_id": self.config.scenario_id,
                "replay_k": self.config.replay_k,
            },
        ))

        # Main loop
        for _epoch in range(self.config.n_epochs):
            epoch_metrics = self._run_epoch()
            self._epoch_metrics.append(epoch_metrics)

            # Callbacks
            for callback in self._on_epoch_end:
                callback(epoch_metrics)

        # Log simulation end
        self._emit_event(Event(
            event_type=EventType.SIMULATION_ENDED,
            payload={
                "total_epochs": self.config.n_epochs,
                "final_metrics": self._epoch_metrics[-1].__dict__ if self._epoch_metrics else {},
            },
        ))

        return self._epoch_metrics

    def _run_epoch(self) -> EpochMetrics:
        """Run a single epoch."""
        epoch_start = self.state.current_epoch

        self._update_adaptive_governance()

        # Apply epoch-start governance (reputation decay, unfreezes)
        if self.governance_engine:
            gov_effect = self.governance_engine.apply_epoch_start(
                self.state, self.state.current_epoch
            )
            self._apply_governance_effect(gov_effect)

        for _step in range(self.config.steps_per_epoch):
            if self.state.is_paused:
                break

            self._run_step()
            self.state.advance_step()

        # Marketplace epoch maintenance
        if self._marketplace_handler is not None:
            self._marketplace_handler.on_epoch_end(self.state)

        # Apply network edge decay
        if self.network is not None:
            pruned = self.network.decay_edges()
            if pruned > 0:
                self._emit_event(Event(
                    event_type=EventType.EPOCH_COMPLETED,
                    payload={"network_edges_pruned": pruned},
                    epoch=epoch_start,
                ))

        # Compute epoch metrics
        metrics = self._compute_epoch_metrics()

        # Log epoch completion
        self._emit_event(Event(
            event_type=EventType.EPOCH_COMPLETED,
            payload=metrics.__dict__,
            epoch=epoch_start,
        ))

        # Advance to next epoch
        self.state.advance_epoch()

        return metrics

    def _run_step(self) -> None:
        """Run a single step within an epoch."""
        if (
            self.governance_engine
            and self.governance_engine.config.adaptive_use_behavioral_features
        ):
            self._update_adaptive_governance(include_behavioral=True)

        # Step-level governance hooks (e.g., decomposition checkpoints)
        if self.governance_engine:
            step_effect = self.governance_engine.apply_step(
                self.state, self.state.current_step
            )
            self._apply_governance_effect(step_effect)

        # Get agent schedule for this step
        agent_order = self._get_agent_schedule()

        actions_this_step = 0

        for agent_id in agent_order:
            if actions_this_step >= self.config.max_actions_per_step:
                break

            if not self.state.can_agent_act(agent_id):
                continue

            # Check governance admission control (staking)
            if self.governance_engine and not self.governance_engine.can_agent_act(agent_id, self.state):
                continue

            agent = self._agents[agent_id]

            # Build observation for agent
            observation = self._build_observation(agent_id)

            # Get agent action
            action = self._select_action(agent, observation)

            # Execute action
            success = self._execute_action(action)

            if success:
                actions_this_step += 1

        # Resolve pending interactions
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

    async def _select_action_async(self, agent: BaseAgent, observation: Observation) -> Action:
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
                key=lambda aid: (agent_st.reputation if (agent_st := self.state.get_agent(aid)) else 0),
                reverse=True,
            )
        # else: round_robin (default order)

        return agent_ids

    def _build_observation(self, agent_id: str) -> Observation:
        """Build observation for an agent."""
        agent_state = self.state.get_agent(agent_id)
        rate_limit = self.state.get_rate_limit_state(agent_id)

        # Get visible posts
        visible_posts = [
            p.to_dict() for p in self.feed.get_ranked_posts(limit=20)
        ]

        # Get pending proposals for this agent
        pending_proposals = [
            {
                "proposal_id": p.proposal_id,
                "initiator_id": p.initiator_id,
                "interaction_type": p.interaction_type,
                "content": p.content,
                "offered_transfer": p.metadata.get("offered_transfer", 0),
            }
            for p in self.state.get_proposals_for_agent(agent_id)
        ]

        # Get available tasks
        available_tasks = [
            t.to_dict() for t in self.task_pool.get_claimable_tasks(
                agent_reputation=agent_state.reputation if agent_state else 0,
                current_epoch=self.state.current_epoch,
            )
        ]

        # Get agent's active tasks
        active_tasks = [
            t.to_dict() for t in self.task_pool.get_tasks_for_agent(agent_id)
            if t.status in (TaskStatus.CLAIMED, TaskStatus.IN_PROGRESS)
        ]

        # Get visible agents (filtered by network if enabled)
        active_agents = self.state.get_active_agents()

        if self.network is not None:
            # Only show network neighbors
            neighbor_ids = set(self.network.neighbors(agent_id))
            active_agents = [s for s in active_agents if s.agent_id in neighbor_ids]

        visible_agents = [
            {
                "agent_id": s.agent_id,
                "name": s.name,
                "agent_type": s.agent_type.value,
                "reputation": s.reputation,
                "resources": s.resources,
                "edge_weight": self.network.edge_weight(agent_id, s.agent_id) if self.network else 1.0,
            }
            for s in active_agents
            if s.agent_id != agent_id
        ]

        visible_agents = [self._apply_observation_noise(record) for record in visible_agents]

        # Build marketplace observation via handler
        available_bounties: List[Dict] = []
        active_bids: List[Dict] = []
        active_escrows: List[Dict] = []
        pending_bid_decisions: List[Dict] = []

        if self._marketplace_handler is not None:
            mkt_obs = self._marketplace_handler.build_observation_fields(
                agent_id, self.state,
            )
            available_bounties = mkt_obs["available_bounties"]
            active_bids = mkt_obs["active_bids"]
            active_escrows = mkt_obs["active_escrows"]
            pending_bid_decisions = mkt_obs["pending_bid_decisions"]

        return Observation(
            agent_state=agent_state or AgentState(),
            current_epoch=self.state.current_epoch,
            current_step=self.state.current_step,
            can_post=rate_limit.can_post(self.state.rate_limits) if self.config.enable_rate_limits else True,
            can_interact=rate_limit.can_interact(self.state.rate_limits) if self.config.enable_rate_limits else True,
            can_vote=rate_limit.can_vote(self.state.rate_limits) if self.config.enable_rate_limits else True,
            can_claim_task=rate_limit.can_claim_task(self.state.rate_limits) if self.config.enable_rate_limits else True,
            visible_posts=visible_posts,
            pending_proposals=pending_proposals,
            available_tasks=available_tasks,
            active_tasks=active_tasks,
            visible_agents=visible_agents,
            available_bounties=available_bounties,
            active_bids=active_bids,
            active_escrows=active_escrows,
            pending_bid_decisions=pending_bid_decisions,
            ecosystem_metrics=self._apply_observation_noise(
                self.state.get_epoch_metrics_snapshot()
            ),
        )

    def _apply_observation_noise(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply configurable gaussian noise to numeric observation fields.

        Noise is only applied when enabled and never to boolean fields.
        """
        if (
            self.config.observation_noise_probability <= 0
            or self.config.observation_noise_std <= 0
        ):
            return record

        noisy: Dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, bool):
                noisy[key] = value
                continue
            if isinstance(value, (int, float)):
                if self._rng.random() < self.config.observation_noise_probability:
                    noisy[key] = float(value) + self._rng.gauss(
                        0.0, self.config.observation_noise_std
                    )
                else:
                    noisy[key] = value
                continue
            noisy[key] = value
        return noisy

    def _execute_action(self, action: Action) -> bool:
        """
        Execute an agent action.

        Returns:
            True if action was successful
        """
        agent_id = action.agent_id
        rate_limit = self.state.get_rate_limit_state(agent_id)

        if action.action_type == ActionType.NOOP:
            return True

        elif action.action_type == ActionType.POST:
            if not rate_limit.can_post(self.state.rate_limits):
                return False

            try:
                self.feed.create_post(
                    author_id=agent_id,
                    content=action.content[:self.config.max_content_length],
                )
                rate_limit.record_post()
                return True
            except ValueError:
                return False

        elif action.action_type == ActionType.REPLY:
            if not rate_limit.can_post(self.state.rate_limits):
                return False

            try:
                self.feed.create_post(
                    author_id=agent_id,
                    content=action.content[:self.config.max_content_length],
                    parent_id=action.target_id,
                )
                rate_limit.record_post()
                return True
            except ValueError:
                return False

        elif action.action_type == ActionType.VOTE:
            if not rate_limit.can_vote(self.state.rate_limits):
                return False

            vote_type = VoteType.UPVOTE if action.vote_direction > 0 else VoteType.DOWNVOTE
            vote = self.feed.vote(action.target_id, agent_id, vote_type)
            if vote:
                rate_limit.record_vote()
                return True
            return False

        elif action.action_type == ActionType.PROPOSE_INTERACTION:
            if not rate_limit.can_interact(self.state.rate_limits):
                return False

            # Validate network constraint
            if self.network is not None:
                if not self.network.has_edge(agent_id, action.counterparty_id):
                    # Cannot interact with non-neighbors
                    return False

            proposal = InteractionProposal(
                initiator_id=agent_id,
                counterparty_id=action.counterparty_id,
                interaction_type=action.interaction_type.value,
                content=action.content,
                metadata=action.metadata,
            )
            self.state.add_proposal(proposal)
            rate_limit.record_interaction()

            # Log proposal event
            self._emit_event(interaction_proposed_event(
                interaction_id=proposal.proposal_id,
                initiator_id=agent_id,
                counterparty_id=action.counterparty_id,
                interaction_type=action.interaction_type.value,
                v_hat=0.0,  # Computed later
                p=0.5,
                epoch=self.state.current_epoch,
                step=self.state.current_step,
            ))

            return True

        elif action.action_type == ActionType.ACCEPT_INTERACTION:
            accept_proposal: Optional[InteractionProposal] = self.state.remove_proposal(action.target_id)
            if accept_proposal:
                self._complete_interaction(accept_proposal, accepted=True)
                return True
            return False

        elif action.action_type == ActionType.REJECT_INTERACTION:
            proposal_rej: Optional[InteractionProposal] = self.state.remove_proposal(action.target_id)
            if proposal_rej:
                self._complete_interaction(proposal_rej, accepted=False)
                return True
            return False

        elif action.action_type == ActionType.CLAIM_TASK:
            agent_state = self.state.get_agent(agent_id)
            if not agent_state:
                return False

            success = self.task_pool.claim_task(
                task_id=action.target_id,
                agent_id=agent_id,
                agent_reputation=agent_state.reputation,
            )
            if success:
                rate_limit.record_task_claim()
            return success

        elif action.action_type == ActionType.SUBMIT_OUTPUT:
            task = self.task_pool.get_task(action.target_id)
            if task and task.claimed_by == agent_id:
                task.submit_output(agent_id, action.content)
                return True
            return False

        # Marketplace actions — delegate to handler
        elif action.action_type == ActionType.POST_BOUNTY:
            if self._marketplace_handler is None:
                return False
            return self._marketplace_handler.handle_post_bounty(
                action, self.state,
                enable_rate_limits=self.config.enable_rate_limits,
            )

        elif action.action_type == ActionType.PLACE_BID:
            if self._marketplace_handler is None:
                return False
            return self._marketplace_handler.handle_place_bid(
                action, self.state,
                enable_rate_limits=self.config.enable_rate_limits,
            )

        elif action.action_type == ActionType.ACCEPT_BID:
            if self._marketplace_handler is None:
                return False
            return self._marketplace_handler.handle_accept_bid(action, self.state)

        elif action.action_type == ActionType.REJECT_BID:
            if self._marketplace_handler is None:
                return False
            return self._marketplace_handler.handle_reject_bid(action, self.state)

        elif action.action_type == ActionType.WITHDRAW_BID:
            if self._marketplace_handler is None:
                return False
            return self._marketplace_handler.handle_withdraw_bid(action)

        elif action.action_type == ActionType.FILE_DISPUTE:
            if self._marketplace_handler is None:
                return False
            return self._marketplace_handler.handle_file_dispute(action, self.state)

        return False

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
        # Generate observables via the injectable generator
        observables = self._observable_generator.generate(
            proposal, accepted, self.state,
        )

        # Compute v_hat and p
        v_hat, p = self.proxy_computer.compute_labels(observables)

        # Create SoftInteraction
        interaction = SoftInteraction(
            interaction_id=proposal.proposal_id,
            initiator=proposal.initiator_id,
            counterparty=proposal.counterparty_id,
            interaction_type=InteractionType(proposal.interaction_type),
            accepted=accepted,
            task_progress_delta=observables.task_progress_delta,
            rework_count=observables.rework_count,
            verifier_rejections=observables.verifier_rejections,
            tool_misuse_flags=observables.tool_misuse_flags,
            counterparty_engagement_delta=observables.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
            tau=proposal.metadata.get("offered_transfer", 0),
        )

        # Apply governance costs to interaction
        if self.governance_engine:
            gov_effect = self.governance_engine.apply_interaction(interaction, self.state)
            interaction.c_a += gov_effect.cost_a
            interaction.c_b += gov_effect.cost_b
            self._apply_governance_effect(gov_effect)

            # Log governance event
            if gov_effect.cost_a > 0 or gov_effect.cost_b > 0:
                self._emit_event(Event(
                    event_type=EventType.GOVERNANCE_COST_APPLIED,
                    interaction_id=proposal.proposal_id,
                    initiator_id=proposal.initiator_id,
                    counterparty_id=proposal.counterparty_id,
                    payload={
                        "cost_a": gov_effect.cost_a,
                        "cost_b": gov_effect.cost_b,
                        "levers": [e.lever_name for e in gov_effect.lever_effects],
                    },
                    epoch=self.state.current_epoch,
                    step=self.state.current_step,
                ))

        # Compute payoffs
        payoff_init = self.payoff_engine.payoff_initiator(interaction)
        payoff_counter = self.payoff_engine.payoff_counterparty(interaction)

        # Update agent states
        if accepted:
            initiator_state = self.state.get_agent(proposal.initiator_id)
            counterparty_state = self.state.get_agent(proposal.counterparty_id)

            if initiator_state:
                initiator_state.record_initiated(accepted=True, p=p)
                initiator_state.total_payoff += payoff_init
                # Reputation delta accounts for governance costs so that
                # tax, audit penalties, etc. feed back through the
                # reputation → observables → p loop to affect toxicity.
                rep_delta = (p - 0.5) - interaction.c_a
                self._update_reputation(proposal.initiator_id, rep_delta)

            if counterparty_state:
                counterparty_state.record_received(accepted=True, p=p)
                counterparty_state.total_payoff += payoff_counter

        # Update agent memory
        if proposal.initiator_id in self._agents:
            self._agents[proposal.initiator_id].update_from_outcome(interaction, payoff_init)
        if proposal.counterparty_id in self._agents:
            self._agents[proposal.counterparty_id].update_from_outcome(interaction, payoff_counter)

        # Record interaction
        self.state.record_interaction(interaction)

        # Log events
        self._emit_event(interaction_completed_event(
            interaction_id=proposal.proposal_id,
            accepted=accepted,
            payoff_initiator=payoff_init,
            payoff_counterparty=payoff_counter,
            epoch=self.state.current_epoch,
            step=self.state.current_step,
        ))

        self._emit_event(payoff_computed_event(
            interaction_id=proposal.proposal_id,
            initiator_id=proposal.initiator_id,
            counterparty_id=proposal.counterparty_id,
            payoff_initiator=payoff_init,
            payoff_counterparty=payoff_counter,
            components={
                "p": p,
                "v_hat": v_hat,
                "tau": interaction.tau,
                "accepted": accepted,
            },
            epoch=self.state.current_epoch,
            step=self.state.current_step,
        ))

        # Strengthen network edge after interaction
        if self.network is not None and accepted:
            self.network.strengthen_edge(proposal.initiator_id, proposal.counterparty_id)

        # Callbacks
        for callback in self._on_interaction_complete:
            callback(interaction, payoff_init, payoff_counter)

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

    def _update_reputation(self, agent_id: str, delta: float) -> None:
        """Update agent reputation."""
        agent_state = self.state.get_agent(agent_id)
        if not agent_state:
            return

        old_rep = agent_state.reputation
        agent_state.update_reputation(delta)

        self._emit_event(reputation_updated_event(
            agent_id=agent_id,
            old_reputation=old_rep,
            new_reputation=agent_state.reputation,
            delta=delta,
            reason="interaction_outcome",
            epoch=self.state.current_epoch,
            step=self.state.current_step,
        ))

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
        # Freeze agents
        for agent_id in effect.agents_to_freeze:
            self.state.freeze_agent(agent_id)

        # Unfreeze agents
        for agent_id in effect.agents_to_unfreeze:
            self.state.unfreeze_agent(agent_id)

        # Apply reputation deltas
        for agent_id, delta in effect.reputation_deltas.items():
            agent_state = self.state.get_agent(agent_id)
            if agent_state:
                old_rep = agent_state.reputation
                agent_state.update_reputation(delta)
                self._emit_event(reputation_updated_event(
                    agent_id=agent_id,
                    old_reputation=old_rep,
                    new_reputation=agent_state.reputation,
                    delta=delta,
                    reason="governance",
                    epoch=self.state.current_epoch,
                    step=self.state.current_step,
                ))

        # Apply resource deltas
        for agent_id, delta in effect.resource_deltas.items():
            agent_state = self.state.get_agent(agent_id)
            if agent_state:
                agent_state.update_resources(delta)

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
        """Emit an event to the log."""
        if event.seed is None:
            event.seed = self.config.seed
        if event.scenario_id is None:
            event.scenario_id = self.config.scenario_id
        if event.replay_k is None:
            event.replay_k = self.config.replay_k

        if self.event_log is not None and self.config.log_events:
            self.event_log.append(event)

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
        return hasattr(agent, 'act_async') and hasattr(agent, 'accept_interaction_async')

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
        self._emit_event(Event(
            event_type=EventType.SIMULATION_STARTED,
            payload={
                "n_epochs": self.config.n_epochs,
                "steps_per_epoch": self.config.steps_per_epoch,
                "n_agents": len(self._agents),
                "seed": self.config.seed,
                "async": True,
            },
        ))

        # Main loop
        for _epoch in range(self.config.n_epochs):
            epoch_metrics = await self._run_epoch_async()
            self._epoch_metrics.append(epoch_metrics)

            # Callbacks
            for callback in self._on_epoch_end:
                callback(epoch_metrics)

        # Log simulation end
        self._emit_event(Event(
            event_type=EventType.SIMULATION_ENDED,
            payload={
                "total_epochs": self.config.n_epochs,
                "final_metrics": self._epoch_metrics[-1].__dict__ if self._epoch_metrics else {},
            },
        ))

        return self._epoch_metrics

    async def _run_epoch_async(self) -> EpochMetrics:
        """Run a single epoch asynchronously."""
        epoch_start = self.state.current_epoch

        self._update_adaptive_governance()

        # Apply epoch-start governance (reputation decay, unfreezes)
        if self.governance_engine:
            gov_effect = self.governance_engine.apply_epoch_start(
                self.state, self.state.current_epoch
            )
            self._apply_governance_effect(gov_effect)

        for _step in range(self.config.steps_per_epoch):
            if self.state.is_paused:
                break

            await self._run_step_async()
            self.state.advance_step()

        # Apply network edge decay
        if self.network is not None:
            pruned = self.network.decay_edges()
            if pruned > 0:
                self._emit_event(Event(
                    event_type=EventType.EPOCH_COMPLETED,
                    payload={"network_edges_pruned": pruned},
                    epoch=epoch_start,
                ))

        # Compute epoch metrics
        metrics = self._compute_epoch_metrics()

        # Log epoch completion
        self._emit_event(Event(
            event_type=EventType.EPOCH_COMPLETED,
            payload=metrics.__dict__,
            epoch=epoch_start,
        ))

        # Advance to next epoch
        self.state.advance_epoch()

        return metrics

    async def _run_step_async(self) -> None:
        """Run a single step asynchronously with concurrent LLM calls."""
        if (
            self.governance_engine
            and self.governance_engine.config.adaptive_use_behavioral_features
        ):
            self._update_adaptive_governance(include_behavioral=True)

        # Step-level governance hooks (e.g., decomposition checkpoints)
        if self.governance_engine:
            step_effect = self.governance_engine.apply_step(
                self.state, self.state.current_step
            )
            self._apply_governance_effect(step_effect)

        # Get agent schedule for this step
        agent_order = self._get_agent_schedule()

        actions_this_step = 0

        # Collect agents that can act this step
        agents_to_act = []
        for agent_id in agent_order:
            if actions_this_step >= self.config.max_actions_per_step:
                break

            if not self.state.can_agent_act(agent_id):
                continue

            # Check governance admission control (staking)
            if self.governance_engine and not self.governance_engine.can_agent_act(agent_id, self.state):
                continue

            agents_to_act.append(agent_id)
            actions_this_step += 1

        # Get actions concurrently for LLM agents
        async def get_agent_action(agent_id: str) -> Tuple[str, Action]:
            agent = self._agents[agent_id]
            observation = self._build_observation(agent_id)
            action = await self._select_action_async(agent, observation)
            return agent_id, action

        # Execute all agent actions concurrently
        tasks = [get_agent_action(agent_id) for agent_id in agents_to_act]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                # Log error but continue
                continue

            agent_id, action = result  # type: ignore[misc]
            self._execute_action(action)

        # Resolve pending interactions asynchronously
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
                accept = await counterparty.accept_interaction_async(agent_proposal, observation)  # type: ignore[attr-defined]
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
            1 for agent in agents
            if agent.agent_type in (AgentType.ADVERSARIAL, AgentType.DECEPTIVE)
        )
        structural = extract_structural_features(
            horizon_length=self.config.steps_per_epoch,
            agent_count=len(agents),
            action_space_size=len(ActionType),
            adversarial_fraction=(
                adversarial_count / len(agents) if agents else 0.0
            ),
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
            if hasattr(agent, 'get_usage_stats'):
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
        """
        Get strategy reports from all adaptive adversaries.

        Returns:
            Dictionary mapping agent_id to strategy report
        """
        reports = {}
        for agent_id, agent in self._agents.items():
            if hasattr(agent, 'get_strategy_report'):
                reports[agent_id] = agent.get_strategy_report()
        return reports

    def notify_adversary_detection(
        self,
        agent_id: str,
        penalty: float = 0.0,
        detected: bool = True,
    ) -> None:
        """
        Notify an adaptive adversary of detection/penalty.

        This allows adversaries to learn from governance feedback.

        Args:
            agent_id: The agent that was detected
            penalty: Penalty amount applied
            detected: Whether the agent was detected
        """
        agent = self._agents.get(agent_id)
        if agent is not None and hasattr(agent, 'update_adversary_outcome'):
            # Get recent payoff for this agent
            recent_payoff = 0.0
            if self.state.completed_interactions:
                agent_interactions = [
                    i for i in self.state.completed_interactions
                    if i.initiator == agent_id or i.counterparty == agent_id
                ]
                if agent_interactions:
                    last = agent_interactions[-1]
                    if last.initiator == agent_id:
                        recent_payoff = last.payoff_initiator or 0.0  # type: ignore[attr-defined]
                    else:
                        recent_payoff = last.payoff_counterparty or 0.0  # type: ignore[attr-defined]

            agent.update_adversary_outcome(
                payoff=recent_payoff,
                penalty=penalty,
                detected=detected,
            )

    def get_evasion_metrics(self) -> Dict:
        """
        Get evasion metrics for adversarial agents.

        Returns:
            Dictionary with evasion statistics
        """
        metrics: Dict[str, Any] = {
            "total_adversaries": 0,
            "adaptive_adversaries": 0,
            "avg_detection_rate": 0.0,
            "avg_heat_level": 0.0,
            "strategies_used": {},
            "by_agent": {},
        }

        detection_rates = []
        heat_levels = []

        for agent_id, agent in self._agents.items():
            agent_state = self.state.get_agent(agent_id)
            if agent_state and agent_state.agent_type == AgentType.ADVERSARIAL:
                metrics["total_adversaries"] += 1

                if hasattr(agent, 'get_strategy_report'):
                    metrics["adaptive_adversaries"] += 1
                    report = agent.get_strategy_report()
                    metrics["by_agent"][agent_id] = report

                    # Aggregate strategy usage
                    for strategy, stats in report.get("strategy_stats", {}).items():
                        if strategy not in metrics["strategies_used"]:
                            metrics["strategies_used"][strategy] = {
                                "total_attempts": 0,
                                "total_detections": 0,
                            }
                        attempts = stats.get("attempts", 0)
                        detection_rate = stats.get("detection_rate", 0)
                        metrics["strategies_used"][strategy]["total_attempts"] += attempts
                        metrics["strategies_used"][strategy]["total_detections"] += int(
                            attempts * detection_rate
                        )

                    heat_levels.append(report.get("heat_level", 0))

                    # Calculate detection rate from strategy stats
                    total_attempts = sum(
                        s.get("attempts", 0) for s in report.get("strategy_stats", {}).values()
                    )
                    total_detected = sum(
                        s.get("attempts", 0) * s.get("detection_rate", 0)
                        for s in report.get("strategy_stats", {}).values()
                    )
                    if total_attempts > 0:
                        detection_rates.append(total_detected / total_attempts)

        if detection_rates:
            metrics["avg_detection_rate"] = sum(detection_rates) / len(detection_rates)
        if heat_levels:
            metrics["avg_heat_level"] = sum(heat_levels) / len(heat_levels)

        return metrics

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
