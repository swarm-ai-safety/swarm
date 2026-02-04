"""Orchestrator for running the multi-agent simulation."""

import asyncio
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.agents.base import Action, ActionType, BaseAgent, Observation

if TYPE_CHECKING:
    from src.agents.llm_agent import LLMAgent
from src.core.payoff import PayoffConfig, SoftPayoffEngine
from src.core.proxy import ProxyComputer, ProxyObservables
from src.env.composite_tasks import (
    CapabilityType,
    CompositeTask,
    CompositeTaskPool,
    CompositeTaskStatus,
)
from src.env.feed import Feed, VoteType
from src.env.network import AgentNetwork, NetworkConfig
from src.env.state import EnvState, InteractionProposal
from src.env.tasks import TaskPool, TaskStatus
from src.metrics.capabilities import CapabilityAnalyzer, EmergentCapabilityMetrics
from src.logging.event_log import EventLog
from src.metrics.soft_metrics import SoftMetrics
from src.models.agent import AgentState, AgentType
from src.models.events import (
    Event,
    EventType,
    interaction_completed_event,
    interaction_proposed_event,
    payoff_computed_event,
    reputation_updated_event,
)
from src.models.interaction import InteractionType, SoftInteraction
from src.governance.config import GovernanceConfig
from src.governance.engine import GovernanceEffect, GovernanceEngine
from src.boundaries.external_world import ExternalWorld, ExternalEntity
from src.boundaries.information_flow import FlowTracker, FlowDirection, FlowType, InformationFlow
from src.boundaries.policies import PolicyEngine, CrossingDecision
from src.boundaries.leakage import LeakageDetector, LeakageReport
from src.env.marketplace import Marketplace, MarketplaceConfig, EscrowStatus


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

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
    payoff_config: PayoffConfig = field(default_factory=PayoffConfig)

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
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        state: Optional[EnvState] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            config: Orchestrator configuration
            state: Initial environment state (optional)
        """
        self.config = config or OrchestratorConfig()

        # Set random seed
        if self.config.seed is not None:
            random.seed(self.config.seed)

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

        # Computation engines
        self.payoff_engine = SoftPayoffEngine(self.config.payoff_config)
        self.proxy_computer = ProxyComputer()
        self.metrics_calculator = SoftMetrics(self.payoff_engine)

        # Governance engine
        if self.config.governance_config is not None:
            self.governance_engine: Optional[GovernanceEngine] = GovernanceEngine(
                self.config.governance_config,
                seed=self.config.seed,
            )
        else:
            self.governance_engine = None

        # Marketplace
        if self.config.marketplace_config is not None:
            self.marketplace: Optional[Marketplace] = Marketplace(
                self.config.marketplace_config
            )
        else:
            self.marketplace = None

        # Boundary components
        if self.config.enable_boundaries:
            self.external_world: Optional[ExternalWorld] = ExternalWorld().create_default_world()
            self.flow_tracker: Optional[FlowTracker] = FlowTracker(
                sensitivity_threshold=self.config.boundary_sensitivity_threshold
            )
            self.policy_engine: Optional[PolicyEngine] = PolicyEngine().create_default_policies()
            self.leakage_detector: Optional[LeakageDetector] = LeakageDetector()
        else:
            self.external_world = None
            self.flow_tracker = None
            self.policy_engine = None
            self.leakage_detector = None

        # Event logging
        if self.config.log_path:
            self.event_log = EventLog(self.config.log_path)
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
            agent_type=agent.agent_type,
        )

        # Log event
        self._emit_event(Event(
            event_type=EventType.AGENT_CREATED,
            agent_id=agent.agent_id,
            payload={
                "agent_type": agent.agent_type.value,
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
            },
        ))

        # Main loop
        for epoch in range(self.config.n_epochs):
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

        # Apply epoch-start governance (reputation decay, unfreezes)
        if self.governance_engine:
            gov_effect = self.governance_engine.apply_epoch_start(
                self.state, self.state.current_epoch
            )
            self._apply_governance_effect(gov_effect)

        for step in range(self.config.steps_per_epoch):
            if self.state.is_paused:
                break

            self._run_step()
            self.state.advance_step()

        # Marketplace epoch maintenance
        if self.marketplace is not None:
            expired_bounties = self.marketplace.expire_bounties(self.state.current_epoch)
            for bounty_id in expired_bounties:
                bounty = self.marketplace.get_bounty(bounty_id)
                if bounty:
                    poster_state = self.state.get_agent(bounty.poster_id)
                    if poster_state:
                        poster_state.update_resources(bounty.reward_amount)

            resolved_disputes = self.marketplace.auto_resolve_disputes(self.state.current_epoch)
            for dispute_id in resolved_disputes:
                dispute = self.marketplace.get_dispute(dispute_id)
                if dispute:
                    escrow = self.marketplace.get_escrow(dispute.escrow_id)
                    if escrow:
                        worker_state = self.state.get_agent(escrow.worker_id)
                        poster_state = self.state.get_agent(escrow.poster_id)
                        if worker_state:
                            worker_state.update_resources(escrow.released_amount)
                        if poster_state:
                            poster_state.update_resources(escrow.refunded_amount)
                        self._emit_event(Event(
                            event_type=EventType.DISPUTE_RESOLVED,
                            payload={
                                "dispute_id": dispute_id,
                                "escrow_id": escrow.escrow_id,
                                "worker_share": dispute.worker_share,
                                "auto_resolved": True,
                            },
                            epoch=self.state.current_epoch,
                        ))

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
            action = agent.act(observation)

            # Execute action
            success = self._execute_action(action)

            if success:
                actions_this_step += 1

        # Resolve pending interactions
        self._resolve_pending_interactions()

    def _get_agent_schedule(self) -> List[str]:
        """Get the order of agents for this step."""
        agent_ids = list(self._agents.keys())

        if self.config.schedule_mode == "random":
            random.shuffle(agent_ids)
        elif self.config.schedule_mode == "priority":
            # Sort by reputation (higher reputation goes first)
            agent_ids.sort(
                key=lambda aid: self.state.get_agent(aid).reputation if self.state.get_agent(aid) else 0,
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
                "agent_type": s.agent_type.value,
                "reputation": s.reputation,
                "resources": s.resources,
                "edge_weight": self.network.edge_weight(agent_id, s.agent_id) if self.network else 1.0,
            }
            for s in active_agents
            if s.agent_id != agent_id
        ]

        # Build marketplace observation
        available_bounties = []
        active_bids = []
        active_escrows = []
        pending_bid_decisions = []

        if self.marketplace is not None:
            agent_rep = agent_state.reputation if agent_state else 0
            available_bounties = [
                b.to_dict()
                for b in self.marketplace.get_open_bounties(
                    current_epoch=self.state.current_epoch,
                    min_reputation=agent_rep,
                )
                if b.poster_id != agent_id
            ]

            active_bids = [
                b.to_dict()
                for b in self.marketplace.get_agent_bids(agent_id)
                if b.status.value == "pending"
            ]

            active_escrows = [
                e.to_dict()
                for e in self.marketplace.get_agent_escrows(agent_id)
                if e.status == EscrowStatus.HELD
            ]

            # Bids on this agent's bounties awaiting decision
            for bounty in self.marketplace.get_agent_bounties(agent_id):
                if bounty.status.value == "open":
                    for bid in self.marketplace.get_bids_for_bounty(bounty.bounty_id):
                        if bid.status.value == "pending":
                            bid_dict = bid.to_dict()
                            bid_dict["bounty_reward"] = bounty.reward_amount
                            pending_bid_decisions.append(bid_dict)

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
            ecosystem_metrics=self.state.get_epoch_metrics_snapshot(),
        )

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
                post = self.feed.create_post(
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
                post = self.feed.create_post(
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
            proposal = self.state.remove_proposal(action.target_id)
            if proposal:
                self._complete_interaction(proposal, accepted=True)
                return True
            return False

        elif action.action_type == ActionType.REJECT_INTERACTION:
            proposal = self.state.remove_proposal(action.target_id)
            if proposal:
                self._complete_interaction(proposal, accepted=False)
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

        elif action.action_type == ActionType.POST_BOUNTY:
            return self._handle_post_bounty(action)

        elif action.action_type == ActionType.PLACE_BID:
            return self._handle_place_bid(action)

        elif action.action_type == ActionType.ACCEPT_BID:
            return self._handle_accept_bid(action)

        elif action.action_type == ActionType.REJECT_BID:
            return self._handle_reject_bid(action)

        elif action.action_type == ActionType.WITHDRAW_BID:
            return self._handle_withdraw_bid(action)

        elif action.action_type == ActionType.FILE_DISPUTE:
            return self._handle_file_dispute(action)

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
            from src.agents.base import InteractionProposal as AgentProposal

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
        # Create observables (simulated based on acceptance and agent types)
        observables = self._generate_observables(proposal, accepted)

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

        Observables are based on the initiator's agent type and modulated
        by the initiator's current reputation.  This creates a feedback
        loop: reputation decay degrades observable quality, which lowers
        the soft label p, which in turn reduces payoffs and welfare.
        """
        # Base observables depend on agent types
        initiator_state = self.state.get_agent(proposal.initiator_id)
        counterparty_state = self.state.get_agent(proposal.counterparty_id)

        initiator_type = initiator_state.agent_type if initiator_state else AgentType.HONEST
        counterparty_type = counterparty_state.agent_type if counterparty_state else AgentType.HONEST

        # Generate observables based on agent type combination
        if initiator_type == AgentType.HONEST:
            task_progress = random.uniform(0.3, 0.8)
            rework = random.randint(0, 1)
            rejections = 0
            misuse = 0
            engagement = random.uniform(0.2, 0.6)
        elif initiator_type == AgentType.OPPORTUNISTIC:
            task_progress = random.uniform(0.1, 0.5)
            rework = random.randint(0, 2)
            rejections = random.randint(0, 1)
            misuse = 0
            engagement = random.uniform(-0.2, 0.3)
        elif initiator_type == AgentType.DECEPTIVE:
            task_progress = random.uniform(0.2, 0.6)  # Appears moderate
            rework = random.randint(0, 2)
            rejections = random.randint(0, 1)
            misuse = 0
            engagement = random.uniform(0.0, 0.4)
        else:  # Adversarial
            task_progress = random.uniform(-0.3, 0.2)
            rework = random.randint(1, 3)
            rejections = random.randint(1, 2)
            misuse = random.randint(0, 1)
            engagement = random.uniform(-0.5, -0.1)

        # Modulate by initiator reputation: agents with higher reputation
        # produce better observable signals.  Maps reputation from [-1, 1]
        # to a factor in [0.5, 1.0] so the effect is meaningful but bounded.
        rep = initiator_state.reputation if initiator_state else 0.0
        rep_factor = 0.75 + 0.25 * max(min(rep, 1.0), -1.0)  # [0.5, 1.0]
        task_progress *= rep_factor
        engagement *= rep_factor

        # Modify if not accepted
        if not accepted:
            engagement = min(engagement, 0.0)

        return ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=rework,
            verifier_rejections=rejections,
            tool_misuse_flags=misuse,
            counterparty_engagement_delta=engagement,
        )

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
    # Marketplace Action Handlers
    # =========================================================================

    def _handle_post_bounty(self, action: Action) -> bool:
        """Handle POST_BOUNTY action."""
        if self.marketplace is None:
            return False

        agent_id = action.agent_id
        rate_limit = self.state.get_rate_limit_state(agent_id)

        if self.config.enable_rate_limits and not rate_limit.can_post_bounty(self.state.rate_limits):
            return False

        reward_amount = action.metadata.get("reward_amount", 0)
        min_reputation = action.metadata.get("min_reputation", 0.0)
        deadline_epoch = action.metadata.get("deadline_epoch")

        # Validate agent has enough resources
        agent_state = self.state.get_agent(agent_id)
        if not agent_state or agent_state.resources < reward_amount:
            return False

        try:
            # Create task in pool
            task = self.task_pool.create_task(
                prompt=action.content or "Marketplace bounty task",
                description=action.content or "Marketplace bounty task",
                bounty=reward_amount,
                min_reputation=min_reputation,
                deadline_epoch=deadline_epoch,
            )

            bounty = self.marketplace.post_bounty(
                poster_id=agent_id,
                task_id=task.task_id,
                reward_amount=reward_amount,
                min_reputation=min_reputation,
                deadline_epoch=deadline_epoch,
                current_epoch=self.state.current_epoch,
            )
        except ValueError:
            return False

        # Deduct funds
        agent_state.update_resources(-reward_amount)
        rate_limit.record_bounty()

        self._emit_event(Event(
            event_type=EventType.BOUNTY_POSTED,
            agent_id=agent_id,
            payload={
                "bounty_id": bounty.bounty_id,
                "task_id": task.task_id,
                "reward_amount": reward_amount,
            },
            epoch=self.state.current_epoch,
            step=self.state.current_step,
        ))

        return True

    def _handle_place_bid(self, action: Action) -> bool:
        """Handle PLACE_BID action."""
        if self.marketplace is None:
            return False

        agent_id = action.agent_id
        rate_limit = self.state.get_rate_limit_state(agent_id)

        if self.config.enable_rate_limits and not rate_limit.can_place_bid(self.state.rate_limits):
            return False

        bounty_id = action.target_id
        bid_amount = action.metadata.get("bid_amount", 0)

        bid = self.marketplace.place_bid(
            bounty_id=bounty_id,
            bidder_id=agent_id,
            bid_amount=bid_amount,
            message=action.content,
        )

        if bid is None:
            return False

        rate_limit.record_bid()

        self._emit_event(Event(
            event_type=EventType.BID_PLACED,
            agent_id=agent_id,
            payload={
                "bid_id": bid.bid_id,
                "bounty_id": bounty_id,
                "bid_amount": bid_amount,
            },
            epoch=self.state.current_epoch,
            step=self.state.current_step,
        ))

        return True

    def _handle_accept_bid(self, action: Action) -> bool:
        """Handle ACCEPT_BID action."""
        if self.marketplace is None:
            return False

        agent_id = action.agent_id
        bounty_id = action.target_id
        bid_id = action.metadata.get("bid_id", "")

        escrow = self.marketplace.accept_bid(
            bounty_id=bounty_id,
            bid_id=bid_id,
            poster_id=agent_id,
        )

        if escrow is None:
            return False

        # Claim the task for the worker
        bounty = self.marketplace.get_bounty(bounty_id)
        if bounty:
            worker_state = self.state.get_agent(escrow.worker_id)
            if worker_state:
                self.task_pool.claim_task(
                    task_id=bounty.task_id,
                    agent_id=escrow.worker_id,
                    agent_reputation=worker_state.reputation,
                )

        self._emit_event(Event(
            event_type=EventType.ESCROW_CREATED,
            agent_id=agent_id,
            payload={
                "escrow_id": escrow.escrow_id,
                "bounty_id": bounty_id,
                "bid_id": bid_id,
                "worker_id": escrow.worker_id,
                "amount": escrow.amount,
            },
            epoch=self.state.current_epoch,
            step=self.state.current_step,
        ))

        return True

    def _handle_reject_bid(self, action: Action) -> bool:
        """Handle REJECT_BID action."""
        if self.marketplace is None:
            return False

        success = self.marketplace.reject_bid(
            bid_id=action.target_id,
            poster_id=action.agent_id,
        )

        if success:
            self._emit_event(Event(
                event_type=EventType.BID_REJECTED,
                agent_id=action.agent_id,
                payload={"bid_id": action.target_id},
                epoch=self.state.current_epoch,
                step=self.state.current_step,
            ))

        return success

    def _handle_withdraw_bid(self, action: Action) -> bool:
        """Handle WITHDRAW_BID action."""
        if self.marketplace is None:
            return False

        return self.marketplace.withdraw_bid(
            bid_id=action.target_id,
            bidder_id=action.agent_id,
        )

    def _handle_file_dispute(self, action: Action) -> bool:
        """Handle FILE_DISPUTE action."""
        if self.marketplace is None:
            return False

        dispute = self.marketplace.file_dispute(
            escrow_id=action.target_id,
            filed_by=action.agent_id,
            reason=action.content,
            current_epoch=self.state.current_epoch,
        )

        if dispute is None:
            return False

        self._emit_event(Event(
            event_type=EventType.DISPUTE_FILED,
            agent_id=action.agent_id,
            payload={
                "dispute_id": dispute.dispute_id,
                "escrow_id": action.target_id,
                "reason": action.content,
            },
            epoch=self.state.current_epoch,
            step=self.state.current_step,
        ))

        return True

    def settle_marketplace_task(
        self,
        task_id: str,
        success: bool,
        quality_score: float = 1.0,
    ) -> Optional[Dict]:
        """
        Settle a marketplace bounty/escrow after task completion.

        Called after VERIFY_OUTPUT succeeds. Checks if the task has
        an associated bounty/escrow and settles it.

        Args:
            task_id: The completed task ID
            success: Whether the task was completed successfully
            quality_score: Quality score from verifier

        Returns:
            Settlement details, or None if no marketplace bounty
        """
        if self.marketplace is None:
            return None

        bounty = self.marketplace.get_bounty_for_task(task_id)
        if not bounty or not bounty.escrow_id:
            return None

        settlement = self.marketplace.settle_escrow(
            escrow_id=bounty.escrow_id,
            success=success,
            quality_score=quality_score,
        )

        if not settlement:
            return None

        # Apply resource changes
        if success:
            worker_id = settlement["worker_id"]
            poster_id = settlement["poster_id"]
            released = settlement["released_to_worker"]
            refund_to_poster = settlement.get("refund_to_poster", 0.0)

            worker_state = self.state.get_agent(worker_id)
            poster_state = self.state.get_agent(poster_id)

            if worker_state:
                worker_state.update_resources(released)
            if poster_state and refund_to_poster > 0:
                poster_state.update_resources(refund_to_poster)

            # Create SoftInteraction for governance tax
            interaction = SoftInteraction(
                initiator=poster_id,
                counterparty=worker_id,
                interaction_type=InteractionType.TRADE,
                accepted=True,
                task_progress_delta=quality_score,
                rework_count=0,
                verifier_rejections=0,
                tool_misuse_flags=0,
                counterparty_engagement_delta=quality_score * 0.5,
                v_hat=quality_score * 2 - 1,  # Map [0,1] to [-1,1]
                p=quality_score,
                tau=released,
            )

            # Apply governance taxes
            if self.governance_engine:
                gov_effect = self.governance_engine.apply_interaction(
                    interaction, self.state
                )
                # Deduct tax costs from settlement parties
                if gov_effect.cost_a > 0:
                    if poster_state:
                        poster_state.update_resources(-gov_effect.cost_a)
                if gov_effect.cost_b > 0:
                    if worker_state:
                        worker_state.update_resources(-gov_effect.cost_b)

            self._emit_event(Event(
                event_type=EventType.ESCROW_RELEASED,
                payload={
                    "escrow_id": bounty.escrow_id,
                    "worker_id": worker_id,
                    "amount": released,
                    "quality_score": quality_score,
                },
                epoch=self.state.current_epoch,
                step=self.state.current_step,
            ))
        else:
            poster_id = settlement["poster_id"]
            refunded = settlement["refunded_to_poster"]
            poster_state = self.state.get_agent(poster_id)
            if poster_state:
                poster_state.update_resources(refunded)

            self._emit_event(Event(
                event_type=EventType.ESCROW_REFUNDED,
                payload={
                    "escrow_id": bounty.escrow_id,
                    "poster_id": poster_id,
                    "amount": refunded,
                },
                epoch=self.state.current_epoch,
                step=self.state.current_step,
            ))

        return settlement

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
        for epoch in range(self.config.n_epochs):
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

        # Apply epoch-start governance (reputation decay, unfreezes)
        if self.governance_engine:
            gov_effect = self.governance_engine.apply_epoch_start(
                self.state, self.state.current_epoch
            )
            self._apply_governance_effect(gov_effect)

        for step in range(self.config.steps_per_epoch):
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

            if self._is_llm_agent(agent):
                action = await agent.act_async(observation)
            else:
                action = agent.act(observation)

            return agent_id, action

        # Execute all agent actions concurrently
        tasks = [get_agent_action(agent_id) for agent_id in agents_to_act]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                # Log error but continue
                continue

            agent_id, action = result
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
            from src.agents.base import InteractionProposal as AgentProposal

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
                accept = await counterparty.accept_interaction_async(agent_proposal, observation)
            else:
                accept = counterparty.accept_interaction(agent_proposal, observation)

            return accept

        # Resolve all proposals concurrently
        tasks = [resolve_proposal(p) for p in proposals]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for proposal, result in zip(proposals, results):
            if isinstance(result, Exception) or result is None:
                continue

            accept = result
            # Remove and complete
            self.state.remove_proposal(proposal.proposal_id)
            self._complete_interaction(proposal, accepted=accept)

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
                        recent_payoff = last.payoff_initiator or 0.0
                    else:
                        recent_payoff = last.payoff_counterparty or 0.0

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
        metrics = {
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
    # Boundary Support
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

        This is the main interface for agents to interact with the external world.
        The request goes through policy evaluation and leakage detection.

        Args:
            agent_id: The agent making the request
            entity_id: The external entity to interact with
            action: The type of action (call, query, send, etc.)
            payload: Optional data to send

        Returns:
            Result of the interaction (or denial information)
        """
        if self.external_world is None:
            return {"success": False, "error": "Boundaries not enabled"}

        payload = payload or {}
        metadata = {
            "sensitivity": payload.get("sensitivity", 0.0),
            "action": action,
        }

        # Check policy
        if self.policy_engine is not None:
            decision = self.policy_engine.evaluate(
                agent_id=agent_id,
                direction="outbound",
                flow_type=action,
                content=payload,
                metadata=metadata,
            )

            if not decision.allowed:
                if self.external_world:
                    self.external_world.blocked_attempts += 1
                return {
                    "success": False,
                    "blocked": True,
                    "reason": decision.reason,
                    "policy": decision.policy_name,
                }

        # Check for leakage
        if self.leakage_detector is not None:
            leakage_events = self.leakage_detector.scan(
                content=payload,
                agent_id=agent_id,
                destination_id=entity_id,
            )
            if leakage_events:
                # Log but don't necessarily block
                for event in leakage_events:
                    if event.severity >= 0.9:
                        return {
                            "success": False,
                            "blocked": True,
                            "reason": f"Critical leakage detected: {event.description}",
                            "leakage_type": event.leakage_type.value,
                        }

        # Record outbound flow
        if self.flow_tracker is not None:
            flow = InformationFlow.create(
                direction=FlowDirection.OUTBOUND,
                flow_type=FlowType.QUERY,
                source_id=agent_id,
                destination_id=entity_id,
                content=payload,
                sensitivity_score=metadata.get("sensitivity", 0.0),
            )
            self.flow_tracker.record_flow(flow)

        # Execute the interaction
        result = self.external_world.interact(
            agent_id=agent_id,
            entity_id=entity_id,
            action=action,
            payload=payload,
            rng=random.Random(self.config.seed) if self.config.seed else None,
        )

        # Record inbound flow if successful
        if result.get("success") and self.flow_tracker is not None:
            flow = InformationFlow.create(
                direction=FlowDirection.INBOUND,
                flow_type=FlowType.RESPONSE,
                source_id=entity_id,
                destination_id=agent_id,
                content=result.get("data", {}),
                sensitivity_score=result.get("sensitivity", 0.0) if isinstance(result.get("sensitivity"), (int, float)) else 0.0,
            )
            self.flow_tracker.record_flow(flow)

        return result

    def get_external_entities(
        self,
        entity_type: Optional[str] = None,
        min_trust: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get available external entities.

        Args:
            entity_type: Filter by entity type
            min_trust: Minimum trust level

        Returns:
            List of entity information dictionaries
        """
        if self.external_world is None:
            return []

        from src.boundaries.external_world import ExternalEntityType

        type_filter = None
        if entity_type:
            try:
                type_filter = ExternalEntityType(entity_type)
            except ValueError:
                pass

        entities = self.external_world.list_entities(
            entity_type=type_filter,
            min_trust=min_trust,
        )

        return [
            {
                "entity_id": e.entity_id,
                "name": e.name,
                "type": e.entity_type.value,
                "trust_level": e.trust_level,
            }
            for e in entities
        ]

    def add_external_entity(self, entity: ExternalEntity) -> None:
        """
        Add an external entity to the world.

        Args:
            entity: The entity to add
        """
        if self.external_world is not None:
            self.external_world.add_entity(entity)

    def get_boundary_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive boundary metrics.

        Returns:
            Dictionary with boundary statistics
        """
        metrics: Dict[str, Any] = {
            "boundaries_enabled": self.external_world is not None,
        }

        if self.external_world is None:
            return metrics

        # External world stats
        metrics["external_world"] = self.external_world.get_interaction_stats()

        # Flow tracker stats
        if self.flow_tracker is not None:
            summary = self.flow_tracker.get_summary()
            metrics["flows"] = {
                "total": summary.total_flows,
                "inbound": summary.inbound_flows,
                "outbound": summary.outbound_flows,
                "bytes_in": summary.total_bytes_in,
                "bytes_out": summary.total_bytes_out,
                "blocked": summary.blocked_flows,
                "sensitive": summary.sensitive_flows,
                "avg_sensitivity": summary.avg_sensitivity,
            }

            # Anomalies
            anomalies = self.flow_tracker.detect_anomalies()
            metrics["anomalies"] = anomalies

        # Policy stats
        if self.policy_engine is not None:
            metrics["policies"] = self.policy_engine.get_statistics()

        # Leakage stats
        if self.leakage_detector is not None:
            report = self.leakage_detector.generate_report()
            metrics["leakage"] = {
                "total_events": report.total_events,
                "blocked": report.blocked_count,
                "by_type": report.events_by_type,
                "avg_severity": report.avg_severity,
                "max_severity": report.max_severity,
                "recommendations": report.recommendations,
            }

        return metrics

    def get_agent_boundary_activity(self, agent_id: str) -> Dict[str, Any]:
        """
        Get boundary activity for a specific agent.

        Args:
            agent_id: The agent to query

        Returns:
            Dictionary with agent's boundary activity
        """
        activity: Dict[str, Any] = {"agent_id": agent_id}

        if self.flow_tracker is not None:
            activity["flows"] = self.flow_tracker.get_agent_flows(agent_id)

        if self.leakage_detector is not None:
            events = self.leakage_detector.get_events(agent_id=agent_id)
            activity["leakage_events"] = len(events)
            activity["leakage_severity"] = (
                max(e.severity for e in events) if events else 0.0
            )

        return activity

    def get_leakage_report(self) -> Optional[LeakageReport]:
        """
        Get the full leakage detection report.

        Returns:
            LeakageReport if boundaries enabled, None otherwise
        """
        if self.leakage_detector is None:
            return None
        return self.leakage_detector.generate_report()
