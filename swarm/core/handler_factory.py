"""Factory for constructing handlers from OrchestratorConfig.

Moves the ~200 lines of conditional handler instantiation out of
``Orchestrator.__init__`` into a focused builder that returns a
populated ``HandlerRegistry`` plus named references for handlers
that need direct access (e.g. marketplace, tierra).
"""

from __future__ import annotations

import random
from typing import Any, Optional

from swarm.core.boundary_handler import BoundaryHandler
from swarm.core.coding_handler import CodingHandler
from swarm.core.core_interaction_handler import CoreInteractionHandler
from swarm.core.feed_handler import FeedHandler
from swarm.core.handler_registry import HandlerRegistry
from swarm.core.kernel_handler import KernelOracleHandler
from swarm.core.marketplace_handler import MarketplaceHandler
from swarm.core.memory_handler import MemoryHandler
from swarm.core.moltbook_handler import MoltbookHandler
from swarm.core.moltipedia_handler import MoltipediaHandler
from swarm.core.rivals_handler import RivalsHandler
from swarm.core.scholar_handler import ScholarHandler
from swarm.core.task_handler import TaskHandler
from swarm.env.feed import Feed
from swarm.env.marketplace import Marketplace
from swarm.env.tasks import TaskPool
from swarm.logging.event_bus import EventBus


class HandlerSet:
    """Container for the constructed handler registry and named references.

    Attributes are ``None`` when the corresponding feature is disabled.
    """

    def __init__(self) -> None:
        self.registry = HandlerRegistry()

        # Named references for handlers needing direct orchestrator access
        self.marketplace: Optional[Marketplace] = None
        self.marketplace_handler: Optional[MarketplaceHandler] = None
        self.moltipedia_handler: Optional[Any] = None
        self.moltbook_handler: Optional[Any] = None
        self.memory_handler: Optional[Any] = None
        self.scholar_handler: Optional[Any] = None
        self.kernel_handler: Optional[Any] = None
        self.rivals_handler: Optional[Any] = None
        self.awm_handler: Optional[Any] = None
        self.evo_game_handler: Optional[Any] = None
        self.tierra_handler: Optional[Any] = None
        self.boundary_handler: Optional[BoundaryHandler] = None
        self.feed_handler: Optional[FeedHandler] = None
        self.core_interaction_handler: Optional[CoreInteractionHandler] = None
        self.task_handler: Optional[TaskHandler] = None
        self.coding_handler: Optional[CodingHandler] = None

        # Boundary sub-components exposed for public API
        self.external_world: Optional[Any] = None
        self.flow_tracker: Optional[Any] = None
        self.policy_engine: Optional[Any] = None
        self.leakage_detector: Optional[Any] = None


def build_handlers(
    config: Any,  # OrchestratorConfig
    *,
    event_bus: EventBus,
    feed: Feed,
    task_pool: TaskPool,
    finalizer: Any,  # InteractionFinalizer
    network: Optional[Any],
    governance_engine: Optional[Any],
    rng: random.Random,
) -> HandlerSet:
    """Build all handlers from *config* and return a populated ``HandlerSet``."""
    hs = HandlerSet()

    # -- Marketplace --
    if config.marketplace_config is not None:
        marketplace = Marketplace(config.marketplace_config)
        hs.marketplace = marketplace
        hs.marketplace_handler = MarketplaceHandler(
            marketplace=marketplace,
            task_pool=task_pool,
            event_bus=event_bus,
            enable_rate_limits=config.enable_rate_limits,
        )
        hs.registry.register(hs.marketplace_handler)

    # -- Moltipedia --
    if config.moltipedia_config is not None:
        hs.moltipedia_handler = MoltipediaHandler(
            config=config.moltipedia_config,
            event_bus=event_bus,
        )
        hs.registry.register(hs.moltipedia_handler)

    # -- Moltbook --
    if config.moltbook_config is not None:
        rate_limit_lever = None
        challenge_lever = None
        if governance_engine is not None:
            rate_limit_lever = governance_engine.get_moltbook_rate_limit_lever()
            challenge_lever = governance_engine.get_moltbook_challenge_lever()
        hs.moltbook_handler = MoltbookHandler(
            config=config.moltbook_config,
            governance_config=config.governance_config,
            rate_limit_lever=rate_limit_lever,
            challenge_lever=challenge_lever,
            event_bus=event_bus,
        )
        hs.registry.register(hs.moltbook_handler)

    # -- Memory tier --
    if config.memory_tier_config is not None:
        hs.memory_handler = MemoryHandler(
            config=config.memory_tier_config,
            event_bus=event_bus,
        )
        hs.registry.register(hs.memory_handler)

    # -- Scholar --
    if config.scholar_config is not None:
        hs.scholar_handler = ScholarHandler(
            config=config.scholar_config,
            event_bus=event_bus,
        )
        hs.registry.register(hs.scholar_handler)

    # -- Kernel oracle --
    if config.kernel_oracle_config is not None:
        hs.kernel_handler = KernelOracleHandler(
            config=config.kernel_oracle_config,
            event_bus=event_bus,
        )
        hs.registry.register(hs.kernel_handler)

    # -- Rivals --
    if config.rivals_config is not None:
        hs.rivals_handler = RivalsHandler(
            config=config.rivals_config,
            event_bus=event_bus,
        )
        hs.registry.register(hs.rivals_handler)

    # -- AWM (lazy import) --
    if config.awm_config is not None:
        from swarm.core.awm_handler import AWMHandler

        hs.awm_handler = AWMHandler(
            config=config.awm_config,
            event_bus=event_bus,
            seed=config.seed,
        )
        hs.registry.register(hs.awm_handler)

    # -- Evolutionary game (lazy import) --
    if config.evo_game_config is not None:
        from swarm.core.evo_game_handler import EvoGameConfig, EvolutionaryGameHandler

        evo_cfg = config.evo_game_config
        if not isinstance(evo_cfg, EvoGameConfig):
            evo_cfg = (
                EvoGameConfig(**evo_cfg)
                if isinstance(evo_cfg, dict)
                else evo_cfg
            )
        hs.evo_game_handler = EvolutionaryGameHandler(
            config=evo_cfg,
            event_bus=event_bus,
            rng=rng,
        )
        hs.registry.register(hs.evo_game_handler)

    # -- Tierra (lazy import) --
    if config.tierra_config is not None:
        from swarm.core.tierra_handler import TierraConfig, TierraHandler

        tierra_cfg = config.tierra_config
        if not isinstance(tierra_cfg, TierraConfig):
            tierra_cfg = (
                TierraConfig(**tierra_cfg)
                if isinstance(tierra_cfg, dict)
                else tierra_cfg
            )
        hs.tierra_handler = TierraHandler(
            config=tierra_cfg,
            event_bus=event_bus,
            rng=rng,
        )
        hs.registry.register(hs.tierra_handler)

    # -- Boundary --
    if config.enable_boundaries:
        from swarm.boundaries.external_world import ExternalWorld
        from swarm.boundaries.information_flow import FlowTracker
        from swarm.boundaries.leakage import LeakageDetector
        from swarm.boundaries.policies import PolicyEngine

        external_world = ExternalWorld().create_default_world()
        flow_tracker = FlowTracker(
            sensitivity_threshold=config.boundary_sensitivity_threshold
        )
        policy_engine = PolicyEngine().create_default_policies()
        leakage_detector = LeakageDetector()

        hs.external_world = external_world
        hs.flow_tracker = flow_tracker
        hs.policy_engine = policy_engine
        hs.leakage_detector = leakage_detector

        hs.boundary_handler = BoundaryHandler(
            external_world=external_world,
            flow_tracker=flow_tracker,
            policy_engine=policy_engine,
            leakage_detector=leakage_detector,
            event_bus=event_bus,
            seed=config.seed,
        )
        # Note: BoundaryHandler is not registered in the action registry
        # because it uses a separate request_external_interaction path.

    # -- Core handlers (always present) --
    hs.feed_handler = FeedHandler(
        feed=feed,
        max_content_length=config.max_content_length,
        event_bus=event_bus,
    )
    hs.registry.register(hs.feed_handler)

    hs.core_interaction_handler = CoreInteractionHandler(
        finalizer=finalizer,
        network=network,
        event_bus=event_bus,
    )
    hs.registry.register(hs.core_interaction_handler)

    hs.task_handler = TaskHandler(
        task_pool=task_pool,
        event_bus=event_bus,
    )
    hs.registry.register(hs.task_handler)

    hs.coding_handler = CodingHandler(
        event_bus=event_bus,
        rng=rng,
    )
    hs.registry.register(hs.coding_handler)

    return hs
