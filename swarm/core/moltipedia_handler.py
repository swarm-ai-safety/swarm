"""Moltipedia heartbeat handler for wiki actions."""

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, model_validator

from swarm.agents.base import Action, ActionType
from swarm.core.moltipedia_observables import (
    MoltipediaEditOutcome,
    MoltipediaObservableGenerator,
)
from swarm.core.proxy import ProxyObservables
from swarm.env.state import EnvState
from swarm.env.wiki import (
    EditType,
    PageStatus,
    PolicyViolationType,
    WikiPage,
    WikiTaskPool,
)
from swarm.models.agent import AgentType
from swarm.models.events import Event, EventType


class MoltipediaConfig(BaseModel):
    """Configuration for Moltipedia handler."""

    enabled: bool = True
    initial_pages: int = 50
    contested_queue_size: int = 6
    random_queue_size: int = 6
    search_queue_size: int = 6
    page_cooldown_steps: int = 3
    stub_length_threshold: int = 120
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _run_validation(self) -> "MoltipediaConfig":
        if self.initial_pages < 0:
            raise ValueError("initial_pages must be non-negative")
        if self.contested_queue_size < 0:
            raise ValueError("contested_queue_size must be non-negative")
        if self.random_queue_size < 0:
            raise ValueError("random_queue_size must be non-negative")
        if self.search_queue_size < 0:
            raise ValueError("search_queue_size must be non-negative")
        if self.page_cooldown_steps < 0:
            raise ValueError("page_cooldown_steps must be non-negative")
        if self.stub_length_threshold < 1:
            raise ValueError("stub_length_threshold must be >= 1")
        return self


class MoltipediaScorer:
    """Score edits by type."""

    POINTS = {
        EditType.CREATE: 25.0,
        EditType.EDIT: 15.0,
        EditType.CONTESTED_RESOLVE: 20.0,
        EditType.POLICY_FIX: 8.0,
    }

    def score(self, edit_type: Optional[EditType]) -> float:
        if edit_type is None:
            return 0.0
        return float(self.POINTS.get(edit_type, 0.0))


@dataclass
class MoltipediaActionResult:
    """Result of a Moltipedia action."""

    success: bool
    observables: Optional[ProxyObservables] = None
    initiator_id: str = ""
    counterparty_id: str = ""
    points: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = True


class MoltipediaHandler:
    """Handles wiki actions and lifecycle events."""

    def __init__(
        self,
        config: MoltipediaConfig,
        emit_event: Callable[[Event], None],
    ):
        self.config = config
        self._emit_event = emit_event
        self._rng = random.Random(config.seed)
        self.task_pool = WikiTaskPool(seed=config.seed)
        self.scorer = MoltipediaScorer()
        self.observable_generator = MoltipediaObservableGenerator()

        if self.config.initial_pages > 0:
            self.task_pool.seed_pages(self.config.initial_pages)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def build_observation_fields(
        self,
        agent_id: str,
        state: EnvState,
    ) -> Dict[str, Any]:
        """Build Moltipedia-related observation fields."""
        contested = self.task_pool.get_contested_pages(
            self.config.contested_queue_size,
            current_step=state.current_step,
        )
        random_pages = self.task_pool.get_random_pages(
            self.config.random_queue_size,
            current_step=state.current_step,
        )
        search_pages = self.task_pool.get_low_quality_pages(
            self.config.search_queue_size,
            current_step=state.current_step,
        )

        leaderboard = [
            {"agent_id": aid, "points": pts}
            for aid, pts in sorted(
                self.task_pool.leaderboard.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        return {
            "contested_pages": [p.to_dict() for p in contested],
            "random_pages": [p.to_dict() for p in random_pages],
            "search_results": [p.to_dict() for p in search_pages],
            "leaderboard": leaderboard,
            "agent_points": float(self.task_pool.leaderboard.get(agent_id, 0.0)),
            "heartbeat_status": {
                "epoch": state.current_epoch,
                "step": state.current_step,
                "pages": len(self.task_pool.all_pages()),
            },
        }

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def handle_action(self, action: Action, state: EnvState) -> MoltipediaActionResult:
        """Handle a Moltipedia wiki action."""
        if action.action_type == ActionType.CREATE_PAGE:
            return self._handle_create(action, state)
        if action.action_type == ActionType.EDIT_PAGE:
            return self._handle_edit(action, state)
        if action.action_type == ActionType.FILE_OBJECTION:
            return self._handle_objection(action, state)
        if action.action_type == ActionType.POLICY_FLAG:
            return self._handle_policy_flag(action, state)

        return MoltipediaActionResult(success=False)

    def record_points(
        self,
        agent_id: str,
        points_awarded: float,
        state: EnvState,
        *,
        page_id: Optional[str] = None,
        edit_type: Optional[str] = None,
    ) -> None:
        """Update leaderboard and emit points event."""
        if points_awarded <= 0:
            return
        self.task_pool.award_points(agent_id, points_awarded)
        self._emit_event(Event(
            event_type=EventType.POINTS_AWARDED,
            agent_id=agent_id,
            payload={
                "points": points_awarded,
                "page_id": page_id,
                "edit_type": edit_type,
            },
            epoch=state.current_epoch,
            step=state.current_step,
        ))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_create(self, action: Action, state: EnvState) -> MoltipediaActionResult:
        title = action.metadata.get("title", action.content or "Untitled Page")
        content = action.content or action.metadata.get("content", "")
        status = (
            PageStatus.STUB
            if len(content) < self.config.stub_length_threshold
            else PageStatus.DRAFT
        )
        agent_state = state.get_agent(action.agent_id)
        agent_type = agent_state.agent_type if agent_state else AgentType.HONEST

        quality_delta = self._quality_delta_for_agent(agent_type, is_create=True)
        quality_score = max(0.1, min(1.0, 0.2 + quality_delta))
        violations = self._policy_violations_for_agent(agent_type)

        page = WikiPage(
            title=title,
            content=content,
            status=status,
            quality_score=quality_score,
            policy_violations=violations,
            created_by=action.agent_id,
            last_editor=action.agent_id,
        )
        self.task_pool.add_page(page)

        edit_type = EditType.CREATE
        points = self.scorer.score(edit_type)
        outcome = self._build_outcome(quality_delta, violations, agent_type)
        observables = self.observable_generator.generate(outcome)

        self._emit_event(Event(
            event_type=EventType.PAGE_CREATED,
            agent_id=action.agent_id,
            payload={"page_id": page.page_id, "title": page.title},
            epoch=state.current_epoch,
            step=state.current_step,
        ))

        return MoltipediaActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id="moltipedia",
            points=points,
            metadata={
                "moltipedia": True,
                "page_id": page.page_id,
                "edit_type": edit_type.value,
                "created_by": page.created_by,
                "points": points,
            },
        )

    def _handle_edit(self, action: Action, state: EnvState) -> MoltipediaActionResult:
        page = self.task_pool.get_page(action.target_id)
        if page is None:
            return MoltipediaActionResult(success=False)

        agent_state = state.get_agent(action.agent_id)
        agent_type = agent_state.agent_type if agent_state else AgentType.HONEST

        old_quality = page.quality_score
        old_status = page.status
        old_violations = list(page.policy_violations)
        new_content = action.content or page.content
        violations = self._updated_policy_violations(page, agent_type)

        quality_delta = self._quality_delta_for_agent(agent_type)
        edit_type = self._classify_edit_type(old_status, old_violations, violations)
        page.apply_edit(
            editor_id=action.agent_id,
            new_content=new_content,
            edit_type=edit_type,
            delta_quality=quality_delta,
            policy_violations=violations,
            epoch=state.current_epoch,
            step=state.current_step,
        )
        points = self.scorer.score(edit_type)

        page.status = self._status_for_page(page)
        page.cooldown_until = state.current_step + self.config.page_cooldown_steps

        outcome = self._build_outcome(page.quality_score - old_quality, violations, agent_type)
        observables = self.observable_generator.generate(outcome)

        if page.created_by and page.created_by != action.agent_id:
            counterparty_id = page.created_by
        else:
            counterparty_id = "moltipedia"

        self._emit_event(Event(
            event_type=EventType.PAGE_EDITED,
            agent_id=action.agent_id,
            payload={
                "page_id": page.page_id,
                "edit_type": edit_type.value,
                "quality_delta": page.quality_score - old_quality,
            },
            epoch=state.current_epoch,
            step=state.current_step,
        ))

        return MoltipediaActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id=counterparty_id,
            points=points,
            metadata={
                "moltipedia": True,
                "page_id": page.page_id,
                "edit_type": edit_type.value,
                "created_by": page.created_by,
                "points": points,
            },
        )

    def _handle_objection(self, action: Action, state: EnvState) -> MoltipediaActionResult:
        page = self.task_pool.get_page(action.target_id)
        if page is None:
            return MoltipediaActionResult(success=False)

        page.status = PageStatus.CONTESTED
        page.cooldown_until = state.current_step + self.config.page_cooldown_steps

        self._emit_event(Event(
            event_type=EventType.OBJECTION_FILED,
            agent_id=action.agent_id,
            payload={"page_id": page.page_id, "reason": action.content},
            epoch=state.current_epoch,
            step=state.current_step,
        ))

        return MoltipediaActionResult(
            success=True,
            observables=None,
            initiator_id=action.agent_id,
            counterparty_id=(
                page.created_by if page.created_by and page.created_by != action.agent_id else "moltipedia"
            ),
            points=0.0,
            metadata={
                "moltipedia": True,
                "page_id": page.page_id,
                "edit_type": None,
                "created_by": page.created_by,
                "points": 0.0,
            },
            accepted=True,
        )

    def _handle_policy_flag(self, action: Action, state: EnvState) -> MoltipediaActionResult:
        page = self.task_pool.get_page(action.target_id)
        if page is None:
            return MoltipediaActionResult(success=False)

        violation = self._violation_from_action(action)
        if violation not in page.policy_violations:
            page.policy_violations.append(violation)
        page.status = PageStatus.CONTESTED

        self._emit_event(Event(
            event_type=EventType.POLICY_VIOLATION_FLAGGED,
            agent_id=action.agent_id,
            payload={
                "page_id": page.page_id,
                "violation": violation.value,
            },
            epoch=state.current_epoch,
            step=state.current_step,
        ))

        return MoltipediaActionResult(
            success=True,
            observables=None,
            initiator_id=action.agent_id,
            counterparty_id=(
                page.created_by if page.created_by and page.created_by != action.agent_id else "moltipedia"
            ),
            points=0.0,
            metadata={
                "moltipedia": True,
                "page_id": page.page_id,
                "edit_type": None,
                "created_by": page.created_by,
                "points": 0.0,
            },
            accepted=True,
        )

    def _quality_delta_for_agent(self, agent_type: AgentType, is_create: bool = False) -> float:
        if agent_type == AgentType.ADVERSARIAL:
            return self._rng.uniform(-0.35, -0.1)
        if agent_type == AgentType.OPPORTUNISTIC:
            return self._rng.uniform(0.05, 0.2) if not is_create else self._rng.uniform(0.05, 0.15)
        if agent_type == AgentType.DECEPTIVE:
            return self._rng.uniform(0.05, 0.18)
        return self._rng.uniform(0.15, 0.35) if not is_create else self._rng.uniform(0.1, 0.25)

    def _policy_violations_for_agent(self, agent_type: AgentType) -> list[PolicyViolationType]:
        if agent_type == AgentType.ADVERSARIAL:
            return [self._rng.choice([
                PolicyViolationType.NEUTRALITY,
                PolicyViolationType.SCOPE,
                PolicyViolationType.SPECULATION,
                PolicyViolationType.SOURCING,
            ])]
        if agent_type == AgentType.OPPORTUNISTIC:
            return [PolicyViolationType.SOURCING] if self._rng.random() < 0.3 else []
        if agent_type == AgentType.DECEPTIVE:
            return [PolicyViolationType.SCOPE] if self._rng.random() < 0.25 else []
        return []

    def _updated_policy_violations(
        self, page: WikiPage, agent_type: AgentType
    ) -> list[PolicyViolationType]:
        if agent_type == AgentType.ADVERSARIAL:
            violations = list(set(page.policy_violations) | set(self._policy_violations_for_agent(agent_type)))
            return violations
        if agent_type == AgentType.OPPORTUNISTIC and self._rng.random() < 0.2:
            return list(set(page.policy_violations) | {PolicyViolationType.SOURCING})
        if agent_type == AgentType.DECEPTIVE and self._rng.random() < 0.2:
            return list(set(page.policy_violations) | {PolicyViolationType.SCOPE})
        return []

    def _classify_edit_type(
        self,
        old_status: PageStatus,
        old_violations: list[PolicyViolationType],
        new_violations: list[PolicyViolationType],
    ) -> EditType:
        if old_status == PageStatus.CONTESTED and not new_violations:
            return EditType.CONTESTED_RESOLVE
        if old_violations and not new_violations:
            return EditType.POLICY_FIX
        return EditType.EDIT

    def _status_for_page(self, page: WikiPage) -> PageStatus:
        if page.policy_violations:
            return PageStatus.CONTESTED
        if page.quality_score >= 0.7:
            return PageStatus.PUBLISHED
        if len(page.content) < self.config.stub_length_threshold:
            return PageStatus.STUB
        return PageStatus.DRAFT

    def _build_outcome(
        self,
        quality_delta: float,
        violations: list[PolicyViolationType],
        agent_type: AgentType,
    ) -> MoltipediaEditOutcome:
        rework_count = 1 if quality_delta < 0 or violations else 0
        misuse_flags = 1 if agent_type == AgentType.ADVERSARIAL or quality_delta < 0 else 0
        engagement_delta = max(-0.6, min(0.6, quality_delta * 0.8))
        return MoltipediaEditOutcome(
            quality_delta=quality_delta,
            rework_count=rework_count,
            verifier_rejections=len(violations),
            tool_misuse_flags=misuse_flags,
            engagement_delta=engagement_delta,
        )

    def _violation_from_action(self, action: Action) -> PolicyViolationType:
        raw = action.metadata.get("violation", "")
        for violation in PolicyViolationType:
            if violation.value == raw:
                return violation
        return PolicyViolationType.SOURCING
