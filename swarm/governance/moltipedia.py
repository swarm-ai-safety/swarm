"""Moltipedia-specific governance levers."""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from swarm.governance.levers import GovernanceLever, LeverEffect

if TYPE_CHECKING:
    from swarm.env.state import EnvState
    from swarm.governance.config import GovernanceConfig
    from swarm.models.interaction import SoftInteraction


def _points_from_interaction(interaction: "SoftInteraction") -> float:
    meta = getattr(interaction, "metadata", {}) or {}
    return float(meta.get("points", 0.0))


def _page_id_from_interaction(interaction: "SoftInteraction") -> Optional[str]:
    meta = getattr(interaction, "metadata", {}) or {}
    page_id = meta.get("page_id")
    return str(page_id) if page_id else None


def _edit_type_from_interaction(interaction: "SoftInteraction") -> Optional[str]:
    meta = getattr(interaction, "metadata", {}) or {}
    edit_type = meta.get("edit_type")
    return str(edit_type) if edit_type else None


def _created_by_from_interaction(interaction: "SoftInteraction") -> Optional[str]:
    meta = getattr(interaction, "metadata", {}) or {}
    created_by = meta.get("created_by")
    return str(created_by) if created_by else None


class PairCapLever(GovernanceLever):
    """Max scored fixes per agent pair per epoch."""

    def __init__(self, config: "GovernanceConfig"):
        super().__init__(config)
        self._pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    @property
    def name(self) -> str:
        return "moltipedia_pair_cap"

    def on_epoch_start(self, state: "EnvState", epoch: int) -> LeverEffect:
        self._pair_counts.clear()
        return LeverEffect(lever_name=self.name)

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        if not getattr(self.config, "moltipedia_pair_cap_enabled", False):
            return LeverEffect(lever_name=self.name)

        points = _points_from_interaction(interaction)
        if points <= 0:
            return LeverEffect(lever_name=self.name)

        if not interaction.counterparty or interaction.counterparty == "moltipedia":
            return LeverEffect(lever_name=self.name)

        if interaction.initiator <= interaction.counterparty:
            pair = (interaction.initiator, interaction.counterparty)
        else:
            pair = (interaction.counterparty, interaction.initiator)
        self._pair_counts[pair] += 1

        max_per_pair = getattr(self.config, "moltipedia_pair_cap_max", 2)
        if self._pair_counts[pair] > max_per_pair:
            return LeverEffect(
                lever_name=self.name,
                cost_a=points,
                details={
                    "pair": pair,
                    "count": self._pair_counts[pair],
                    "max": max_per_pair,
                },
            )

        return LeverEffect(lever_name=self.name)


class PageCooldownLever(GovernanceLever):
    """Max 1 scored fix per page per N steps."""

    def __init__(self, config: "GovernanceConfig"):
        super().__init__(config)
        self._last_scored_step: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "moltipedia_page_cooldown"

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        if not getattr(self.config, "moltipedia_page_cooldown_enabled", False):
            return LeverEffect(lever_name=self.name)

        points = _points_from_interaction(interaction)
        page_id = _page_id_from_interaction(interaction)
        if points <= 0 or not page_id:
            return LeverEffect(lever_name=self.name)

        cooldown_steps = getattr(self.config, "moltipedia_page_cooldown_steps", 3)
        last_step = self._last_scored_step.get(page_id)
        current_step = state.current_step

        if last_step is not None and current_step - last_step < cooldown_steps:
            return LeverEffect(
                lever_name=self.name,
                cost_a=points,
                details={
                    "page_id": page_id,
                    "last_step": last_step,
                    "current_step": current_step,
                    "cooldown_steps": cooldown_steps,
                },
            )

        self._last_scored_step[page_id] = current_step
        return LeverEffect(lever_name=self.name)


class DailyPointCapLever(GovernanceLever):
    """Max daily policy-fix points per agent per epoch."""

    def __init__(self, config: "GovernanceConfig"):
        super().__init__(config)
        self._policy_fix_points: Dict[str, float] = defaultdict(float)

    @property
    def name(self) -> str:
        return "moltipedia_daily_cap"

    def on_epoch_start(self, state: "EnvState", epoch: int) -> LeverEffect:
        self._policy_fix_points.clear()
        return LeverEffect(lever_name=self.name)

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        if not getattr(self.config, "moltipedia_daily_cap_enabled", False):
            return LeverEffect(lever_name=self.name)

        edit_type = _edit_type_from_interaction(interaction)
        if edit_type != "policy_fix":
            return LeverEffect(lever_name=self.name)

        points = _points_from_interaction(interaction)
        if points <= 0:
            return LeverEffect(lever_name=self.name)

        cap = getattr(self.config, "moltipedia_daily_policy_fix_cap", 24.0)
        current = self._policy_fix_points[interaction.initiator]
        remaining = cap - current

        if remaining <= 0:
            return LeverEffect(
                lever_name=self.name,
                cost_a=points,
                details={"cap": cap, "current": current, "points": points},
            )

        if points > remaining:
            self._policy_fix_points[interaction.initiator] += remaining
            return LeverEffect(
                lever_name=self.name,
                cost_a=points - remaining,
                details={"cap": cap, "current": current, "points": points},
            )

        self._policy_fix_points[interaction.initiator] += points
        return LeverEffect(lever_name=self.name)


class NoSelfFixLever(GovernanceLever):
    """Cancels scoring when agent edits own page."""

    @property
    def name(self) -> str:
        return "moltipedia_no_self_fix"

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        if not getattr(self.config, "moltipedia_no_self_fix", False):
            return LeverEffect(lever_name=self.name)

        points = _points_from_interaction(interaction)
        created_by = _created_by_from_interaction(interaction)
        edit_type = _edit_type_from_interaction(interaction)
        if points <= 0 or not created_by:
            return LeverEffect(lever_name=self.name)
        if edit_type == "create":
            return LeverEffect(lever_name=self.name)

        if created_by == interaction.initiator:
            return LeverEffect(
                lever_name=self.name,
                cost_a=points,
                details={"created_by": created_by},
            )

        return LeverEffect(lever_name=self.name)
