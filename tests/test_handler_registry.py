"""Tests for the handler registry and plugin architecture."""

from __future__ import annotations

from typing import Any, FrozenSet

import pytest

from swarm.agents.base import ActionType
from swarm.core.handler import Handler, HandlerActionResult
from swarm.core.handler_registry import HandlerRegistry
from swarm.logging.event_bus import EventBus


class _StubHandler(Handler):
    """Minimal handler for testing."""

    _action_types: FrozenSet[ActionType]

    def __init__(self, action_types: FrozenSet[ActionType], *, name: str = "stub"):
        super().__init__(event_bus=EventBus())
        self._action_types = action_types
        self.name = name
        self.epoch_starts = 0
        self.epoch_ends = 0
        self.steps: list = []
        self.pre_observations: list = []
        self.post_finalizes: list = []

    @staticmethod
    def handled_action_types() -> FrozenSet[ActionType]:
        # Will be overridden per-instance
        raise NotImplementedError

    def handle_action(self, action: Any, state: Any) -> HandlerActionResult:
        return HandlerActionResult(success=True)

    def on_epoch_start(self, state: Any) -> None:
        self.epoch_starts += 1

    def on_epoch_end(self, state: Any) -> None:
        self.epoch_ends += 1

    def on_step(self, state: Any, step: int) -> None:
        self.steps.append(step)

    def on_pre_observation(self, agent_id: str, state: Any) -> None:
        self.pre_observations.append(agent_id)


def _make_handler(
    action_types: FrozenSet[ActionType], name: str = "stub"
) -> _StubHandler:
    """Create a stub handler with the given action types."""
    handler = _StubHandler(action_types, name=name)
    # Patch handled_action_types to return the instance's value
    handler.handled_action_types = staticmethod(lambda: action_types)  # type: ignore[assignment]
    return handler


class TestHandlerRegistry:
    def test_register_and_lookup(self):
        registry = HandlerRegistry()
        handler = _make_handler(frozenset({ActionType.POST}))

        registry.register(handler)

        assert registry.get_handler(ActionType.POST) is handler
        assert registry.has_handler(ActionType.POST)
        assert not registry.has_handler(ActionType.VOTE)

    def test_all_handlers_preserves_order(self):
        registry = HandlerRegistry()
        h1 = _make_handler(frozenset({ActionType.POST}), name="first")
        h2 = _make_handler(frozenset({ActionType.VOTE}), name="second")

        registry.register(h1)
        registry.register(h2)

        handlers = registry.all_handlers()
        assert len(handlers) == 2
        assert handlers[0] is h1
        assert handlers[1] is h2

    def test_conflict_raises(self):
        registry = HandlerRegistry()
        h1 = _make_handler(frozenset({ActionType.POST}), name="first")
        h2 = _make_handler(frozenset({ActionType.POST}), name="second")

        registry.register(h1)
        with pytest.raises(ValueError, match="already claimed"):
            registry.register(h2)

    def test_same_handler_registered_twice_is_deduplicated(self):
        registry = HandlerRegistry()
        handler = _make_handler(frozenset({ActionType.POST}))

        registry.register(handler)
        registry.register(handler)

        assert len(registry.all_handlers()) == 1

    def test_get_handler_returns_none_for_unregistered(self):
        registry = HandlerRegistry()
        assert registry.get_handler(ActionType.POST) is None

    def test_multiple_action_types_per_handler(self):
        registry = HandlerRegistry()
        handler = _make_handler(
            frozenset({ActionType.POST, ActionType.REPLY, ActionType.VOTE})
        )

        registry.register(handler)

        assert registry.get_handler(ActionType.POST) is handler
        assert registry.get_handler(ActionType.REPLY) is handler
        assert registry.get_handler(ActionType.VOTE) is handler
        assert len(registry.all_handlers()) == 1


class TestHandlerActionResult:
    def test_default_values(self):
        result = HandlerActionResult(success=True)
        assert result.success is True
        assert result.observables is None
        assert result.initiator_id == ""
        assert result.counterparty_id == ""
        assert result.accepted is True
        assert result.tau == 0.0
        assert result.ground_truth is None
        assert result.metadata == {}

    def test_with_observables(self):
        from swarm.core.proxy import ProxyObservables

        obs = ProxyObservables(task_progress_delta=0.5)
        result = HandlerActionResult(
            success=True,
            observables=obs,
            initiator_id="a1",
            counterparty_id="a2",
            tau=-10.0,
        )
        assert result.observables is obs
        assert result.tau == -10.0


class TestHandlerLifecycleHooks:
    """Test that Handler's default hook methods are no-ops."""

    def test_default_on_step_is_noop(self):
        """Default on_step does not raise."""

        class MinimalHandler(Handler):
            @staticmethod
            def handled_action_types() -> FrozenSet:
                return frozenset()

            def handle_action(self, action: Any, state: Any) -> Any:
                return HandlerActionResult(success=False)

        h = MinimalHandler(event_bus=EventBus())
        # Should not raise
        h.on_step(None, 0)
        h.on_pre_observation("agent_1", None)
        h.post_finalize(None, None, None, None)

    def test_observation_field_mapping_default_empty(self):
        class MinimalHandler(Handler):
            @staticmethod
            def handled_action_types() -> FrozenSet:
                return frozenset()

            def handle_action(self, action: Any, state: Any) -> Any:
                return HandlerActionResult(success=False)

        h = MinimalHandler(event_bus=EventBus())
        assert h.observation_field_mapping() == {}
        assert h.build_observation_fields("a", None) == {}
