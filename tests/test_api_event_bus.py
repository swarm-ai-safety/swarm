import asyncio

import pytest

from swarm.api.event_bus import EventBus, SimEvent, SimEventType


def _event() -> SimEvent:
    return SimEvent(
        event_type=SimEventType.STEP_COMPLETE,
        simulation_id="sim-1",
        agent_id="agent-1",
    )


@pytest.mark.asyncio
async def test_publish_sync_drops_cross_loop_events_when_queue_full():
    bus = EventBus()
    queue = bus.subscribe("sim-1", "agent-1")
    for _ in range(queue.maxsize):
        queue.put_nowait(_event())

    count = await asyncio.to_thread(bus.publish_sync, _event())
    await asyncio.sleep(0)

    assert count == 1
    assert queue.qsize() == queue.maxsize
