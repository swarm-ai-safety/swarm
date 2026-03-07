"""Tests for telemetry: traces and sinks."""

import json
import tempfile
from pathlib import Path

from swarm_gym.telemetry.sinks import FileSink
from swarm_gym.telemetry.trace import Trace, TraceSpan


class TestTrace:
    def test_trace_serialization(self):
        trace = Trace(episode_id="test_ep")
        span = TraceSpan(
            t=0,
            actions=[{"agent_id": "a0", "type": "cooperate"}],
            governance={"before": {"tax_rate": 0.05}, "after": {"tax_rate": 0.05}},
            events=[{"type": "AUDIT", "severity": 0.1}],
            metrics_step={"cooperation": 0.5},
        )
        trace.add_span(span)

        d = trace.to_dict()
        assert d["episode_id"] == "test_ep"
        assert len(d["spans"]) == 1
        assert d["spans"][0]["t"] == 0
        assert d["spans"][0]["actions"][0]["type"] == "cooperate"


class TestFileSink:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            sink = FileSink(path)
            data = {"episode_id": "test", "spans": []}
            sink.write(data)

            with open(path) as f:
                loaded = json.load(f)
            assert loaded["episode_id"] == "test"
