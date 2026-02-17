"""Tests for unified provenance trace schema and exporters."""

from datetime import datetime

from swarm.logging.event_log import EventLog
from swarm.models.events import (
    Event,
    EventType,
    agent_message_event,
    artifact_created_event,
    audit_event,
    generate_deterministic_id,
    intervention_event,
    tool_call_executed_event,
)

# =============================================================================
# Tests for deterministic ID generation
# =============================================================================


class TestDeterministicIds:
    """Tests for generate_deterministic_id function."""

    def test_same_inputs_same_id(self):
        """Same inputs generate same ID."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        id1 = generate_deterministic_id(
            event_type="tool_call",
            agent_id="agent_1",
            timestamp=ts,
        )
        id2 = generate_deterministic_id(
            event_type="tool_call",
            agent_id="agent_1",
            timestamp=ts,
        )
        assert id1 == id2

    def test_different_inputs_different_id(self):
        """Different inputs generate different IDs."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        id1 = generate_deterministic_id(
            event_type="tool_call",
            agent_id="agent_1",
            timestamp=ts,
        )
        id2 = generate_deterministic_id(
            event_type="tool_call",
            agent_id="agent_2",
            timestamp=ts,
        )
        assert id1 != id2

    def test_id_length(self):
        """Generated ID is 12 characters."""
        id_ = generate_deterministic_id(
            event_type="test",
            agent_id="agent_1",
        )
        assert len(id_) == 12

    def test_kwargs_included_in_hash(self):
        """Additional kwargs affect the hash."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        id1 = generate_deterministic_id(
            event_type="tool_call",
            agent_id="agent_1",
            timestamp=ts,
            tool_name="query",
        )
        id2 = generate_deterministic_id(
            event_type="tool_call",
            agent_id="agent_1",
            timestamp=ts,
            tool_name="execute",
        )
        assert id1 != id2


# =============================================================================
# Tests for provenance-aware event factories
# =============================================================================


class TestProvenanceEventFactories:
    """Tests for provenance-aware event creation functions."""

    def test_tool_call_event_has_ids(self):
        """Tool call event includes tool_call_id and provenance_id."""
        event = tool_call_executed_event(
            agent_id="agent_1",
            tool_name="query_database",
            arguments={"query": "SELECT 1"},
            result={"rows": 1},
            success=True,
        )
        assert event.tool_call_id is not None
        assert event.provenance_id is not None
        assert event.event_type == EventType.AWM_TOOL_CALL_EXECUTED
        assert event.agent_id == "agent_1"
        assert event.payload["tool_name"] == "query_database"
        assert event.payload["success"] is True

    def test_tool_call_event_with_parent(self):
        """Tool call event can reference parent event."""
        event = tool_call_executed_event(
            agent_id="agent_1",
            tool_name="query",
            arguments={},
            result={},
            success=True,
            parent_event_id="parent_123",
        )
        assert event.parent_event_id == "parent_123"

    def test_artifact_event_has_ids(self):
        """Artifact event includes artifact_id and provenance_id."""
        event = artifact_created_event(
            agent_id="agent_1",
            artifact_id="artifact_123",
            artifact_type="memory",
            artifact_data={"title": "Test", "summary": "Summary"},
        )
        assert event.artifact_id == "artifact_123"
        assert event.provenance_id is not None
        assert event.event_type == EventType.MEMORY_WRITTEN
        assert event.payload["artifact_type"] == "memory"

    def test_audit_event_has_ids(self):
        """Audit event includes audit_id and provenance_id."""
        event = audit_event(
            agent_id="auditor_1",
            audit_type="council",
            audit_result={"decision": "approve", "confidence": 0.9},
            audited_event_id="event_456",
        )
        assert event.audit_id is not None
        assert event.provenance_id is not None
        assert event.event_type == EventType.COUNCIL_AUDIT
        assert event.payload["audit_type"] == "council"
        assert event.payload["audited_event_id"] == "event_456"

    def test_intervention_event_has_ids(self):
        """Intervention event includes intervention_id and provenance_id."""
        event = intervention_event(
            agent_id="governance",
            intervention_type="freeze",
            intervention_action="freeze_agent",
            intervention_reason="suspicious_behavior",
            affected_agents=["agent_2", "agent_3"],
        )
        assert event.intervention_id is not None
        assert event.provenance_id is not None
        assert event.event_type == EventType.GOVERNANCE_COST_APPLIED
        assert event.payload["intervention_type"] == "freeze"
        assert event.payload["affected_agents"] == ["agent_2", "agent_3"]

    def test_agent_message_event_has_ids(self):
        """Agent message event includes provenance_id."""
        event = agent_message_event(
            agent_id="agent_1",
            message_role="assistant",
            message_content="Hello, world!",
            tool_calls=[{"name": "search", "args": {}}],
        )
        assert event.provenance_id is not None
        assert event.event_type == EventType.AGENT_STATE_UPDATED
        assert event.payload["message_role"] == "assistant"
        assert event.payload["message_content"] == "Hello, world!"
        assert len(event.payload["tool_calls"]) == 1


# =============================================================================
# Tests for Event serialization with provenance
# =============================================================================


class TestEventSerialization:
    """Tests for Event to_dict and from_dict with provenance fields."""

    def test_to_dict_includes_provenance_fields(self):
        """to_dict includes all provenance fields."""
        event = Event(
            event_type=EventType.AWM_TOOL_CALL_EXECUTED,
            agent_id="agent_1",
            provenance_id="prov_123",
            parent_event_id="parent_456",
            tool_call_id="tool_789",
            artifact_id="artifact_012",
            audit_id="audit_345",
            intervention_id="intervention_678",
        )
        d = event.to_dict()
        assert d["provenance_id"] == "prov_123"
        assert d["parent_event_id"] == "parent_456"
        assert d["tool_call_id"] == "tool_789"
        assert d["artifact_id"] == "artifact_012"
        assert d["audit_id"] == "audit_345"
        assert d["intervention_id"] == "intervention_678"

    def test_from_dict_restores_provenance_fields(self):
        """from_dict correctly restores provenance fields."""
        data = {
            "event_id": "evt_123",
            "timestamp": "2024-01-01T12:00:00",
            "event_type": "awm_tool_call_executed",
            "agent_id": "agent_1",
            "provenance_id": "prov_123",
            "parent_event_id": "parent_456",
            "tool_call_id": "tool_789",
            "artifact_id": "artifact_012",
            "audit_id": "audit_345",
            "intervention_id": "intervention_678",
            "payload": {},
        }
        event = Event.from_dict(data)
        assert event.provenance_id == "prov_123"
        assert event.parent_event_id == "parent_456"
        assert event.tool_call_id == "tool_789"
        assert event.artifact_id == "artifact_012"
        assert event.audit_id == "audit_345"
        assert event.intervention_id == "intervention_678"

    def test_from_dict_backwards_compatible(self):
        """from_dict works with old events without provenance fields."""
        data = {
            "event_id": "evt_123",
            "timestamp": "2024-01-01T12:00:00",
            "event_type": "interaction_proposed",
            "agent_id": "agent_1",
            "payload": {},
        }
        event = Event.from_dict(data)
        assert event.provenance_id is None
        assert event.parent_event_id is None
        assert event.tool_call_id is None


# =============================================================================
# Tests for CSV export
# =============================================================================


class TestCSVExport:
    """Tests for event log CSV export."""

    def test_to_csv_creates_file(self, tmp_path):
        """to_csv creates a CSV file with correct structure."""
        log = EventLog(tmp_path / "log.jsonl")
        event = tool_call_executed_event(
            agent_id="agent_1",
            tool_name="test_tool",
            arguments={},
            result={},
            success=True,
            epoch=1,
            step=5,
        )
        log.append(event)

        csv_path = tmp_path / "events.csv"
        result = log.to_csv(csv_path)
        assert result.exists()
        assert csv_path.exists()

    def test_to_csv_includes_provenance_columns(self, tmp_path):
        """CSV includes all provenance columns."""
        log = EventLog(tmp_path / "log.jsonl")
        event = tool_call_executed_event(
            agent_id="agent_1",
            tool_name="test",
            arguments={},
            result={},
            success=True,
        )
        log.append(event)

        csv_path = tmp_path / "events.csv"
        log.to_csv(csv_path)

        # Read and check headers
        with open(csv_path) as f:
            header = f.readline().strip()
            assert "provenance_id" in header
            assert "parent_event_id" in header
            assert "tool_call_id" in header
            assert "artifact_id" in header
            assert "audit_id" in header
            assert "intervention_id" in header

    def test_to_csv_preserves_provenance_data(self, tmp_path):
        """CSV export preserves provenance IDs."""
        import csv

        log = EventLog(tmp_path / "log.jsonl")
        event = tool_call_executed_event(
            agent_id="agent_1",
            tool_name="test",
            arguments={},
            result={},
            success=True,
        )
        log.append(event)

        csv_path = tmp_path / "events.csv"
        log.to_csv(csv_path)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            row = rows[0]
            assert row["provenance_id"] == event.provenance_id
            assert row["tool_call_id"] == event.tool_call_id

    def test_to_csv_with_payload(self, tmp_path):
        """to_csv can optionally include payload."""
        log = EventLog(tmp_path / "log.jsonl")
        event = Event(
            event_type=EventType.INTERACTION_PROPOSED,
            agent_id="agent_1",
            payload={"key": "value"},
        )
        log.append(event)

        csv_path = tmp_path / "events.csv"
        log.to_csv(csv_path, include_payload=True)

        with open(csv_path) as f:
            header = f.readline()
            assert "payload" in header


class TestProvenanceCSVExport:
    """Tests for provenance-specific CSV export."""

    def test_to_provenance_csv_filters_events(self, tmp_path):
        """to_provenance_csv only includes events with provenance_id."""
        log = EventLog(tmp_path / "log.jsonl")

        # Event without provenance
        event1 = Event(
            event_type=EventType.INTERACTION_PROPOSED,
            agent_id="agent_1",
        )
        log.append(event1)

        # Event with provenance
        event2 = tool_call_executed_event(
            agent_id="agent_1",
            tool_name="test",
            arguments={},
            result={},
            success=True,
        )
        log.append(event2)

        csv_path = tmp_path / "provenance.csv"
        log.to_provenance_csv(csv_path)

        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Should only include event2
            assert len(rows) == 1
            assert rows[0]["provenance_id"] == event2.provenance_id

    def test_to_provenance_csv_extracts_key_fields(self, tmp_path):
        """to_provenance_csv extracts tool_name, audit_type, etc."""
        import csv

        log = EventLog(tmp_path / "log.jsonl")

        # Tool call event
        tool_event = tool_call_executed_event(
            agent_id="agent_1",
            tool_name="query_db",
            arguments={},
            result={},
            success=True,
        )
        log.append(tool_event)

        # Audit event
        audit_evt = audit_event(
            agent_id="auditor",
            audit_type="council",
            audit_result={},
        )
        log.append(audit_evt)

        # Intervention event
        intervention_evt = intervention_event(
            agent_id="governance",
            intervention_type="freeze",
            intervention_action="freeze_agent",
            intervention_reason="suspicious",
            affected_agents=["agent_2"],
        )
        log.append(intervention_evt)

        csv_path = tmp_path / "provenance.csv"
        log.to_provenance_csv(csv_path)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3

            # Check tool call row
            tool_row = rows[0]
            assert tool_row["tool_name"] == "query_db"
            assert tool_row["success"] == "True"

            # Check audit row
            audit_row = rows[1]
            assert audit_row["audit_type"] == "council"

            # Check intervention row
            intervention_row = rows[2]
            assert intervention_row["intervention_type"] == "freeze"

    def test_provenance_csv_includes_parent_links(self, tmp_path):
        """Provenance CSV preserves parent-child relationships."""
        import csv

        log = EventLog(tmp_path / "log.jsonl")

        # Parent event
        parent = tool_call_executed_event(
            agent_id="agent_1",
            tool_name="parent",
            arguments={},
            result={},
            success=True,
        )
        log.append(parent)

        # Child event
        child = tool_call_executed_event(
            agent_id="agent_1",
            tool_name="child",
            arguments={},
            result={},
            success=True,
            parent_event_id=parent.provenance_id,
        )
        log.append(child)

        csv_path = tmp_path / "provenance.csv"
        log.to_provenance_csv(csv_path)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2

            child_row = rows[1]
            assert child_row["parent_event_id"] == parent.provenance_id


# =============================================================================
# Integration tests
# =============================================================================


class TestProvenanceIntegration:
    """Integration tests for full provenance workflow."""

    def test_full_provenance_chain(self, tmp_path):
        """Test complete provenance chain from tool call to intervention."""
        log = EventLog(tmp_path / "log.jsonl")

        # 1. Tool call
        tool_evt = tool_call_executed_event(
            agent_id="agent_1",
            tool_name="risky_operation",
            arguments={"risk": "high"},
            result={"status": "executed"},
            success=True,
            epoch=1,
            step=1,
        )
        log.append(tool_evt)

        # 2. Audit of tool call
        audit_evt = audit_event(
            agent_id="auditor",
            audit_type="safety",
            audit_result={"flagged": True, "risk_level": 0.8},
            audited_event_id=tool_evt.event_id,
            parent_event_id=tool_evt.provenance_id,
            epoch=1,
            step=2,
        )
        log.append(audit_evt)

        # 3. Intervention based on audit
        intervention_evt = intervention_event(
            agent_id="governance",
            intervention_type="throttle",
            intervention_action="rate_limit",
            intervention_reason="high_risk_detected",
            affected_agents=["agent_1"],
            parent_event_id=audit_evt.provenance_id,
            epoch=1,
            step=3,
        )
        log.append(intervention_evt)

        # Export to CSV
        csv_path = tmp_path / "provenance.csv"
        log.to_provenance_csv(csv_path)

        # Verify chain is preserved
        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3

            # Verify tool call
            assert rows[0]["tool_name"] == "risky_operation"
            assert rows[0]["success"] == "True"

            # Verify audit links to tool call
            assert rows[1]["audit_type"] == "safety"
            assert rows[1]["parent_event_id"] == tool_evt.provenance_id

            # Verify intervention links to audit
            assert rows[2]["intervention_type"] == "throttle"
            assert rows[2]["parent_event_id"] == audit_evt.provenance_id

    def test_jsonl_and_csv_consistency(self, tmp_path):
        """JSONL and CSV exports contain same provenance data."""
        import csv
        import json

        log = EventLog(tmp_path / "log.jsonl")

        events = [
            tool_call_executed_event(
                agent_id=f"agent_{i}",
                tool_name=f"tool_{i}",
                arguments={},
                result={},
                success=True,
            )
            for i in range(5)
        ]
        log.append_many(events)

        # Export to both formats
        csv_path = tmp_path / "events.csv"
        log.to_csv(csv_path)

        # Read JSONL
        with open(log.path) as f:
            jsonl_events = [json.loads(line) for line in f]

        # Read CSV
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            csv_events = list(reader)

        # Same number of events
        assert len(jsonl_events) == len(csv_events) == 5

        # Provenance IDs match
        for jsonl_evt, csv_evt in zip(jsonl_events, csv_events, strict=False):
            assert jsonl_evt["provenance_id"] == csv_evt["provenance_id"]
            assert jsonl_evt["tool_call_id"] == csv_evt["tool_call_id"]
