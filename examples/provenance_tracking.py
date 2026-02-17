"""Example demonstrating unified provenance trace schema and CSV export.

This example shows how to:
1. Create events with provenance tracking (tool calls, artifacts, audits, interventions)
2. Link events together via parent_event_id
3. Export to CSV format for analysis
4. Query provenance chains from JSONL logs
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from swarm.logging.event_log import EventLog
from swarm.models.events import (
    agent_message_event,
    artifact_created_event,
    audit_event,
    intervention_event,
    tool_call_executed_event,
)


def example_provenance_workflow():
    """Demonstrate a complete provenance tracking workflow."""

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Initialize event log
        log = EventLog(tmpdir / "events.jsonl")

        print("=" * 60)
        print("Unified Provenance Trace Example")
        print("=" * 60)

        # 1. Agent sends a message with tool calls
        msg_event = agent_message_event(
            agent_id="agent_alice",
            message_role="assistant",
            message_content="I'll query the database for you",
            tool_calls=[{"name": "query_db", "args": {"query": "SELECT * FROM users"}}],
            epoch=1,
            step=1,
        )
        log.append(msg_event)
        print(f"\n1. Agent message created")
        print(f"   Provenance ID: {msg_event.provenance_id}")

        # 2. Tool call is executed
        tool_event = tool_call_executed_event(
            agent_id="agent_alice",
            tool_name="query_db",
            arguments={"query": "SELECT * FROM users"},
            result={"rows": [{"id": 1, "name": "Alice"}]},
            success=True,
            parent_event_id=msg_event.provenance_id,  # Link to parent
            epoch=1,
            step=2,
        )
        log.append(tool_event)
        print(f"\n2. Tool call executed")
        print(f"   Tool Call ID: {tool_event.tool_call_id}")
        print(f"   Provenance ID: {tool_event.provenance_id}")
        print(f"   Parent: {tool_event.parent_event_id}")

        # 3. Create an artifact from the result
        artifact_event = artifact_created_event(
            agent_id="agent_alice",
            artifact_id="memory_001",
            artifact_type="memory",
            artifact_data={
                "title": "Database Query Result",
                "summary": "Retrieved user data successfully",
                "content": {"rows": 1},
            },
            parent_event_id=tool_event.provenance_id,  # Link to tool call
            epoch=1,
            step=3,
        )
        log.append(artifact_event)
        print(f"\n3. Artifact created")
        print(f"   Artifact ID: {artifact_event.artifact_id}")
        print(f"   Provenance ID: {artifact_event.provenance_id}")
        print(f"   Parent: {artifact_event.parent_event_id}")

        # 4. Audit the tool call for safety
        audit_evt = audit_event(
            agent_id="auditor_system",
            audit_type="safety_check",
            audit_result={
                "decision": "approve",
                "risk_level": 0.1,
                "checks_passed": ["sql_injection", "data_access"],
            },
            audited_event_id=tool_event.event_id,
            parent_event_id=tool_event.provenance_id,
            epoch=1,
            step=4,
        )
        log.append(audit_evt)
        print(f"\n4. Audit performed")
        print(f"   Audit ID: {audit_evt.audit_id}")
        print(f"   Provenance ID: {audit_evt.provenance_id}")
        print(f"   Audited Event: {audit_evt.payload.get('audited_event_id', 'N/A')}")

        # 5. Simulate a risky tool call that triggers intervention
        risky_tool = tool_call_executed_event(
            agent_id="agent_bob",
            tool_name="delete_all_records",
            arguments={},
            result={"status": "blocked"},
            success=False,
            epoch=1,
            step=5,
        )
        log.append(risky_tool)
        print(f"\n5. Risky tool call detected")
        print(f"   Tool Call ID: {risky_tool.tool_call_id}")

        # 6. Governance intervention
        intervention_evt = intervention_event(
            agent_id="governance_system",
            intervention_type="throttle",
            intervention_action="block_tool",
            intervention_reason="destructive_operation_attempted",
            affected_agents=["agent_bob"],
            parent_event_id=risky_tool.provenance_id,
            epoch=1,
            step=6,
        )
        log.append(intervention_evt)
        print(f"\n6. Intervention applied")
        print(f"   Intervention ID: {intervention_evt.intervention_id}")
        print(f"   Type: {intervention_evt.payload['intervention_type']}")
        print(f"   Affected agents: {intervention_evt.payload['affected_agents']}")

        # Export to different formats
        print(f"\n{'=' * 60}")
        print("Exporting provenance data")
        print("=" * 60)

        # Full CSV export
        csv_path = tmpdir / "events_full.csv"
        log.to_csv(csv_path)
        print(f"\n✓ Full CSV exported to: {csv_path}")
        print(f"  Columns: event_id, timestamp, event_type, provenance_id, ...")

        # Provenance-focused CSV export
        prov_csv_path = tmpdir / "provenance.csv"
        log.to_provenance_csv(prov_csv_path)
        print(f"\n✓ Provenance CSV exported to: {prov_csv_path}")
        print(f"  Columns: provenance_id, tool_call_id, artifact_id, audit_id, ...")

        # Show a few lines from provenance CSV
        print(f"\n{'=' * 60}")
        print("Provenance Chain Summary")
        print("=" * 60)

        import csv

        with open(prov_csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            for i, row in enumerate(rows, 1):
                event_type = row["event_type"]
                prov_id = row["provenance_id"][:8] + "..."
                parent = row["parent_event_id"][:8] + "..." if row["parent_event_id"] else "None"

                print(f"\n{i}. {event_type}")
                print(f"   Provenance: {prov_id}")
                print(f"   Parent: {parent}")

                if row["tool_name"]:
                    print(f"   Tool: {row['tool_name']}")
                if row["artifact_id"]:
                    print(f"   Artifact: {row['artifact_id']}")
                if row["audit_id"]:
                    print(f"   Audit: {row['audit_id'][:8]}...")
                if row["intervention_id"]:
                    print(f"   Intervention: {row['intervention_id'][:8]}...")

        print(f"\n{'=' * 60}")
        print("Provenance tracking complete!")
        print("=" * 60)

        # JSONL log path
        print(f"\nJSONL event log: {log.path}")
        print(f"Total events: {log.count()}")


if __name__ == "__main__":
    example_provenance_workflow()
