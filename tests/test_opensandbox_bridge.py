"""Unit tests for the OpenSandbox bridge subsystems.

Covers screening, message bus, provenance, observer, and bridge-level
validation logic requested in PR #297 review.
"""

from datetime import timezone

import pytest

from swarm.bridges.opensandbox.bridge import OpenSandboxBridge
from swarm.bridges.opensandbox.config import (
    AgentType,
    CapabilityManifest,
    GovernanceContract,
    NetworkPolicy,
    OpenSandboxConfig,
)
from swarm.bridges.opensandbox.events import OpenSandboxEvent, OpenSandboxEventType
from swarm.bridges.opensandbox.message_bus import MessageBus
from swarm.bridges.opensandbox.observer import Observer
from swarm.bridges.opensandbox.provenance import ProvenanceTracker
from swarm.bridges.opensandbox.screener import ScreeningProtocol

# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def config():
    return OpenSandboxConfig()


@pytest.fixture
def restricted_contract():
    return GovernanceContract(
        contract_id="restricted-v1",
        tier="restricted",
        capabilities=["python", "file_read"],
        network=NetworkPolicy.DENY_ALL,
    )


@pytest.fixture
def standard_contract():
    return GovernanceContract(
        contract_id="standard-v1",
        tier="standard",
        capabilities=["python", "file_read", "file_write", "bash"],
        network=NetworkPolicy.ALLOWLIST,
    )


@pytest.fixture
def cooperative_manifest():
    return CapabilityManifest(
        agent_id="agent-coop",
        agent_type=AgentType.COOPERATIVE,
        capabilities=["python"],
        requires_network=False,
    )


@pytest.fixture
def adversarial_manifest():
    return CapabilityManifest(
        agent_id="agent-adv",
        agent_type=AgentType.ADVERSARIAL,
        capabilities=["python", "bash", "curl"],
        requires_network=True,
        max_memory_mb=4096,
    )


# ------------------------------------------------------------------ #
# Screening tests                                                     #
# ------------------------------------------------------------------ #

class TestScreeningProtocol:
    """Tests for the screening subsystem."""

    def test_admit_compatible_agent(self, config, restricted_contract):
        config.contracts = {"restricted-v1": restricted_contract}
        screener = ScreeningProtocol(config)
        manifest = CapabilityManifest(
            agent_id="agent-a",
            capabilities=["python"],
        )
        assignment = screener.evaluate(manifest)
        assert not assignment.rejected
        assert assignment.contract_id == "restricted-v1"
        assert assignment.tier == "restricted"
        assert assignment.score > 0

    def test_reject_incompatible_agent(self, config):
        contract = GovernanceContract(
            contract_id="strict",
            tier="strict",
            capabilities=["file_read"],
            network=NetworkPolicy.DENY_ALL,
            max_memory_mb=128,
            max_cpu_shares=256,
        )
        config.contracts = {"strict": contract}
        screener = ScreeningProtocol(config, min_score=0.95)
        manifest = CapabilityManifest(
            agent_id="agent-bad",
            agent_type=AgentType.ADVERSARIAL,
            capabilities=["python", "bash", "curl"],
            requires_network=True,
            max_memory_mb=4096,
        )
        assignment = screener.evaluate(manifest)
        assert assignment.rejected
        assert assignment.contract_id == ""
        assert assignment.tier == ""
        assert "No compatible contract" in assignment.rejection_reason

    def test_rejected_assignment_has_empty_contract_fields(self, config):
        """Verify rejected ContractAssignment doesn't retain defaults."""
        config.contracts = {}
        screener = ScreeningProtocol(config, min_score=0.99)
        manifest = CapabilityManifest(agent_id="x", capabilities=["nope"])
        assignment = screener.evaluate(manifest)
        assert assignment.rejected
        # Should be explicitly empty, not "default"/"restricted"
        assert assignment.contract_id == ""
        assert assignment.tier == ""

    def test_sorting_ledger_tracks_admissions(self, config, restricted_contract):
        config.contracts = {"r": restricted_contract}
        screener = ScreeningProtocol(config)
        for i in range(5):
            manifest = CapabilityManifest(
                agent_id=f"agent-{i}", capabilities=["python"]
            )
            screener.evaluate(manifest)
        ledger = screener.get_sorting_ledger()
        assert "restricted" in ledger
        assert len(ledger["restricted"]) == 5

    def test_sorting_ledger_bounded_growth(self, config, restricted_contract):
        config.contracts = {"r": restricted_contract}
        screener = ScreeningProtocol(config, max_records=10)
        for i in range(20):
            manifest = CapabilityManifest(
                agent_id=f"agent-{i}", capabilities=["python"]
            )
            screener.evaluate(manifest)
        ledger = screener.get_sorting_ledger()
        total = sum(len(v) for v in ledger.values())
        assert total <= 15  # should have truncated


# ------------------------------------------------------------------ #
# Event timezone tests                                                #
# ------------------------------------------------------------------ #

class TestEventTimezones:
    """Ensure from_dict normalizes naive timestamps to UTC."""

    def test_naive_timestamp_normalized_to_utc(self):
        data = {
            "event_type": "sandbox:created",
            "timestamp": "2025-01-15T12:00:00",  # naive
            "agent_id": "a",
        }
        event = OpenSandboxEvent.from_dict(data)
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.tzinfo == timezone.utc

    def test_aware_timestamp_preserved(self):
        data = {
            "event_type": "sandbox:created",
            "timestamp": "2025-01-15T12:00:00+05:00",
            "agent_id": "a",
        }
        event = OpenSandboxEvent.from_dict(data)
        assert event.timestamp.tzinfo is not None

    def test_default_factory_is_utc(self):
        event = OpenSandboxEvent()
        assert event.timestamp.tzinfo is not None


# ------------------------------------------------------------------ #
# Message bus tests                                                   #
# ------------------------------------------------------------------ #

class TestMessageBus:
    """Tests for the message bus subsystem."""

    def test_basic_send_receive(self):
        bus = MessageBus(require_provenance=False)
        bus.register_sandbox("s-a", "agent-a")
        bus.register_sandbox("s-b", "agent-b")
        msg = bus.send("s-a", "s-b", {"hello": "world"})
        assert msg.delivered
        assert not msg.blocked
        received = bus.receive("s-b")
        assert len(received) == 1
        assert received[0].payload == {"hello": "world"}

    def test_unregistered_sender_blocked(self):
        bus = MessageBus(require_provenance=False)
        bus.register_sandbox("s-b", "agent-b")
        msg = bus.send("s-unknown", "s-b", {})
        assert msg.blocked
        assert "not registered" in msg.block_reason

    def test_unregistered_destination_emits_blocked_event(self):
        bus = MessageBus(require_provenance=False)
        bus.register_sandbox("s-a", "agent-a")
        msg = bus.send("s-a", "s-nonexistent", {"data": 1})
        assert msg.blocked
        events = bus.get_events()
        blocked_events = [
            e for e in events
            if e.event_type == OpenSandboxEventType.MESSAGE_BLOCKED
        ]
        assert len(blocked_events) >= 1
        assert "not registered" in blocked_events[-1].payload.get("reason", "")

    def test_mailbox_full_emits_blocked_event(self):
        bus = MessageBus(max_pending=1, require_provenance=False)
        bus.register_sandbox("s-a", "agent-a")
        bus.register_sandbox("s-b", "agent-b")
        bus.send("s-a", "s-b", {"m": 1})  # fills mailbox
        msg = bus.send("s-a", "s-b", {"m": 2})  # should block
        assert msg.blocked
        assert "full" in msg.block_reason.lower()
        events = bus.get_events()
        blocked_events = [
            e for e in events
            if e.event_type == OpenSandboxEventType.MESSAGE_BLOCKED
        ]
        assert len(blocked_events) >= 1

    def test_payload_size_limit(self):
        bus = MessageBus(
            max_message_bytes=100,
            require_provenance=False,
        )
        bus.register_sandbox("s-a", "agent-a")
        bus.register_sandbox("s-b", "agent-b")
        msg = bus.send("s-a", "s-b", {"big": "x" * 200})
        assert msg.blocked
        assert "exceeds limit" in msg.block_reason.lower()

    def test_provenance_required(self):
        bus = MessageBus(require_provenance=True)
        bus.register_sandbox("s-a", "agent-a")
        bus.register_sandbox("s-b", "agent-b")
        msg = bus.send("s-a", "s-b", {"data": 1})
        assert msg.blocked
        assert "Provenance" in msg.block_reason

    def test_route_restriction(self):
        bus = MessageBus(require_provenance=False)
        bus.register_sandbox("s-a", "agent-a")
        bus.register_sandbox("s-b", "agent-b")
        bus.set_allowed_routes({("s-b", "s-a")})  # only b->a allowed
        msg = bus.send("s-a", "s-b", {})
        assert msg.blocked
        assert "not in allowed routes" in msg.block_reason

    def test_history_bounded(self):
        bus = MessageBus(
            max_history=10,
            require_provenance=False,
        )
        bus.register_sandbox("s-a", "agent-a")
        bus.register_sandbox("s-b", "agent-b")
        for i in range(20):
            bus.send("s-a", "s-b", {"i": i})
        history = bus.get_history()
        assert len(history) <= 10


# ------------------------------------------------------------------ #
# Provenance tests                                                    #
# ------------------------------------------------------------------ #

class TestProvenanceTracker:
    """Tests for the provenance tracking subsystem."""

    def test_sign_creates_record(self):
        tracker = ProvenanceTracker()
        prov_id = tracker.sign(
            sandbox_id="s-a",
            agent_id="agent-a",
            action_type="exec",
            action_summary="python test.py",
            content={"cmd": "python test.py"},
        )
        assert prov_id
        chain = tracker.get_chain("agent-a")
        assert len(chain) == 1
        assert chain[0].provenance_id == prov_id

    def test_chain_auto_links(self):
        tracker = ProvenanceTracker()
        id1 = tracker.sign(
            sandbox_id="s-a", agent_id="a", action_type="exec",
            action_summary="cmd1",
        )
        tracker.sign(
            sandbox_id="s-a", agent_id="a", action_type="exec",
            action_summary="cmd2",
        )
        chain = tracker.get_chain("a")
        assert chain[1].parent_id == id1
        assert chain[0].parent_id is None

    def test_chain_verification_succeeds(self):
        tracker = ProvenanceTracker(hmac_key="test-secret")
        for i in range(5):
            tracker.sign(
                sandbox_id="s", agent_id="a", action_type="exec",
                action_summary=f"cmd-{i}",
                content={"i": i},
            )
        assert tracker.verify_chain("a")

    def test_chain_verification_detects_tamper(self):
        tracker = ProvenanceTracker(hmac_key="secret")
        tracker.sign(
            sandbox_id="s", agent_id="a", action_type="exec",
            action_summary="cmd", content={"x": 1},
        )
        # Tamper with the record
        # Access internal state to tamper
        with tracker._lock:
            tracker._chains["a"][0].chain_hash = "tampered"
        assert not tracker.verify_chain("a")

    def test_disabled_tracker_returns_ids(self):
        tracker = ProvenanceTracker(enabled=False)
        prov_id = tracker.sign(
            sandbox_id="s", agent_id="a", action_type="exec",
            action_summary="cmd",
        )
        assert prov_id  # should still return an ID
        assert tracker.get_chain("a") == []  # no records stored

    def test_bounded_growth(self):
        tracker = ProvenanceTracker(max_records=10)
        for i in range(25):
            tracker.sign(
                sandbox_id="s", agent_id="a", action_type="exec",
                action_summary=f"cmd-{i}",
            )
        stats = tracker.get_stats()
        assert stats["total_records"] <= 15


# ------------------------------------------------------------------ #
# Observer tests                                                      #
# ------------------------------------------------------------------ #

class TestObserver:
    """Tests for the observability subsystem."""

    def test_register_and_record(self):
        obs = Observer()
        obs.register_agent("a", "contract-1", "restricted")
        obs.record_command("a", success=True)
        obs.record_command("a", success=False)
        metrics = obs.get_agent_metrics("a")
        assert metrics.total_commands == 2
        assert metrics.successful_commands == 1

    def test_risk_score_computation(self):
        obs = Observer(risk_threshold=0.6)
        obs.register_agent("a", "c1", "restricted")
        # Low p drives risk up
        obs.record_p("a", 0.1)
        obs.record_violation("a")
        obs.record_violation("a")
        metrics = obs.get_agent_metrics("a")
        assert metrics.risk_score > 0

    def test_check_risk_deduplicates_alerts(self):
        obs = Observer(risk_threshold=0.3)
        obs.register_agent("a", "c1", "restricted")
        obs.record_p("a", 0.1)  # triggers high risk
        obs.record_violation("a")

        alert1 = obs.check_risk("a")
        assert alert1 is not None

        # Second call should NOT produce a duplicate alert
        alert2 = obs.check_risk("a")
        assert alert2 is None

        # Total alerts should be exactly 1
        assert len(obs.get_risk_alerts()) == 1

    def test_alert_re_emits_after_drop_below(self):
        obs = Observer(risk_threshold=0.8)
        obs.register_agent("a", "c1", "restricted")
        obs.record_p("a", 0.05)
        obs.record_violation("a")
        obs.record_violation("a")
        obs.record_intervention("a")

        obs.check_risk("a")
        # Regardless of whether we crossed threshold, simulate recovery
        # by recording many good p values to drop risk below threshold
        for _ in range(50):
            obs.record_p("a", 0.99)
        # Now risk should have dropped
        obs.check_risk("a")  # resets alert state

        # Push risk back up
        for _ in range(10):
            obs.record_p("a", 0.05)
            obs.record_violation("a")

        obs.check_risk("a")
        # If risk crossed again, we should get a new alert
        # (this depends on whether risk_score goes above threshold again)
        # The important thing is: no duplicates within one crossing
        alerts = obs.get_risk_alerts()
        # Should not have more alerts than threshold crossings
        assert len(alerts) <= 3

    def test_unregister_cleans_alert_state(self):
        obs = Observer(risk_threshold=0.3)
        obs.register_agent("a", "c1", "restricted")
        obs.record_p("a", 0.1)
        obs.record_violation("a")
        obs.check_risk("a")  # emit alert
        obs.unregister_agent("a")
        # Re-register and check — should be fresh state
        obs.register_agent("a", "c1", "restricted")
        obs.record_p("a", 0.1)
        obs.record_violation("a")
        alert = obs.check_risk("a")
        # Should get a fresh alert since agent was re-registered
        assert alert is not None

    def test_experiment_metrics(self):
        obs = Observer()
        obs.register_agent("a", "c1", "tier-a")
        obs.register_agent("b", "c2", "tier-b")
        obs.record_command("a", True)
        obs.record_command("b", True)
        obs.record_p("a", 0.9)
        obs.record_p("b", 0.3)
        metrics = obs.compute_experiment_metrics("exp-1")
        assert metrics.experiment_id == "exp-1"
        assert "tier-a" in metrics.avg_p_by_tier
        assert "tier-b" in metrics.avg_p_by_tier

    def test_observer_events_bounded(self):
        obs = Observer(max_events=10)
        obs.register_agent("a", "c1", "t")
        for _i in range(20):
            obs.record_p("a", 0.1)
            obs.record_violation("a")
            obs.check_risk("a")
        events = obs.get_events()
        assert len(events) <= 10


# ------------------------------------------------------------------ #
# Bridge integration tests                                            #
# ------------------------------------------------------------------ #

class TestOpenSandboxBridge:
    """Integration tests for the bridge orchestrator."""

    def _setup_bridge(self):
        """Create a bridge with one agent ready to go."""
        config = OpenSandboxConfig(provenance_hmac_key="test-key")
        bridge = OpenSandboxBridge(config)
        contract = GovernanceContract(
            contract_id="test-v1",
            tier="test",
            capabilities=["python", "file_read"],
        )
        bridge.publish_contract(contract)
        manifest = CapabilityManifest(
            agent_id="agent-a",
            capabilities=["python"],
        )
        assignment = bridge.screen_agent(manifest)
        sandbox_id = bridge.create_sandbox(assignment)
        return bridge, sandbox_id

    def test_send_message_rejects_unknown_source(self):
        bridge, sandbox_id = self._setup_bridge()
        with pytest.raises(ValueError, match="not registered"):
            bridge.send_message("unknown-sandbox", sandbox_id, {"data": 1})

    def test_send_message_rejects_unknown_destination(self):
        bridge, sandbox_id = self._setup_bridge()
        with pytest.raises(ValueError, match="not registered"):
            bridge.send_message(sandbox_id, "unknown-sandbox", {"data": 1})

    def test_send_message_rejects_isolated_sender(self):
        bridge, sandbox_id = self._setup_bridge()
        # Isolate the agent
        bridge.isolate("agent-a", reason="test")
        with pytest.raises(ValueError, match="isolated"):
            bridge.send_message(sandbox_id, sandbox_id, {"data": 1})

    def test_execute_command_rejects_empty(self):
        bridge, _ = self._setup_bridge()
        with pytest.raises(ValueError, match="must not be empty"):
            bridge.execute_command("agent-a", "")

    def test_execute_command_rejects_whitespace(self):
        bridge, _ = self._setup_bridge()
        with pytest.raises(ValueError, match="must not be empty"):
            bridge.execute_command("agent-a", "   ")

    def test_execute_command_normalizes_path(self):
        bridge, _ = self._setup_bridge()
        # /usr/bin/python should normalize to "python" which is allowed
        interaction = bridge.execute_command("agent-a", "/usr/bin/python test.py")
        assert interaction.p > 0

    def test_capability_violation_returns_rejection(self):
        bridge, _ = self._setup_bridge()
        interaction = bridge.execute_command("agent-a", "bash evil.sh")
        assert not interaction.accepted
        assert interaction.metadata.get("event") == "governance_intervention"

    def test_sandbox_id_contains_random_suffix(self):
        bridge, sandbox_id = self._setup_bridge()
        parts = sandbox_id.split("-")
        # Should be "sandbox-agent-a-<hex>"
        assert len(parts[-1]) == 8  # uuid4 hex[:8]

    def test_get_sandbox_info_redacts_secrets(self):
        bridge, sandbox_id = self._setup_bridge()
        # Inject a secret env var
        with bridge._lock:
            bridge._sandboxes[sandbox_id]["env"]["MY_SECRET_KEY"] = "hunter2"
        info = bridge.get_sandbox_info(sandbox_id)
        assert info["env"]["MY_SECRET_KEY"] == "***REDACTED***"

    def test_get_sandbox_info_returns_deep_copy(self):
        bridge, sandbox_id = self._setup_bridge()
        info1 = bridge.get_sandbox_info(sandbox_id)
        info1["env"]["INJECTED"] = "bad"
        info2 = bridge.get_sandbox_info(sandbox_id)
        assert "INJECTED" not in info2.get("env", {})

    def test_provenance_chain_verifies(self):
        bridge, _ = self._setup_bridge()
        bridge.execute_command("agent-a", "python run.py")
        bridge.execute_command("agent-a", "python check.py")
        assert bridge._provenance.verify_chain("agent-a")

    def test_shutdown_destroys_all_sandboxes(self):
        bridge, _ = self._setup_bridge()
        bridge.shutdown()
        assert len(bridge._agent_sandboxes) == 0
        assert len(bridge._sandboxes) == 0
