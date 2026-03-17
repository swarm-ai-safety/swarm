"""Tests for the Attestation and Relational Messaging API."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest

from swarm.attestation.receipt import (
    AdmissibilityReceipt,
    ExecutionBounds,
    PolicyCompliance,
    ReceiptStatus,
)
from swarm.attestation.signer import ReceiptSigner, ReceiptVerifier
from swarm.attestation.middleware import AttestationMiddleware
from swarm.attestation.relay import ReceiptRelay, RelayMessage
from swarm.models.events import Event, EventType


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def signer():
    return ReceiptSigner(signer_id="test-platform")


@pytest.fixture
def verifier(signer):
    return ReceiptVerifier(signer.secret_key_hex)


@pytest.fixture
def sample_event():
    return Event(
        event_type=EventType.INTERACTION_COMPLETED,
        agent_id="agent-1",
        interaction_id="ix-1",
        payload={"accepted": True, "payoff_initiator": 1.0, "payoff_counterparty": 0.5},
        epoch=0,
        step=1,
    )


@pytest.fixture
def sample_payload():
    return {"accepted": True, "payoff_initiator": 1.0, "payoff_counterparty": 0.5}


def _make_pending_receipt(agent_id="agent-1", payload=None):
    payload = payload or {"key": "value"}
    now = datetime.utcnow()
    payload_hash = AdmissibilityReceipt.hash_payload(payload)
    receipt_id = AdmissibilityReceipt.generate_receipt_id(
        agent_id=agent_id,
        action_type="interaction_completed",
        payload_hash=payload_hash,
        timestamp=now,
    )
    return AdmissibilityReceipt(
        receipt_id=receipt_id,
        timestamp=now,
        agent_id=agent_id,
        action_type="interaction_completed",
        payload_hash=payload_hash,
    )


# ------------------------------------------------------------------ #
# Receipt model tests
# ------------------------------------------------------------------ #


class TestAdmissibilityReceipt:
    def test_hash_payload_deterministic(self):
        p = {"b": 2, "a": 1}
        h1 = AdmissibilityReceipt.hash_payload(p)
        h2 = AdmissibilityReceipt.hash_payload({"a": 1, "b": 2})
        assert h1 == h2

    def test_generate_receipt_id_deterministic(self):
        ts = datetime(2025, 1, 1, 0, 0, 0)
        id1 = AdmissibilityReceipt.generate_receipt_id("a", "t", "h", ts)
        id2 = AdmissibilityReceipt.generate_receipt_id("a", "t", "h", ts)
        assert id1 == id2
        assert len(id1) == 24

    def test_is_admissible_pending(self):
        r = _make_pending_receipt()
        assert not r.is_admissible()

    def test_is_admissible_sealed_no_policies(self, signer):
        r = _make_pending_receipt()
        sealed = signer.seal(r)
        assert sealed.is_admissible()

    def test_is_admissible_fails_when_policy_fails(self, signer):
        r = _make_pending_receipt()
        r = r.model_copy(
            update={
                "policy_results": [
                    PolicyCompliance(policy_id="p1", passed=False)
                ]
            }
        )
        sealed = signer.seal(r)
        assert not sealed.is_admissible()

    def test_canonical_bytes_excludes_signature(self):
        r = _make_pending_receipt()
        b1 = r.canonical_bytes()
        r2 = r.model_copy(update={"signature": "deadbeef", "signer_id": "x"})
        b2 = r2.canonical_bytes()
        assert b1 == b2

    def test_content_hash(self):
        r = _make_pending_receipt()
        h = r.content_hash()
        assert len(h) == 64  # SHA-256 hex

    def test_execution_bounds_validation(self):
        with pytest.raises(ValueError, match="non-negative"):
            ExecutionBounds(max_resource_spend=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            ExecutionBounds(max_delegation_depth=-1)


# ------------------------------------------------------------------ #
# Signer / Verifier tests
# ------------------------------------------------------------------ #


class TestSignerVerifier:
    def test_seal_transitions_status(self, signer):
        r = _make_pending_receipt()
        sealed = signer.seal(r)
        assert sealed.status == ReceiptStatus.SEALED
        assert sealed.signature is not None
        assert sealed.signer_id == "test-platform"

    def test_seal_rejects_non_pending(self, signer):
        r = _make_pending_receipt()
        sealed = signer.seal(r)
        with pytest.raises(ValueError, match="PENDING"):
            signer.seal(sealed)

    def test_verify_valid(self, signer, verifier):
        r = _make_pending_receipt()
        sealed = signer.seal(r)
        assert verifier.verify(sealed)

    def test_verify_tampered(self, signer, verifier):
        r = _make_pending_receipt()
        sealed = signer.seal(r)
        tampered = sealed.model_copy(update={"agent_id": "evil-agent"})
        assert not verifier.verify(tampered)

    def test_verify_wrong_key(self, signer):
        other = ReceiptVerifier("00" * 32)
        r = _make_pending_receipt()
        sealed = signer.seal(r)
        assert not other.verify(sealed)

    def test_verify_and_mark_valid(self, signer, verifier):
        r = _make_pending_receipt()
        sealed = signer.seal(r)
        verified = verifier.verify_and_mark(sealed)
        assert verified.status == ReceiptStatus.VERIFIED

    def test_verify_and_mark_invalid(self, signer):
        other = ReceiptVerifier("00" * 32)
        r = _make_pending_receipt()
        sealed = signer.seal(r)
        rejected = other.verify_and_mark(sealed)
        assert rejected.status == ReceiptStatus.REJECTED

    def test_auto_generated_key(self):
        s = ReceiptSigner()
        assert len(s.secret_key_hex) == 64


# ------------------------------------------------------------------ #
# Middleware tests
# ------------------------------------------------------------------ #


class TestAttestationMiddleware:
    def test_attest_creates_sealed_receipt(self, signer, sample_event, sample_payload):
        mw = AttestationMiddleware(signer=signer)
        receipt = mw.attest(sample_event, sample_payload)
        assert receipt.status == ReceiptStatus.SEALED
        assert receipt.agent_id == "agent-1"
        assert receipt.signature is not None

    def test_attest_stores_receipt(self, signer, sample_event, sample_payload):
        mw = AttestationMiddleware(signer=signer)
        receipt = mw.attest(sample_event, sample_payload)
        assert mw.receipt_count == 1
        found = mw.get_receipt(receipt.receipt_id)
        assert found is not None
        assert found.receipt_id == receipt.receipt_id

    def test_attest_with_policy_evaluator(self, signer, sample_event, sample_payload):
        def always_pass(event: Event) -> PolicyCompliance:
            return PolicyCompliance(policy_id="test-policy", passed=True)

        mw = AttestationMiddleware(signer=signer, policy_evaluators=[always_pass])
        receipt = mw.attest(sample_event, sample_payload)
        assert len(receipt.policy_results) == 1
        assert receipt.policy_results[0].passed
        assert receipt.is_admissible()

    def test_attest_with_failing_policy(self, signer, sample_event, sample_payload):
        def always_fail(event: Event) -> PolicyCompliance:
            return PolicyCompliance(
                policy_id="strict-policy",
                passed=False,
                details={"reason": "test failure"},
            )

        mw = AttestationMiddleware(signer=signer, policy_evaluators=[always_fail])
        receipt = mw.attest(sample_event, sample_payload)
        assert not receipt.is_admissible()
        assert receipt.status == ReceiptStatus.SEALED  # still sealed

    def test_attest_with_custom_bounds(self, signer, sample_event, sample_payload):
        bounds = ExecutionBounds(max_resource_spend=100.0, max_delegation_depth=2)
        mw = AttestationMiddleware(signer=signer)
        receipt = mw.attest(sample_event, sample_payload, bounds=bounds)
        assert receipt.bounds.max_resource_spend == 100.0
        assert receipt.bounds.max_delegation_depth == 2

    def test_get_receipts_for_agent(self, signer):
        mw = AttestationMiddleware(signer=signer)
        e1 = Event(event_type=EventType.PAYOFF_COMPUTED, agent_id="a1", payload={"x": 1})
        e2 = Event(event_type=EventType.PAYOFF_COMPUTED, agent_id="a2", payload={"x": 2})
        mw.attest(e1, {"x": 1})
        mw.attest(e2, {"x": 2})
        assert len(mw.get_receipts_for_agent("a1")) == 1
        assert len(mw.get_receipts_for_agent("a2")) == 1

    def test_get_receipts_for_event(self, signer, sample_event, sample_payload):
        mw = AttestationMiddleware(signer=signer)
        receipt = mw.attest(sample_event, sample_payload)
        found = mw.get_receipts_for_event(sample_event.event_id)
        assert len(found) == 1
        assert found[0].receipt_id == receipt.receipt_id


# ------------------------------------------------------------------ #
# Relay tests
# ------------------------------------------------------------------ #


class TestReceiptRelay:
    def test_ingest_and_get(self, signer, verifier):
        relay = ReceiptRelay(verifier=verifier)
        r = _make_pending_receipt()
        sealed = signer.seal(r)
        assert relay.ingest(sealed)
        assert relay.get(sealed.receipt_id) is not None

    def test_ingest_rejects_invalid(self):
        verifier = ReceiptVerifier("00" * 32)
        relay = ReceiptRelay(verifier=verifier)
        r = _make_pending_receipt()
        # Not sealed → should fail
        assert not relay.ingest(r)

    def test_query_by_agent(self, signer, verifier):
        relay = ReceiptRelay(verifier=verifier)
        r1 = signer.seal(_make_pending_receipt("agent-a"))
        r2 = signer.seal(_make_pending_receipt("agent-b"))
        relay.ingest(r1)
        relay.ingest(r2)
        assert len(relay.query_by_agent("agent-a")) == 1

    def test_query_admissible(self, signer, verifier):
        relay = ReceiptRelay(verifier=verifier)
        r = signer.seal(_make_pending_receipt())
        relay.ingest(r)
        assert len(relay.query_admissible()) == 1

    def test_chain_valid_simple(self, signer, verifier):
        relay = ReceiptRelay(verifier=verifier)
        r = signer.seal(_make_pending_receipt())
        relay.ingest(r)
        assert relay.chain_valid(r.receipt_id)

    def test_chain_valid_with_parent(self, signer, verifier):
        relay = ReceiptRelay(verifier=verifier)
        parent = signer.seal(_make_pending_receipt("agent-1", {"step": 1}))
        relay.ingest(parent)

        child_pending = _make_pending_receipt("agent-1", {"step": 2})
        child_pending = child_pending.model_copy(
            update={"parent_receipt_ids": [parent.receipt_id]}
        )
        child = signer.seal(child_pending)
        relay.ingest(child)

        assert relay.chain_valid(child.receipt_id)

    def test_chain_invalid_missing_parent(self, signer, verifier):
        relay = ReceiptRelay(verifier=verifier)
        child_pending = _make_pending_receipt()
        child_pending = child_pending.model_copy(
            update={"parent_receipt_ids": ["nonexistent"]}
        )
        child = signer.seal(child_pending)
        relay.ingest(child)
        assert not relay.chain_valid(child.receipt_id)


# ------------------------------------------------------------------ #
# Relational messaging tests
# ------------------------------------------------------------------ #


class TestRelayMessaging:
    def _setup_relay_with_receipt(self, signer, verifier):
        relay = ReceiptRelay(verifier=verifier)
        r = signer.seal(_make_pending_receipt("sender"))
        relay.ingest(r)
        return relay, r

    def test_send_message_accepted(self, signer, verifier):
        relay, r = self._setup_relay_with_receipt(signer, verifier)
        msg = RelayMessage(
            message_id=str(uuid.uuid4()),
            from_agent="sender",
            to_agent="receiver",
            body="hello",
            receipt_ids=[r.receipt_id],
        )
        assert relay.send_message(msg)
        assert relay.message_count == 1

    def test_send_message_rejected_bad_receipt(self, signer, verifier):
        relay, _ = self._setup_relay_with_receipt(signer, verifier)
        msg = RelayMessage(
            message_id=str(uuid.uuid4()),
            from_agent="sender",
            to_agent="receiver",
            body="hello",
            receipt_ids=["nonexistent"],
        )
        assert not relay.send_message(msg)

    def test_inbox(self, signer, verifier):
        relay, r = self._setup_relay_with_receipt(signer, verifier)
        msg = RelayMessage(
            message_id="msg-1",
            from_agent="sender",
            to_agent="receiver",
            body="hello",
            receipt_ids=[r.receipt_id],
        )
        relay.send_message(msg)
        inbox = relay.inbox("receiver")
        assert len(inbox) == 1
        assert inbox[0].body == "hello"

    def test_inbox_broadcast(self, signer, verifier):
        relay, r = self._setup_relay_with_receipt(signer, verifier)
        msg = RelayMessage(
            message_id="msg-broadcast",
            from_agent="sender",
            to_agent="#broadcast",
            body="announcement",
            receipt_ids=[r.receipt_id],
        )
        relay.send_message(msg)
        assert len(relay.inbox("anyone")) == 1

    def test_acknowledge(self, signer, verifier):
        relay, r = self._setup_relay_with_receipt(signer, verifier)
        msg = RelayMessage(
            message_id="msg-ack",
            from_agent="sender",
            to_agent="receiver",
            body="ack me",
            receipt_ids=[r.receipt_id],
        )
        relay.send_message(msg)
        assert relay.acknowledge("msg-ack")
        assert len(relay.inbox("receiver", unacknowledged_only=True)) == 0
        assert len(relay.inbox("receiver", unacknowledged_only=False)) == 1

    def test_acknowledge_nonexistent(self, signer, verifier):
        relay = ReceiptRelay(verifier=verifier)
        assert not relay.acknowledge("nope")


# ------------------------------------------------------------------ #
# API router tests
# ------------------------------------------------------------------ #


class TestAttestationAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from swarm.api.routers import attestation as att_module

        # Reset module-level singletons for test isolation
        att_module._signer = None
        att_module._middleware = None
        att_module._relay = None

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(att_module.router, prefix="/api/v1/attestation")
        return TestClient(app)

    def test_attest_action(self, client):
        resp = client.post(
            "/api/v1/attestation/attest",
            json={
                "agent_id": "agent-1",
                "action_type": "interaction_completed",
                "payload": {"accepted": True},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "sealed"
        assert data["admissible"] is True
        assert data["agent_id"] == "agent-1"

    def test_get_receipt(self, client):
        # Create a receipt first
        resp = client.post(
            "/api/v1/attestation/attest",
            json={
                "agent_id": "agent-2",
                "action_type": "payoff_computed",
                "payload": {"x": 1},
            },
        )
        receipt_id = resp.json()["receipt_id"]

        resp2 = client.get(f"/api/v1/attestation/receipts/{receipt_id}")
        assert resp2.status_code == 200
        assert resp2.json()["receipt_id"] == receipt_id

    def test_get_receipt_not_found(self, client):
        resp = client.get("/api/v1/attestation/receipts/nonexistent")
        assert resp.status_code == 404

    def test_validate_chain(self, client):
        resp = client.post(
            "/api/v1/attestation/attest",
            json={
                "agent_id": "agent-3",
                "action_type": "agent_state_updated",
                "payload": {"state": "active"},
            },
        )
        receipt_id = resp.json()["receipt_id"]

        resp2 = client.get(f"/api/v1/attestation/receipts/{receipt_id}/chain")
        assert resp2.status_code == 200
        assert resp2.json()["chain_valid"] is True

    def test_send_and_receive_message(self, client):
        # Attest first to get a receipt
        resp = client.post(
            "/api/v1/attestation/attest",
            json={
                "agent_id": "sender",
                "action_type": "agent_state_updated",
                "payload": {"msg": "backing"},
            },
        )
        receipt_id = resp.json()["receipt_id"]

        # Send message
        resp2 = client.post(
            "/api/v1/attestation/messages",
            json={
                "from_agent": "sender",
                "to_agent": "receiver",
                "body": "hello via API",
                "receipt_ids": [receipt_id],
            },
        )
        assert resp2.status_code == 200
        msg_id = resp2.json()["message_id"]

        # Check inbox
        resp3 = client.get("/api/v1/attestation/messages/inbox/receiver")
        assert resp3.status_code == 200
        msgs = resp3.json()
        assert len(msgs) == 1
        assert msgs[0]["body"] == "hello via API"

        # Ack
        resp4 = client.post(f"/api/v1/attestation/messages/{msg_id}/ack")
        assert resp4.status_code == 200

        # Inbox should be empty (unacked only)
        resp5 = client.get("/api/v1/attestation/messages/inbox/receiver")
        assert resp5.status_code == 200
        assert len(resp5.json()) == 0

    def test_send_message_bad_receipt(self, client):
        resp = client.post(
            "/api/v1/attestation/messages",
            json={
                "from_agent": "sender",
                "to_agent": "receiver",
                "body": "should fail",
                "receipt_ids": ["nonexistent"],
            },
        )
        assert resp.status_code == 422
