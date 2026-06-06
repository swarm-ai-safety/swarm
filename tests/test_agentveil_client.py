"""Targeted regression tests for AVPClient.

Covers the two Codex P2 bugs on PR #508:
  - _is_valid_hex accepted any-length hex, but the E3 contract is a
    SHA-256 digest (64 hex chars). Inputs like "deadbeef" got receipts.
  - Mock fixture tiers bypassed _validate_tier, so a typo like "trustd"
    propagated into TrustDecision payloads.

Not the full security-validator suite the rsavitt review asked for
(B5 did:key whitelist, D6 deterministic pubkey, E3 pre-hashed evidence,
plus the four input validators). Filed as a follow-up.
"""

from __future__ import annotations

import hashlib

import pytest

from swarm.bridges.agentveil.client import AVPClient

VALID_DID = "did:key:z6MkqaliceTestDID0000000000000000000000000"


@pytest.fixture
def client() -> AVPClient:
    return AVPClient()


def _sha256_hex(payload: bytes = b"x") -> str:
    return hashlib.sha256(payload).hexdigest()


class TestEvidenceHashLengthEnforced:
    """E3: evidence_hash must be a 64-char lowercase SHA-256 hex digest."""

    def test_valid_sha256_accepted(self, client):
        r = client.submit_attestation(VALID_DID, +1, _sha256_hex())
        assert r.accepted is True
        assert r.attestation_id.startswith("mock-")

    @pytest.mark.parametrize(
        "bad_hex",
        [
            "",                  # empty
            "deadbeef",          # 8 chars (valid hex, wrong length)
            "a" * 63,            # 63 chars (one short)
            "a" * 65,            # 65 chars (one long)
            "a" * 128,           # 128 chars (SHA-512 length)
            _sha256_hex().upper(),  # uppercase form (canonicalization)
        ],
    )
    def test_wrong_length_or_case_rejected(self, client, bad_hex):
        with pytest.raises(ValueError, match="evidence_hash"):
            client.submit_attestation(VALID_DID, +1, bad_hex)

    def test_non_hex_rejected(self, client):
        with pytest.raises(ValueError, match="evidence_hash"):
            client.submit_attestation(VALID_DID, +1, "z" * 64)


class TestFixtureTierValidated:
    """Mock fixtures must route through _validate_tier so typo'd tiers
    can't leak into TrustDecision payloads."""

    def test_valid_fixture_tier_accepted(self):
        c = AVPClient(fixtures={VALID_DID: {"tier": "elite"}})
        d = c.can_trust(VALID_DID, min_tier="basic")
        assert d.tier == "elite"
        assert d.allowed is True

    def test_typoed_fixture_tier_rejected(self):
        c = AVPClient(fixtures={VALID_DID: {"tier": "trustd"}})
        with pytest.raises(ValueError, match="Invalid tier"):
            c.can_trust(VALID_DID, min_tier="basic")

    def test_empty_fixture_tier_falls_back_to_newcomer(self):
        c = AVPClient(fixtures={VALID_DID: {}})
        d = c.can_trust(VALID_DID, min_tier="newcomer")
        assert d.tier == "newcomer"
