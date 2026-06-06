"""Targeted regression tests for AVPMapper.

Currently covers the negative-attestation penalty bug (Codex P2 on
PR #507): without populating the penalty fields, outcome_sign=-1 mapped
to v_hat ≈ +0.08 / p ≈ 0.54 — a negative attestation recorded as
beneficial. The fix mirrors the negative signal across rework_count,
verifier_rejections, tool_misuse_flags, and engagement_delta so v_hat
tracks the attestation's sign.

These are not the full mitigation test suite the rsavitt review asked for
(D5 clamp, C4 cap, D1 reputation suppression, D2 ordinal mapping) —
those need the agentveil v1 spec context. Filed as a follow-up.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from swarm.bridges.agentveil.mapper import AVPMapper
from swarm.core.proxy import ProxyComputer


def _att(outcome_sign: int) -> SimpleNamespace:
    return SimpleNamespace(
        outcome_sign=outcome_sign,
        interaction_id="",
        evidence_hash="",
    )


class TestAttestationOutcomeSignFidelity:
    """Negative attestations must produce v_hat < 0 / p < 0.5."""

    def setup_method(self) -> None:
        self.mapper = AVPMapper()
        self.computer = ProxyComputer()

    def _map(self, sign: int):
        return self.mapper.attestation_received_to_interaction(
            _att(sign),
            initiator_id="a",
            counterparty_id="b",
            computer=self.computer,
        )

    def test_negative_attestation_records_as_negative(self):
        ix = self._map(-1)
        assert ix.v_hat < 0, f"expected v_hat<0, got {ix.v_hat}"
        assert ix.p < 0.5, f"expected p<0.5, got {ix.p}"
        assert ix.accepted is False

    def test_positive_attestation_records_as_positive(self):
        ix = self._map(+1)
        assert ix.v_hat > 0
        assert ix.p > 0.5
        assert ix.accepted is True

    def test_negative_attestation_penalty_fields_populated(self):
        """Mitigation: the penalty fields *must* be non-zero so the proxy
        weighting can't read them as 'good'. This is the load-bearing
        invariant for the bug."""
        ix = self._map(-1)
        assert ix.rework_count > 0
        assert ix.verifier_rejections > 0
        assert ix.tool_misuse_flags > 0
        assert ix.counterparty_engagement_delta < 0

    def test_positive_attestation_no_penalty_fields(self):
        """Conversely, positive attestations should not carry penalty signals."""
        ix = self._map(+1)
        assert ix.rework_count == 0
        assert ix.verifier_rejections == 0
        assert ix.tool_misuse_flags == 0

    @pytest.mark.parametrize("sign", [-1, +1])
    def test_p_in_valid_range(self, sign):
        ix = self._map(sign)
        assert 0.0 <= ix.p <= 1.0
