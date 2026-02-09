"""Tests for identity and trust infrastructure."""

import pytest

from swarm.models.identity import (
    AgentIdentity,
    CredentialIssuer,
    IdentityConfig,
    IdentityRegistry,
    VerifiableCredential,
)


class TestVerifiableCredential:
    """Tests for VerifiableCredential."""

    def test_valid_credential(self):
        cred = VerifiableCredential(
            issued_epoch=0,
            expires_epoch=10,
            revoked=False,
        )
        assert cred.is_valid(5) is True

    def test_expired_credential(self):
        cred = VerifiableCredential(
            issued_epoch=0,
            expires_epoch=10,
            revoked=False,
        )
        assert cred.is_valid(10) is False
        assert cred.is_valid(15) is False

    def test_revoked_credential(self):
        cred = VerifiableCredential(
            issued_epoch=0,
            expires_epoch=10,
            revoked=True,
        )
        assert cred.is_valid(5) is False

    def test_no_expiry_always_valid(self):
        cred = VerifiableCredential(
            issued_epoch=0,
            expires_epoch=None,
            revoked=False,
        )
        assert cred.is_valid(1000) is True

    def test_serialization_roundtrip(self):
        cred = VerifiableCredential(
            issuer_id="system",
            subject_id="agent_1",
            claim_type="audit_pass",
            claim_value={"epoch": 5},
            issued_epoch=5,
            expires_epoch=55,
        )
        d = cred.to_dict()
        cred2 = VerifiableCredential.from_dict(d)
        assert cred2.issuer_id == "system"
        assert cred2.subject_id == "agent_1"
        assert cred2.claim_type == "audit_pass"
        assert cred2.expires_epoch == 55


class TestAgentIdentity:
    """Tests for AgentIdentity."""

    def test_has_credential(self):
        cred = VerifiableCredential(
            claim_type="audit_pass",
            issued_epoch=0,
            expires_epoch=10,
        )
        identity = AgentIdentity(
            agent_id="a1",
            credentials=[cred],
        )
        assert identity.has_credential("audit_pass", current_epoch=5) is True
        assert identity.has_credential("reputation_history", current_epoch=5) is False

    def test_has_credential_expired(self):
        cred = VerifiableCredential(
            claim_type="audit_pass",
            issued_epoch=0,
            expires_epoch=5,
        )
        identity = AgentIdentity(
            agent_id="a1",
            credentials=[cred],
        )
        assert identity.has_credential("audit_pass", current_epoch=10) is False

    def test_get_credential_value(self):
        cred = VerifiableCredential(
            claim_type="reputation_history",
            claim_value={"reputation": 0.8},
            issued_epoch=0,
            expires_epoch=10,
        )
        identity = AgentIdentity(
            agent_id="a1",
            credentials=[cred],
        )
        val = identity.get_credential_value("reputation_history", current_epoch=5)
        assert val == {"reputation": 0.8}

    def test_compute_trust_score_base(self):
        identity = AgentIdentity(agent_id="a1")
        score = identity.compute_trust_score()
        assert score == pytest.approx(0.3)

    def test_compute_trust_score_with_pop(self):
        identity = AgentIdentity(agent_id="a1", proof_of_personhood=True)
        score = identity.compute_trust_score()
        assert score == pytest.approx(0.5)

    def test_compute_trust_score_with_credentials(self):
        creds = [
            VerifiableCredential(
                claim_type=f"type_{i}",
                issued_epoch=0,
                expires_epoch=100,
            )
            for i in range(3)
        ]
        identity = AgentIdentity(agent_id="a1", credentials=creds)
        score = identity.compute_trust_score()
        assert score == pytest.approx(0.6)  # 0.3 base + 0.3 from 3 creds

    def test_compute_trust_score_capped(self):
        creds = [
            VerifiableCredential(
                claim_type=f"type_{i}",
                issued_epoch=0,
                expires_epoch=100,
            )
            for i in range(10)
        ]
        identity = AgentIdentity(
            agent_id="a1",
            credentials=creds,
            proof_of_personhood=True,
        )
        score = identity.compute_trust_score()
        assert score == 1.0

    def test_serialization_roundtrip(self):
        identity = AgentIdentity(
            agent_id="a1",
            identity_cost=10.0,
            creation_epoch=5,
            proof_of_personhood=True,
            trust_score=0.7,
        )
        d = identity.to_dict()
        identity2 = AgentIdentity.from_dict(d)
        assert identity2.agent_id == "a1"
        assert identity2.proof_of_personhood is True
        assert identity2.trust_score == 0.7


class TestCredentialIssuer:
    """Tests for CredentialIssuer."""

    def test_issue_credential(self):
        issuer = CredentialIssuer()
        cred = issuer.issue_credential(
            subject_id="a1",
            claim_type="audit_pass",
            claim_value={"score": 0.9},
            current_epoch=5,
        )
        assert cred.subject_id == "a1"
        assert cred.claim_type == "audit_pass"
        assert cred.issued_epoch == 5
        assert cred.expires_epoch == 55  # 5 + 50 (default expiry)

    def test_verify_credential(self):
        issuer = CredentialIssuer()
        cred = issuer.issue_credential("a1", "audit_pass", None, 5)
        assert issuer.verify_credential(cred.credential_id, 10) is True
        assert issuer.verify_credential(cred.credential_id, 100) is False

    def test_verify_nonexistent(self):
        issuer = CredentialIssuer()
        assert issuer.verify_credential("nonexistent", 0) is False

    def test_revoke_credential(self):
        issuer = CredentialIssuer()
        cred = issuer.issue_credential("a1", "audit_pass", None, 5)
        assert issuer.revoke_credential(cred.credential_id) is True
        assert issuer.verify_credential(cred.credential_id, 10) is False

    def test_revoke_nonexistent(self):
        issuer = CredentialIssuer()
        assert issuer.revoke_credential("nonexistent") is False

    def test_issue_reputation_credential(self):
        issuer = CredentialIssuer()
        cred = issuer.issue_reputation_credential("a1", 0.85, 10)
        assert cred.claim_type == "reputation_history"
        assert cred.claim_value["reputation"] == 0.85

    def test_issue_audit_pass(self):
        issuer = CredentialIssuer()
        cred = issuer.issue_audit_pass("a1", 10)
        assert cred.claim_type == "audit_pass"


class TestIdentityConfig:
    """Tests for IdentityConfig validation."""

    def test_default_config_valid(self):
        config = IdentityConfig()
        config.validate()

    def test_invalid_creation_cost(self):
        with pytest.raises(ValueError, match="identity_creation_cost"):
            IdentityConfig(identity_creation_cost=-1).validate()

    def test_invalid_expiry(self):
        with pytest.raises(ValueError, match="credential_expiry_epochs"):
            IdentityConfig(credential_expiry_epochs=0).validate()

    def test_invalid_max_identities(self):
        with pytest.raises(ValueError, match="max_identities_per_entity"):
            IdentityConfig(max_identities_per_entity=0).validate()

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="behavioral_similarity_threshold"):
            IdentityConfig(behavioral_similarity_threshold=1.5).validate()


class TestIdentityRegistry:
    """Tests for the IdentityRegistry."""

    def test_create_identity(self):
        registry = IdentityRegistry()
        identity = registry.create_identity("a1")
        assert identity is not None
        assert identity.agent_id == "a1"
        assert identity.identity_cost == 10.0

    def test_duplicate_identity_rejected(self):
        registry = IdentityRegistry()
        assert registry.create_identity("a1") is not None
        assert registry.create_identity("a1") is None

    def test_max_identities_per_entity(self):
        registry = IdentityRegistry(IdentityConfig(max_identities_per_entity=2))
        assert registry.create_identity("a1", entity_id="entity_1") is not None
        assert registry.create_identity("a2", entity_id="entity_1") is not None
        assert registry.create_identity("a3", entity_id="entity_1") is None

    def test_pop_required(self):
        registry = IdentityRegistry(IdentityConfig(proof_of_personhood_required=True))
        assert registry.create_identity("a1", proof_of_personhood=False) is None
        assert registry.create_identity("a1", proof_of_personhood=True) is not None

    def test_get_identity(self):
        registry = IdentityRegistry()
        registry.create_identity("a1")
        identity = registry.get_identity("a1")
        assert identity is not None
        assert identity.agent_id == "a1"

    def test_get_nonexistent_identity(self):
        registry = IdentityRegistry()
        assert registry.get_identity("nonexistent") is None

    def test_sybil_detection_identical_patterns(self):
        registry = IdentityRegistry(IdentityConfig(behavioral_similarity_threshold=0.7))
        patterns = {
            "a1": {"target_1": 10, "target_2": 5, "target_3": 3},
            "a2": {"target_1": 10, "target_2": 5, "target_3": 3},  # Identical
            "a3": {"target_4": 10, "target_5": 5},  # Different
        }
        clusters = registry.detect_sybil_clusters(patterns)
        assert len(clusters) == 1
        assert {"a1", "a2"} == clusters[0]

    def test_sybil_detection_no_clusters(self):
        registry = IdentityRegistry(IdentityConfig(behavioral_similarity_threshold=0.9))
        patterns = {
            "a1": {"target_1": 10},
            "a2": {"target_2": 10},
            "a3": {"target_3": 10},
        }
        clusters = registry.detect_sybil_clusters(patterns)
        assert len(clusters) == 0

    def test_sybil_detection_disabled(self):
        registry = IdentityRegistry(IdentityConfig(sybil_detection_enabled=False))
        patterns = {
            "a1": {"target_1": 10},
            "a2": {"target_1": 10},
        }
        clusters = registry.detect_sybil_clusters(patterns)
        assert len(clusters) == 0

    def test_pop_verification_rate(self):
        registry = IdentityRegistry()
        registry.create_identity("a1", proof_of_personhood=True)
        registry.create_identity("a2", proof_of_personhood=False)
        registry.create_identity("a3", proof_of_personhood=True)

        rate = registry.pop_verification_rate()
        assert rate == pytest.approx(2 / 3)

    def test_pop_verification_rate_empty(self):
        registry = IdentityRegistry()
        assert registry.pop_verification_rate() == 0.0

    def test_trust_distribution(self):
        registry = IdentityRegistry()
        registry.create_identity("a1")
        registry.create_identity("a2", proof_of_personhood=True)

        dist = registry.get_trust_distribution()
        assert "a1" in dist
        assert "a2" in dist
        assert dist["a2"] > dist["a1"]  # PoP adds trust
