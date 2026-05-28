"""Cryptographic agent identity and delegation chains for AgentGit.

The MVP signs provenance bundles with a shared HMAC key (see
``swarm.attestation.signer``). That proves a bundle was sealed by *someone
holding the key* — it cannot prove *which agent* produced a change, because
anyone with the key can impersonate any ``agent_id`` string.

This module adds verifiable identity using Ed25519 (asymmetric) signatures:

- An :class:`AgentIdentity` carries a DID derived from a public key, plus the
  owner/org and model/runtime/version provenance and the tools the agent is
  allowed to use.
- A :class:`DelegationChain` of signed :class:`DelegationLink` objects encodes
  ``human -> org policy -> task agent``. Each link is signed by its issuer, and
  permissions may only narrow down the chain. Because we use ``did:key``, every
  issuer's public key is embedded in its DID, so a verifier needs no external
  key registry: it extracts the key from the DID and checks the signature.

DID format (simplified ``did:key``): ``did:key:ed25519:<hex public key>``. This
is intentionally simpler than the full multibase/multicodec ``did:key`` spec;
it is unambiguous and self-describing for our purposes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

DID_PREFIX = "did:key:ed25519:"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(payload: Dict[str, Any]) -> bytes:
    """Stable, signature-ready encoding of a mapping."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def did_from_public_key(public_key_hex: str) -> str:
    return f"{DID_PREFIX}{public_key_hex}"


def public_key_from_did(did: str) -> str:
    if not did.startswith(DID_PREFIX):
        raise ValueError(f"Unsupported DID method: {did!r}")
    return did[len(DID_PREFIX) :]


class AgentKeypair:
    """An Ed25519 keypair backing one agent (or org/human) identity."""

    def __init__(self, signing_key: SigningKey) -> None:
        self._signing_key = signing_key
        self._verify_key = signing_key.verify_key

    @classmethod
    def generate(cls) -> "AgentKeypair":
        return cls(SigningKey.generate())

    @classmethod
    def from_seed_hex(cls, seed_hex: str) -> "AgentKeypair":
        """Deterministically rebuild a keypair from a 32-byte hex seed."""

        seed = bytes.fromhex(seed_hex)
        if len(seed) != 32:
            raise ValueError("Ed25519 seed must be 32 bytes (64 hex chars)")
        return cls(SigningKey(seed))

    @property
    def seed_hex(self) -> str:
        """Hex-encoded private seed. Treat as a secret."""

        return bytes(self._signing_key).hex()

    @property
    def public_key_hex(self) -> str:
        return bytes(self._verify_key).hex()

    @property
    def did(self) -> str:
        return did_from_public_key(self.public_key_hex)

    def sign(self, message: bytes) -> str:
        """Return a detached signature as hex."""

        signature: bytes = self._signing_key.sign(message).signature
        return signature.hex()


def verify_signature(did: str, message: bytes, signature_hex: str) -> bool:
    """Verify a detached hex signature against the key embedded in ``did``."""

    try:
        verify_key = VerifyKey(bytes.fromhex(public_key_from_did(did)))
        verify_key.verify(message, bytes.fromhex(signature_hex))
        return True
    except (BadSignatureError, ValueError):
        return False


@dataclass(frozen=True)
class AgentIdentity:
    """Verifiable identity for one agent, owner, or org."""

    did: str
    owner: str
    org: str
    model: str = ""
    runtime: str = ""
    version: str = ""
    allowed_tools: List[str] = field(default_factory=list)

    @classmethod
    def for_keypair(
        cls,
        keypair: AgentKeypair,
        *,
        owner: str,
        org: str,
        model: str = "",
        runtime: str = "",
        version: str = "",
        allowed_tools: Optional[List[str]] = None,
    ) -> "AgentIdentity":
        return cls(
            did=keypair.did,
            owner=owner,
            org=org,
            model=model,
            runtime=runtime,
            version=version,
            allowed_tools=list(allowed_tools or []),
        )

    @property
    def public_key_hex(self) -> str:
        return public_key_from_did(self.did)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "did": self.did,
            "owner": self.owner,
            "org": self.org,
            "model": self.model,
            "runtime": self.runtime,
            "version": self.version,
            "allowed_tools": list(self.allowed_tools),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentIdentity":
        return cls(
            did=data["did"],
            owner=data.get("owner", ""),
            org=data.get("org", ""),
            model=data.get("model", ""),
            runtime=data.get("runtime", ""),
            version=data.get("version", ""),
            allowed_tools=list(data.get("allowed_tools", [])),
        )


@dataclass(frozen=True)
class DelegationLink:
    """One signed grant: ``issuer`` delegates ``permissions`` to ``subject``."""

    issuer_did: str
    subject_did: str
    permissions: List[str]
    issued_at: str
    not_after: Optional[str] = None
    signature: str = ""

    def _signing_payload(self) -> Dict[str, Any]:
        return {
            "issuer_did": self.issuer_did,
            "subject_did": self.subject_did,
            "permissions": sorted(self.permissions),
            "issued_at": self.issued_at,
            "not_after": self.not_after,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self._signing_payload())

    def to_dict(self) -> Dict[str, Any]:
        return {**self._signing_payload(), "signature": self.signature}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DelegationLink":
        return cls(
            issuer_did=data["issuer_did"],
            subject_did=data["subject_did"],
            permissions=list(data.get("permissions", [])),
            issued_at=data["issued_at"],
            not_after=data.get("not_after"),
            signature=data.get("signature", ""),
        )


def sign_link(
    issuer: AgentKeypair,
    *,
    subject_did: str,
    permissions: List[str],
    issued_at: Optional[str] = None,
    not_after: Optional[str] = None,
) -> DelegationLink:
    """Create a link signed by ``issuer`` granting ``permissions`` to subject."""

    unsigned = DelegationLink(
        issuer_did=issuer.did,
        subject_did=subject_did,
        permissions=list(permissions),
        issued_at=issued_at or _now_iso(),
        not_after=not_after,
    )
    signature = issuer.sign(unsigned.canonical_bytes())
    return DelegationLink(
        issuer_did=unsigned.issuer_did,
        subject_did=unsigned.subject_did,
        permissions=unsigned.permissions,
        issued_at=unsigned.issued_at,
        not_after=unsigned.not_after,
        signature=signature,
    )


@dataclass(frozen=True)
class DelegationChain:
    """An ordered ``human -> org -> ... -> agent`` chain of signed links."""

    links: List[DelegationLink] = field(default_factory=list)

    @property
    def root_did(self) -> Optional[str]:
        return self.links[0].issuer_did if self.links else None

    @property
    def subject_did(self) -> Optional[str]:
        return self.links[-1].subject_did if self.links else None

    def effective_permissions(self) -> List[str]:
        """Permissions granted to the final subject (the last link)."""

        return list(self.links[-1].permissions) if self.links else []

    def verify(
        self,
        *,
        expected_subject_did: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Tuple[bool, List[str]]:
        """Verify signatures, connectivity, permission narrowing, and expiry."""

        errors: List[str] = []
        if not self.links:
            return False, ["delegation chain is empty"]

        now = now or datetime.now(timezone.utc)
        prev_permissions: Optional[set[str]] = None
        prev_subject: Optional[str] = None

        for index, link in enumerate(self.links):
            if not verify_signature(link.issuer_did, link.canonical_bytes(), link.signature):
                errors.append(f"link {index}: invalid issuer signature")

            if prev_subject is not None and link.issuer_did != prev_subject:
                errors.append(
                    f"link {index}: issuer {link.issuer_did} does not match "
                    f"prior subject {prev_subject}"
                )

            permissions = set(link.permissions)
            if prev_permissions is not None and not permissions <= prev_permissions:
                widened = sorted(permissions - prev_permissions)
                errors.append(
                    f"link {index}: permissions widen beyond parent grant: {widened}"
                )

            if link.not_after is not None:
                # verify() is a security boundary over untrusted bundle data;
                # a malformed timestamp must surface as an error, never raise.
                try:
                    expiry = datetime.fromisoformat(link.not_after)
                except ValueError:
                    errors.append(f"link {index}: malformed not_after {link.not_after!r}")
                else:
                    if expiry.tzinfo is None:
                        expiry = expiry.replace(tzinfo=timezone.utc)
                    if now > expiry:
                        errors.append(
                            f"link {index}: delegation expired at {link.not_after}"
                        )

            prev_permissions = permissions
            prev_subject = link.subject_did

        if expected_subject_did is not None and self.subject_did != expected_subject_did:
            errors.append(
                f"chain subject {self.subject_did} does not match expected "
                f"agent {expected_subject_did}"
            )

        return not errors, errors

    def to_dict(self) -> Dict[str, Any]:
        return {"links": [link.to_dict() for link in self.links]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DelegationChain":
        return cls(links=[DelegationLink.from_dict(item) for item in data.get("links", [])])
