"""Build and write AgentGit provenance bundles."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from swarm.agentgit.git import GitSnapshot, collect_snapshot
from swarm.agentgit.identity import (
    AgentIdentity,
    AgentKeypair,
    DelegationChain,
    verify_signature,
)
from swarm.agentgit.policy import AgentGitPolicy, PolicyDecision, decisions_passed
from swarm.attestation.receipt import (
    AdmissibilityReceipt,
    ExecutionBounds,
    PolicyCompliance,
    PolicySeverity,
    ReceiptStatus,
)
from swarm.attestation.signer import ReceiptSigner, ReceiptVerifier

DEFAULT_DEV_SIGNING_KEY = "0" * 64


def build_bundle(
    *,
    repo: Path,
    task_id: str,
    agent_id: str,
    policy: AgentGitPolicy,
    base_ref: str = "HEAD",
    check_results: Dict[str, bool] | None = None,
    signing_key: str | None = None,
    signer_id: str = "agentgit-local",
    identity: AgentIdentity | None = None,
    agent_keypair: AgentKeypair | None = None,
    delegation: DelegationChain | None = None,
) -> Dict[str, Any]:
    """Build a signed provenance bundle for the current git diff.

    When ``identity`` and ``agent_keypair`` are supplied, the agent's Ed25519
    key signs the receipt ``payload_hash``, binding a *verifiable* identity to
    this exact diff. An optional ``delegation`` chain records the
    ``human -> org -> agent`` authority under which the agent acted.
    """

    snapshot = collect_snapshot(repo, base_ref=base_ref)
    decisions = policy.evaluate(snapshot, check_results=check_results)
    payload = _payload(task_id, agent_id, snapshot, decisions, check_results or {})
    receipt = _seal_receipt(
        payload=payload,
        agent_id=agent_id,
        signing_key=signing_key or DEFAULT_DEV_SIGNING_KEY,
        signer_id=signer_id,
        decisions=decisions,
    )

    bundle: Dict[str, Any] = {
        "schema_version": "agentgit.provenance.v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": {"task_id": task_id},
        "agent": {"agent_id": agent_id},
        "git": _snapshot_dict(snapshot),
        "policy": {
            "passed": decisions_passed(decisions),
            "decisions": [asdict(decision) for decision in decisions],
        },
        "checks": check_results or {},
        "receipt": receipt.model_dump(mode="json"),
    }

    if identity is not None:
        bundle["identity"] = identity.to_dict()
        if agent_keypair is not None:
            if agent_keypair.did != identity.did:
                raise ValueError("agent_keypair does not match identity.did")
            bundle["identity_signature"] = {
                "did": identity.did,
                "over": "receipt.payload_hash",
                "signature": agent_keypair.sign(receipt.payload_hash.encode("utf-8")),
            }
    if delegation is not None:
        bundle["delegation"] = delegation.to_dict()

    return bundle


def write_bundle(bundle: Dict[str, Any], output_path: Path) -> Path:
    """Write a provenance bundle as stable, reviewable JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return output_path


def load_bundle(path: Path) -> Dict[str, Any]:
    """Load a provenance bundle from disk."""

    bundle: Dict[str, Any] = json.loads(path.read_text())
    return bundle


def verify_bundle(
    bundle: Dict[str, Any],
    *,
    signing_key: str | None = None,
    require_policy_pass: bool = True,
) -> tuple[bool, List[str]]:
    """Verify bundle hash, receipt signature, and optional policy pass state."""

    errors: List[str] = []
    if bundle.get("schema_version") != "agentgit.provenance.v0":
        errors.append("unsupported schema_version")

    receipt = AdmissibilityReceipt.model_validate(bundle.get("receipt", {}))
    payload = _payload_from_bundle(bundle)
    expected_hash = AdmissibilityReceipt.hash_payload(payload)
    if receipt.payload_hash != expected_hash:
        errors.append("receipt payload_hash does not match bundle payload")

    key = signing_key or DEFAULT_DEV_SIGNING_KEY
    if not ReceiptVerifier(key).verify(receipt):
        errors.append("receipt signature verification failed")

    if require_policy_pass and not bundle.get("policy", {}).get("passed", False):
        errors.append("policy did not pass")

    errors.extend(_verify_identity(bundle, receipt))

    return not errors, errors


def _verify_identity(bundle: Dict[str, Any], receipt: AdmissibilityReceipt) -> List[str]:
    """Verify the optional Ed25519 identity signature and delegation chain.

    Absent identity blocks are not an error: legacy bundles still verify. When
    present, the agent's signature must cover this bundle's ``payload_hash``,
    the delegation chain must be valid, its final subject must be the signing
    identity, and the identity's tools must stay within the delegated grant.
    """

    errors: List[str] = []
    identity_block = bundle.get("identity")
    signature_block = bundle.get("identity_signature")

    if signature_block is not None:
        if identity_block is None:
            errors.append("identity_signature present without identity block")
            return errors
        identity = AgentIdentity.from_dict(identity_block)
        if signature_block.get("did") != identity.did:
            errors.append("identity_signature did does not match identity block")
        elif not verify_signature(
            identity.did,
            receipt.payload_hash.encode("utf-8"),
            signature_block.get("signature", ""),
        ):
            errors.append("identity signature does not verify against payload_hash")

    delegation_block = bundle.get("delegation")
    if delegation_block is not None:
        chain = DelegationChain.from_dict(delegation_block)
        expected_subject = identity_block.get("did") if identity_block else None
        ok, chain_errors = chain.verify(expected_subject_did=expected_subject)
        errors.extend(f"delegation: {err}" for err in chain_errors)
        if ok and identity_block is not None:
            granted = set(chain.effective_permissions())
            requested = set(identity_block.get("allowed_tools", []))
            if not requested <= granted:
                exceeded = sorted(requested - granted)
                errors.append(f"delegation: allowed_tools exceed delegated grant: {exceeded}")

    return errors


def _payload(
    task_id: str,
    agent_id: str,
    snapshot: GitSnapshot,
    decisions: list[PolicyDecision],
    check_results: Dict[str, bool],
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "agent_id": agent_id,
        "git": _snapshot_dict(snapshot),
        "policy_decisions": [asdict(decision) for decision in decisions],
        "checks": check_results,
    }


def _payload_from_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": bundle.get("task", {}).get("task_id"),
        "agent_id": bundle.get("agent", {}).get("agent_id"),
        "git": bundle.get("git", {}),
        "policy_decisions": bundle.get("policy", {}).get("decisions", []),
        "checks": bundle.get("checks", {}),
    }


def _snapshot_dict(snapshot: GitSnapshot) -> Dict[str, Any]:
    return {
        "repo_path": snapshot.repo_path,
        "base_ref": snapshot.base_ref,
        "head_ref": snapshot.head_ref,
        "base_commit": snapshot.base_commit,
        "head_commit": snapshot.head_commit,
        "branch": snapshot.branch,
        "changed_files": [asdict(file) for file in snapshot.changed_files],
        "totals": {
            "changed_files": len(snapshot.changed_files),
            "additions": snapshot.total_additions,
            "deletions": snapshot.total_deletions,
        },
    }


def _seal_receipt(
    *,
    payload: Dict[str, Any],
    agent_id: str,
    signing_key: str,
    signer_id: str,
    decisions: list[PolicyDecision],
) -> AdmissibilityReceipt:
    payload_hash = AdmissibilityReceipt.hash_payload(payload)
    receipt_id = AdmissibilityReceipt.generate_receipt_id(
        agent_id=agent_id,
        action_type="agentgit.attest",
        payload_hash=payload_hash,
    )
    receipt = AdmissibilityReceipt(  # type: ignore[call-arg]
        receipt_id=receipt_id,
        status=ReceiptStatus.PENDING,
        agent_id=agent_id,
        action_type="agentgit.attest",
        payload_hash=payload_hash,
        policy_results=[
            PolicyCompliance(
                policy_id=decision.policy_id,
                passed=decision.passed,
                severity=PolicySeverity(decision.severity),
                details={"reason": decision.reason, **decision.details},
            )
            for decision in decisions
        ],
        bounds=ExecutionBounds(max_delegation_depth=0),  # type: ignore[call-arg]
    )
    return ReceiptSigner(secret_key=signing_key, signer_id=signer_id).seal(receipt)
