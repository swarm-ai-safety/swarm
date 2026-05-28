"""Tests for AgentGit cryptographic identity and delegation chains."""

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from swarm.agentgit.bundle import build_bundle, verify_bundle
from swarm.agentgit.identity import (
    DID_PREFIX,
    AgentIdentity,
    AgentKeypair,
    DelegationChain,
    sign_link,
    verify_signature,
)
from swarm.agentgit.policy import AgentGitPolicy


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "agentgit@example.com")
    _git(repo, "config", "user.name", "AgentGit Test")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    return repo


def _allowed_diff(repo: Path) -> None:
    src = repo / "swarm"
    src.mkdir()
    (src / "ok.py").write_text("print('ok')\n")


# --- keypair / DID primitives ---


def test_seed_roundtrip_is_deterministic():
    kp = AgentKeypair.generate()
    rebuilt = AgentKeypair.from_seed_hex(kp.seed_hex)
    assert rebuilt.did == kp.did
    assert rebuilt.public_key_hex == kp.public_key_hex


def test_from_seed_hex_rejects_wrong_length():
    with pytest.raises(ValueError):
        AgentKeypair.from_seed_hex("ab")


def test_did_embeds_public_key():
    kp = AgentKeypair.generate()
    assert kp.did == f"{DID_PREFIX}{kp.public_key_hex}"


def test_sign_and_verify_roundtrip():
    kp = AgentKeypair.generate()
    sig = kp.sign(b"hello")
    assert verify_signature(kp.did, b"hello", sig)
    assert not verify_signature(kp.did, b"tampered", sig)


def test_verify_signature_rejects_other_key():
    signer = AgentKeypair.generate()
    other = AgentKeypair.generate()
    sig = signer.sign(b"msg")
    assert not verify_signature(other.did, b"msg", sig)


def test_identity_dict_roundtrip():
    kp = AgentKeypair.generate()
    identity = AgentIdentity.for_keypair(
        kp, owner="alice", org="acme", model="codex", allowed_tools=["pytest"]
    )
    assert AgentIdentity.from_dict(identity.to_dict()) == identity
    assert identity.public_key_hex == kp.public_key_hex


# --- delegation chains ---


def _chain(human, org, agent, *, agent_perms=None, org_perms=None):
    agent_perms = agent_perms if agent_perms is not None else ["read", "test"]
    org_perms = org_perms if org_perms is not None else ["read", "test", "open_pr"]
    link_org = sign_link(human, subject_did=org.did, permissions=org_perms)
    link_agent = sign_link(org, subject_did=agent.did, permissions=agent_perms)
    return DelegationChain(links=[link_org, link_agent])


def test_valid_chain_verifies():
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    chain = _chain(human, org, agent)
    ok, errors = chain.verify(expected_subject_did=agent.did)
    assert ok, errors
    assert chain.root_did == human.did
    assert chain.subject_did == agent.did
    assert set(chain.effective_permissions()) == {"read", "test"}


def test_empty_chain_fails():
    ok, errors = DelegationChain().verify()
    assert not ok
    assert "empty" in errors[0]


def test_tampered_link_signature_fails():
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    chain = _chain(human, org, agent)
    bad = chain.links[1]
    tampered = DelegationChain(
        links=[
            chain.links[0],
            type(bad)(
                issuer_did=bad.issuer_did,
                subject_did=bad.subject_did,
                permissions=["read", "test", "deploy"],  # not what was signed
                issued_at=bad.issued_at,
                not_after=bad.not_after,
                signature=bad.signature,
            ),
        ]
    )
    ok, errors = tampered.verify()
    assert not ok
    assert any("invalid issuer signature" in e for e in errors)


def test_broken_connectivity_fails():
    human, org, agent, stranger = (AgentKeypair.generate() for _ in range(4))
    # Second link issued by a stranger, not the org that was delegated to.
    link_org = sign_link(human, subject_did=org.did, permissions=["read"])
    link_agent = sign_link(stranger, subject_did=agent.did, permissions=["read"])
    ok, errors = DelegationChain(links=[link_org, link_agent]).verify()
    assert not ok
    assert any("does not match prior subject" in e for e in errors)


def test_permission_widening_fails():
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    chain = _chain(human, org, agent, org_perms=["read"], agent_perms=["read", "deploy"])
    ok, errors = chain.verify()
    assert not ok
    assert any("widen beyond parent" in e for e in errors)


def test_expired_link_fails():
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    link_org = sign_link(human, subject_did=org.did, permissions=["read"])
    link_agent = sign_link(org, subject_did=agent.did, permissions=["read"], not_after=past)
    ok, errors = DelegationChain(links=[link_org, link_agent]).verify()
    assert not ok
    assert any("expired" in e for e in errors)


def test_unexpired_link_with_future_expiry_passes():
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    link_org = sign_link(human, subject_did=org.did, permissions=["read"])
    link_agent = sign_link(org, subject_did=agent.did, permissions=["read"], not_after=future)
    ok, errors = DelegationChain(links=[link_org, link_agent]).verify()
    assert ok, errors


def test_wrong_expected_subject_fails():
    human, org, agent, imposter = (AgentKeypair.generate() for _ in range(4))
    chain = _chain(human, org, agent)
    ok, errors = chain.verify(expected_subject_did=imposter.did)
    assert not ok
    assert any("does not match expected agent" in e for e in errors)


def test_chain_dict_roundtrip():
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    chain = _chain(human, org, agent)
    rebuilt = DelegationChain.from_dict(chain.to_dict())
    ok, errors = rebuilt.verify(expected_subject_did=agent.did)
    assert ok, errors


# --- bundle integration ---


def test_bundle_with_identity_and_delegation_verifies(tmp_path):
    repo = _init_repo(tmp_path)
    _allowed_diff(repo)
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    identity = AgentIdentity.for_keypair(
        agent, owner="alice", org="acme", allowed_tools=["read", "test"]
    )
    chain = _chain(human, org, agent)

    bundle = build_bundle(
        repo=repo,
        task_id="issue-id",
        agent_id="codex",
        policy=AgentGitPolicy(allowed_paths=["swarm/**"]),
        identity=identity,
        agent_keypair=agent,
        delegation=chain,
    )

    assert bundle["identity"]["did"] == agent.did
    assert bundle["identity_signature"]["over"] == "receipt.payload_hash"
    ok, errors = verify_bundle(bundle)
    assert ok, errors


def test_bundle_identity_signature_tamper_detected(tmp_path):
    repo = _init_repo(tmp_path)
    _allowed_diff(repo)
    agent = AgentKeypair.generate()
    identity = AgentIdentity.for_keypair(agent, owner="alice", org="acme")
    bundle = build_bundle(
        repo=repo,
        task_id="issue-tamper",
        agent_id="codex",
        policy=AgentGitPolicy(allowed_paths=["swarm/**"]),
        identity=identity,
        agent_keypair=agent,
    )

    # Swap in another agent's identity but keep the original signature.
    other = AgentKeypair.generate()
    bundle["identity"]["did"] = other.did
    bundle["identity_signature"]["did"] = other.did
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("identity signature does not verify" in e for e in errors)


def test_bundle_keypair_identity_mismatch_raises(tmp_path):
    repo = _init_repo(tmp_path)
    _allowed_diff(repo)
    identity = AgentIdentity.for_keypair(AgentKeypair.generate(), owner="a", org="b")
    with pytest.raises(ValueError):
        build_bundle(
            repo=repo,
            task_id="issue-mismatch",
            agent_id="codex",
            policy=AgentGitPolicy(allowed_paths=["swarm/**"]),
            identity=identity,
            agent_keypair=AgentKeypair.generate(),  # different key
        )


def test_bundle_tools_exceeding_delegation_fail(tmp_path):
    repo = _init_repo(tmp_path)
    _allowed_diff(repo)
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    # Agent claims a tool ("deploy") it was never delegated.
    identity = AgentIdentity.for_keypair(
        agent, owner="alice", org="acme", allowed_tools=["read", "deploy"]
    )
    chain = _chain(human, org, agent, agent_perms=["read", "test"])
    bundle = build_bundle(
        repo=repo,
        task_id="issue-scope",
        agent_id="codex",
        policy=AgentGitPolicy(allowed_paths=["swarm/**"]),
        identity=identity,
        agent_keypair=agent,
        delegation=chain,
    )
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert any("allowed_tools exceed delegated grant" in e for e in errors)


def test_legacy_bundle_without_identity_still_verifies(tmp_path):
    repo = _init_repo(tmp_path)
    _allowed_diff(repo)
    bundle = build_bundle(
        repo=repo,
        task_id="issue-legacy",
        agent_id="codex",
        policy=AgentGitPolicy(allowed_paths=["swarm/**"]),
    )
    assert "identity" not in bundle
    ok, errors = verify_bundle(bundle)
    assert ok, errors
