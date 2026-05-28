"""Tests for delegation-derived capability enforcement (7ge5)."""

from datetime import datetime, timedelta, timezone

from swarm.agentgit.capabilities import (
    CAPABILITY_COMMANDS,
    enforced_allowlist_for_chain,
    granted_commands,
    known_capabilities,
)
from swarm.agentgit.identity import AgentKeypair, DelegationChain, sign_link
from swarm.bridges.worktree.config import WorktreeConfig
from swarm.bridges.worktree.policy import WorktreePolicy


def _chain(human, org, agent, *, agent_perms, org_perms=None, not_after=None):
    org_perms = org_perms if org_perms is not None else agent_perms
    link_org = sign_link(human, subject_did=org.did, permissions=org_perms)
    link_agent = sign_link(
        org, subject_did=agent.did, permissions=agent_perms, not_after=not_after
    )
    return DelegationChain(links=[link_org, link_agent])


# --- capability vocabulary ---


def test_granted_commands_unions_capabilities():
    cmds = set(granted_commands(["read", "test"]))
    assert {"ls", "cat", "grep"} <= cmds  # from read
    assert {"pytest", "python"} <= cmds  # from test
    assert "git" not in cmds  # vcs not granted


def test_unknown_permission_grants_nothing():
    assert granted_commands(["open_pr", "deploy"]) == []


def test_granted_commands_is_sorted_and_deduped():
    cmds = granted_commands(["read", "read", "write"])
    assert cmds == sorted(set(cmds))


def test_known_capabilities_matches_map():
    assert known_capabilities() == sorted(CAPABILITY_COMMANDS)


# --- chain -> allowlist ---


def test_valid_chain_yields_granted_commands():
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    chain = _chain(human, org, agent, agent_perms=["read", "test"])
    allowlist, errors = enforced_allowlist_for_chain(
        chain, expected_subject_did=agent.did
    )
    assert errors == []
    assert "pytest" in allowlist and "cat" in allowlist
    assert "git" not in allowlist


def test_invalid_chain_grants_no_commands():
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    # Widen at the agent link beyond what the org was granted -> verify fails.
    chain = _chain(human, org, agent, org_perms=["read"], agent_perms=["read", "vcs"])
    allowlist, errors = enforced_allowlist_for_chain(chain)
    assert allowlist == []
    assert errors


def test_expired_chain_grants_no_commands():
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    chain = _chain(human, org, agent, agent_perms=["read"], not_after=past)
    allowlist, errors = enforced_allowlist_for_chain(chain)
    assert allowlist == []
    assert any("expired" in e for e in errors)


def test_wrong_subject_grants_no_commands():
    human, org, agent, imposter = (AgentKeypair.generate() for _ in range(4))
    chain = _chain(human, org, agent, agent_perms=["read"])
    allowlist, errors = enforced_allowlist_for_chain(
        chain, expected_subject_did=imposter.did
    )
    assert allowlist == []
    assert errors


# --- end-to-end enforcement through WorktreePolicy ---


def _policy() -> WorktreePolicy:
    return WorktreePolicy(WorktreeConfig())


def test_apply_delegation_enforces_granted_commands():
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    chain = _chain(human, org, agent, agent_perms=["read", "test"])
    policy = _policy()

    ok, errors = policy.apply_delegation("codex", chain, expected_subject_did=agent.did)
    assert ok, errors

    # Granted: a read command and a test command are allowed.
    assert policy.evaluate_command("codex", ["pytest", "tests/"]).allowed
    assert policy.evaluate_command("codex", ["cat", "README.md"]).allowed
    # Not granted: vcs/package commands are denied even though they are in the
    # global default allowlist.
    decision = policy.evaluate_command("codex", ["git", "status"])
    assert not decision.allowed
    assert "not in allowlist" in decision.reason
    assert not policy.evaluate_command("codex", ["pip", "install", "x"]).allowed


def test_apply_delegation_invalid_chain_denies_all():
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    chain = _chain(human, org, agent, org_perms=["read"], agent_perms=["read", "vcs"])
    policy = _policy()

    ok, errors = policy.apply_delegation("codex", chain, expected_subject_did=agent.did)
    assert not ok
    assert errors
    # Deny-by-default: even normally-safe commands are blocked.
    assert not policy.evaluate_command("codex", ["cat", "x"]).allowed
    assert not policy.evaluate_command("codex", ["pytest"]).allowed


def test_apply_delegation_rejects_chain_for_other_subject():
    # A chain issued to `agent` must not install capabilities under a sandbox
    # bound to a different DID (P1: subject binding is required).
    human, org, agent, other = (AgentKeypair.generate() for _ in range(4))
    chain = _chain(human, org, agent, agent_perms=["read", "test"])
    policy = _policy()

    ok, errors = policy.apply_delegation("codex", chain, expected_subject_did=other.did)
    assert not ok
    assert errors
    assert not policy.evaluate_command("codex", ["pytest"]).allowed


def test_malformed_not_after_denies_without_raising():
    # P2: a malformed timestamp must surface as an error and deny-by-default,
    # never raise out of verify()/apply_delegation.
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    link_org = sign_link(human, subject_did=org.did, permissions=["read"])
    link_agent = sign_link(
        org, subject_did=agent.did, permissions=["read"], not_after="not-a-date"
    )
    chain = DelegationChain(links=[link_org, link_agent])

    allowlist, errors = enforced_allowlist_for_chain(chain)
    assert allowlist == []
    assert any("malformed not_after" in e for e in errors)

    policy = _policy()
    ok, errors = policy.apply_delegation("codex", chain, expected_subject_did=agent.did)
    assert not ok
    assert not policy.evaluate_command("codex", ["cat", "x"]).allowed


def test_delegation_does_not_override_hard_blocks():
    # Even if "vcs" is granted, network-reaching git stays blocked.
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    chain = _chain(human, org, agent, agent_perms=["vcs"])
    policy = _policy()
    policy.apply_delegation("codex", chain, expected_subject_did=agent.did)

    assert policy.evaluate_command("codex", ["git", "status"]).allowed
    push = policy.evaluate_command("codex", ["git", "push"])
    assert not push.allowed
    assert "unconditionally blocked" in push.reason
    assert not policy.evaluate_command("codex", ["ssh", "host"]).allowed


def test_other_agents_unaffected_by_delegation():
    human, org, agent = (AgentKeypair.generate() for _ in range(3))
    chain = _chain(human, org, agent, agent_perms=["read"])
    policy = _policy()
    policy.apply_delegation("codex", chain, expected_subject_did=agent.did)

    # An agent with no delegation still falls back to the default allowlist.
    assert policy.evaluate_command("other", ["git", "status"]).allowed
