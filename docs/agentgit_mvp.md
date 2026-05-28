# AgentGit MVP

AgentGit is a thin provenance layer over normal git. The first MVP lets an
agent evaluate its current worktree diff against a task-scoped policy and emit
a signed JSON bundle for CI, reviewers, or another agent to inspect.

```bash
python -m swarm.agentgit attest \
  --task issue-123 \
  --agent codex \
  --policy examples/agentgit_policy.yaml \
  --check pytest=pass \
  --output .agentgit/provenance.json
```

The bundle records:

- agent and task identity
- base commit and changed files
- additions/deletions by file
- path and size policy decisions
- required check results
- sealed admissibility receipt with the payload hash

Policy failures return a non-zero exit code by default. Use `--warn-only` when
you want to capture the bundle without blocking the current command.

## Worktree Loop

AgentGit also plugs into the worktree sandbox bridge:

```bash
python -m swarm.bridges.worktree create codex
python -m swarm.bridges.worktree exec codex -- python -m pytest tests/test_agentgit.py
python -m swarm.bridges.worktree attest codex \
  --task issue-123 \
  --policy examples/agentgit_policy.yaml \
  --check pytest=pass \
  --output .agentgit/provenance.json
python -m swarm.agentgit verify .agentgit/provenance.json
```

That gives the first complete local loop:

```text
delegate task -> isolate worktree -> execute checks -> attest diff -> verify bundle
```

## Cryptographic Identity & Delegation

The MVP signs bundles with a shared HMAC key, which proves a bundle was sealed
by *someone holding the key* but not *which agent* produced the change. The
`swarm.agentgit.identity` module adds verifiable identity with Ed25519
(asymmetric) signatures.

- **`AgentKeypair`** — an Ed25519 keypair. Its `did` is `did:key:ed25519:<hex>`,
  so the public key is embedded in the identifier and verifiers need no key
  registry.
- **`AgentIdentity`** — the agent's DID plus owner/org and model/runtime/version
  provenance and its `allowed_tools`.
- **`DelegationChain`** — an ordered, individually-signed `human -> org -> agent`
  chain. `verify()` checks every link's signature, that the chain is connected
  (each link's subject issues the next), that permissions only *narrow* down the
  chain, and that no link has expired.

When `identity` + `agent_keypair` (and optionally `delegation`) are passed to
`build_bundle`, the agent's key signs the receipt `payload_hash`, binding a
verifiable identity to that exact diff:

```python
from swarm.agentgit import AgentIdentity, AgentKeypair, build_bundle, sign_link, DelegationChain

org = AgentKeypair.generate()
agent = AgentKeypair.generate()
identity = AgentIdentity.for_keypair(agent, owner="alice", org="acme", allowed_tools=["read", "test"])
chain = DelegationChain(links=[sign_link(org, subject_did=agent.did, permissions=["read", "test", "open_pr"])])

bundle = build_bundle(..., identity=identity, agent_keypair=agent, delegation=chain)
```

`verify_bundle` then additionally checks the identity signature, the delegation
chain, that the chain's final subject is the signing identity, and that the
identity's `allowed_tools` stay within the delegated grant. Bundles built
without identity blocks still verify (backward compatible).

> The CLI (`attest`/`verify`) does not yet manage keypairs — that key-storage
> surface is tracked as a follow-up. Today identity is wired through the library
> API.
