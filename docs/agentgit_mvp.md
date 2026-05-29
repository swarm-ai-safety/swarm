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
- a `provenance` block of *what happened* producing the diff (see below)
- sealed admissibility receipt with the payload hash

Policy failures return a non-zero exit code by default. Use `--warn-only` when
you want to capture the bundle without blocking the current command.

## Provenance Contents

Beyond the diff and policy verdict, schema `agentgit.provenance.v1` records a
`provenance` block describing how the change was produced:

- `commands` — commands executed (binary + args, `return_code`, OS `isolation`
  backend, `duration_seconds`, and a `timed_out` flag). Build these from a
  worktree `CommandResult` via `CommandRecord.from_command_result(result)`.
- `environment` — model / runtime / version of the producing agent.
- `dependency_changes` — manifest/lockfile edits (`requirements.txt`,
  `pyproject.toml`, `package-lock.json`, `Cargo.lock`, `go.sum`, …) detected
  **automatically** from the diff.
- `sources` — external sources consulted.
- `reviews` — reviewer decisions.
- `overrides` — human overrides.

```python
from swarm.agentgit import CommandRecord, build_bundle

bundle = build_bundle(
    ...,
    commands=[CommandRecord(command=["pytest", "-q"], return_code=0, isolation="bwrap")],
    environment={"model": "claude-opus-4-7", "runtime": "python3.13"},
    sources=["https://example.com/issue/123"],
    reviews=[{"reviewer": "security", "decision": "approve"}],
)
```

The `provenance` block is folded into the signed receipt payload, so it is
**tamper-evident**: editing a recorded command or hiding a dependency change
makes `verify_bundle` fail the `payload_hash` check. Older `v0` bundles (hashed
without provenance) still verify — `verify_bundle` reconstructs the payload per
schema version.

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

## Capability Enforcement

Verifying a delegation chain is still *advisory* — it proves what an agent was
allowed to do without stopping it from doing more. `swarm.agentgit.capabilities`
turns a verified chain into the command allowlist the worktree sandbox
*physically* enforces, closing the loop identity → delegation → enforcement.

`CAPABILITY_COMMANDS` maps permission tokens to the command binaries they
authorize (`read` → `ls/cat/grep/…`, `test` → `pytest/python`, `vcs` → `git`,
etc.). `enforced_allowlist_for_chain` verifies the chain and returns the granted
commands — or, on any verification failure, an empty allowlist (**deny by
default**).

```python
from swarm.bridges.worktree.config import WorktreeConfig
from swarm.bridges.worktree.policy import WorktreePolicy

policy = WorktreePolicy(WorktreeConfig())
ok, errors = policy.apply_delegation("codex", chain, expected_subject_did=agent.did)
# Now only the delegated capabilities execute:
policy.evaluate_command("codex", ["pytest", "tests/"]).allowed   # True  (test granted)
policy.evaluate_command("codex", ["git", "status"]).allowed       # False (vcs not granted)
```

An invalid, expired, or over-scoped chain installs an empty allowlist, so the
agent can run *nothing* until a valid delegation is supplied. Unconditional
hard-blocks (ssh/scp, `git push|fetch|pull|clone`) still apply regardless of
what was delegated.

This slice enforces **command** capabilities (which binary may start).

## OS-Level Isolation

Gating *which* binary starts is not enough: `subprocess.run(cmd, cwd=sandbox)`
runs an ordinary child process, so an allowlisted `python` can still write
anywhere and open sockets. `swarm.bridges.worktree.sandbox_launch` wraps the
executed command in a real OS confinement that limits **filesystem writes to
the sandbox** and **blocks network egress**:

- macOS → `sandbox-exec` with an SBPL profile (deny `file-write*` outside the
  sandbox + temp, deny `network*`).
- Linux → `bwrap` (read-only root, read-write bind on only the sandbox subtree,
  private empty network namespace via `--unshare-net`).

Opt-in via `WorktreeConfig`:

```python
WorktreeConfig(os_isolation_enabled=True)          # wrap when a backend exists
WorktreeConfig(os_isolation_enabled=True, require_os_isolation=True)  # fail-closed
```

When enabled but no backend is available (e.g. CI/Linux without `bwrap`), the
command still runs and `CommandResult.isolation` is recorded as `"none"` — the
isolation status is **never silent**. Set `require_os_isolation=True` to instead
**deny** execution when no backend exists. Reads are not restricted in this
slice (interpreters need their stdlib); a stronger read-confining jail and
short-lived scoped git push tokens remain follow-ups.
