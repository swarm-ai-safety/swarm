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
