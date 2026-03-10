---
description: "Five architectural patterns observed in production agent systems (via Everything Claude Code) and their translations into SWARM's governance and safety framework."
author: "SWARM Team"
date: "2026-03-08"
source: "https://github.com/affaan-m/everything-claude-code"
keywords:
  - agent architecture
  - operational patterns
  - governance design
  - context economics
  - verification hooks
---

# Agent System Patterns

Five structural patterns observed in production agent infrastructure — drawn from
[Everything Claude Code](https://github.com/affaan-m/everything-claude-code) — and
their mappings into SWARM's distributional-safety framework.

These are not "prompting tricks." They are architectural constraints that determine
whether an agent system degrades or stays reliable under sustained operation.

---

## 1. Context as Scarce Resource

**Pattern:** Treat the context window like a budget, not free RAM. Remove ambient
complexity — prefer CLI-wrapped commands over always-on tool integrations. Every
persistent tool imposes cognitive load on the system even when idle.

**Evidence:** ECC recommends replacing MCP servers with CLI-wrapped skills where
possible. Even with lazy loading, ambient integrations consume tokens and increase
the probability of context rot (irrelevant state crowding out relevant state).

**SWARM translation:** This is governance by context budgeting. The question is not
"can the agent access this tool?" but "what persistent cognitive load does this tool
impose on the system?" In a multi-agent economy, each agent's effective intelligence
is bounded by how much of its context window is consumed by coordination overhead vs.
task-relevant reasoning.

**Formal connection:** If an agent's effective decision quality degrades as
irrelevant context grows, then tool-surface minimization is a form of maintaining
proxy accuracy — keeping $\hat{v}$ close to true $v$ by reducing noise in the
observable channel.

---

## 2. Workflow Artifacts over Long Sessions

**Pattern:** Build memory into the workflow (durable artifacts, session summaries,
repeatable commands), not into a single long session. Intelligence lives in the
artifact chain, not in the chat transcript.

**Evidence:** ECC implements session compaction via Stop-phase summaries, persistent
skills, and reusable command surfaces. The system is designed to work across sessions,
not to survive within one enormous session.

**SWARM translation:** This maps to SWARM's 4-tier memory architecture
(`.letta/memory/`): system identity is stable, project context is durable, threads
compress, and run artifacts are checkpointed. Reliability comes from the chain of
committed artifacts — not from trusting the model to remember everything.

**Design principle:** The real unit of intelligence in an agent system is not the
model session. It is the workflow artifact chain: scenario YAML + seed + history
export + metrics output. Reproducibility (a SWARM invariant) depends on this, not on
session continuity.

---

## 3. Mechanical Verification, Not Conversational

**Pattern:** Verification should be automatic and post-action. Use the environment
(hooks, CI, formatters) to catch problems instead of relying on the model to remember
guardrails. Quality becomes a mechanical property of the workflow, not a soft
instruction.

**Evidence:** ECC treats hooks as a first-class reliability layer — formatting, tests,
and review run on the way out. The system enforces `action → validation → trust`
rather than `action → hope`.

**SWARM translation:** This maps directly onto governance hooks and institutional
checks. Agents should not self-certify. In SWARM:

- Pre-commit hooks enforce safety invariants ($p \in [0,1]$, append-only logs)
- Post-write hooks warn on foundational-section edits
- The Reproducibility Sheriff agent role exists precisely because self-certification
  is unreliable

**Governance analog:** In institutional design, this is the separation of execution
from audit. The agent that takes an action is not the agent that validates it.
SWARM's Auditor role and the Reproducibility Sheriff encode this structurally.

---

## 4. Narrow Roles with Explicit Handoff

**Pattern:** Parallelization works when roles are narrow, composable, and connected by
explicit handoff — not when multiple agents share overlapping authority. The
architecture is: narrow role + explicit handoff + reusable command surface + shared
verification layer.

**Evidence:** ECC ships 14 agent roles, 56 skills, and 33 commands — but each role has
a constrained scope. A chief-of-staff agent handles coordination rather than giving
every agent coordination powers. This is role decomposition with orchestration glue,
not role duplication.

**SWARM translation:** SWARM's 6 agent roles (Scenario Architect, Mechanism Designer,
Auditor, Adversary Designer, Reproducibility Sheriff, Research Scout) follow this
pattern by design:

| Role | Scope boundary |
|---|---|
| Scenario Architect | Designs experiments, does not run them |
| Mechanism Designer | Proposes interventions, does not validate them |
| Auditor | Validates claims, does not generate them |
| Adversary Designer | Attacks governance, does not set governance |
| Reproducibility Sheriff | Checks hygiene, does not design experiments |
| Research Scout | Gathers external patterns, does not theorize |

The key is that no role self-validates. Each role's output is another role's input.

**Failure mode:** Multi-agent systems fail when agents have overlapping authority and
fuzzy responsibilities. The fix is not "better coordination prompts" — it is narrower
role definitions and mechanical handoff.

---

## 5. Security as Architecture, Not Afterthought

**Pattern:** Every integration is an attack surface. Security is a design constraint
from day one: minimize channels, constrain tools, isolate execution, scan configs,
watch behavior live.

**Evidence:** ECC's security model treats the following as first-class threats:

- Malicious `CLAUDE.md` / repository config injection
- Transitive prompt injection via fetched documents
- Poisoned MCP tool responses
- Unsandboxed execution environments

Mitigations include allowlists, deny lists, Docker isolation, observability, and
AgentShield (102 rules, 1280 tests).

**SWARM translation:** In SWARM's multi-agent economy, each agent is both a potential
attacker and a potential victim. The Adversary Designer role exists to probe these
boundaries. Key mappings:

| ECC security pattern | SWARM analog |
|---|---|
| Minimize integration channels | Externality internalization ($\rho$) — agents bear cost of the channels they open |
| Constrain tool access | Governance cost ($c_a$) — more powerful tools have higher governance overhead |
| Isolate execution | Worktree isolation for parallel sessions |
| Scan configurations | Pre-commit hooks, scenario validation |
| Watch behavior live | Event logging (append-only JSONL), `SoftMetrics` monitoring |

**Deep insight:** In agent systems, prompt engineering and security engineering are
converging. The architecture *is* the defense model. You cannot bolt security onto an
agent system after the fact any more than you can bolt type safety onto a dynamically
typed language after deployment.

---

## Synthesis

These five patterns share a common structure: **they move reliability from the model's
cognition into the environment's mechanics.**

| Pattern | What moves out of the model | What moves into the environment |
|---|---|---|
| Context budgeting | Tool awareness | CLI-wrapped skills |
| Workflow memory | Session recall | Durable artifacts + summaries |
| Mechanical verification | Guardrail compliance | Hooks + CI + role separation |
| Narrow roles | Coordination reasoning | Explicit handoff protocols |
| Security as architecture | Threat awareness | Sandboxing + allowlists + monitoring |

This is the core design principle for reliable agent systems: **don't ask the model to
be disciplined — build discipline into the system around it.**

In SWARM's formal language: governance mechanisms ($\rho$, $c_a$, $\tau$) exist
precisely because agent self-governance (relying on each agent's $p$ to stay high
without structural incentives) is insufficient. The environment must make safe behavior
the path of least resistance.
