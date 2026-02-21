# /retro

Analyze the current session for repeated workflows, pain points, or manual multi-step patterns that should become new slash commands, hooks, or specialist agents. Can be invoked mid-session or at the end.

## Usage

`/retro`

## Phase 1: Session Health Summary

Compute aggregate statistics from the conversation before mining patterns:

- **Tool calls**: total count, by type (git, file edit, search, test, etc.)
- **Error rate**: failed commands / total commands
- **Corrections**: times the user said "no", "undo", "actually", or re-requested
- **Retries**: same tool called 2+ times with minor variations
- **Files touched**: count of unique files read, edited, created
- **Session complexity**: distinct tasks attempted (estimated from user messages)

Present as a compact block:

```
Session Health
  Tool calls: 47 (git: 18, edit: 12, search: 9, test: 5, other: 3)
  Error rate: 4/47 (8.5%)
  User corrections: 2
  Files touched: 15 (8 edited, 4 created, 3 read-only)
  Tasks: 4
```

## Phase 2: Trajectory Decomposition

Before scanning for patterns, decompose the session into structured steps:

```
Step N: [thought] → [action(s)] → [observation] → [outcome: success/fail/retry]
```

Group consecutive tool calls into logical "steps" by splitting on user messages. This structured view makes pattern detection more precise than scanning raw conversation.

## Phase 3: Pattern Mining

Scan the structured trajectory for:

- **Repeated multi-step sequences** (e.g. branch → commit → push → PR → merge → cleanup appeared 2+ times)
- **Manual workarounds** (e.g. `SKIP_SWARM_HOOKS=1` to avoid chicken-and-egg, `git branch -D` because squash-merge confuses git)
- **Things the user had to ask twice** or correct mid-stream
- **Slow/fragile steps** that were removed or bypassed (e.g. scenario load in pre-commit)
- **Missing automation** — anything done by hand that could be a one-liner
- **Repeated edit patterns** — manual file modifications that could be a hook or template (inspired by live-swe-agent's finding that replacing bash sed with structured tools is the #1 improvement)
- **Repeated search patterns** — same grep/glob exclusions applied manually each time

## Phase 4: Scoring and Classification

For each candidate pattern, compute a priority score:

```
priority = frequency * complexity * error_rate_boost
```

Where:
- `frequency` = number of occurrences in the session
- `complexity` = number of manual steps per occurrence
- `error_rate_boost` = 1.5x if the pattern failed or needed correction at least once, 1.0x otherwise

Also apply a **cost/benefit filter**: if creating the automation has more overhead than doing it manually 2-3 more times, mark it as "not worth automating yet" but still report it.

Classify each candidate into one of four tiers (prefer higher tiers):

| Tier | Where it lives | When to use |
|---|---|---|
| **Extend existing** | `.claude/commands/*.md` (modified) | Pattern fits as a new mode/arg on an existing command |
| **New slash command** | `.claude/commands/*.md` (new file) | Genuinely unrelated to any existing command |
| **Specialist agent** | `.claude/agents/*.md` | Domain-specific multi-step reasoning |
| **Session-scoped** | (noted, not persisted) | Too narrow for permanence; worth remembering for similar future tasks |

## Phase 5: Present Findings

Present as a ranked table (highest priority first):

| # | Pattern | Tier | Priority | Frequency | Complexity | Why |
|---|---|---|---|---|---|---|
| 1 | branch + commit + push + PR | command | 9.0 | 3x | 5 steps | Same sequence every time, errored once |
| 2 | post-merge branch cleanup | command | 4.5 | 2x | 3 steps | squash-merge needs -D, easy to forget |
| 3 | YAML schema validation | session | 2.0 | 2x | 1 step | Only relevant to this task |

## Phase 6: Session Grading

Rate the session on two axes (inspired by SWE-bench's F2P/P2P grading):

- **Goal completion**: Did the intended tasks get done? (FULL / PARTIAL / BLOCKED)
- **Maintenance**: Were existing tests/code kept passing? (CLEAN / REGRESSED)
- **Overall**: CLEAN (all goals met, nothing broken), MESSY (goals met but with corrections/retries), or BLOCKED (goals not fully achieved)

## Phase 7: Generate

**Extend, don't proliferate.** This is a hard rule (see CLAUDE.md "Extend, don't proliferate"). Before proposing ANY new command, agent, or hook:

1. Read `.claude/commands/` and `.claude/agents/` to find the closest existing match.
2. If one exists, propose a new `--flag` or mode section on it — not a new file.
3. Only propose a new file when the pattern is genuinely unrelated to every existing command.
4. Default tier should be "Extend existing", not "New slash command".

Ask the user which candidates to implement. For each selected candidate:
- If extending an existing command: show the proposed additions (new mode/args) and update the `.md` file
- If creating a new file (rare — requires justification): generate the `.claude/commands/*.md` or `.claude/agents/*.md` file
- Note edge cases observed in the session (e.g. stash needed, force-delete for squash)
- Check `.claude/commands/` first to avoid duplicating or fragmenting existing commands

## What to look for specifically

- Git workflows (branch, commit, push, PR, merge, cleanup)
- Test/lint/check sequences run repeatedly
- File exclusion patterns applied manually (`.DS_Store`, `*.db`, secrets)
- Hook modifications that needed reinstall
- Any time the user said "do X" and it took more than 3 tool calls
- Repeated file edits that could be a pre/post hook
- Domain-specific analysis done manually multiple times (candidate for `.claude/agents/`)

## Cross-Session Comparison (optional)

If prior retro outputs exist in `runs/*_retro/` or similar:
- Compare this session's patterns against past sessions
- Flag **chronic patterns** (appeared in 2+ sessions and still not automated)
- Flag **regressions** (a pattern that was automated but broke or was bypassed)

## Constraints

- Do not create or modify commands automatically; always present candidates and let the user choose.
- **Extend over create**: always check `.claude/commands/` and `.claude/agents/` first. If a pattern can be a new mode or arg on an existing command, propose that instead of a new file. Fewer commands with clear modes > many single-purpose commands.
- Apply the cost/benefit filter — not every repeated pattern is worth automating.
- Can be invoked mid-session (not just at end); mid-session invocations should focus on patterns observed so far and suggest tool creation that would help the remainder of the session.
