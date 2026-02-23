# Skill Activation Is the Bottleneck

*Your agent skills work 96% of the time — when they fire. The problem is getting them to fire.*

---

A [controlled experiment by Tessl](https://tessl.io/blog/do-agent-skills-actually-help-a-controlled-experiment/) ran 30 trials per variant measuring whether Claude Code skills improve task performance. The headline numbers: baseline 53%, generic skill 73%, custom skill 80%. But the mechanism underneath is what matters.

When the custom skill activated, it succeeded **96% of the time**. When it didn't activate, the agent succeeded **0% of the time**. The bottleneck isn't skill quality. It's skill activation.

This matched our intuition but we'd never measured it. We run 50+ custom slash commands across the SWARM research framework, and we'd noticed that some commands get picked up reliably while others sit unused even when they're the right tool. So we ran an audit.

## What We Found

We rated all 54 skill descriptions on activation quality — would Claude Code recognize the right moment to suggest or invoke each skill based on the first 2 lines of its description?

| Rating | Count | % |
|--------|-------|---|
| Strong | 29 | 54% |
| Adequate | 18 | 33% |
| Weak | 7 | 13% |

The 29 strong skills shared three properties:

1. **Specific action verb** in the first sentence ("Verify branch identity and detect file conflicts" not "Pre-work safety checks")
2. **Named trigger scenario** ("use after editing `viz/src/` files" not "use when needed")
3. **Explicit differentiation** from competing skills ("Distinct from `/preflight` (pre-commit lint checks) and `/healthcheck` (code health after external changes)")

The 7 weak skills failed on all three.

## The Competing Cluster Problem

The biggest activation risk wasn't individual skill quality — it was **inter-skill confusion**. Several skill clusters competed for the same user phrases:

**"Add something"** triggered four possible skills:

- `/add_metric` — new measurement on existing data
- `/add_scenario` — new parameter config using existing domain infrastructure
- `/add_domain` — new task category requiring new data models and handlers
- `/add_post` — new blog post

None of them said "use this one, not that one." An agent seeing "add a new X to the project" had to guess which of four commands matched.

**"Run experiment"** triggered five:

- `/run_scenario` — single scenario, single seed
- `/sweep` — parameter grid without analysis
- `/analyze_experiment` — statistical analysis on existing data
- `/benchmark` — standardized multi-condition evaluation suite
- `/full_study` — end-to-end pipeline chaining all of the above

**"Create PR"** triggered three:

- `/pr` — create PR from local changes
- `/fix_pr` — resolve conflicts on existing PR
- `/ship` — commit + push to current branch, no PR

Without explicit differentiation clauses, the agent either picks the wrong one or — worse — picks none and does the task manually.

## The Fix: Three Rules

We rewrote all 7 weak descriptions and added differentiation clauses to 7 competing-cluster skills. Every rewrite followed the same pattern:

### Rule 1: Lead with a specific action verb and the artifact it produces

**Before:** "Add a new simulation domain to SWARM following the established handler pattern."

**After:** "Scaffold a new SWARM simulation domain — data models, action types, task handler, agents, metrics, tests, and registry wiring — when adding a fundamentally new task category (e.g. medical triage, code review) whose observables or agent actions don't yet exist."

The before tells the agent *what it is*. The after tells the agent *what it does* and *when to do it*.

### Rule 2: Name the trigger event, not the category

**Before:** "Pre-work safety checks: verify you're on the right branch and/or check that files haven't been modified by concurrent sessions."

**After:** "Verify branch identity and detect file conflicts from concurrent Claude Code worktree sessions before starting destructive work — use when opening a worktree pane, before modifying shared files, or when you suspect another session may have touched the same paths."

"Pre-work safety checks" is a category. "When opening a worktree pane" is a trigger. The agent can match triggers to context; it can't match categories.

### Rule 3: Say "not this — use that instead"

**Before:** "Add a blog post to the mkdocs website (swarm-ai.org/blog/)."

**After:** "Scaffold and publish an original blog post to swarm-ai.org/blog/ — handles MkDocs slug generation, metadata headers, nav wiring, and financial-disclaimer enforcement. Use for manually-written essays and research notes; use `/eval_writeup` instead to auto-generate a post from a Prime Intellect eval run."

Without the differentiation clause, the agent has to decide between `/add_post` and `/eval_writeup` on vibes. With it, the decision is mechanical: "Am I starting from eval output? Use `/eval_writeup`. Writing from scratch? Use `/add_post`."

## The Tessl Finding Applied at Scale

The Tessl experiment tested one skill on one task. We applied the same insight across 54 skills and found that the activation problem compounds:

- **Single skill:** activation rate is the only variable. 83% > 57%.
- **Competing clusters:** activation rate *per skill* drops because the agent's probability mass is split across alternatives. If three skills each have 30% activation probability for the same context, the right one fires less than a third of the time.
- **At scale (50+ skills):** the agent's skill selection becomes a classification problem with 50 classes. Vague descriptions are like training a classifier on noisy labels.

The fix is the same at every scale: make the decision boundary explicit in the description itself.

## Recommendations

If you maintain a set of Claude Code skills (slash commands, CLAUDE.md instructions, MCP tool descriptions):

1. **Audit first-line descriptions.** Read only the first 2 lines of each skill. Could you tell *when* to use it from those 2 lines alone?
2. **Find competing clusters.** Group skills by the user phrases that could trigger them. Any cluster with 3+ skills needs explicit differentiation clauses.
3. **Add "not this — use that" clauses.** Every skill in a cluster should name its siblings and explain the decision boundary.
4. **Name trigger events, not categories.** "Use when the Edit tool reports 'File has been modified since read'" beats "file freshness checking."
5. **Measure, don't guess.** The Tessl study used 30 trials per variant. Single runs are misleading. If you can run your eval harness multiple times, do it.

The quality of your skills doesn't matter if they don't fire. Write descriptions for the classifier, not the human.

---

*This post describes engineering practices for AI coding agents. Nothing here constitutes financial advice, investment recommendations, or endorsement of any trading strategy.*
