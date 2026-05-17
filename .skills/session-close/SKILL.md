---
name: session-close
description: End-of-session ritual — summarize what changed, what was learned, update memory, commit, and push. Ensures no work is lost and the next session has full context.
version: "1.0"
metadata:
  author: swarm-research-os
  generated_from: research-os-v0.1
allowed-tools: Read Write Edit Glob Grep Bash
---

## EXECUTE NOW

Run all steps sequentially. Do not skip any step.

---

## Step 1: Inventory changes

```bash
git status
git diff --stat
```

List all modified, added, and deleted files.

## Step 2: Summarize the session

Answer these questions by reviewing the conversation history and current state:

1. **What changed?** — files modified, runs completed, claims updated
2. **What did we learn?** — key findings, surprises, pattern changes
3. **What's next?** — the most valuable next experiment or task
4. **What should I remember?** — decisions made, preferences expressed, context that would be lost

## Step 3: Update memory

### Update `.letta/memory/threads/research-log.md`

Append a session entry:
```
## {date} — {session focus}

**Ran:** {experiments or tasks completed}
**Found:** {key results}
**Learned:** {insights}
**Next:** {next steps}
**Run pointers:** {run_ids if any}
```

### Update `.letta/memory/threads/current.md`

Replace the content with:
- Current hypothesis (carried forward or updated)
- What we're testing next
- This session's summary (moved to "Last session summary")
- Next experiment
- Any blockers

### Update `.letta/memory/runs/latest.md`

Add any new run pointers to the table.

## Step 4: Commit and push

```bash
git add <changed files>         # stage code changes
bd sync                         # sync beads
git commit -m "<summary>"       # commit with descriptive message
bd sync                         # sync any new beads changes
git push                        # push to remote
```

## Step 5: Confirm

Print:
```
Session closed.

Summary: {one-line summary}
Memory updated: threads/current.md, threads/research-log.md
Committed: {commit hash}
Pushed: {branch}

Next session: {what to do first}
```
