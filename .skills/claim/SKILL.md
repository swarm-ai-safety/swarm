---
name: claim
description: Create or update a claim card in the vault. In create mode, guides through evidence collection, confidence assessment, and boundary conditions. In update mode, adds new evidence or changes status. Triggers on "/claim", "/claim create [proposition]", "/claim update [claim-id]", "/claim status", "add a claim", "update claim evidence".
version: "1.0"
generated_from: "swarm-research-os-v0.1"
user-invocable: true
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
context: fork
model: opus
argument-hint: "create [proposition] | update [claim-id] | status | list"
---

## EXECUTE NOW

**Target: $ARGUMENTS**

Parse immediately:
- `create [proposition]` → create a new claim card (Step 1–6)
- `update [claim-id]` → update an existing claim (Step 7–9)
- `status` → show claim health dashboard (Step 10)
- `list` → list all claims with status and confidence (Step 11)
- Empty → show help with subcommand options

---

## CREATE MODE: Steps 1–6

### Step 1: Parse Proposition

The proposition is the claim stated as a testable sentence. It becomes both the filename and the H1 title.

**Rules:**
- Must be a complete proposition (not a label or question)
- Composability test: "This note argues that [proposition]"
- Lowercase, no trailing period
- Filename: kebab-case with `claim-` prefix. E.g., "circuit breakers alone outperform full governance stacks" → `claim-circuit-breakers-alone-outperform-full-governance-stacks.md`
- Truncate filename to max 80 chars if needed

If no proposition provided, ask: "State the claim as a testable proposition (e.g., 'transaction tax above 5% reduces welfare')."

### Step 2: Gather Evidence

Search for supporting runs:

1. Ask the user: "Which run(s) support this claim? (run_id or 'search')"
2. If user says "search":
   - Ask for keywords related to the claim
   - Search `run-index.yaml` by tags
   - Search `runs/*/summary.json` for relevant metrics
   - Present top candidates with one-line summaries
3. For each supporting run, extract:
   - `run_id`
   - Primary metric and value
   - Effect size (Cohen's d), p-value, correction method, sample size
   - Construct the `detail` string: `"d={d}, p={p}, N={N}, {correction}-corrected"`

### Step 3: Assess Confidence

Based on the evidence gathered, recommend a confidence level:

| Confidence | Criteria |
|-----------|----------|
| **high** | Bonferroni-significant AND replicated across ≥2 independent runs/seeds |
| **medium** | Nominally significant OR single-sweep support with BH correction |
| **low** | Suggestive but underpowered (<20 seeds) or unreplicated |
| **contested** | Conflicting evidence from different runs |

Present the recommendation and let the user confirm or override.

### Step 4: Document Boundary Conditions

Ask: "What are the known limits of this claim's generalizability?"

Prompt with common dimensions:
- Topology tested (e.g., "small-world only")
- Agent count (e.g., "8 agents")
- Adversarial fraction (e.g., "25%")
- Governance config
- Agent type (e.g., "algorithmic, not LLM-powered")

### Step 5: Determine Domain and Related Claims

1. **Domain**: infer from the proposition and evidence. Must be one of: `governance`, `topology`, `agent-behavior`, `decision-theory`, `collusion`, `memory`, `market`, `calibration`, `methodology`
2. **Related claims**: scan `vault/claims/` for claims with overlapping topics or shared run evidence. Present candidates and let user confirm.

### Step 6: Write the Claim Card

Generate `vault/claims/{claim-id}.md` using the template at `vault/templates/claim-card.md`.

**Fill every field:**
- `description`: ≤200 chars summarizing the claim
- `type`: `claim`
- `status`: `active` (default for new claims)
- `confidence`: from Step 3
- `domain`: from Step 5
- `evidence.supporting`: from Step 2
- `evidence.weakening`: `[]` (new claims start with no weakening evidence)
- `evidence.boundary_conditions`: from Step 4
- `sensitivity`: infer from boundary conditions
- `related_claims`: from Step 5
- `created`: today's date
- Body sections: Evidence summary, Boundary conditions, Mechanism, Open questions

**After writing:**
1. Update `vault/_index.md` to include the new claim
2. Report the created file path and a summary

---

## UPDATE MODE: Steps 7–9

### Step 7: Read Existing Claim

Read `vault/claims/{claim-id}.md`. Parse frontmatter. Display current state:
- Status, confidence, domain
- Evidence count (supporting / weakening)
- Boundary conditions
- Last updated date

### Step 8: Determine Update Type

Ask: "What kind of update?"
- **Add supporting evidence** — new run that confirms the claim
- **Add weakening evidence** — new run that contradicts the claim
- **Add boundary condition** — new limitation discovered
- **Change status** — active → weakened → superseded → retracted
- **Change confidence** — upgrade or downgrade based on new evidence

For each type:

**Add supporting/weakening evidence:**
1. Ask for run_id
2. Read the run's summary to extract metrics
3. Add the evidence entry to the appropriate list
4. Reassess confidence (Step 3 logic) and recommend if it should change

**Change status:**
- `active → weakened`: requires at least one weakening evidence entry
- `active → superseded`: requires a `superseded_by` claim_id
- `any → retracted`: requires explanation in the body

### Step 9: Apply Update

Edit the claim file using the Edit tool:
1. Update frontmatter fields
2. Update `updated` date to today
3. If status changed to `weakened`, emit a warning:

```
⚠ CLAIM WEAKENED: {claim-id}
Previous confidence: {old}
New confidence: {new}
Contradiction: {detail}

Consider creating a GitHub issue for investigation.
```

Update `vault/_index.md` if the description changed.

---

## STATUS MODE: Step 10

Generate a claim health dashboard:

1. Glob `vault/claims/*.md`
2. For each claim, read frontmatter
3. Display:

```
## Claim Health Dashboard

| Claim | Status | Confidence | Evidence | Domain | Updated |
|-------|--------|------------|----------|--------|---------|
| circuit-breakers-dominate | active | high | 2S/0W | governance | 2026-02-18 |
| tax-welfare-tradeoff | active | high | 2S/0W | governance | 2026-02-18 |
| smarter-agents-earn-less | active | high | 3S/0W | agent-behavior | 2026-02-18 |
| staking-backfires | active | medium | 1S/0W | governance | 2026-02-18 |

**Summary:**
- {N} active claims ({M} high confidence, {K} medium, {J} low)
- {W} weakened claims
- {S} superseded claims
- {R} retracted claims
- {U} claims not updated in >30 days
```

---

## LIST MODE: Step 11

Quick list of all claims, one per line:

```
vault/claims/claim-circuit-breakers-dominate.md — active/high — governance
vault/claims/claim-tax-welfare-tradeoff.md — active/high — governance
vault/claims/claim-smarter-agents-earn-less.md — active/high — agent-behavior
vault/claims/claim-staking-backfires.md — active/medium — governance
```

---

## Critical Constraints

1. **Prose-as-title.** The claim filename and H1 must be a testable proposition.

2. **Evidence provenance is mandatory.** Every evidence entry must reference a real `run_id` that exists in `runs/`. Verify before writing.

3. **Confidence must be justified.** The confidence level must match the evidence quality criteria in Step 3. Do not assign `high` without Bonferroni-significant replicated evidence.

4. **Boundary conditions are not optional.** Every claim must document what has NOT been tested.

5. **Never silently change status.** Status changes (especially weakening) must be visible and explained.

6. **Description ≤200 chars, no trailing period.** This is a kernel invariant.

7. **Topics footer is mandatory.** Every claim must link to `[[_index]]` at minimum.
