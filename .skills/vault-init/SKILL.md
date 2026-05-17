---
name: vault-init
description: Initialize or extend a SWARM Research OS vault. Creates directory structure, installs schema-enforced templates, generates master index, seeds starter notes, and optionally backfills run.yaml metadata for existing runs. Triggers on "/vault-init", "initialize vault", "set up research os vault", "create vault structure".
version: "1.0"
generated_from: "swarm-research-os-v0.1"
user-invocable: true
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
context: fork
model: opus
argument-hint: "[path] — target directory for vault (default: current working directory)"
---

## EXECUTE NOW

**Target: $ARGUMENTS**

Parse immediately:
- If target is a path: use it as the vault root
- If target is empty: use the current working directory
- If target contains `--backfill`: also generate run.yaml for existing runs
- If target contains `--force`: overwrite existing templates and index
- If target contains `--minimal`: skip seeded content notes, only install templates

**Execute steps 0 through 8 in order. Do not skip steps. Do not ask for confirmation between steps — execute the full sequence, then report.**

---

## Step 0: Detect Vault State

Read the target directory to determine what already exists.

**Check for:**
- `vault/` directory (has vault been initialized before?)
- `vault/_index.md` (has the index been created?)
- `vault/templates/` (have templates been installed?)
- `runs/` directory (are there existing runs?)
- `schemas/` directory (have schemas been installed?)
- `run-index.yaml` (has the run index been built?)

**Decision tree:**
- If `vault/_index.md` exists AND `--force` not set → report "Vault already initialized" with a summary of what exists, then offer to run `--backfill` or `--force`
- If `vault/_index.md` does not exist → proceed with full initialization
- If `--force` is set → proceed, overwriting existing files

Store the state as context for remaining steps.

---

## Step 1: Create Directory Structure

Create the full vault directory tree. Use `mkdir -p` to be idempotent.

```
vault/
├── experiments/          # one note per run (auto-generated)
├── claims/               # claim cards with evidence pointers
├── governance/           # governance mechanism notes
├── topologies/           # topology behavior notes
├── failures/             # named failure patterns from red-team
├── methods/              # reusable methodology notes
├── sweeps/               # cross-run sweep aggregations
└── templates/            # schema-enforced note templates
```

Also ensure these exist:
```
schemas/                  # JSON Schema files for validation
scripts/                  # automation tooling
```

**Critical:** Do NOT create directories that already exist with content. Only create missing ones.

---

## Step 2: Install Templates

Write the following 7 templates to `vault/templates/`. Each template is the **single source of truth** for its note type's schema — the `_schema` block defines required fields, optional fields, enums, and constraints.

### 2a: `vault/templates/experiment-note.md`

```yaml
---
description: One sentence summarizing the experiment and its key finding (~150 chars)
type: experiment
status: completed | failed | partial
run_id: "YYYYMMDD-HHMMSS_slug"
experiment_type: single | sweep | redteam | study | calibration
created: YYYY-MM-DD

_schema:
  required: [description, type, status, run_id, experiment_type, created]
  optional: [swarm_version, git_sha, hypothesis]
  enums:
    type: [experiment]
    status: [completed, failed, partial]
    experiment_type: [single, sweep, redteam, study, calibration]
  constraints:
    description: "max 200 chars, must add info beyond title, no trailing period"
    run_id: "must match folder name in runs/"
---

# prose-as-title expressing the experiment's purpose and outcome

## Design

- **Hypothesis**: what the experiment tests
- **Type**: experiment type
- **Parameters swept**: list of swept parameters and their values (sweeps only)
- **Seeds**: seed count or list
- **Total runs**: number of individual runs
- **SWARM version**: version @ git SHA

## Key results

Summarize findings with effect sizes, p-values, and correction methods.
Use connective reasoning: because, therefore, this suggests, however.

## Claims affected

- Supports: [[claim-id]] — how this run supports the claim
- Weakens: [[claim-id]] — how this run contradicts the claim

## Artifacts

- Summary: `runs/<run_id>/summary.json`
- CSV: `runs/<run_id>/sweep_results.csv`
- Scripts: list of analysis scripts

## Reproduction

\`\`\`bash
python -m swarm <command> <scenario> --seed <N>
\`\`\`

---

Topics:
- [[_index]]
```

### 2b: `vault/templates/claim-card.md`

```yaml
---
description: One sentence stating the claim as a testable proposition (~150 chars)
type: claim
status: active | weakened | superseded | retracted
confidence: high | medium | low | contested
domain: governance | topology | agent-behavior | decision-theory | collusion | memory | market | calibration | methodology

evidence:
  supporting:
    - run: "run_id"
      metric: "metric name"
      detail: "effect size, p-value, N, correction method"
  weakening: []
  boundary_conditions:
    - "known limits of generalizability"

sensitivity:
  topology: "how sensitive to topology changes"
  agent_count: "how sensitive to population size"
  governance_interaction: "how governance mechanisms interact"

supersedes: []
superseded_by: []
related_claims:
  - "claim-id"

created: YYYY-MM-DD
updated: YYYY-MM-DD

_schema:
  required: [description, type, status, confidence, domain, evidence, created]
  optional: [sensitivity, supersedes, superseded_by, related_claims, updated]
  enums:
    type: [claim]
    status: [active, weakened, superseded, retracted]
    confidence: [high, medium, low, contested]
    domain: [governance, topology, agent-behavior, decision-theory, collusion, memory, market, calibration, methodology]
  constraints:
    description: "max 200 chars, states the claim as a proposition, no trailing period"
    confidence: "high = Bonferroni-significant replicated; medium = nominally significant; low = suggestive; contested = conflicting"
    evidence.supporting: "must have at least one entry with a valid run_id"
---

# prose-as-title stating the claim as a complete proposition

## Evidence summary

Present evidence with effect sizes, sample sizes, correction methods.
Link to experiment notes: [[experiment-run-id]]

## Boundary conditions

Known limits of generalizability.

## Mechanism

Why does this happen? Propose or cite the causal mechanism.

## Open questions

What would strengthen, weaken, or supersede this claim?

---

Topics:
- [[_index]]
```

### 2c: `vault/templates/governance-note.md`

```yaml
---
description: One sentence on what this governance mechanism does and when it matters (~150 chars)
type: governance
status: active | experimental | deprecated
mechanism: circuit-breaker | transaction-tax | staking | reputation-decay | audit | collusion-detection | rate-limit | vote-normalization

parameters:
  - name: "parameter_name"
    type: float | int | bool
    default: null
    range: "description of valid range"

created: YYYY-MM-DD
updated: YYYY-MM-DD

_schema:
  required: [description, type, status, mechanism, created]
  optional: [parameters, updated]
  enums:
    type: [governance]
    status: [active, experimental, deprecated]
    mechanism: [circuit-breaker, transaction-tax, staking, reputation-decay, audit, collusion-detection, rate-limit, vote-normalization]
  constraints:
    description: "max 200 chars, no trailing period"
---

# prose-as-title describing the mechanism's role

## How it works
## Parameters
## Evidence
## Interactions
## Known failure modes
## Open questions

---

Topics:
- [[_index]]
```

### 2d: `vault/templates/topology-note.md`

```yaml
---
description: One sentence on how this topology shapes agent behavior and outcomes (~150 chars)
type: topology
status: active | experimental
topology: ring | random | small-world | complete | scale-free | star | grid

properties:
  avg_degree: null
  clustering: null
  diameter: null
  dynamic: false

created: YYYY-MM-DD
updated: YYYY-MM-DD

_schema:
  required: [description, type, status, topology, created]
  optional: [properties, updated]
  enums:
    type: [topology]
    status: [active, experimental]
    topology: [ring, random, small-world, complete, scale-free, star, grid]
  constraints:
    description: "max 200 chars, no trailing period"
---

# prose-as-title describing the topology's behavioral signature

## Structure
## Parameters
## Behavioral signature
## Evidence
## Comparison with other topologies

---

Topics:
- [[_index]]
```

### 2e: `vault/templates/failure-note.md`

```yaml
---
description: One sentence naming the failure pattern and its consequence (~150 chars)
type: failure
status: active | mitigated | resolved
severity: critical | high | medium | low
attack_vector: exploitation | coordination | evasion | resource-exhaustion | information-asymmetry
affected_lever: circuit-breaker | transaction-tax | staking | reputation-decay | audit | collusion-detection | vote-normalization
source_run: "run_id"

created: YYYY-MM-DD
updated: YYYY-MM-DD

_schema:
  required: [description, type, status, severity, attack_vector, affected_lever, source_run, created]
  optional: [updated]
  enums:
    type: [failure]
    status: [active, mitigated, resolved]
    severity: [critical, high, medium, low]
    attack_vector: [exploitation, coordination, evasion, resource-exhaustion, information-asymmetry]
    affected_lever: [circuit-breaker, transaction-tax, staking, reputation-decay, audit, collusion-detection, vote-normalization]
  constraints:
    description: "max 200 chars, no trailing period"
    source_run: "must reference a valid run_id"
---

# prose-as-title naming the failure as a pattern

## Description
## Impact
## Reproduction
## Mitigation
## Related patterns

---

Topics:
- [[_index]]
```

### 2f: `vault/templates/sweep-summary.md`

```yaml
---
description: One sentence summarizing the sweep series and its convergence status (~150 chars)
type: sweep-summary
status: active | superseded
parameter: "dotted.parameter.name"
values_tested: []

created: YYYY-MM-DD
updated: YYYY-MM-DD

_schema:
  required: [description, type, status, parameter, created]
  optional: [values_tested, updated]
  enums:
    type: [sweep-summary]
    status: [active, superseded]
  constraints:
    description: "max 200 chars, no trailing period"
    parameter: "dotted path matching SWARM config"
---

# prose-as-title describing what this sweep series has established

## Runs in this sweep
## Convergence
## Phase transition surface
## Claims supported
## Next decisive experiment

---

Topics:
- [[_index]]
```

### 2g: `vault/templates/method-note.md`

```yaml
---
description: One sentence explaining what this method does and when to use it (~150 chars)
type: method
status: active | experimental | deprecated
category: statistical | simulation | analysis | visualization | governance-design

created: YYYY-MM-DD
updated: YYYY-MM-DD

_schema:
  required: [description, type, status, category, created]
  optional: [updated]
  enums:
    type: [method]
    status: [active, experimental, deprecated]
    category: [statistical, simulation, analysis, visualization, governance-design]
  constraints:
    description: "max 200 chars, no trailing period"
---

# prose-as-title describing the method as a reusable practice

## What it does
## Why it matters
## How to apply it
## Assumptions and limitations
## Evidence of effectiveness

---

Topics:
- [[_index]]
```

---

## Step 3: Install Vocabulary and Categories

### 3a: Write `vault/vocabulary.yaml`

```yaml
vocabulary:
  note: claim
  reduce: extract findings
  reflect: cross-link evidence
  verify: validate provenance
  reweave: update prior claims
  moc: topic map
  description_field: claim summary
  topics_footer: research areas
  inbox: new runs
  pipeline: synthesis pipeline
  wiki_link: evidence link
  thinking_notes: claims
  archive: processed runs
  self_space: research identity
  orient: load run context
  persist: commit to vault
  experiment: experiment
  sweep: parameter sweep
  redteam: adversarial evaluation
  governance: governance mechanism
  topology: network topology
  failure: failure pattern
  calibration: hyperparameter calibration
  canary: regression canary
  drift: behavioral drift
  claim_card: claim card
  run_yaml: run metadata envelope
  provenance: provenance chain

folders:
  notes: vault
  inbox: runs
  archive: runs
  experiments: vault/experiments
  claims: vault/claims
  governance: vault/governance
  topologies: vault/topologies
  failures: vault/failures
  methods: vault/methods
  sweeps: vault/sweeps

templates:
  base_note: claim-card.md
  experiment: experiment-note.md
  claim: claim-card.md
  governance: governance-note.md
  topology: topology-note.md
  failure: failure-note.md
  sweep: sweep-summary.md
  method: method-note.md
  moc: _index.md

skills:
  reduce: /reduce
  reflect: /reflect
  verify: /verify
  reweave: /reweave
```

### 3b: Write `vault/categories.yaml`

```yaml
extraction_categories:
  - claims
  - evidence
  - boundary-conditions
  - governance-effects
  - topology-effects
  - failure-patterns
  - contradictions
  - open-questions
  - phase-transitions
  - mechanism-interactions
  - methodology-notes
```

---

## Step 4: Generate Master Index

Write `vault/_index.md` — the root MOC linking all vault sections.

**Critical rules:**
- Scan `vault/claims/` for existing claim files → list them with descriptions
- Scan `vault/governance/` for existing governance notes → list them
- Scan `vault/topologies/` for existing topology notes → list them
- Scan `vault/failures/` for existing failure notes → list them
- If no notes exist yet in a section, show the directory path with "(empty — auto-populated by synthesis pipeline)"
- Always list all 7 template paths

**Template for `vault/_index.md`:**

```markdown
---
description: "Master index for SWARM Research OS knowledge vault — claims, governance, topologies, experiments"
type: moc
created: {TODAY}
---

# SWARM Research OS — Knowledge Vault

## Claims

Active findings from SWARM experiments, each linked to run evidence.

{FOR EACH .md IN vault/claims/: "- [[{filename}]] — {description from frontmatter}"}

## Governance Mechanisms

Notes on individual governance levers.

{FOR EACH .md IN vault/governance/: "- [[{filename}]] — {description from frontmatter}"}

## Topologies

Network structure notes.

{FOR EACH .md IN vault/topologies/: "- [[{filename}]] — {description from frontmatter}"}

## Failure Patterns

Named vulnerability and attack patterns.

{FOR EACH .md IN vault/failures/: "- [[{filename}]] — {description from frontmatter}"}

## Experiments

One note per run, auto-generated from run.yaml.

- `vault/experiments/` — {COUNT} notes

## Sweeps

Cross-run aggregations tracking parameter sweep convergence.

{FOR EACH .md IN vault/sweeps/: "- [[{filename}]] — {description from frontmatter}"}

## Methods

Reusable statistical and experimental methods.

{FOR EACH .md IN vault/methods/: "- [[{filename}]] — {description from frontmatter}"}

## Templates

Schema-enforced templates for all note types:

- `vault/templates/experiment-note.md`
- `vault/templates/claim-card.md`
- `vault/templates/governance-note.md`
- `vault/templates/topology-note.md`
- `vault/templates/failure-note.md`
- `vault/templates/sweep-summary.md`
- `vault/templates/method-note.md`

<!-- topics: index, vault, research-os -->
```

**Replace `{TODAY}` with the current date in YYYY-MM-DD format.**
**Replace `{FOR EACH ...}` blocks with actual scanned results. If a directory is empty, write "(empty — auto-populated by synthesis pipeline)".**

---

## Step 5: Install Schemas (if missing)

Check if `schemas/` contains `run.schema.yaml`, `claim.schema.yaml`, `sweep.schema.yaml`.

If any are missing, read them from the plugin source at:
- `schemas/run.schema.yaml`
- `schemas/claim.schema.yaml`
- `schemas/sweep.schema.yaml`

Copy them to the vault's `schemas/` directory.

---

## Step 6: Install Scripts (if missing)

Check if `scripts/` contains the automation tooling.

If missing, read from plugin source and install:
- `scripts/backfill-run-yaml.py` — generates run.yaml for existing runs
- `scripts/index-runs.py` — rebuilds run-index.yaml
- `scripts/validate-run.py` — validates run folders against schemas

---

## Step 7: Backfill Existing Runs (conditional)

**Only execute if `--backfill` flag is set OR if `runs/` directory exists with >0 subdirectories that lack `run.yaml`.**

1. Count run folders in `runs/` that do NOT have a `run.yaml`
2. If count > 0, run: `python scripts/backfill-run-yaml.py`
3. Then run: `python scripts/index-runs.py`
4. Report: "{N} run.yaml files generated, index rebuilt with {M} total runs"

**If all runs already have run.yaml, skip and report "All runs already have metadata envelopes."**

---

## Step 8: Report

Print a structured summary of everything that was done:

```
## SWARM Research OS Vault — Initialized

### Directory structure
✓ vault/experiments/
✓ vault/claims/ ({N} notes)
✓ vault/governance/ ({N} notes)
✓ vault/topologies/ ({N} notes)
✓ vault/failures/ ({N} notes)
✓ vault/methods/ ({N} notes)
✓ vault/sweeps/ ({N} notes)

### Templates installed
✓ 7 templates in vault/templates/

### Configuration
✓ vault/vocabulary.yaml
✓ vault/categories.yaml

### Index
✓ vault/_index.md (master MOC)

### Schemas
✓ schemas/run.schema.yaml
✓ schemas/claim.schema.yaml
✓ schemas/sweep.schema.yaml

### Scripts
✓ scripts/backfill-run-yaml.py
✓ scripts/index-runs.py
✓ scripts/validate-run.py

### Runs
{IF backfill ran}: ✓ {N} run.yaml files generated, {M} total runs indexed
{ELSE}: ○ Run backfill skipped (use --backfill to generate run.yaml for existing runs)

### Next steps
- Add claims: create files in vault/claims/ using vault/templates/claim-card.md
- Add governance notes: create files in vault/governance/ using vault/templates/governance-note.md
- Run synthesis: when new SWARM runs complete, the Inngest pipeline auto-generates experiment notes
- Validate: python scripts/validate-run.py --all
```

---

## Critical Constraints

1. **Never overwrite existing notes** in claims/, governance/, topologies/, failures/, methods/, sweeps/ — these contain human-authored or synthesis-generated content. Only overwrite templates and config files when `--force` is set.

2. **Templates are the single source of truth for schema.** The `_schema` block in each template defines what validation checks. Do not create notes that violate their template's schema.

3. **Every note must have a description field** that adds information beyond the title. This is a kernel invariant (primitive #5: description-field).

4. **Every non-MOC note must have a Topics footer** with at least one wiki-link to a parent MOC. This is a kernel invariant (primitive #6: topics-footer).

5. **Prose-as-title convention.** Note filenames and H1 headings are lowercase prose propositions. The composability test: can you complete "This note argues that [title]"?

6. **Wiki-links are used inline as prose**, not as bare references. Example: "Since [[circuit breakers alone outperform full governance stacks]], the question becomes..."

7. **All dates use YYYY-MM-DD format.** All timestamps use ISO 8601 UTC.
